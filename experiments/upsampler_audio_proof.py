"""
Audible before/after for the decoder's upsampler, on real Deep Noise renders.

    PYTHONPATH=. python experiments/upsampler_audio_proof.py \
        --input work/corpus --out proofs/upsampler

What this does, and what it does not do
---------------------------------------
It isolates **one operator**: the decoder's final upsampling stage, which
takes the signal from fs/4 to fs. It does not run a trained model, because no
DeepGEN checkpoint exists. Nothing here should be described as "the model
sounds better".

The experiment:

1. Take a real render. Low-pass it at the rate the decimated representation
   can actually carry (fs/8) -- this band-limited signal is the *reference*,
   the best any 4x upsampler could return.
2. Decimate by 4 with a high-quality filter. This stands in for the signal
   arriving at the decoder's last stage.
3. Upsample back by 4 through both arms, with **identical weights** set to
   linear interpolation -- the best a 2-tap-per-phase kernel can do smoothly,
   so the baseline is given its strongest form rather than its weakest.
4. Everything the output contains above fs/8 is an image the upsampler
   invented, because the reference has nothing there. That residual is
   measured, and written out as audio so it can be heard on its own.

Attributability guard, inherited from the anti-aliasing session: an artefact
number measured through a reconstruction that is itself broken says nothing.
Each clip therefore carries the SNR of its own reconstruction against the
reference, and readings are marked ``attributable: false`` below 10 dB.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import firwin, resample_poly, sosfiltfilt, butter

from synthgen.model.resample import BandlimitedUpsample1d, TransposedUpsample1d

STRIDE = 4
ATTRIBUTABLE_SNR_DB = 10.0


def set_linear_interp(module, stride: int) -> None:
    with torch.no_grad():
        w = module.conv.weight if hasattr(module, "conv") else module.weight
        b = module.conv.bias if hasattr(module, "conv") else module.bias
        taps = 1.0 - torch.abs(torch.arange(2 * stride, dtype=torch.float32) - stride) / stride
        w.fill_(0.0)
        for i in range(w.shape[0]):
            w[i, i, :] = taps
        b.fill_(0.0)


def band_energy_db(x: np.ndarray, sr: int, low: float, high: float | None) -> float:
    spec = np.fft.rfft(x * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    mask = freqs >= low
    if high is not None:
        mask &= freqs < high
    return float(10 * np.log10(np.sum(np.abs(spec[mask]) ** 2) + 1e-30))


def _snr_at(reference: np.ndarray, test: np.ndarray) -> float:
    # Scale-invariant: the arms may differ by a constant gain.
    alpha = np.dot(test, reference) / (np.dot(reference, reference) + 1e-20)
    err = test - alpha * reference
    return float(
        10 * np.log10((np.sum((alpha * reference) ** 2) + 1e-20) / (np.sum(err**2) + 1e-20))
    )


def snr_db(
    reference: np.ndarray,
    test: np.ndarray,
    max_shift: int = 96,
    trim: int = 1024,
) -> tuple[float, int]:
    """
    Scale-invariant SNR, maximised over integer delay.

    The alignment search is not optional. Both arms are linear-phase FIRs
    with their own group delay, and the band-limiting filter's delay is
    larger. Comparing un-aligned waveforms measures the delay difference, not
    the fidelity difference -- and it does so in a way that happens to
    flatter the filtered arm, which is precisely the wrong direction for an
    honest result.
    """
    n = min(len(reference), len(test))
    reference, test = reference[:n], test[:n]
    # Trim generously at both ends. A long FIR rings for half its length at a
    # clip boundary, so a short trim charges long filters for an edge
    # transient and makes a 17-tap filter look more faithful than a 257-tap
    # one -- which is backwards.
    edge = max(max_shift + 1, trim)
    best = (-np.inf, 0)
    for shift in range(-max_shift, max_shift + 1):
        candidate = np.roll(test, -shift)
        value = _snr_at(reference[edge:-edge], candidate[edge:-edge])
        if value > best[0]:
            best = (value, shift)
    return float(best[0]), int(best[1])


def highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(8, cutoff / (sr / 2), btype="highpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def process_clip(path: Path, out_dir: Path) -> dict:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T  # (channels, samples)

    # 1. Reference: band-limited to what the decimated signal can carry.
    edge = sr / (2 * STRIDE * 2)  # fs/16 in Hz for stride 4 => fs/8 band
    taps = firwin(511, edge / (sr / 2), window=("kaiser", 8.0))
    reference = np.stack([np.convolve(ch, taps, mode="same") for ch in audio])

    # 2. Decimate by STRIDE with a high-quality polyphase filter.
    decimated = np.stack([resample_poly(ch, 1, STRIDE) for ch in reference])

    # 3. Upsample back through both arms with identical weights.
    channels = decimated.shape[0]
    baseline = TransposedUpsample1d(channels, channels, STRIDE)
    fixed = BandlimitedUpsample1d(channels, channels, STRIDE)
    set_linear_interp(baseline, STRIDE)
    with torch.no_grad():
        fixed.weight.copy_(baseline.conv.weight)
        fixed.bias.copy_(baseline.conv.bias)

    x = torch.from_numpy(decimated).float().unsqueeze(0)
    with torch.no_grad():
        y_base = baseline(x)[0].numpy()
        y_fixed = fixed(x)[0].numpy()

    n = min(reference.shape[-1], y_base.shape[-1], y_fixed.shape[-1])
    reference, y_base, y_fixed = reference[:, :n], y_base[:, :n], y_fixed[:, :n]

    # 4. Measure. Everything above `edge` is invented.
    row: dict = {"clip": path.stem, "sample_rate": sr, "band_edge_hz": edge}
    for name, y in (("baseline", y_base), ("fixed", y_fixed)):
        isr = []
        snrs = []
        shifts = []
        for ch in range(channels):
            above = band_energy_db(y[ch], sr, edge * 1.25, None)
            below = band_energy_db(y[ch], sr, 0.0, edge * 0.8)
            isr.append(above - below)
            value, shift = snr_db(reference[ch], y[ch])
            snrs.append(value)
            shifts.append(shift)
        row[f"{name}_image_to_signal_db"] = float(np.mean(isr))
        row[f"{name}_snr_db"] = float(np.mean(snrs))
        row[f"{name}_delay_samples"] = shifts

    row["improvement_db"] = row["baseline_image_to_signal_db"] - row["fixed_image_to_signal_db"]
    row["attributable"] = bool(
        row["baseline_snr_db"] > ATTRIBUTABLE_SNR_DB
        and row["fixed_snr_db"] > ATTRIBUTABLE_SNR_DB
    )

    # 5. Write audio: the three takes, plus each arm's isolated artefact.
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def write(tag: str, data: np.ndarray) -> str:
        name = f"{path.stem}__{tag}.ogg"
        sf.write(str(audio_dir / name), data.T, sr, format="OGG", subtype="VORBIS")
        return name

    peak = max(np.abs(reference).max(), 1e-9)
    row["files"] = {
        "reference": write("1_reference", reference / peak * 0.9),
        "baseline": write("2_baseline", y_base / peak * 0.9),
        "fixed": write("3_bandlimited", y_fixed / peak * 0.9),
    }

    # The artefact on its own. Written at the SAME gain as the takes above so
    # the level is honest, and again normalised so it can be heard at all.
    art_base = np.stack([highpass(y_base[c], sr, edge * 1.25) for c in range(channels)])
    art_fixed = np.stack([highpass(y_fixed[c], sr, edge * 1.25) for c in range(channels)])
    row["files"]["baseline_artefact"] = write("4_baseline_artefact", art_base / peak * 0.9)
    row["files"]["fixed_artefact"] = write("5_bandlimited_artefact", art_fixed / peak * 0.9)

    art_peak = max(np.abs(art_base).max(), 1e-9)
    row["artefact_boost_db"] = float(20 * math.log10(peak / art_peak))
    row["files"]["baseline_artefact_amplified"] = write(
        "6_baseline_artefact_amplified", art_base / art_peak * 0.9
    )
    row["files"]["fixed_artefact_amplified"] = write(
        "7_bandlimited_artefact_amplified", art_fixed / art_peak * 0.9
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="proofs/upsampler")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(Path(args.input).glob("*.ogg")) + sorted(Path(args.input).glob("*.wav"))
    if not clips:
        raise SystemExit(f"no audio under {args.input}")

    rows = [process_clip(c, out) for c in clips]
    for r in rows:
        print(
            f"{r['clip']:>14}  baseline {r['baseline_image_to_signal_db']:7.1f} dB  "
            f"fixed {r['fixed_image_to_signal_db']:7.1f} dB  "
            f"gain {r['improvement_db']:6.1f} dB  "
            f"snr {r['baseline_snr_db']:5.1f}/{r['fixed_snr_db']:5.1f} dB  "
            f"{'ok' if r['attributable'] else 'NOT ATTRIBUTABLE'}"
        )

    good = [r for r in rows if r["attributable"]]
    summary = {
        "stride": STRIDE,
        "clips": len(rows),
        "attributable": len(good),
        "mean_improvement_db": float(np.mean([r["improvement_db"] for r in good])) if good else None,
        "mean_baseline_image_to_signal_db": float(
            np.mean([r["baseline_image_to_signal_db"] for r in good])
        )
        if good
        else None,
        "mean_fixed_image_to_signal_db": float(
            np.mean([r["fixed_image_to_signal_db"] for r in good])
        )
        if good
        else None,
        "rows": rows,
    }
    (out / "audio_proof.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean improvement over {len(good)} attributable clips: {summary['mean_improvement_db']:.1f} dB")


if __name__ == "__main__":
    main()
