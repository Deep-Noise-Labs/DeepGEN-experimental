"""
Choose the anti-imaging filter's length and transition width from data.

    PYTHONPATH=. python experiments/upsampler_filter_ablation.py \
        --input work/corpus --out proofs/upsampler

Sweeps taps-per-stride against transition half-width on real renders and
reports, for each pair: image-to-signal ratio, reconstruction SNR against a
band-limited reference, pass-band droop, and filter length.

This exists because the first two attempts at picking these numbers by
reasoning were both wrong. The reasoning was that a wide transition must
dull the pass-band, so the filter should be long and sharp. In fact a wider
transition raises the Kaiser beta and deepens the stop-band far faster than
the droop costs, and the apparent fidelity penalty of filtering turned out
to be the half-sample delay of an even-length FIR rather than droop at all.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import firwin, resample_poly

from synthgen.model.resample import (
    BandlimitedUpsample1d,
    TransposedUpsample1d,
    resampling_filter,
)

from experiments.upsampler_audio_proof import (
    STRIDE,
    band_energy_db,
    set_linear_interp,
    snr_db,
)

TAPS_PER_STRIDE = (4, 8, 16, 32, 64)
TRANSITIONS = (0.15, 0.25, 0.5)


def prepare(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T
    edge = sr / (2 * STRIDE * 2)
    taps = firwin(511, edge / (sr / 2), window=("kaiser", 8.0))
    reference = np.stack([np.convolve(ch, taps, mode="same") for ch in audio])
    decimated = np.stack([resample_poly(ch, 1, STRIDE) for ch in reference])
    return reference, decimated, sr, edge


def passband_droop_db(stride: int, kernel_size: int, transition: float) -> float:
    taps = resampling_filter(stride, kernel_size, transition)[0, 0].to(torch.float64)
    response = torch.fft.rfft(taps, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192)
    band = response[freqs < (0.5 / stride) * 0.8]
    return float(20 * np.log10((band.min() / band.max()).item()))


def spectral_split(reference: np.ndarray, test: np.ndarray, sr: int, edge: float) -> tuple[float, float]:
    """
    Split the comparison into the two questions that actually matter.

    Returns ``(in_band_fidelity_db, out_of_band_energy_db)``:

    - **in-band fidelity** -- error against the reference *inside* the band
      the reference occupies, after a single complex scale. This is what the
      filter could damage, and the number that decides whether it is free.
    - **out-of-band energy** -- everything above the reference's band,
      relative to in-band signal. The reference has nothing there, so this is
      pure invented image content.

    Reporting one aggregate SNR instead of these two conflates "the filter
    dulled the signal" with "the operator invented content", which are
    opposite failures and move in opposite directions.
    """
    trim = 1024
    ref = reference[trim:-trim]
    n = len(ref)
    test = np.roll(test, -2)[trim : trim + n]
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec_ref = np.fft.rfft(ref * window)
    spec_test = np.fft.rfft(test * window)

    alpha = np.vdot(spec_ref, spec_test) / np.vdot(spec_ref, spec_ref)
    in_band = freqs < edge
    out_band = freqs > edge * 1.25

    signal = np.sum(np.abs(alpha * spec_ref[in_band]) ** 2)
    error = np.sum(np.abs(spec_test[in_band] - alpha * spec_ref[in_band]) ** 2)
    out = np.sum(np.abs(spec_test[out_band]) ** 2)
    return (
        float(10 * np.log10(signal / (error + 1e-30))),
        float(10 * np.log10((out + 1e-30) / signal)),
    )


def evaluate(clips, taps_per_stride: int | None, transition: float) -> dict:
    """``taps_per_stride=None`` measures the unfiltered baseline."""
    isrs, snrs = [], []
    for reference, decimated, sr, edge in clips:
        channels = decimated.shape[0]
        baseline = TransposedUpsample1d(channels, channels, STRIDE)
        set_linear_interp(baseline, STRIDE)

        if taps_per_stride is None:
            module = baseline
        else:
            module = BandlimitedUpsample1d(
                channels,
                channels,
                STRIDE,
                filter_kernel_size=taps_per_stride * STRIDE,
                transition=transition,
            )
            with torch.no_grad():
                module.weight.copy_(baseline.conv.weight)
                module.bias.copy_(baseline.conv.bias)

        with torch.no_grad():
            y = module(torch.from_numpy(decimated).float().unsqueeze(0))[0].numpy()

        n = min(reference.shape[-1], y.shape[-1])
        for ch in range(channels):
            fidelity, out_of_band = spectral_split(reference[ch, :n], y[ch, :n], sr, edge)
            snrs.append(fidelity)
            isrs.append(out_of_band)

    row = {
        "taps_per_stride": taps_per_stride,
        "transition": transition if taps_per_stride is not None else None,
        "out_of_band_db": float(np.mean(isrs)),
        "in_band_fidelity_db": float(np.mean(snrs)),
    }
    if taps_per_stride is None:
        row["filter_taps"] = 0
        row["passband_droop_db"] = 0.0
    else:
        row["filter_taps"] = int(
            resampling_filter(STRIDE, taps_per_stride * STRIDE, transition).shape[-1]
        )
        row["passband_droop_db"] = passband_droop_db(
            STRIDE, taps_per_stride * STRIDE, transition
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="proofs/upsampler")
    args = parser.parse_args()

    paths = sorted(Path(args.input).glob("*.ogg")) + sorted(Path(args.input).glob("*.wav"))
    if not paths:
        raise SystemExit(f"no audio under {args.input}")
    clips = [prepare(p) for p in paths]

    rows = [evaluate(clips, None, 0.0)]
    for taps, transition in itertools.product(TAPS_PER_STRIDE, TRANSITIONS):
        rows.append(evaluate(clips, taps, transition))

    print(
        f"{'taps/s':>7} {'trans':>6} {'N':>5} {'out-of-band':>12} "
        f"{'in-band fid':>12} {'droop':>7}"
    )
    for r in rows:
        taps = "none" if r["taps_per_stride"] is None else str(r["taps_per_stride"])
        trans = "-" if r["transition"] is None else f"{r['transition']:.2f}"
        print(
            f"{taps:>7} {trans:>6} {r['filter_taps']:>5} "
            f"{r['out_of_band_db']:>12.1f} {r['in_band_fidelity_db']:>12.1f} "
            f"{r['passband_droop_db']:>7.2f}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "filter_ablation.json").write_text(
        json.dumps({"stride": STRIDE, "clips": len(clips), "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
