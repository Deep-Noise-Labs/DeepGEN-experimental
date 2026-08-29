#!/usr/bin/env python3
"""
Objective-blindness probe: what can the VAE objective actually hear?

Takes real audio, applies controlled degradations that mimic how a
magnitude-trained audio decoder fails, and scores each one with the legacy
objective and the proposed one. Anything an objective scores as cheap is inside
its optimum, so the decoder is free to converge there.

Writes the degraded WAVs so the numbers can be checked against ears.

    python scripts/vae_objective_probe.py path/to/a.wav path/to/b.wav \
        --output-dir ./probe_out

Background and results: docs/VAE_OBJECTIVE.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal as sps

# Runnable from a plain checkout, not only from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthgen.training.losses import (  # noqa: E402
    MultiResolutionSTFTLoss,
    MultiScaleMelSpectrogramLoss,
)


# --------------------------------------------------------------------- audio --
def load(path: Path, sample_rate: int, duration: float, offset: float) -> np.ndarray:
    audio, file_sr = sf.read(str(path), always_2d=True, dtype="float64")
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if file_sr != sample_rate:
        g = np.gcd(int(file_sr), sample_rate)
        audio = sps.resample_poly(audio, sample_rate // g, file_sr // g, axis=-1)

    n = int(duration * sample_rate)
    start = int(offset * sample_rate)
    audio = audio[:, start:start + n]
    if audio.shape[-1] < n:
        audio = np.pad(audio, ((0, 0), (0, n - audio.shape[-1])))

    fade = int(0.005 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    audio[:, :fade] *= ramp
    audio[:, -fade:] *= ramp[::-1]
    return audio / (np.max(np.abs(audio)) + 1e-12) * 0.89


def save(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    out = audio.T.astype(np.float32)
    peak = np.max(np.abs(out))
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(path), out, sample_rate, subtype="PCM_16")


# -------------------------------------------------------------- degradations --
def hf_rolloff(x, sr, fc=9000.0, order=4):
    """The 'dull' failure. CONTROL: a magnitude loss should, and does, catch it."""
    b, a = sps.butter(order, fc / (sr / 2), btype="low")
    return sps.filtfilt(b, a, x, axis=-1)


def all_pass_dispersion(x, sr, strength=900.0):
    """
    The 'smeared / phasey' failure: frequency-dependent group delay applied by a
    pure all-pass. |H(f)| = 1 at every f, so the magnitude spectrum is unchanged
    by construction and only phase moves. strength=900 gives ~41 ms of HF lag.
    """
    n = x.shape[-1]
    nfft = 1 << int(math.ceil(math.log2(n * 2)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    w = freqs / (sr / 2)
    h = np.exp(1j * (-strength * (w ** 2) * np.pi))
    spec = np.fft.rfft(x, n=nfft, axis=-1)
    return np.fft.irfft(spec * h, n=nfft, axis=-1)[:, :n]


def stereo_collapse(x, sr, side_gain=0.15):
    """The 'narrow / mono-ish' failure."""
    mid = (x[0] + x[1]) / 2
    side = (x[0] - x[1]) / 2 * side_gain
    return np.stack([mid + side, mid - side])


def transient_soften(x, sr, attack_ms=30.0):
    """
    The 'no snap' failure. A CAUSAL one-pole reshapes the envelope so the attack
    genuinely lags - a zero-phase filter would smooth symmetrically and leave
    the peak where it was.
    """
    env = np.abs(sps.hilbert(x, axis=-1))
    a_fast = np.exp(-1.0 / (sr * 0.001))
    a_slow = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    env_fast = sps.lfilter([1 - a_fast], [1, -a_fast], env, axis=-1)
    env_slow = sps.lfilter([1 - a_slow], [1, -a_slow], env, axis=-1)
    ratio = np.clip((env_slow + 1e-5) / (env_fast + 1e-5), 0.05, 3.0)
    return x * ratio


DEGRADATIONS = [
    ("hf_rolloff", hf_rolloff, "HF roll-off @ 9 kHz - the 'dull' failure (CONTROL)"),
    ("dispersion", all_pass_dispersion, "All-pass smear ~41 ms - magnitude spectrum unchanged"),
    ("dispersion_mild", lambda x, sr: all_pass_dispersion(x, sr, 300.0), "All-pass smear ~14 ms"),
    ("stereo_collapse", stereo_collapse, "Side x0.15 - the 'narrow' failure"),
    ("transient_soften", transient_soften, "Attack slowed to ~30 ms - the 'no snap' failure"),
]


# -------------------------------------------------------------- diagnostics ---
def spectral_delta_db(pred, ref, sr):
    """Max deviation of the smoothed long-term magnitude spectrum, in dB."""
    p = np.abs(np.fft.rfft(pred, axis=-1)).mean(0)
    r = np.abs(np.fft.rfft(ref, axis=-1)).mean(0)
    k = 257
    p = np.convolve(p, np.ones(k) / k, mode="same")
    r = np.convolve(r, np.ones(k) / k, mode="same")
    freqs = np.fft.rfftfreq(pred.shape[-1], 1 / sr)
    band = (freqs > 40) & (freqs < 18000)
    return float(np.max(np.abs(20 * np.log10((p[band] + 1e-12) / (r[band] + 1e-12)))))


def stereo_width(x):
    side = (x[0] - x[1]) / 2
    mid = (x[0] + x[1]) / 2
    return float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))


def crest_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)))


# --------------------------------------------------------------------- main ---
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Reference audio files")
    parser.add_argument("--output-dir", default="./probe_out")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy = MultiResolutionSTFTLoss()
    proposed = MultiScaleMelSpectrogramLoss(
        sample_rate=args.sample_rate, mid_side=True
    )

    def score(pred, ref):
        p = torch.from_numpy(pred).float().unsqueeze(0)
        r = torch.from_numpy(ref).float().unsqueeze(0)
        with torch.no_grad():
            l1 = torch.nn.functional.l1_loss(p, r)
            return {
                # legacy VAELoss supervised terms: 0.1*L1 + 1.0*MRSTFT
                "legacy": float(0.1 * l1 + legacy(p, r)),
                "proposed": float(proposed(p, r)),
            }

    report = {"sample_rate": args.sample_rate, "sources": []}

    for path in args.inputs:
        path = Path(path)
        ref = load(path, args.sample_rate, args.duration, args.offset)
        save(out_dir / f"{path.stem}__reference.wav", ref, args.sample_rate)

        rows = []
        for key, fn, desc in DEGRADATIONS:
            deg = fn(ref, args.sample_rate)
            # RMS-match so gross level is never the thing being scored
            deg = deg * (
                np.sqrt(np.mean(ref ** 2)) / (np.sqrt(np.mean(deg ** 2)) + 1e-12)
            )
            save(out_dir / f"{path.stem}__{key}.wav", deg, args.sample_rate)
            row = {"key": key, "desc": desc, **score(deg, ref)}
            row["spectral_delta_db"] = spectral_delta_db(deg, ref, args.sample_rate)
            row["width"] = stereo_width(deg)
            row["crest_db"] = crest_db(deg)
            rows.append(row)

        # Absolute values of the two objectives are on different scales; only
        # their rankings are comparable. Normalise by each objective's own
        # penalty for the audible control.
        base_l = next(r["legacy"] for r in rows if r["key"] == "hf_rolloff")
        base_p = next(r["proposed"] for r in rows if r["key"] == "hf_rolloff")
        for row in rows:
            row["legacy_rel"] = row["legacy"] / base_l
            row["proposed_rel"] = row["proposed"] / base_p

        print(f"\n=== {path.name}  (width {stereo_width(ref):.3f}, "
              f"crest {crest_db(ref):.1f} dB)")
        print(f"{'degradation':<18}{'LEGACY':>9}{'rel':>7}{'PROPOSED':>10}{'rel':>7}"
              f"{'specD dB':>10}{'width':>8}{'crest':>8}")
        for row in rows:
            print(f"{row['key']:<18}{row['legacy']:>9.4f}{row['legacy_rel']:>7.2f}"
                  f"{row['proposed']:>10.4f}{row['proposed_rel']:>7.2f}"
                  f"{row['spectral_delta_db']:>10.2f}{row['width']:>8.3f}"
                  f"{row['crest_db']:>8.1f}")

        report["sources"].append({"file": str(path), "degradations": rows})

    with open(out_dir / "probe_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWAVs and probe_report.json -> {out_dir}")


if __name__ == "__main__":
    main()
