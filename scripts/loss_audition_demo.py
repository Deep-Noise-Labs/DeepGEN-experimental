"""
Audition demo for the VAE reconstruction loss upgrade.

Renders three deterministic degradation pairs that professional sample
libraries would reject, then scores each pair with the legacy spectral loss
(256-2048 windows, per-channel only) and the upgraded loss (64-8192 windows
plus mid/side term). Every pair is scored against an inaudible control
(+0.1 dB gain) so the numbers read as "how much louder than an inaudible
change does the objective think this error is".

The three cases:

1. sub_pitch      - an 808 sub bass played one semitone flat.
2. attack_smear   - a pluck whose attack transient is destroyed by phase
                    randomisation inside 2048-sample STFT windows (magnitude
                    at that resolution is preserved by construction).
3. stereo_collapse - a wide supersaw pad collapsed to mono.

No trained model is involved: this measures what the training objective
itself can and cannot hear. Usage:

    uv run python scripts/loss_audition_demo.py --out demo_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from synthgen.training.losses import MultiResolutionSTFTLoss

SAMPLE_RATE = 44100
DURATION = 3.0
N = int(SAMPLE_RATE * DURATION)

LEGACY_KWARGS = dict(
    fft_sizes=(2048, 1024, 512, 256),
    hop_sizes=(512, 256, 128, 64),
    win_sizes=(2048, 1024, 512, 256),
    sum_and_difference=False,
    log_eps=1e-8,
)


def peak_normalize(x: np.ndarray, peak_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x * (10 ** (peak_db / 20.0) / peak)
    return x.astype(np.float32)


def render_808(f0: float, seed: int = 0) -> np.ndarray:
    """808-style sub bass: pitch glide, exponential decay, mild saturation."""
    t = np.arange(N) / SAMPLE_RATE
    glide = f0 * (1.0 + 1.5 * np.exp(-t / 0.04))
    phase = 2 * np.pi * np.cumsum(glide) / SAMPLE_RATE
    env = np.minimum(t / 0.005, 1.0) * np.exp(-t / 1.2)
    mono = np.tanh(1.5 * np.sin(phase)) * env
    return np.stack([mono, mono])


def render_pluck(seed: int = 1) -> np.ndarray:
    """Bright pluck: noise click plus decaying detuned saw partials, 3 hits."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SAMPLE_RATE
    mono = np.zeros(N)
    for onset in (0.1, 1.1, 2.1):
        gate = t >= onset
        tt = np.where(gate, t - onset, 0.0)
        hit = np.zeros(N)
        # Attack click: 4 ms of noise with a sharp decay
        click = rng.standard_normal(N) * np.exp(-tt / 0.001) * gate
        hit += 0.6 * click
        # Saw-like partial stack at 220 Hz
        for k in range(1, 24):
            hit += (
                (1.0 / k)
                * np.sin(2 * np.pi * 220.0 * k * tt + rng.uniform(0, 2 * np.pi))
                * np.exp(-tt / (0.35 / np.sqrt(k)))
                * gate
            )
        mono += hit
    return np.stack([mono, mono])


def phase_scramble(audio: np.ndarray, n_fft: int = 2048, seed: int = 2) -> np.ndarray:
    """Randomise STFT phases per channel; magnitudes at n_fft are preserved."""
    rng = np.random.default_rng(seed)
    hop = n_fft // 4
    window = torch.hann_window(n_fft)
    out = []
    for ch in audio:
        spec = torch.stft(
            torch.from_numpy(ch.astype(np.float32)),
            n_fft, hop, n_fft, window, return_complex=True,
        )
        phases = torch.from_numpy(
            rng.uniform(0, 2 * np.pi, spec.shape).astype(np.float32)
        )
        scrambled = spec.abs() * torch.exp(1j * phases)
        rec = torch.istft(scrambled, n_fft, hop, n_fft, window, length=len(ch))
        out.append(rec.numpy())
    return np.stack(out)


def render_supersaw(seed: int = 3) -> np.ndarray:
    """Wide supersaw pad: 7 detuned saws panned across the stereo field."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SAMPLE_RATE
    left = np.zeros(N)
    right = np.zeros(N)
    detunes_cents = np.linspace(-25, 25, 7)
    pans = np.linspace(-1.0, 1.0, 7)
    env = np.minimum(t / 0.4, 1.0) * np.minimum((DURATION - t) / 0.4, 1.0)
    for cents, pan in zip(detunes_cents, pans):
        f = 110.0 * 2 ** (cents / 1200.0)
        voice = np.zeros(N)
        for k in range(1, 40):
            if f * k >= SAMPLE_RATE / 2:
                break
            voice += (1.0 / k) * np.sin(
                2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)
            )
        voice *= env
        left += voice * np.sqrt((1 - pan) / 2)
        right += voice * np.sqrt((1 + pan) / 2)
    return np.stack([left, right])


def mono_collapse(audio: np.ndarray) -> np.ndarray:
    mid = 0.5 * (audio[0] + audio[1])
    return np.stack([mid, mid])


def score_pair(
    loss_fn: MultiResolutionSTFTLoss,
    degraded: np.ndarray,
    reference: np.ndarray,
) -> float:
    pred = torch.from_numpy(degraded)[None]
    target = torch.from_numpy(reference)[None]
    return float(loss_fn(pred, target).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="demo_out")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    legacy = MultiResolutionSTFTLoss(**LEGACY_KWARGS)
    upgraded = MultiResolutionSTFTLoss()

    cases = {}

    ref_808 = peak_normalize(render_808(41.2))  # E1
    flat_808 = peak_normalize(render_808(41.2 * 2 ** (-1 / 12)))  # one semitone flat
    cases["sub_pitch"] = (ref_808, flat_808)

    ref_pluck = peak_normalize(render_pluck())
    smeared = peak_normalize(phase_scramble(ref_pluck))
    cases["attack_smear"] = (ref_pluck, smeared)

    ref_saw = peak_normalize(render_supersaw())
    collapsed = peak_normalize(mono_collapse(ref_saw))
    cases["stereo_collapse"] = (ref_saw, collapsed)

    results = {}
    for name, (reference, degraded) in cases.items():
        control = reference * 10 ** (0.1 / 20.0)  # +0.1 dB: inaudible
        row = {}
        for label, loss_fn in (("legacy", legacy), ("upgraded", upgraded)):
            d = score_pair(loss_fn, degraded, reference)
            c = score_pair(loss_fn, control, reference)
            row[label] = {
                "loss_degraded": d,
                "loss_control": c,
                "discrimination": d / c if c > 0 else float("inf"),
            }
        # The current objective's only phase-sensitive term, at its weight
        row["l1_term_at_weight"] = 0.1 * float(
            F.l1_loss(
                torch.from_numpy(degraded), torch.from_numpy(reference)
            ).item()
        )
        results[name] = row

        if sf is not None:
            sf.write(out_dir / f"{name}_reference.wav", reference.T, SAMPLE_RATE)
            sf.write(out_dir / f"{name}_degraded.wav", degraded.T, SAMPLE_RATE)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWAVs and metrics written to {out_dir}/")


if __name__ == "__main__":
    main()
