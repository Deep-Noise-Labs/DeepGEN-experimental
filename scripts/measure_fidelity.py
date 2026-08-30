#!/usr/bin/env python
"""
Reproduce the measurements in docs/AUDIO_FIDELITY.md.

    uv run python scripts/measure_fidelity.py
    uv run python scripts/measure_fidelity.py --audio path/to/stereo.wav

Table 1 quantifies the aliasing a Snake nonlinearity folds back into the
audible band, at the native rate versus oversampled.

Table 2 scores three reconstructions that a magnitude-only spectral objective
regards as perfect or near-perfect, under both objectives. It runs on a
synthetic stereo signal by default; pass --audio to use real material.
"""

from __future__ import annotations

import argparse
import math

import torch

from synthgen.model.vae import AliasFreeSnake, SnakeBeta
from synthgen.training.losses import MultiResolutionSTFTLoss

SAMPLE_RATE = 44100
STFT_KWARGS = dict(
    fft_sizes=(2048, 1024, 512, 256),
    hop_sizes=(512, 256, 128, 64),
    win_sizes=(2048, 1024, 512, 256),
)


def band_energy_db(signal: torch.Tensor, freq: float, bandwidth: float = 60.0) -> float:
    x = signal.detach().reshape(-1)
    window = torch.hann_window(x.numel())
    spectrum = torch.fft.rfft(x * window).abs()
    freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / SAMPLE_RATE)
    band = (freqs >= freq - bandwidth) & (freqs <= freq + bandwidth)
    return 10 * math.log10(float(spectrum[band].pow(2).sum()) + 1e-30)


def measure_aliasing() -> None:
    fundamental = 9000.0
    n = SAMPLE_RATE // 2
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    tone = (3.0 * torch.sin(2 * math.pi * fundamental * t)).view(1, 1, n)

    native = SnakeBeta(1)
    oversampled = AliasFreeSnake(1)
    # Identical nonlinearity parameters: the only difference is oversampling.
    oversampled.activation.load_state_dict(native.state_dict())

    with torch.no_grad():
        before, after = native(tone), oversampled(tone)

    bands = [
        (fundamental, "Fundamental, 9 kHz"),
        (2 * fundamental, "2nd harmonic, 18 kHz"),
        (SAMPLE_RATE - 4 * fundamental, "4th harmonic folding 36 kHz -> 8.1 kHz"),
        (SAMPLE_RATE - 3 * fundamental, "3rd harmonic folding 27 kHz -> 17.1 kHz"),
    ]

    print("\nTable 1: aliasing from one Snake activation on a 9 kHz tone\n")
    print(f"{'Band':44s} {'Native':>9s} {'Oversamp':>9s} {'Change':>9s}")
    for freq, label in bands:
        b, a = band_energy_db(before, freq), band_energy_db(after, freq)
        print(f"{label:44s} {b:8.1f}dB {a:8.1f}dB {a - b:+8.1f}dB")


def synthetic_stereo(seconds: float = 3.0) -> torch.Tensor:
    n = int(SAMPLE_RATE * seconds)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    left = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate([220, 440, 660, 880], 1))
    right = sum(torch.cos(2 * math.pi * f * t) / k for k, f in enumerate([221, 442, 663, 884], 1))
    audio = torch.stack([left, right]).unsqueeze(0)
    return audio / audio.abs().max() * 0.75


def measure_objective(target: torch.Tensor) -> None:
    magnitude_only = MultiResolutionSTFTLoss(
        phase_weight=0.0, stereo_weight=0.0, **STFT_KWARGS
    )
    perceptual = MultiResolutionSTFTLoss(**STFT_KWARGS)

    polarity = target.clone()
    polarity[:, 1] *= -1

    torch.manual_seed(0)
    spectrum = torch.fft.rfft(target, dim=-1)
    phase = torch.rand(spectrum.shape[-1]) * 2 * math.pi
    smoothing = torch.ones(1, 1, 129) / 129
    phase = torch.nn.functional.conv1d(
        phase.view(1, 1, -1), smoothing, padding=64
    ).view(-1) * 40
    dispersed = torch.fft.irfft(
        spectrum * torch.polar(torch.ones_like(phase), phase),
        n=target.shape[-1], dim=-1,
    )

    cases = [
        ("Right channel polarity inverted", polarity),
        ("Phase dispersed (all-pass)", dispersed),
        ("Left/right swapped", target.flip(dims=[1])),
    ]

    print("\nTable 2: what the magnitude-only objective scores these at\n")
    print(f"{'Reconstruction':36s} {'Magnitude-only':>15s} {'With phase + M/S':>18s}")
    with torch.no_grad():
        for label, candidate in cases:
            old = float(magnitude_only(candidate, target))
            new = float(perceptual(candidate, target))
            print(f"{label:36s} {old:15.4f} {new:18.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, default=None,
                        help="Stereo audio file to score in Table 2")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    measure_aliasing()

    if args.audio:
        from synthgen.utils.audio import load_audio

        audio = load_audio(
            args.audio, sample_rate=SAMPLE_RATE, channels=2,
            offset=args.offset, duration=args.duration,
        )
        target = torch.from_numpy(audio.copy()).unsqueeze(0)
        target = target / target.abs().max().clamp_min(1e-9) * 0.75
    else:
        target = synthetic_stereo(args.duration)

    measure_objective(target)


if __name__ == "__main__":
    main()
