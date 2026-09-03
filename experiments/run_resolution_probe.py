"""
Experiment 2 -- can the loss tell a right bass note from a wrong one?

This isolates a single variable: the longest STFT window. Everything else about
the two losses is held constant.

At 44.1 kHz a 2048-sample window resolves 21.5 Hz per bin. E1 (41.20 Hz) and
F1 (43.65 Hz) are a semitone apart -- unmistakably a wrong note to any listener
-- but only 2.45 Hz apart in absolute terms, so both land in the same bin. A
loss built on 2048-sample windows therefore cannot represent the difference,
and a decoder trained under it gets no gradient telling it to fix bass tuning.
An 8192-sample window resolves 5.4 Hz and separates them.

The probe is synthetic on purpose: two pure tones are the cleanest way to show
a resolution limit, with no other spectral content to confound it. The response
is compared against a control pair an octave up (E3 vs F3, 164.8 vs 174.6 Hz),
where the same semitone is 9.8 Hz wide and both window sizes resolve it.

Usage::

    python experiments/run_resolution_probe.py --out <results-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synthgen.training.losses import MultiResolutionSTFTLoss, PerceptualSampleLoss

SR = 44100
DURATION = 2.0

# (name, correct Hz, wrong-by-a-semitone Hz, bin spacing needed)
NOTE_PAIRS = [
    ("E1 vs F1  (41.20 -> 43.65 Hz)", 41.20, 43.65),
    ("E2 vs F2  (82.41 -> 87.31 Hz)", 82.41, 87.31),
    ("E3 vs F3  (164.81 -> 174.61 Hz)", 164.81, 174.61),
]


def tone(freq: float, sr: int = SR, seconds: float = DURATION) -> torch.Tensor:
    """A stereo sine at ``freq``, with a soft fade to avoid edge clicks."""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    fade = int(sr * 0.01)
    window = np.ones_like(x)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    x = x * window
    return torch.from_numpy(np.stack([x, x])).float().unsqueeze(0)


def build_losses() -> dict:
    """Two MRSTFT losses differing only in window size, plus the new loss."""
    return {
        "mrstft_max2048": MultiResolutionSTFTLoss(
            fft_sizes=(2048, 1024, 512, 256),
            hop_sizes=(512, 256, 128, 64),
            win_sizes=(2048, 1024, 512, 256),
        ),
        "mrstft_max8192": MultiResolutionSTFTLoss(
            fft_sizes=(8192, 4096, 2048, 1024),
            hop_sizes=(2048, 1024, 512, 256),
            win_sizes=(8192, 4096, 2048, 1024),
        ),
        "perceptual": PerceptualSampleLoss(sample_rate=SR),
    }


def run(out_dir: Path) -> dict:
    losses = build_losses()
    results: list[dict] = []

    for label, correct_hz, wrong_hz in NOTE_PAIRS:
        correct = tone(correct_hz)
        wrong = tone(wrong_hz)
        # Reference point: the same tone at half amplitude. Any loss sees this,
        # so it calibrates "a difference this loss considers meaningful".
        quieter = correct * 0.5

        entry = {
            "pair": label,
            "correct_hz": correct_hz,
            "wrong_hz": wrong_hz,
            "separation_hz": round(wrong_hz - correct_hz, 2),
        }
        for name, fn in losses.items():
            wrong_note = float(fn(wrong, correct))
            gain_change = float(fn(quieter, correct))
            entry[name] = {
                "wrong_note_loss": wrong_note,
                "gain_change_loss": gain_change,
                # >1 means the loss treats a wrong note as worse than a 6 dB
                # level error. <1 means it considers the wrong note the
                # smaller problem of the two.
                "ratio": wrong_note / gain_change if gain_change else float("nan"),
            }
        results.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sample_rate": SR, "results": results}
    (out_dir / "resolution_probe.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = run(args.out)

    print("Wrong-note detectability: loss(wrong note) / loss(-6 dB level change)")
    print("A value near 0 means the loss cannot see the wrong note at all.")
    print("-" * 86)
    header = f"{'note pair':<34}{'gap':>7}{'max2048':>12}{'max8192':>12}{'perceptual':>14}"
    print(header)
    print("-" * 86)
    for entry in payload["results"]:
        print(
            f"{entry['pair']:<34}"
            f"{entry['separation_hz']:>6.1f}Hz"
            f"{entry['mrstft_max2048']['ratio']:>12.4f}"
            f"{entry['mrstft_max8192']['ratio']:>12.4f}"
            f"{entry['perceptual']['ratio']:>14.4f}"
        )


if __name__ == "__main__":
    main()
