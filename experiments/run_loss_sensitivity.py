"""
Experiment 1 -- how much of an audible degradation does each loss actually see?

A training loss can only teach a decoder to preserve what the loss responds to.
This measures the response of the legacy ``MultiResolutionSTFTLoss`` and the new
``PerceptualSampleLoss`` to a set of degradations that are obviously audible,
applied to real audio.

Raw loss values from two different losses live on different scales and cannot
be compared directly, so two scale-free measures are reported instead. Both are
computed *within* a single loss, which is what makes them comparable across
losses.

**Wasted sensitivity.** The response to ``inaudible_dither`` (noise at -90
dBFS, which no listener can hear) divided by the mean response to the audible
degradations. This is the fraction of the loss's attention spent on content
that cannot matter. Lower is better; zero is ideal.

**Allocation share.** Each audible degradation's response as a fraction of the
total across all audible degradations. This shows how the loss divides its
attention between the things a producer would actually complain about. It sums
to 1 by construction for each loss, so a rise in one share is a fall in
another -- that is the point: gradient budget is finite.

The -6 dB gain change is reported as a reference point but excluded from both
measures, because a level offset is not a quality defect.

Usage::

    python experiments/run_loss_sensitivity.py --input <audio-dir> --out <results-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.degradations import DEGRADATIONS
from synthgen.training.losses import MultiResolutionSTFTLoss, PerceptualSampleLoss

CONTROL = "gain_minus_6db"
INAUDIBLE = "inaudible_dither"
LOSS_NAMES = ("legacy", "perceptual")


def audible_degradations() -> list[str]:
    """The degradations a listener would actually complain about."""
    return [n for n in DEGRADATIONS if n not in (CONTROL, INAUDIBLE)]


def build_losses(sample_rate: int, fft_sizes: tuple[int, ...]) -> dict:
    """The legacy loss at its shipped settings, and the new one."""
    return {
        "legacy": MultiResolutionSTFTLoss(
            fft_sizes=(2048, 1024, 512, 256),
            hop_sizes=(512, 256, 128, 64),
            win_sizes=(2048, 1024, 512, 256),
        ),
        "perceptual": PerceptualSampleLoss(
            fft_sizes=fft_sizes, sample_rate=sample_rate
        ),
    }


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)


def run(input_dir: Path, out_dir: Path, max_seconds: float) -> dict:
    files = sorted(input_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files under {input_dir}")

    rows: list[dict] = []
    losses = None

    for path in files:
        audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
        audio = audio.T[:, : int(sr * max_seconds)]
        if losses is None:
            losses = build_losses(sr, (2048, 1024, 512, 256, 128))

        target = to_tensor(audio)
        for deg_name, (deg_fn, _) in DEGRADATIONS.items():
            degraded = to_tensor(deg_fn(audio, sr))
            for loss_name, fn in losses.items():
                rows.append(
                    {
                        "file": path.name,
                        "degradation": deg_name,
                        "loss": loss_name,
                        "value": float(fn(degraded, target)),
                    }
                )
        print(f"  {path.name} done")

    result = {"n_files": len(files), "rows": rows, **analyse(rows)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loss_sensitivity.json").write_text(json.dumps(result, indent=2))
    return result


def analyse(rows: list[dict]) -> dict:
    """Reduce per-clip losses to mean values, allocation shares and waste."""

    def mean_value(deg: str, loss: str) -> float:
        return float(
            np.mean([r["value"] for r in rows if r["degradation"] == deg and r["loss"] == loss])
        )

    audible = audible_degradations()
    means = {
        loss: {deg: mean_value(deg, loss) for deg in DEGRADATIONS}
        for loss in LOSS_NAMES
    }

    shares, waste = {}, {}
    for loss in LOSS_NAMES:
        total = sum(means[loss][d] for d in audible)
        shares[loss] = {d: means[loss][d] / total for d in audible}
        waste[loss] = means[loss][INAUDIBLE] / (total / len(audible))

    return {"mean_values": means, "allocation_share": shares, "wasted_sensitivity": waste}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    result = run(args.input, args.out, args.seconds)
    shares = result["allocation_share"]
    waste = result["wasted_sensitivity"]

    print(f"\nWasted sensitivity  (response to inaudible -90 dBFS noise, "
          f"relative to the mean audible defect; lower is better)")
    print("-" * 72)
    print(f"  legacy      {waste['legacy']:.4f}")
    print(f"  perceptual  {waste['perceptual']:.4f}"
          f"   ({waste['legacy'] / max(waste['perceptual'], 1e-12):.0f}x less waste)")

    print("\nAllocation share across audible defects (each column sums to 1.00)")
    print("-" * 72)
    print(f"{'defect':<26}{'legacy':>12}{'perceptual':>14}{'change':>14}")
    print("-" * 72)
    for name in audible_degradations():
        legacy, new = shares["legacy"][name], shares["perceptual"][name]
        print(f"{name:<26}{legacy:>12.3f}{new:>14.3f}{new / legacy:>13.2f}x")


if __name__ == "__main__":
    main()
