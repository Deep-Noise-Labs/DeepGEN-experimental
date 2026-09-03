"""
Experiment 3 -- what does each loss actually pull audio towards?

Experiments 1 and 2 measure what a loss *responds to*. This one makes the
consequence audible.

A decoder starts out producing something imperfect and is pushed toward the
target by whatever gradient the loss provides. Here that process is run
directly on the waveform, with no network in the way: take a real clip, damage
it in two specific ways, then optimise the damaged audio back toward the
original under each loss and listen to where each one gets to.

Isolating the loss like this is the point. There is no decoder capacity, no
latent bottleneck and no training schedule to argue about -- whatever
difference appears between the two outputs is attributable to the objective
and nothing else. The flip side is that this is *not* a claim about trained
model quality: it shows the direction each loss pulls in, not what a trained
DeepGEN would sound like. Only a training run can show that.

The damage is deliberately chosen to sit where the two losses disagree most:
a 12 dB shelf above 8 kHz (dulling) plus a collapse of the stereo image to
mono. Both are recoverable in principle -- the information is in the target --
so whether they *are* recovered depends entirely on whether the loss can see
them.

Usage::

    python experiments/run_loss_inversion.py --input <audio-dir> --out <dir> --steps 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.degradations import mono_collapse, shelf
from synthgen.eval.metrics import (
    band_error_db,
    si_sdr_db,
    stereo_width,
)
from synthgen.training.losses import MultiResolutionSTFTLoss, PerceptualSampleLoss


def build_losses(sample_rate: int) -> dict:
    return {
        "legacy": MultiResolutionSTFTLoss(
            fft_sizes=(2048, 1024, 512, 256),
            hop_sizes=(512, 256, 128, 64),
            win_sizes=(2048, 1024, 512, 256),
        ),
        "perceptual": PerceptualSampleLoss(
            fft_sizes=(2048, 1024, 512, 256, 128), sample_rate=sample_rate
        ),
    }


def damage(audio: np.ndarray, sr: int) -> np.ndarray:
    """Dull the top end by 12 dB and flatten the stereo image."""
    return mono_collapse(shelf(audio, sr, 8000.0, -12.0, high=True), sr)


def optimise(
    start: np.ndarray,
    target: np.ndarray,
    loss_fn: torch.nn.Module,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, list[float]]:
    """Gradient descent on the waveform itself, under one loss."""
    target_t = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
    x = torch.nn.Parameter(
        torch.from_numpy(np.ascontiguousarray(start)).float().unsqueeze(0).clone()
    )
    optimiser = torch.optim.Adam([x], lr=lr)

    history: list[float] = []
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(x, target_t)
        loss.backward()
        optimiser.step()
        history.append(loss.detach().item())
        if step % 50 == 0:
            print(f"      step {step:>4}  loss {history[-1]:.5f}")
    return x.detach().squeeze(0).numpy(), history


def score(candidate: np.ndarray, target: np.ndarray, sr: int) -> dict:
    """The three things this experiment is actually testing."""
    bands = band_error_db(candidate, target, sr)
    return {
        "air_band_error_db": bands["air"],
        "presence_band_error_db": bands["presence"],
        "stereo_width_db": stereo_width(candidate),
        "stereo_width_error_db": abs(stereo_width(candidate) - stereo_width(target)),
        "si_sdr_db": si_sdr_db(candidate, target),
    }


def run(
    input_dir: Path,
    out_dir: Path,
    steps: int,
    lr: float,
    seconds: float,
    n_clips: int,
) -> dict:
    files = sorted(input_dir.glob("*.wav"))[:n_clips]
    if not files:
        raise FileNotFoundError(f"No .wav files under {input_dir}")

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for path in files:
        raw, sr = sf.read(str(path), dtype="float64", always_2d=True)
        target = raw.T[:, : int(sr * seconds)]
        start = damage(target, sr)
        stem = path.stem
        print(f"  {stem}")

        sf.write(str(audio_dir / f"{stem}__1_target.wav"), target.T, sr)
        sf.write(str(audio_dir / f"{stem}__2_damaged.wav"), start.T, sr)

        entry = {
            "file": path.name,
            "sample_rate": sr,
            "scores": {
                "target": score(target, target, sr),
                "damaged": score(start, target, sr),
            },
            "loss_history": {},
        }

        for name, loss_fn in build_losses(sr).items():
            print(f"    optimising under {name}")
            result, history = optimise(start, target, loss_fn, steps, lr)
            sf.write(str(audio_dir / f"{stem}__3_{name}.wav"), result.T, sr)
            entry["scores"][name] = score(result, target, sr)
            entry["loss_history"][name] = history

        results.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"steps": steps, "lr": lr, "seconds": seconds, "results": results}
    (out_dir / "loss_inversion.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--clips", type=int, default=3)
    args = parser.parse_args()

    payload = run(args.input, args.out, args.steps, args.lr, args.seconds, args.clips)

    print("\nRecovery after optimisation (target values in brackets)")
    print("-" * 78)
    for entry in payload["results"]:
        print(f"\n{entry['file']}")
        target_width = entry["scores"]["target"]["stereo_width_db"]
        print(f"{'stage':<14}{'air err dB':>12}{'width dB':>12}"
              f"{'width err':>12}{'SI-SDR dB':>12}")
        for stage in ("damaged", "legacy", "perceptual"):
            s = entry["scores"][stage]
            print(
                f"{stage:<14}{s['air_band_error_db']:>12.2f}"
                f"{s['stereo_width_db']:>12.2f}"
                f"{s['stereo_width_error_db']:>12.2f}"
                f"{s['si_sdr_db']:>12.2f}"
            )
        print(f"{'(target)':<14}{0.0:>12.2f}{target_width:>12.2f}"
              f"{0.0:>12.2f}{'inf':>12}")


if __name__ == "__main__":
    main()
