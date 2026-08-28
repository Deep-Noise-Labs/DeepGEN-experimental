"""
Controlled ablation of the Stage-1 VAE objective.

Two experiments, both on real audio:

``probe``
    Score a set of deliberate, audible degradations under the objective as it
    exists on ``main`` and under the new one. No training. Answers: *what can
    each objective actually hear?*

``train``
    Train the same small AudioVAE twice — identical architecture, seed, data,
    step count and optimiser — changing only the objective, then dump
    reconstructions and third-party metrics. Answers: *what does each objective
    make the autoencoder keep?*

The training arm runs at deliberately reduced scale (a ~1M parameter VAE, a
handful of clips, CPU-feasible step counts) and is an *objective ablation*, not
a quality claim about the production model. It is run in the overfit regime —
reconstruction is measured on the clips it trained on — because the question is
what the objective preserves, not how the model generalises.

Usage:
    python -m experiments.vae_objective_ablation probe --out runs/probe
    python -m experiments.vae_objective_ablation train --out runs/train --steps 900
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.degradations import DEGRADATIONS, MEAN_SEEKING_PAIR
from experiments.legacy_objective import LegacyVAELoss
from experiments.metrics import (
    band_energy_error_db,
    envelope_error_db,
    log_spectral_distance_db,
    si_sdr_db,
    stereo_width,
)
from synthgen.model.vae import AudioVAE
from synthgen.training.losses import VAELoss
from synthgen.utils.audio import load_audio, save_audio

SAMPLE_RATE = 44100

# Real audio checked into the Deep Noise repositories. Each entry is
# (label, path, offset_seconds) chosen for a distinct acoustic character.
DEFAULT_SOURCES = [
    (
        "sub_bass_pad",
        "/home/user/deepnoise-web/public/assets/player/track1.mp3",
        6.0,
    ),
    (
        "wide_synth",
        "/home/user/deep-noise-studio-frontend/public/sample2.wav",
        4.0,
    ),
    (
        "glissando",
        "/home/user/deep-noise-studio-frontend/public/sample6glissando1.wav",
        12.0,
    ),
    (
        "electronic",
        "/home/user/audiocraft/assets/electronic.mp3",
        0.5,
    ),
    (
        "guitar",
        "/home/user/aisynth-vst/assets/guitar.wav",
        0.15,
    ),
]


# =============================================================================
# Audio helpers
# =============================================================================


def load_clip(path: str, offset: float, seconds: float) -> np.ndarray:
    """Load a stereo clip at the project sample rate, peak-normalised to -3 dBFS."""
    audio = load_audio(path, sample_rate=SAMPLE_RATE, channels=2, offset=offset,
                       duration=seconds)
    needed = int(seconds * SAMPLE_RATE)
    if audio.shape[-1] < needed:
        audio = np.pad(audio, ((0, 0), (0, needed - audio.shape[-1])))
    audio = audio[:, :needed].astype(np.float32)

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (10 ** (-3.0 / 20.0) / peak)
    return audio


def fit_group(clips: dict[str, np.ndarray], headroom: float = 0.9) -> dict[str, np.ndarray]:
    """
    Scale a group of related clips by one common factor so none of them clips.

    Phase randomisation raises crest factor, so a candidate can peak well above
    its source even though nothing was added to it. Writing that to 16-bit
    clamps it, and the clamping is audible distortion that would be mistaken for
    the effect under test. Applying a *single* factor across the group keeps
    every level relationship — including the 1 dB calibration step — intact, and
    a uniform gain leaves both objectives' scores unchanged: spectral
    convergence is a ratio, and a constant cancels inside a log difference.
    """
    peak = max(float(np.abs(clip).max()) for clip in clips.values())
    scale = headroom / peak if peak > headroom else 1.0
    return {name: (clip * scale).astype(np.float32) for name, clip in clips.items()}


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(audio).float().unsqueeze(0)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().squeeze(0).cpu().numpy()


# =============================================================================
# Experiment 1 — objective probe
# =============================================================================


def spectral_only(objective, pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Score using only the objective's spectral terms.

    Both objectives include a waveform L1 term, which is hypersensitive to phase
    in a way that has no relationship to audibility — a one-sample shift is
    inaudible but moves L1 enormously. Separating the spectral terms is what
    makes the phase-blindness result legible rather than masked by L1 noise.
    """
    zeros = torch.zeros(1, 8, 8)
    losses = objective(pred, target, zeros, zeros)
    total = losses["spectral_loss"]
    for key in ("mel_loss", "stereo_loss"):
        if key in losses:
            total = total + losses[key]
    return float(total)


def run_probe(out_dir: Path, sources: list, seconds: float) -> dict:
    legacy = LegacyVAELoss()
    improved = VAELoss(sample_rate=SAMPLE_RATE)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for label, path, offset in sources:
        group = {"original": load_clip(path, offset, seconds)}
        for name, fn in DEGRADATIONS.items():
            group[name] = np.ascontiguousarray(fn(group["original"]))
        group = fit_group(group)

        clip = group["original"]
        target = to_tensor(clip)
        save_audio(audio_dir / f"{label}__original.wav", clip, SAMPLE_RATE)

        per_source: dict[str, dict] = {}
        for name in DEGRADATIONS:
            degraded = group[name]
            save_audio(audio_dir / f"{label}__{name}.wav", degraded, SAMPLE_RATE)

            pred = to_tensor(degraded)
            per_source[name] = {
                "legacy_spectral": spectral_only(legacy, pred, target),
                "improved_spectral": spectral_only(improved, pred, target),
                "legacy_total": float(
                    legacy(pred, target, torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))[
                        "loss"
                    ]
                ),
                "improved_total": float(
                    improved(pred, target, torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))[
                        "loss"
                    ]
                ),
                "si_sdr_db": si_sdr_db(degraded, clip),
                "lsd_low_db": log_spectral_distance_db(degraded, clip, fmax=200.0),
                "envelope_error_db": envelope_error_db(degraded, clip),
                "stereo_width": stereo_width(degraded),
            }

        # Two objectives with different term counts have different absolute
        # scales, so raw scores cannot be compared. Two normalisations are
        # reported instead:
        #
        # x_ref   — in units of the objective's own response to a 1 dB
        #           broadband level error: a real but barely audible mistake.
        # share   — the fraction of the objective's total sensitivity (across
        #           the real degradations) that this degradation attracts. This
        #           is the scale-free question: where does the objective spend
        #           its attention?
        real = [name for name in DEGRADATIONS if name not in ("sample_shift", "gain_1db")]
        for objective in ("legacy", "improved"):
            unit = per_source["gain_1db"][f"{objective}_spectral"]
            total = sum(per_source[name][f"{objective}_spectral"] for name in real)
            for name in per_source:
                score = per_source[name][f"{objective}_spectral"]
                per_source[name][f"{objective}_x_ref"] = (
                    score / unit if unit > 0 else float("nan")
                )
                per_source[name][f"{objective}_share"] = (
                    score / total if total > 0 else float("nan")
                )

        per_source["_reference"] = {"stereo_width": stereo_width(clip)}
        results[label] = per_source

    return results


def run_mean_seeking(out_dir: Path, path: str, offset: float, seconds: float) -> dict:
    """
    Show that a purely reconstructive objective prefers a dull average to a
    plausible alternative — the reason a critic is needed at all.

    Two candidates are scored against the same noise-like target:

    ``texture_redraw``  keeps the magnitude spectrum exactly and re-draws every
        phase. On noise-like material a listener cannot reliably tell it from
        the original: it is a *different realisation of the same sound*.
    ``spectral_blur``   smooths the magnitude spectrum. It is audibly duller —
        this is what predicting the conditional mean sounds like.

    If a reconstruction objective scores the dull candidate better than the
    perceptually equivalent one, then optimising it harder cannot produce
    realistic detail, no matter how well weighted it is. That is an argument no
    reweighting can answer, and it is what the discriminator exists to fix.
    """
    legacy = LegacyVAELoss()
    improved = VAELoss(sample_rate=SAMPLE_RATE)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    group = {"original": load_clip(path, offset, seconds)}
    for name, fn in MEAN_SEEKING_PAIR.items():
        group[name] = np.ascontiguousarray(fn(group["original"]))
    group = fit_group(group)

    clip = group["original"]
    target = to_tensor(clip)
    save_audio(audio_dir / "texture__original.wav", clip, SAMPLE_RATE)

    scores: dict[str, dict] = {}
    for name in MEAN_SEEKING_PAIR:
        candidate = group[name]
        save_audio(audio_dir / f"texture__{name}.wav", candidate, SAMPLE_RATE)
        pred = to_tensor(candidate)
        scores[name] = {
            "legacy_spectral": spectral_only(legacy, pred, target),
            "improved_spectral": spectral_only(improved, pred, target),
            "si_sdr_db": si_sdr_db(candidate, clip),
        }

    return scores


# =============================================================================
# Experiment 2 — training ablation
# =============================================================================


def build_vae(seed: int) -> AudioVAE:
    torch.manual_seed(seed)
    return AudioVAE(
        in_channels=2,
        latent_dim=32,
        base_channels=16,
        encoder_channel_multipliers=(1, 2, 4, 8),
        decoder_channel_multipliers=(8, 4, 2, 1),
        strides=(4, 4, 4, 8),
        num_residual_per_block=2,
    )


def build_dataset(sources: list, seconds: float, crops_per_source: int) -> torch.Tensor:
    """Fixed set of crops, identical for both arms."""
    clips = []
    for _, path, offset in sources:
        for index in range(crops_per_source):
            start = offset + index * seconds
            clips.append(load_clip(path, start, seconds))
    return torch.from_numpy(np.stack(clips)).float()


def train_arm(
    name: str,
    objective,
    dataset: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[AudioVAE, list]:
    vae = build_vae(seed)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.95))

    generator = torch.Generator().manual_seed(seed)
    history = []
    started = time.time()

    for step in range(steps):
        # Cosine decay to a tenth of the peak rate.
        progress = step / max(steps - 1, 1)
        for group in optimizer.param_groups:
            group["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

        idx = torch.randint(
            0, dataset.shape[0], (batch_size,), generator=generator
        )
        batch = dataset[idx]

        reconstruction, target, mean, log_var = vae(batch)
        losses = objective(reconstruction, target, mean, log_var)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(losses["loss"])})
            print(
                f"  [{name}] step {step:4d}/{steps}  loss={float(losses['loss']):.4f}"
                f"  ({time.time() - started:.0f}s)",
                flush=True,
            )

    return vae, history


@torch.no_grad()
def reconstruct(vae: AudioVAE, clip: np.ndarray) -> np.ndarray:
    vae.eval()
    mean, _ = vae.encode(to_tensor(clip))
    return to_numpy(vae.decode(mean))


def evaluate(pred: np.ndarray, target: np.ndarray) -> dict:
    n = min(pred.shape[-1], target.shape[-1])
    pred, target = pred[..., :n], target[..., :n]
    return {
        "si_sdr_db": si_sdr_db(pred, target),
        "lsd_full_db": log_spectral_distance_db(pred, target),
        "lsd_low_db": log_spectral_distance_db(pred, target, fmax=200.0),
        "lsd_mid_db": log_spectral_distance_db(pred, target, fmin=200.0, fmax=4000.0),
        "lsd_high_db": log_spectral_distance_db(pred, target, fmin=4000.0),
        "envelope_error_db": envelope_error_db(pred, target),
        "sub_band_error_db": band_energy_error_db(pred, target, 20.0, 200.0),
        "stereo_width": stereo_width(pred),
    }


def run_training(
    out_dir: Path,
    sources: list,
    seconds: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    crops_per_source: int,
) -> dict:
    dataset = build_dataset(sources, seconds, crops_per_source)
    print(f"Dataset: {tuple(dataset.shape)} ({dataset.shape[0]} clips)", flush=True)

    arms = {
        "legacy": LegacyVAELoss(),
        "improved": VAELoss(sample_rate=SAMPLE_RATE),
    }

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    histories = {}
    for name, objective in arms.items():
        print(f"Training arm: {name}", flush=True)
        models[name], histories[name] = train_arm(
            name, objective, dataset, steps, batch_size, lr, seed
        )

    results: dict[str, dict] = {}
    for index, (label, path, offset) in enumerate(sources):
        clip = to_numpy(dataset[index * crops_per_source].unsqueeze(0))

        per_source = {"_reference": {"stereo_width": stereo_width(clip)}}
        # The decoder has no output non-linearity, so a reconstruction can
        # overshoot the target's peak. Fit the whole group by one factor before
        # writing, or 16-bit clamping adds distortion that is not the model's.
        group = fit_group(
            {"original": clip, **{name: reconstruct(vae, clip)
                                  for name, vae in models.items()}}
        )
        save_audio(audio_dir / f"{label}__original.wav", group["original"], SAMPLE_RATE)
        for name in models:
            save_audio(audio_dir / f"{label}__{name}.wav", group[name], SAMPLE_RATE)
            per_source[name] = evaluate(group[name], group["original"])
        results[label] = per_source

    # Aggregate over the whole training set, not just the exported examples.
    aggregate: dict[str, dict] = {}
    for name, vae in models.items():
        scores: dict[str, list] = {}
        for index in range(dataset.shape[0]):
            clip = to_numpy(dataset[index].unsqueeze(0))
            for key, value in evaluate(reconstruct(vae, clip), clip).items():
                scores.setdefault(key, []).append(value)
        aggregate[name] = {
            key: float(np.nanmean(values)) for key, values in scores.items()
        }

    return {"per_source": results, "aggregate": aggregate, "history": histories}


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "train"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--crops-per-source", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "probe":
        payload = {
            "config": vars(args),
            "sources": [
                {"label": s[0], "path": s[1], "offset": s[2]} for s in DEFAULT_SOURCES
            ],
            "results": run_probe(out_dir, DEFAULT_SOURCES, args.seconds),
            "mean_seeking": run_mean_seeking(
                out_dir,
                "/home/user/aisynth-vst/assets/whitenoise.wav",
                0.5,
                args.seconds,
            ),
        }
    else:
        payload = {
            "config": vars(args),
            "sources": [
                {"label": s[0], "path": s[1], "offset": s[2]} for s in DEFAULT_SOURCES
            ],
            **run_training(
                out_dir,
                DEFAULT_SOURCES,
                args.seconds,
                args.steps,
                args.batch_size,
                args.lr,
                args.seed,
                args.crops_per_source,
            ),
        }

    report = out_dir / "report.json"
    report.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
