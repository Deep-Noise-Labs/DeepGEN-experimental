"""
Controlled A/B: train the SynthGen VAE with and without anti-aliased
activations, everything else held identical.

Both arms share the same seed, the same weight initialisation, the same
data order, the same optimiser and the same number of steps. The *only*
difference is whether the Snake activations are evaluated at the base rate
(``--no-antialias``, the pre-change model) or inside an oversample/
filter/decimate sandwich (``--antialias``, the change).

Run as:

    python experiments/train_antialias_ab.py --arm baseline  --steps 3000
    python experiments/train_antialias_ab.py --arm antialias --steps 3000

Deliberately small: this is a *controlled comparison* on CPU, not a
production training run. It answers "does the change help a real trained
codec", not "is this checkpoint shippable".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from synthgen.model.vae import AudioVAE
from synthgen.training.losses import VAELoss

SAMPLE_RATE = 44100
CROP_SECONDS = 0.5
CROP = int(SAMPLE_RATE * CROP_SECONDS)


def load_corpus(paths: list[Path]) -> list[np.ndarray]:
    """Load every file as mono float32 at ``SAMPLE_RATE``."""
    clips = []
    for path in paths:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            # Linear resample is adequate here: the corpus is only ever used
            # to give the codec something real to learn, and both arms see
            # byte-identical input.
            n = int(round(len(mono) * SAMPLE_RATE / sr))
            mono = np.interp(
                np.linspace(0, len(mono) - 1, n),
                np.arange(len(mono)),
                mono,
            ).astype(np.float32)
        peak = np.max(np.abs(mono))
        if peak > 0:
            mono = mono / peak * 0.7
        if len(mono) >= CROP:
            clips.append(mono.astype(np.float32))
    return clips


def batches(clips: list[np.ndarray], batch_size: int, seed: int):
    """Infinite stream of random crops, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    while True:
        rows = []
        for _ in range(batch_size):
            clip = clips[rng.integers(len(clips))]
            start = rng.integers(0, len(clip) - CROP + 1)
            rows.append(clip[start : start + CROP])
        yield torch.from_numpy(np.stack(rows)).unsqueeze(1)


def _save(model, args, n_params: int, step: int, history: list) -> None:
    """Write the checkpoint atomically so a reaped run leaves a usable file."""
    target = args.out_dir / f"vae_{args.arm}.pt"
    tmp = target.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arm": args.arm,
            "params": n_params,
            "step": step,
        },
        tmp,
    )
    tmp.replace(target)
    (args.out_dir / f"history_{args.arm}.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "antialias"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/out"))
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Write a resumable checkpoint this often. A long CPU run can be "
        "reaped by the environment; without this, everything is lost.",
    )
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_corpus(list(args.corpus))
    total_seconds = sum(len(c) for c in clips) / SAMPLE_RATE
    print(f"[{args.arm}] corpus: {len(clips)} clips, {total_seconds:.1f}s")

    # Identical init across arms: same seed, same construction order.
    torch.manual_seed(args.seed)
    model = AudioVAE(
        in_channels=1,
        latent_dim=32,
        base_channels=16,
        encoder_channel_multipliers=(1, 2, 4),
        decoder_channel_multipliers=(4, 2, 1),
        strides=(4, 4, 4),
        num_residual_per_block=3,
        antialias=(args.arm == "antialias"),
        antialias_ratio=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] params: {n_params/1e6:.3f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = VAELoss(kl_weight=1e-6, l1_weight=1.0, spectral_weight=1.0)
    stream = batches(clips, args.batch_size, args.seed)

    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        x = next(stream)
        recon, target, mean, log_var = model(x)
        losses = loss_fn(recon, target, mean, log_var)
        optimiser.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % 25 == 0 or step == 1:
            history.append(
                {
                    "step": step,
                    "loss": float(losses["loss"]),
                    "l1": float(losses["l1_loss"]),
                    "spectral": float(losses["spectral_loss"]),
                    "elapsed_s": time.time() - start,
                }
            )
        if step % 250 == 0:
            rate = step / (time.time() - start)
            print(
                f"[{args.arm}] step {step}/{args.steps} "
                f"loss={float(losses['loss']):.4f} "
                f"spectral={float(losses['spectral_loss']):.4f} "
                f"({rate:.2f} steps/s)",
                flush=True,
            )

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            _save(model, args, n_params, step, history)

    _save(model, args, n_params, args.steps, history)
    print(f"[{args.arm}] done in {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
