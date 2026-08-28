"""
Learning-rate control for the objective ablation.

The obvious objection to the ablation: the new objective has more terms and so a
larger raw gradient at equal reconstruction quality, which means comparing the
two arms at one shared learning rate might be comparing effective step sizes
rather than objectives. Gradient clipping at norm 1.0 equalises much of it, but
"much" is not "all".

This sweeps both objectives across learning rates and reports each one's *best*
result. If the new objective still wins at every arm's own best setting, the gap
is not a step-size artefact.

    python -m experiments.lr_control --out runs/lr_control --steps 500
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from experiments.legacy_objective import LegacyVAELoss
from experiments.metrics import log_spectral_distance_db, si_sdr_db
from experiments.vae_objective_ablation import (
    DEFAULT_SOURCES,
    SAMPLE_RATE,
    build_dataset,
    build_vae,
    reconstruct,
    to_numpy,
)
from synthgen.training.losses import VAELoss

LEARNING_RATES = (5e-4, 1e-3, 2e-3, 4e-3)


def run_one(objective, dataset, steps, lr, seed, strides, base_channels, latent_dim):
    vae = build_vae(seed, strides, base_channels, latent_dim)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.95))
    diverged = False

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        for group in optimizer.param_groups:
            group["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

        reconstruction, target, mean, log_var = vae(dataset)
        loss = objective(reconstruction, target, mean, log_var)["loss"]
        if not torch.isfinite(loss):
            diverged = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    vae.eval()
    scores = []
    for index in range(dataset.shape[0]):
        clip = to_numpy(dataset[index].unsqueeze(0))
        recon = reconstruct(vae, clip)
        scores.append((si_sdr_db(recon, clip), log_spectral_distance_db(recon, clip)))

    return {
        "lr": lr,
        "diverged": diverged,
        "si_sdr_db": sum(s for s, _ in scores) / len(scores),
        "lsd_db": sum(d for _, d in scores) / len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--strides", type=str, default="2,2,4,4")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    strides = tuple(int(s) for s in args.strides.split(","))
    sources = DEFAULT_SOURCES[: args.sources]
    dataset = build_dataset(sources, args.seconds, 1)

    arms = {
        "legacy": lambda: LegacyVAELoss(),
        "improved": lambda: VAELoss(sample_rate=SAMPLE_RATE),
    }

    results: dict[str, list] = {}
    started = time.time()
    for name, make in arms.items():
        results[name] = []
        for lr in LEARNING_RATES:
            outcome = run_one(
                make(), dataset, args.steps, lr, args.seed,
                strides, args.base_channels, args.latent_dim,
            )
            results[name].append(outcome)
            print(
                f"  [{name}] lr={lr:<7g} si-sdr={outcome['si_sdr_db']:7.2f} dB  "
                f"lsd={outcome['lsd_db']:5.2f} dB"
                f"{'  DIVERGED' if outcome['diverged'] else ''}"
                f"  ({time.time() - started:.0f}s)",
                flush=True,
            )

    best = {
        name: max(runs, key=lambda r: r["si_sdr_db"]) for name, runs in results.items()
    }
    print("\nBest over the sweep:")
    for name, outcome in best.items():
        print(f"  {name:9s} lr={outcome['lr']:<7g} si-sdr={outcome['si_sdr_db']:7.2f} dB")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"config": vars(args), "sweep": results, "best": best}, indent=2)
    )


if __name__ == "__main__":
    main()
