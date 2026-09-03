"""
Experiment 4 -- the Snake activation's singularity, and the log-space fix.

Snake is ``x + (1/alpha) * sin^2(alpha * x)``. The original implementation
stored ``alpha`` as a raw learnable parameter and computed the reciprocal as
``1 / (alpha + 1e-8)``, with nothing constraining alpha to stay positive.

The obvious hypothesis is that this explodes near zero. **It does not, and this
script is what showed that.** As alpha approaches zero, ``sin^2(alpha*x)``
vanishes quadratically while the denominator vanishes linearly, so the ratio
tends to ``alpha * x^2`` and both the output and the gradient stay bounded.
The sweep below records that directly: nothing blows up.

What the sweep *does* find is narrower and still worth fixing:

1. ``alpha == -1e-8`` is exactly singular -- that one value returns ``inf``
   for the output and for the gradient.
2. Negative alpha silently changes which function is being computed. ``sin^2``
   is even, so a sign flip on alpha flips the sign of the entire periodic term
   and mirrors the activation about the origin. It is not a mis-scaled Snake;
   it is a different activation, and the old parameterisation let an optimiser
   wander into it unannounced.

Storing ``log_alpha`` makes both unreachable by construction. This is a
correctness guard, not a fix for measured instability -- stated that way here
because the measurement is what settled it.

Usage::

    python experiments/run_snake_stability.py --out <results-dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from synthgen.model.vae import Snake


class LegacySnake(torch.nn.Module):
    """The pre-fix implementation, kept here so the failure is reproducible."""

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


def probe(module: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Max |output| and max |d output / d input| for one activation."""
    x = x.clone().requires_grad_(True)
    y = module(x)
    y.sum().backward()
    return y.abs().max().item(), x.grad.abs().max().item()


def mirror_check() -> dict:
    """
    Show that negative alpha computes a mirrored function, not a scaled one.

    If negative alpha merely rescaled the periodic term, ``f_{-a}(x)`` would
    stay close to ``f_{a}(x)``. It does not: it equals ``-f_{a}(-x)``.
    """
    x = torch.linspace(-3, 3, 129).view(1, 1, -1)

    def legacy_forward(alpha: float, inp: torch.Tensor) -> torch.Tensor:
        a = torch.tensor([[[alpha]]])
        return inp + (1.0 / (a + 1e-8)) * torch.sin(a * inp) ** 2

    positive = legacy_forward(1.0, x)
    negative = legacy_forward(-1.0, x)
    mirrored = -legacy_forward(1.0, -x)
    return {
        "max_abs_diff_pos_vs_neg": (positive - negative).abs().max().item(),
        "max_abs_diff_neg_vs_mirrored_pos": (negative - mirrored).abs().max().item(),
    }


def singularity_check() -> dict:
    """The one exactly-singular value of alpha in the old formulation."""
    x = torch.randn(1, 1, 256) * 0.5
    a = torch.tensor([[[-1e-8]]], requires_grad=True)
    y = x + (1.0 / (a + 1e-8)) * torch.sin(a * x) ** 2
    y.sum().backward()
    return {
        "alpha": -1e-8,
        "output_finite": bool(torch.isfinite(y).all()),
        "grad_finite": bool(torch.isfinite(a.grad).all()),
    }


def run(out_dir: Path) -> dict:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 512) * 0.5

    # Alphas approaching and crossing zero, as unconstrained training can.
    alphas = [1.0, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6, 0.0, -1e-6, -1e-3, -0.1]
    rows = []
    for alpha in alphas:
        legacy = LegacySnake(1, alpha_init=1.0)
        with torch.no_grad():
            legacy.alpha.fill_(alpha)
        legacy_out, legacy_grad = probe(legacy, x)

        # The fixed version is parameterised by log_alpha, so the same *effective*
        # alpha is only reachable for alpha > 0. Below that it does not exist,
        # which is the entire point of the fix.
        if alpha > 0:
            fixed = Snake(1, alpha_init=1.0)
            with torch.no_grad():
                fixed.log_alpha.fill_(math.log(alpha))
            fixed_out, fixed_grad = probe(fixed, x)
        else:
            fixed_out = fixed_grad = float("nan")

        rows.append(
            {
                "alpha": alpha,
                "legacy_max_output": legacy_out,
                "legacy_max_grad": legacy_grad,
                "fixed_max_output": fixed_out,
                "fixed_max_grad": fixed_grad,
                "fixed_reachable": alpha > 0,
            }
        )

    # What an optimiser step actually does: a large negative update on the
    # parameter. Raw alpha crosses zero; log_alpha cannot.
    legacy = LegacySnake(1, alpha_init=1.0)
    fixed = Snake(1, alpha_init=1.0)
    with torch.no_grad():
        legacy.alpha -= 20.0
        fixed.log_alpha -= 20.0
    step_test = {
        "legacy_alpha_after": float(legacy.alpha.item()),
        "legacy_output_finite": bool(torch.isfinite(legacy(x)).all()),
        "legacy_max_output": float(legacy(x).abs().max()),
        "fixed_alpha_after": float(fixed.alpha.item()),
        "fixed_output_finite": bool(torch.isfinite(fixed(x)).all()),
        "fixed_max_output": float(fixed(x).abs().max()),
    }

    payload = {
        "sweep": rows,
        "large_negative_step": step_test,
        "mirror_check": mirror_check(),
        "singularity_check": singularity_check(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snake_stability.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = run(args.out)

    print("Snake activation as alpha approaches and crosses zero")
    print("-" * 76)
    print(f"{'alpha':>10}{'legacy |out|':>16}{'legacy |grad|':>16}"
          f"{'fixed |out|':>16}{'fixed |grad|':>16}")
    print("-" * 76)
    for row in payload["sweep"]:
        fixed_out = (
            f"{row['fixed_max_output']:.3f}" if row["fixed_reachable"] else "unreachable"
        )
        fixed_grad = (
            f"{row['fixed_max_grad']:.3f}" if row["fixed_reachable"] else "-"
        )
        print(
            f"{row['alpha']:>10.1e}{row['legacy_max_output']:>16.3e}"
            f"{row['legacy_max_grad']:>16.3e}{fixed_out:>16}{fixed_grad:>16}"
        )

    print("\nNo blow-up anywhere in the sweep: the hypothesised singularity")
    print("cancels, because sin^2(alpha*x)/alpha -> alpha*x^2 as alpha -> 0.")

    step = payload["large_negative_step"]
    print("\nAfter a single -20 update to the parameter")
    print("-" * 76)
    print(f"  legacy: alpha = {step['legacy_alpha_after']:+.4f}, "
          f"max |out| = {step['legacy_max_output']:.4g}, "
          f"finite = {step['legacy_output_finite']}")
    print(f"  fixed:  alpha = {step['fixed_alpha_after']:.3e}, "
          f"max |out| = {step['fixed_max_output']:.4g}, "
          f"finite = {step['fixed_output_finite']}")

    sing = payload["singularity_check"]
    print(f"\nThe one genuinely singular value, alpha = {sing['alpha']:.0e}")
    print("-" * 76)
    print(f"  output finite: {sing['output_finite']}    "
          f"gradient finite: {sing['grad_finite']}")

    mirror = payload["mirror_check"]
    print("\nNegative alpha computes a mirrored function, not a rescaled one")
    print("-" * 76)
    print(f"  max |f(+1, x) - f(-1, x)|      = "
          f"{mirror['max_abs_diff_pos_vs_neg']:.4f}   (large: different curves)")
    print(f"  max |f(-1, x) - (-f(+1, -x))|  = "
          f"{mirror['max_abs_diff_neg_vs_mirrored_pos']:.2e}   (zero: it is the mirror)")


if __name__ == "__main__":
    main()
