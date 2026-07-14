"""Helpers for logging inference/generation runs (params + local paths only)."""

from __future__ import annotations

from typing import Any

from synthgen.tracking.tracker import ExperimentTracker


def log_generation_run(
    tracker: ExperimentTracker,
    prompt: str,
    duration: float,
    num_steps: int,
    cfg_scale: float,
    seed: int | None,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """Log generation hyperparameters and local artifact paths (no media upload)."""
    params: dict[str, Any] = {
        "prompt": prompt,
        "duration": duration,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "output_path": output_path,
    }
    tracker.log_params(params)
    tracker.log_artifact_json("generation_run", params)
