"""Experiment tracker protocol, composite, and factory."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from synthgen.tracking.clearml_backend import ClearMLTracker
from synthgen.tracking.null import NullTracker
from synthgen.tracking.wandb_backend import WandBTracker

logger = logging.getLogger(__name__)


@runtime_checkable
class ExperimentTracker(Protocol):
    """Common interface for experiment tracking backends."""

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None: ...

    def log_checkpoint_ref(self, path: str, step: int) -> None: ...

    def finish(self) -> None: ...


class CompositeTracker:
    """Fan-out tracker; ClearML is listed first when both are enabled."""

    def __init__(self, trackers: list[ExperimentTracker]):
        self._trackers = trackers

    def log_params(self, params: dict[str, Any]) -> None:
        for t in self._trackers:
            t.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for t in self._trackers:
            t.log_metrics(metrics, step)

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        for t in self._trackers:
            t.log_artifact_json(name, payload)

    def log_checkpoint_ref(self, path: str, step: int) -> None:
        for t in self._trackers:
            t.log_checkpoint_ref(path, step)

    def finish(self) -> None:
        for t in self._trackers:
            t.finish()

    @property
    def trackers(self) -> list[ExperimentTracker]:
        return list(self._trackers)


def config_as_dict(config: Any) -> dict[str, Any]:
    """Serialize a TrainingConfig-like object to a flat dict."""
    if isinstance(config, dict):
        return dict(config)
    data: dict[str, Any] = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        data[key] = value
    return data


def build_tracker(config: Any, rank: int = 0) -> ExperimentTracker:
    """
    Build the appropriate tracker for this process.

    ClearML is the primary backend when enabled. WandB is optional secondary.
    Non-rank-0 workers always get NullTracker.
    """
    if rank != 0:
        return NullTracker()

    use_clearml = bool(getattr(config, "use_clearml", False))
    use_wandb = bool(getattr(config, "use_wandb", False))

    if not use_clearml and not use_wandb:
        return NullTracker()

    params = config_as_dict(config)
    trackers: list[ExperimentTracker] = []

    if use_clearml:
        stage = getattr(config, "stage", "train")
        task_name = getattr(config, "clearml_task_name", None) or f"synthgen-{stage}"
        trackers.append(
            ClearMLTracker(
                project_name=getattr(config, "clearml_project", "synthgen"),
                task_name=task_name,
                params=params,
                tags=list(getattr(config, "clearml_tags", None) or []),
                upload_checkpoints=bool(
                    getattr(config, "clearml_upload_checkpoints", False)
                ),
            )
        )

    if use_wandb:
        trackers.append(
            WandBTracker(
                project=getattr(config, "wandb_project", "synthgen"),
                config=params,
            )
        )

    if len(trackers) == 1:
        return trackers[0]
    return CompositeTracker(trackers)


def get_clearml_task(tracker: ExperimentTracker) -> Any | None:
    """Return the underlying ClearML Task if present on the tracker tree."""
    if isinstance(tracker, ClearMLTracker):
        return tracker.task
    if isinstance(tracker, CompositeTracker):
        for t in tracker.trackers:
            if isinstance(t, ClearMLTracker):
                return t.task
    return None
