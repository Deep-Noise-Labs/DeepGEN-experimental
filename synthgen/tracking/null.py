"""No-op experiment tracker for disabled tracking or non-rank-0 workers."""

from __future__ import annotations

from typing import Any


class NullTracker:
    """Discard all tracking calls."""

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        return None

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        return None

    def log_checkpoint_ref(self, path: str, step: int) -> None:
        return None

    def finish(self) -> None:
        return None
