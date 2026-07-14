"""Weights & Biases tracker (optional secondary backend)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandBTracker:
    """Thin wrapper around wandb.init / wandb.log."""

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        wandb_module: Any = None,
    ):
        self._wandb = wandb_module
        self._ok = False

        if self._wandb is None:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                logger.warning(
                    "wandb is not installed; WandB tracking disabled. "
                    "Install with: uv sync --extra train"
                )
                return

        try:
            self._wandb.init(project=project, config=config or {})
            self._ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.init failed (%s); continuing without WandB.", exc)
            self._ok = False

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._ok:
            return
        try:
            self._wandb.config.update(params, allow_val_change=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self._ok:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb log_metrics failed: %s", exc)

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        if not self._ok:
            return
        try:
            # Keep lightweight: log as a summary nested dict, not a file upload
            self._wandb.summary[f"artifact/{name}"] = payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb log_artifact_json failed: %s", exc)

    def log_checkpoint_ref(self, path: str, step: int) -> None:
        if not self._ok:
            return
        try:
            self._wandb.summary["checkpoint/latest_path"] = path
            self._wandb.summary["checkpoint/latest_step"] = step
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb log_checkpoint_ref failed: %s", exc)

    def finish(self) -> None:
        if not self._ok:
            return
        try:
            self._wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.finish failed: %s", exc)
        finally:
            self._ok = False
