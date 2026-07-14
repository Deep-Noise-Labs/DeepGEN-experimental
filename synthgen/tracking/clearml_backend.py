"""ClearML experiment tracker (primary backend).

Never uploads audio/media. PyTorch auto model capture is disabled so checkpoint
binaries are not sent to the ClearML fileserver unless explicitly opted in.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ClearMLTracker:
    """ClearML Task-backed tracker with metadata-only uploads."""

    def __init__(
        self,
        project_name: str,
        task_name: str,
        params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        upload_checkpoints: bool = False,
        task_cls: Any = None,
    ):
        self._task = None
        self._logger = None
        self._upload_checkpoints = upload_checkpoints
        self._output_model = None
        self._closed = False

        if task_cls is None:
            try:
                from clearml import Task as task_cls  # type: ignore
            except ImportError:
                logger.warning(
                    "clearml is not installed; ClearML tracking disabled. "
                    "Install with: uv sync --extra train"
                )
                return

        try:
            self._task = task_cls.init(
                project_name=project_name,
                task_name=task_name,
                auto_connect_frameworks={"pytorch": False},
            )
            if tags:
                self._task.add_tags(tags)
            if params:
                connected = self._task.connect(dict(params))
                # Prefer connected values when ClearML overrides remotely
                if isinstance(connected, dict):
                    params.update(connected)
            self._logger = self._task.get_logger()

            if upload_checkpoints:
                try:
                    from clearml import OutputModel

                    self._output_model = OutputModel(task=self._task)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ClearML OutputModel unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ClearML Task.init failed (%s); continuing without ClearML.",
                exc,
            )
            self._task = None
            self._logger = None

    @property
    def task(self) -> Any:
        return self._task

    def log_params(self, params: dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            self._task.connect(dict(params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClearML log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._logger is None:
            return
        try:
            for name, value in metrics.items():
                title, _, series = name.partition("/")
                if not series:
                    title, series = "train", name
                self._logger.report_scalar(
                    title=title,
                    series=series,
                    iteration=step,
                    value=float(value),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClearML log_metrics failed: %s", exc)

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            self._task.upload_artifact(name=name, artifact_object=payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClearML log_artifact_json failed: %s", exc)

    def log_checkpoint_ref(self, path: str, step: int) -> None:
        if self._task is None:
            return
        try:
            self._task.set_parameter("checkpoint/latest_path", path)
            self._task.set_parameter("checkpoint/latest_step", step)
            if self._upload_checkpoints and self._output_model is not None:
                self._output_model.update_weights(weights_filename=path, auto_delete_file=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClearML log_checkpoint_ref failed: %s", exc)

    def finish(self) -> None:
        if self._task is None or self._closed:
            return
        try:
            self._task.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ClearML task.close failed: %s", exc)
        finally:
            self._closed = True
