"""Unit tests for ClearML / WandB experiment tracking (mocked, no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from synthgen.tracking.null import NullTracker
from synthgen.tracking.tracker import CompositeTracker, ExperimentTracker, build_tracker


@dataclass
class FakeTrackingConfig:
    use_clearml: bool = False
    use_wandb: bool = False
    clearml_project: str = "synthgen"
    clearml_task_name: str | None = None
    clearml_tags: list[str] = field(default_factory=list)
    clearml_dataset_name: str = "synthgen-data"
    clearml_register_dataset: bool = True
    clearml_upload_checkpoints: bool = False
    wandb_project: str = "synthgen"
    stage: str = "dit"


class TestBuildTrackerFactory:
    def test_factory_returns_null_when_tracking_disabled(self):
        tracker = build_tracker(FakeTrackingConfig(), rank=0)
        assert isinstance(tracker, NullTracker)

    def test_factory_returns_null_for_non_rank_zero(self):
        cfg = FakeTrackingConfig(use_clearml=True, use_wandb=True)
        tracker = build_tracker(cfg, rank=1)
        assert isinstance(tracker, NullTracker)

    def test_factory_returns_clearml_when_enabled(self):
        cfg = FakeTrackingConfig(use_clearml=True)
        instance = MagicMock(spec=ExperimentTracker)
        with patch("synthgen.tracking.tracker.ClearMLTracker", return_value=instance) as mocked:
            tracker = build_tracker(cfg, rank=0)
            mocked.assert_called_once()
            assert tracker is instance

    def test_factory_returns_wandb_when_only_wandb_enabled(self):
        cfg = FakeTrackingConfig(use_wandb=True)
        with patch("synthgen.tracking.tracker.WandBTracker") as wb:
            instance = MagicMock(spec=ExperimentTracker)
            wb.return_value = instance
            tracker = build_tracker(cfg, rank=0)
            wb.assert_called_once()
            assert tracker is instance

    def test_factory_composite_when_both_enabled(self):
        cfg = FakeTrackingConfig(use_clearml=True, use_wandb=True)
        clearml_instance = MagicMock(spec=ExperimentTracker)
        wandb_instance = MagicMock(spec=ExperimentTracker)
        with (
            patch("synthgen.tracking.tracker.ClearMLTracker", return_value=clearml_instance),
            patch("synthgen.tracking.tracker.WandBTracker", return_value=wandb_instance),
        ):
            tracker = build_tracker(cfg, rank=0)
            assert isinstance(tracker, CompositeTracker)
            tracker.log_metrics({"loss": 1.0}, step=1)
            clearml_instance.log_metrics.assert_called_once_with({"loss": 1.0}, 1)
            wandb_instance.log_metrics.assert_called_once_with({"loss": 1.0}, 1)


class TestNullTracker:
    def test_null_tracker_methods_are_noop(self):
        tracker = NullTracker()
        tracker.log_params({"a": 1})
        tracker.log_metrics({"loss": 0.1}, step=1)
        tracker.log_artifact_json("name", {"k": "v"})
        tracker.log_checkpoint_ref("/tmp/ckpt.pt", step=10)
        tracker.finish()


class TestClearMLTracker:
    def test_clearml_init_connects_params_and_disables_pytorch_auto(self):
        mock_task = MagicMock()
        mock_logger = MagicMock()
        mock_task.get_logger.return_value = mock_logger
        mock_task.connect.side_effect = lambda params: params

        mock_task_cls = MagicMock()
        mock_task_cls.init.return_value = mock_task

        with patch.dict("sys.modules", {"clearml": MagicMock(Task=mock_task_cls)}):
            # Force re-import path to use patched module
            from synthgen.tracking.clearml_backend import ClearMLTracker

            tracker = ClearMLTracker(
                project_name="synthgen",
                task_name="synthgen-dit",
                tags=["test"],
                params={"batch_size": 16, "stage": "dit"},
                upload_checkpoints=False,
                task_cls=mock_task_cls,
            )

        mock_task_cls.init.assert_called_once()
        init_kwargs = mock_task_cls.init.call_args.kwargs
        assert init_kwargs["project_name"] == "synthgen"
        assert init_kwargs["task_name"] == "synthgen-dit"
        assert init_kwargs["auto_connect_frameworks"] == {"pytorch": False}
        mock_task.connect.assert_called()
        mock_task.add_tags.assert_called_once_with(["test"])

        tracker.log_metrics({"loss": 0.5, "learning_rate": 1e-4}, step=100)
        assert mock_logger.report_scalar.call_count == 2

        tracker.log_artifact_json("config", {"stage": "dit"})
        mock_task.upload_artifact.assert_called()
        artifact_name, artifact_obj = (
            mock_task.upload_artifact.call_args.kwargs.get("name")
            or mock_task.upload_artifact.call_args.args[0],
            mock_task.upload_artifact.call_args.kwargs.get("artifact_object")
            or mock_task.upload_artifact.call_args.args[1],
        )
        assert artifact_name == "config"
        assert artifact_obj == {"stage": "dit"}

        tracker.log_checkpoint_ref("/tmp/checkpoint-100.pt", step=100)
        mock_task.set_parameter.assert_called()

        # Must never upload binary via OutputModel when upload_checkpoints=False
        assert not hasattr(tracker, "_output_model") or tracker._output_model is None

        tracker.finish()
        mock_task.close.assert_called_once()

    def test_clearml_init_failure_degrades_to_noop(self):
        mock_task_cls = MagicMock()
        mock_task_cls.init.side_effect = RuntimeError("not configured")

        from synthgen.tracking.clearml_backend import ClearMLTracker

        tracker = ClearMLTracker(
            project_name="synthgen",
            task_name="fail",
            params={},
            task_cls=mock_task_cls,
        )
        # Should not raise
        tracker.log_metrics({"loss": 1.0}, step=1)
        tracker.log_params({"a": 1})
        tracker.finish()


class TestWandBTracker:
    def test_wandb_logs_metrics(self):
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from synthgen.tracking.wandb_backend import WandBTracker

            tracker = WandBTracker(project="synthgen", config={"lr": 1e-4}, wandb_module=mock_wandb)
            mock_wandb.init.assert_called_once_with(project="synthgen", config={"lr": 1e-4})
            tracker.log_metrics({"loss": 0.2}, step=5)
            mock_wandb.log.assert_called_once_with({"loss": 0.2}, step=5)
            tracker.finish()
            mock_wandb.finish.assert_called_once()


class TestDatasetRegistry:
    def test_build_manifest_includes_file_metadata(self, tmp_path):
        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)
        text = tmp_path / "meta.txt"
        text.write_text("kick drum")

        from synthgen.tracking.dataset_registry import build_dataset_manifest

        manifest = build_dataset_manifest(tmp_path)
        assert manifest["file_count"] == 2
        assert manifest["total_bytes"] == audio.stat().st_size + text.stat().st_size
        paths = {f["relative_path"] for f in manifest["files"]}
        assert "sample.wav" in paths
        assert "meta.txt" in paths
        for f in manifest["files"]:
            assert "size_bytes" in f
            assert "sha256" in f
        assert "manifest_sha256" in manifest

    def test_register_dataset_uploads_only_json_manifest(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"audio-bytes")
        mock_task = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "ds-123"
        mock_dataset_cls = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        from synthgen.tracking.dataset_registry import register_dataset_metadata

        result = register_dataset_metadata(
            data_dir=tmp_path,
            dataset_name="synthgen-data",
            dataset_project="synthgen",
            task=mock_task,
            dataset_cls=mock_dataset_cls,
        )

        assert result["dataset_id"] == "ds-123"
        # Must not sync/add audio files
        mock_dataset.add_files.assert_not_called()
        mock_dataset.sync_folder.assert_not_called()
        # Upload only happens for manifest JSON via task artifact
        mock_task.upload_artifact.assert_called()
        args = mock_task.upload_artifact.call_args
        name = args.kwargs.get("name") or args.args[0]
        payload = args.kwargs.get("artifact_object") or args.args[1]
        assert name == "dataset_manifest"
        assert isinstance(payload, dict)
        assert payload["file_count"] == 1
        mock_task.set_parameter.assert_any_call("Datasets/synthgen-data", "ds-123")

    def test_register_missing_data_dir_warns_and_returns_none(self, tmp_path, caplog):
        from synthgen.tracking.dataset_registry import register_dataset_metadata

        missing = tmp_path / "nope"
        with caplog.at_level("WARNING"):
            result = register_dataset_metadata(
                data_dir=missing,
                dataset_name="synthgen-data",
                dataset_project="synthgen",
                task=MagicMock(),
            )
        assert result is None


class TestInferenceTrackingHelper:
    def test_log_generation_run_params_only(self):
        mock_tracker = MagicMock(spec=ExperimentTracker)
        from synthgen.tracking.inference import log_generation_run

        log_generation_run(
            tracker=mock_tracker,
            prompt="warm pad",
            duration=8.0,
            num_steps=25,
            cfg_scale=3.5,
            seed=42,
            checkpoint_path="/ckpt.pt",
            output_path="/out.wav",
        )
        mock_tracker.log_params.assert_called_once()
        params = mock_tracker.log_params.call_args.args[0]
        assert params["prompt"] == "warm pad"
        assert params["output_path"] == "/out.wav"
        # Never report media
        assert not hasattr(mock_tracker, "report_media") or not mock_tracker.report_media.called


class TestTrainingConfigTracking:
    def test_as_dict_includes_clearml_fields(self):
        from synthgen.training.trainer import TrainingConfig

        cfg = TrainingConfig()
        cfg.use_clearml = True
        cfg.clearml_project = "synthgen-vae"
        data = cfg.as_dict()
        assert data["use_clearml"] is True
        assert data["clearml_project"] == "synthgen-vae"
        assert data["clearml_upload_checkpoints"] is False

    def test_scalar_metrics_include_component_losses(self):
        from synthgen.training.trainer import SynthGenTrainer

        losses = {
            "loss": torch.tensor(1.0),
            "spectral_loss": torch.tensor(0.3),
            "kl_loss": torch.tensor(0.01),
        }
        metrics = SynthGenTrainer._scalar_metrics(
            accumulated_loss=0.5,
            losses=losses,
            lr=1e-4,
            steps_per_sec=2.5,
            epoch=1,
        )
        assert metrics["loss"] == 0.5
        assert metrics["spectral_loss"] == pytest.approx(0.3)
        assert metrics["kl_loss"] == pytest.approx(0.01)
        assert metrics["learning_rate"] == pytest.approx(1e-4)

    def test_trainer_logs_metrics_via_injected_tracker(self):
        """Verify logging path without running full train (data package may be absent)."""
        from synthgen.training.trainer import SynthGenTrainer, TrainingConfig

        mock_tracker = MagicMock(spec=ExperimentTracker)
        cfg = TrainingConfig()
        cfg.use_clearml = True
        cfg.log_every_steps = 1
        cfg.save_every_steps = 10_000
        cfg.max_steps = 1

        trainer = SynthGenTrainer.__new__(SynthGenTrainer)
        trainer.config = cfg
        trainer.rank = 0
        trainer.world_size = 1
        trainer.global_step = 0
        trainer.epoch = 0
        trainer.tracker = mock_tracker
        trainer.device = torch.device("cpu")
        trainer.scaler = None

        metrics = trainer._scalar_metrics(0.25, {"loss": 0.25, "l1_loss": 0.1}, 1e-4, 1.0, 0)
        trainer.tracker.log_metrics(metrics, step=1)
        mock_tracker.log_metrics.assert_called_once()
        assert mock_tracker.log_metrics.call_args.args[0]["l1_loss"] == pytest.approx(0.1)
