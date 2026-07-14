"""
Training loop for SynthGen.

Supports:
- Two-stage training (VAE first, then DiT)
- Mixed precision (bf16/fp16)
- Gradient accumulation
- Distributed training via PyTorch DDP
- Checkpointing and resumption
- ClearML experiment tracking (primary) and optional Weights & Biases
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from synthgen.model.synthgen import SynthGen
from synthgen.model.vae import AudioVAE
from synthgen.tracking import build_tracker
from synthgen.tracking.dataset_registry import register_dataset_metadata
from synthgen.tracking.null import NullTracker
from synthgen.tracking.tracker import ExperimentTracker, config_as_dict, get_clearml_task
from synthgen.training.losses import FlowMatchingLoss, VAELoss
from synthgen.training.scheduler import WarmupCosineScheduler

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class TrainingConfig:
    """Training configuration with sensible defaults."""

    # Model
    vae_latent_dim: int = 64
    dit_model_dim: int = 1024
    dit_num_heads: int = 16
    dit_num_layers: int = 20
    dit_mlp_ratio: float = 4.0

    # Audio
    sample_rate: int = 44100
    audio_channels: int = 2
    max_duration: float = 15.0

    # Training
    stage: str = "dit"  # "vae" or "dit"
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 500000
    warmup_steps: int = 5000
    min_lr: float = 1e-6

    # Mixed precision
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_steps: int = 5000
    resume_from: str | None = None

    # Logging / experiment tracking
    log_every_steps: int = 100
    eval_every_steps: int = 2500
    use_clearml: bool = False
    clearml_project: str = "synthgen"
    clearml_task_name: str | None = None
    clearml_tags: list | None = None
    clearml_dataset_name: str = "synthgen-data"
    clearml_register_dataset: bool = True
    clearml_upload_checkpoints: bool = False
    use_wandb: bool = False
    wandb_project: str = "synthgen"

    # Data
    data_dir: str = "./data"
    num_workers: int = 8

    # Distributed
    distributed: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        import yaml

        config = cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def as_dict(self) -> dict[str, Any]:
        """Serialize config fields for checkpointing and trackers."""
        return config_as_dict(self)


# =============================================================================
# Trainer
# =============================================================================


class SynthGenTrainer:
    """
    Main trainer class for SynthGen.

    Handles the full training loop including:
    - Model initialization
    - Data loading
    - Optimization
    - Checkpointing
    - Experiment logging (ClearML primary, WandB optional)
    """

    def __init__(
        self,
        config: TrainingConfig,
        tracker: ExperimentTracker | None = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0

        # Setup distributed training
        self.rank = 0
        self.world_size = 1
        if config.distributed:
            self._setup_distributed()

        # Experiment tracker as early as possible after rank is known
        self.tracker: ExperimentTracker = (
            tracker if tracker is not None else build_tracker(config, rank=self.rank)
        )
        self._register_dataset_metadata()
        if not isinstance(self.tracker, NullTracker):
            self.tracker.log_artifact_json("training_config", config.as_dict())

        # Initialize model
        self._init_model()

        # Initialize optimizer and scheduler
        self._init_optimizer()

        # Initialize data
        self._init_data()

        # Initialize loss
        self._init_loss()

        # Mixed precision
        self.scaler = None
        if config.mixed_precision == "fp16":
            self.scaler = GradScaler()

        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _register_dataset_metadata(self) -> None:
        """Attach metadata-only dataset lineage to ClearML (no media upload)."""
        if self.rank != 0 or not self.config.use_clearml:
            return
        if not self.config.clearml_register_dataset:
            return
        task = get_clearml_task(self.tracker)
        register_dataset_metadata(
            data_dir=self.config.data_dir,
            dataset_name=self.config.clearml_dataset_name,
            dataset_project=self.config.clearml_project,
            task=task,
        )

    def _setup_distributed(self):
        """Initialize distributed training."""
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)

    def _init_model(self):
        """Initialize the model."""
        config = self.config

        if config.stage == "vae":
            self.model = AudioVAE(
                in_channels=config.audio_channels,
                latent_dim=config.vae_latent_dim,
            ).to(self.device)
        else:
            self.model = SynthGen(
                vae_latent_dim=config.vae_latent_dim,
                dit_model_dim=config.dit_model_dim,
                dit_num_heads=config.dit_num_heads,
                dit_num_layers=config.dit_num_layers,
                dit_mlp_ratio=config.dit_mlp_ratio,
                use_dummy_text_encoder=False,
            ).to(self.device)

            # Freeze VAE during DiT training
            if hasattr(self.model, "vae"):
                for param in self.model.vae.parameters():
                    param.requires_grad = False

        if config.distributed:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        self.tracker.log_params(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
        )

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        config = self.config

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr=config.min_lr,
        )

    def _init_data(self):
        """Initialize data loaders."""
        from synthgen.data.dataset import AudioTextDataset, SynthGenCollator

        config = self.config

        dataset = AudioTextDataset(
            data_dir=Path(config.data_dir),
            sample_rate=config.sample_rate,
            duration=config.max_duration,
            channels=config.audio_channels,
            augment=True,
        )

        sampler = None
        if config.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=SynthGenCollator(),
            pin_memory=True,
            drop_last=True,
        )

    def _init_loss(self):
        """Initialize loss functions."""
        if self.config.stage == "vae":
            self.loss_fn = VAELoss()
        else:
            self.loss_fn = FlowMatchingLoss(weighting="min_snr")

    @staticmethod
    def _scalar_metrics(
        accumulated_loss: float,
        losses: dict[str, Any],
        lr: float,
        steps_per_sec: float,
        epoch: int,
    ) -> dict[str, float]:
        """Build the metric dict logged each log_every_steps."""
        metrics: dict[str, float] = {
            "loss": float(accumulated_loss),
            "learning_rate": float(lr),
            "steps_per_second": float(steps_per_sec),
            "epoch": float(epoch),
        }
        for key, value in losses.items():
            if key == "loss":
                continue
            if torch.is_tensor(value):
                metrics[key] = float(value.detach().item())
            else:
                metrics[key] = float(value)
        return metrics

    def train(self):
        """Main training loop."""
        config = self.config
        logger.info(f"Starting training: stage={config.stage}, max_steps={config.max_steps}")

        self.model.train()
        data_iter = iter(self.dataloader)
        accumulated_loss = 0.0
        step_start_time = time.time()
        last_losses: dict[str, Any] = {}

        try:
            while self.global_step < config.max_steps:
                # Get batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    if config.distributed:
                        self.dataloader.sampler.set_epoch(self.epoch)
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Move to device
                audio = batch["audio"].to(self.device)
                captions = batch["captions"]
                durations = batch["durations"].to(self.device)

                # Forward pass with mixed precision
                amp_dtype = {
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                    "no": None,
                }[config.mixed_precision]

                with autocast(dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    if config.stage == "vae":
                        reconstruction, target, mean, log_var = self.model(audio)
                        losses = self.loss_fn(reconstruction, target, mean, log_var)
                    else:
                        losses = self.model.compute_loss(audio, captions, durations)

                last_losses = losses
                loss = losses["loss"] / config.gradient_accumulation_steps

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item()

                # Optimizer step (with gradient accumulation)
                if (self.global_step + 1) % config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % config.log_every_steps == 0 and self.rank == 0:
                    elapsed = time.time() - step_start_time
                    steps_per_sec = config.log_every_steps / elapsed
                    lr = self.scheduler.get_lr()[0]

                    log_msg = (
                        f"Step {self.global_step}/{config.max_steps} | "
                        f"Loss: {accumulated_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )
                    logger.info(log_msg)

                    metrics = self._scalar_metrics(
                        accumulated_loss=accumulated_loss,
                        losses=last_losses,
                        lr=lr,
                        steps_per_sec=steps_per_sec,
                        epoch=self.epoch,
                    )
                    self.tracker.log_metrics(metrics, step=self.global_step)

                    accumulated_loss = 0.0
                    step_start_time = time.time()

                # Save checkpoint
                if self.global_step % config.save_every_steps == 0 and self.rank == 0:
                    self._save_checkpoint()

            logger.info("Training complete!")
        finally:
            self.tracker.finish()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint-{self.global_step}.pt"

        model_state = self.model.module.state_dict() if isinstance(
            self.model, DDP
        ) else self.model.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.as_dict(),
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        self.tracker.log_checkpoint_ref(str(checkpoint_path), step=self.global_step)

        # Keep only last 3 checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed from step {self.global_step}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train SynthGen model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--stage", type=str, choices=["vae", "dit"], default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Enable ClearML experiment tracking (primary)",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default=None,
        help="ClearML project name",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases as secondary tracker",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Override with CLI args only when explicitly provided
    if args.stage is not None:
        config.stage = args.stage
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.clearml:
        config.use_clearml = True
    if args.clearml_project is not None:
        config.clearml_project = args.clearml_project
    if args.wandb:
        config.use_wandb = True

    # Check for distributed environment
    if "RANK" in os.environ:
        config.distributed = True

    trainer = SynthGenTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
