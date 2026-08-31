"""
End-to-end smoke tests for the adversarial Stage-1 training path.

These run a handful of real optimiser steps on a tiny synthetic dataset. They
are slow-ish for unit tests but they are the only thing that catches the class
of bug that matters here: a GAN loop that runs but silently trains nothing,
because a critic gradient leaked into the wrong optimiser or the warm-up gate
never opened.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from synthgen.training.trainer import SynthGenTrainer, TrainingConfig

SR = 16000


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Four one-second clips of band-limited noise with captions."""
    root = tmp_path / "data"
    (root / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)

    with open(root / "metadata.jsonl", "w") as manifest:
        for i in range(4):
            name = f"audio/{i:03d}.wav"
            audio = rng.normal(0, 0.1, size=(SR, 2)).astype(np.float32)
            sf.write(root / name, audio, SR)
            manifest.write(json.dumps({"file_name": name, "caption": f"tone {i}"}) + "\n")

    return root


def _config(data_dir: Path, checkpoint_dir: Path, **overrides) -> TrainingConfig:
    config = TrainingConfig()
    config.stage = "vae"
    config.sample_rate = SR
    config.max_duration = 0.5
    config.audio_channels = 2
    config.vae_latent_dim = 8
    config.batch_size = 2
    config.gradient_accumulation_steps = 1
    config.max_steps = 4
    config.warmup_steps = 1
    config.mixed_precision = "no"
    config.num_workers = 0
    config.data_dir = str(data_dir)
    config.checkpoint_dir = str(checkpoint_dir)
    config.save_every_steps = 1000
    config.log_every_steps = 1000
    # Keep the critic banks tiny; this is a plumbing test, not a quality test.
    config.disc_periods = (2, 3)
    config.disc_fft_sizes = (256,)
    config.disc_hop_sizes = (64,)
    config.mel_weight = 1.0

    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestAdversarialTrainingLoop:
    def test_runs_and_updates_both_players(self, tiny_dataset, tmp_path):
        config = _config(
            tiny_dataset,
            tmp_path / "ckpt",
            vae_adversarial=True,
            disc_start_step=0,
        )
        trainer = SynthGenTrainer(config)
        assert trainer.discriminator is not None

        before_g = next(trainer.model.parameters()).detach().clone()
        before_d = next(trainer.discriminator.parameters()).detach().clone()

        trainer.train()

        assert not torch.equal(before_g, next(trainer.model.parameters()))
        assert not torch.equal(before_d, next(trainer.discriminator.parameters()))

    def test_warmup_gate_holds_the_critic_back(self, tiny_dataset, tmp_path):
        config = _config(
            tiny_dataset,
            tmp_path / "ckpt",
            vae_adversarial=True,
            disc_start_step=1000,  # never reached in 4 steps
        )
        trainer = SynthGenTrainer(config)
        assert not trainer.adversarial_active

        before_d = next(trainer.discriminator.parameters()).detach().clone()
        trainer.train()

        assert torch.equal(before_d, next(trainer.discriminator.parameters()))

    def test_generator_gradients_are_not_polluted_by_the_critic_step(
        self, tiny_dataset, tmp_path
    ):
        """
        The critic's own backward runs on a detached reconstruction, so it must
        leave nothing on the generator's parameters.
        """
        config = _config(
            tiny_dataset, tmp_path / "ckpt", vae_adversarial=True, disc_start_step=0
        )
        trainer = SynthGenTrainer(config)

        audio = torch.randn(1, 2, 4096) * 0.1
        reconstruction, target, _, _ = trainer.model(audio)
        trainer._discriminator_step(reconstruction, target)

        assert all(p.grad is None for p in trainer.model.parameters())

    def test_checkpoint_round_trips_the_critic(self, tiny_dataset, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        config = _config(
            tiny_dataset,
            checkpoint_dir,
            vae_adversarial=True,
            disc_start_step=0,
            max_steps=2,
        )
        trainer = SynthGenTrainer(config)
        trainer.train()

        saved = sorted(checkpoint_dir.glob("checkpoint-*.pt"))
        assert saved, "training left no checkpoint"
        payload = torch.load(saved[-1], map_location="cpu", weights_only=False)
        assert "discriminator_state_dict" in payload
        assert "disc_optimizer_state_dict" in payload

        resumed = _config(
            tiny_dataset,
            checkpoint_dir,
            vae_adversarial=True,
            disc_start_step=0,
            max_steps=2,
            resume_from=str(saved[-1]),
        )
        trainer_resumed = SynthGenTrainer(resumed)
        torch.testing.assert_close(
            next(trainer_resumed.discriminator.parameters()),
            next(trainer.discriminator.parameters()),
        )

    def test_adversarial_can_be_disabled(self, tiny_dataset, tmp_path):
        config = _config(tiny_dataset, tmp_path / "ckpt", vae_adversarial=False)
        trainer = SynthGenTrainer(config)
        assert trainer.discriminator is None
        assert trainer.disc_optimizer is None
        trainer.train()  # must not raise
