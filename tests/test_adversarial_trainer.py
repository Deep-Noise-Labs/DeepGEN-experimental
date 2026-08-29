"""
End-to-end smoke tests for the adversarial VAE training stage.

These build a real ``SynthGenTrainer`` against a tiny on-disk dataset and run a
handful of steps, so the wiring that unit tests cannot see - two optimizers,
the warmup gate, gradient isolation between generator and discriminator,
checkpoint round-trips - is actually exercised.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from synthgen.training.trainer import SynthGenTrainer, TrainingConfig

SR = 16000  # keep the smoke tests fast; the objective is sample-rate agnostic


@pytest.fixture
def tiny_dataset(tmp_path):
    """Four short noise-plus-tone clips with captions."""
    root = tmp_path / "data"
    (root / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        n = SR // 2
        t = np.arange(n) / SR
        tone = 0.3 * np.sin(2 * np.pi * (110 * (i + 1)) * t)
        stereo = np.stack([tone + 0.02 * rng.standard_normal(n),
                           tone * 0.8 + 0.02 * rng.standard_normal(n)], axis=-1)
        name = f"audio/{i:03d}.wav"
        sf.write(root / name, stereo.astype(np.float32), SR)
        rows.append({"file_name": name, "caption": f"test tone {i}"})
    with open(root / "metadata.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return root


def _config(tmp_path, tiny_dataset, **overrides):
    config = TrainingConfig()
    config.stage = "vae"
    config.sample_rate = SR
    config.max_duration = 0.5
    config.vae_latent_dim = 8
    config.batch_size = 2
    config.gradient_accumulation_steps = 1
    config.num_workers = 0
    config.mixed_precision = "no"
    config.max_steps = 4
    config.warmup_steps = 1
    config.log_every_steps = 100
    config.save_every_steps = 10_000
    config.eval_every_steps = 10_000
    config.data_dir = str(tiny_dataset)
    config.checkpoint_dir = str(tmp_path / "ckpt")
    config.adv_start_step = 0
    config.disc_periods = (2, 3)
    config.disc_mpd_channels = (8, 16)
    config.disc_stft_channels = 8
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_adversarial_vae_stage_runs(tmp_path, tiny_dataset):
    trainer = SynthGenTrainer(_config(tmp_path, tiny_dataset))
    assert trainer.discriminator is not None
    assert trainer.disc_optimizer is not None

    trainer.train()

    assert trainer.global_step == 4
    for param in trainer.model.parameters():
        assert torch.isfinite(param).all()
    for param in trainer.discriminator.parameters():
        assert torch.isfinite(param).all()


def test_warmup_gate_defers_the_discriminator(tmp_path, tiny_dataset):
    """Before ``adv_start_step`` the objective must be reconstruction-only."""
    trainer = SynthGenTrainer(_config(tmp_path, tiny_dataset, adv_start_step=1000))
    assert not trainer._adversarial_active()

    audio = next(iter(trainer.dataloader))["audio"].to(trainer.device)
    losses = trainer._vae_step(audio, None)
    assert "adv_loss" not in losses and "disc_loss" not in losses

    trainer.global_step = 1000
    assert trainer._adversarial_active()
    losses = trainer._vae_step(audio, None)
    assert "adv_loss" in losses and "fm_loss" in losses and "disc_loss" in losses


def test_generator_backward_does_not_touch_discriminator_weights(tmp_path, tiny_dataset):
    """
    The generator's loss is computed *through* the discriminator. If leaf
    accumulation is not switched off, the generator's backward pollutes the
    discriminator's gradients and corrupts its next update.
    """
    trainer = SynthGenTrainer(_config(tmp_path, tiny_dataset))
    audio = next(iter(trainer.dataloader))["audio"].to(trainer.device)

    losses = trainer._vae_step(audio, None)          # runs the discriminator step
    trainer.disc_optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()                         # generator backward

    leaked = [
        name for name, param in trainer.discriminator.named_parameters()
        if param.grad is not None and param.grad.abs().sum() > 0
    ]
    assert not leaked, f"generator backward leaked gradients into: {leaked[:5]}"


def test_legacy_objective_disables_the_discriminator(tmp_path, tiny_dataset):
    trainer = SynthGenTrainer(
        _config(tmp_path, tiny_dataset, legacy_vae_objective=True)
    )
    assert trainer.discriminator is None
    assert trainer.disc_optimizer is None

    audio = next(iter(trainer.dataloader))["audio"].to(trainer.device)
    losses = trainer._vae_step(audio, None)
    assert set(losses) == {"loss", "l1_loss", "spectral_loss", "kl_loss"}


def test_checkpoint_round_trips_the_discriminator(tmp_path, tiny_dataset):
    trainer = SynthGenTrainer(_config(tmp_path, tiny_dataset))
    trainer.train()
    trainer._save_checkpoint()

    path = sorted((tmp_path / "ckpt").glob("checkpoint-*.pt"))[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "discriminator_state_dict" in payload
    assert "disc_optimizer_state_dict" in payload

    resumed = SynthGenTrainer(_config(tmp_path, tiny_dataset, resume_from=str(path)))
    before = dict(trainer.discriminator.named_parameters())
    for name, param in resumed.discriminator.named_parameters():
        assert torch.allclose(param, before[name]), f"{name} did not round-trip"
