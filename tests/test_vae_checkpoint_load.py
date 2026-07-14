"""Tests for loading Stage-1 VAE weights into SynthGen."""

from pathlib import Path

import torch

from synthgen.model.synthgen import SynthGen
from synthgen.model.vae import AudioVAE
from synthgen.training.trainer import TrainingConfig, load_vae_weights_into_synthgen


def test_load_bare_vae_checkpoint(tmp_path: Path):
    vae = AudioVAE(
        in_channels=2,
        latent_dim=32,
        base_channels=16,
        strides=(4, 4, 4, 4),
    )
    # Mark a weight so we can detect a successful load
    with torch.no_grad():
        for p in vae.parameters():
            p.fill_(0.123)

    ckpt_path = tmp_path / "vae.pt"
    torch.save({"model_state_dict": vae.state_dict()}, ckpt_path)

    model = SynthGen(
        vae_latent_dim=32,
        vae_base_channels=16,
        vae_strides=(4, 4, 4, 4),
        dit_model_dim=64,
        dit_num_heads=4,
        dit_num_layers=2,
        use_dummy_text_encoder=True,
    )
    # Ensure different init
    with torch.no_grad():
        for p in model.vae.parameters():
            p.zero_()

    load_vae_weights_into_synthgen(model, str(ckpt_path), torch.device("cpu"))
    loaded = next(model.vae.parameters()).reshape(-1)[0].item()
    assert abs(loaded - 0.123) < 1e-5


def test_load_prefixed_vae_keys(tmp_path: Path):
    model_src = SynthGen(
        vae_latent_dim=32,
        vae_base_channels=16,
        vae_strides=(4, 4, 4, 4),
        dit_model_dim=64,
        dit_num_heads=4,
        dit_num_layers=2,
        use_dummy_text_encoder=True,
    )
    with torch.no_grad():
        for p in model_src.vae.parameters():
            p.fill_(0.456)

    ckpt_path = tmp_path / "synthgen.pt"
    torch.save({"model_state_dict": model_src.state_dict()}, ckpt_path)

    model = SynthGen(
        vae_latent_dim=32,
        vae_base_channels=16,
        vae_strides=(4, 4, 4, 4),
        dit_model_dim=64,
        dit_num_heads=4,
        dit_num_layers=2,
        use_dummy_text_encoder=True,
    )
    with torch.no_grad():
        for p in model.vae.parameters():
            p.zero_()

    load_vae_weights_into_synthgen(model, str(ckpt_path), torch.device("cpu"))
    loaded = next(model.vae.parameters()).reshape(-1)[0].item()
    assert abs(loaded - 0.456) < 1e-5


def test_config_yaml_new_fields():
    cfg = TrainingConfig.from_yaml("configs/dit_audiocaps.yaml")
    assert cfg.stage == "dit"
    assert cfg.vae_checkpoint is not None
    assert cfg.cfg_dropout_prob == 0.1
    assert cfg.max_steps == 10000

    vae_cfg = TrainingConfig.from_yaml("configs/vae_audiocaps.yaml")
    assert vae_cfg.stage == "vae"
    assert vae_cfg.data_dir.endswith("audiocaps")
