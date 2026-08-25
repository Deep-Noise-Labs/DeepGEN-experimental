"""
Unit tests for the multi-scale STFT discriminator and adversarial losses.
"""

import torch

from synthgen.model.discriminator import (
    MultiScaleSTFTDiscriminator,
    STFTSubDiscriminator,
)
from synthgen.training.losses import DiscriminatorLoss, GeneratorAdversarialLoss


class TestSTFTSubDiscriminator:
    def test_output_shapes(self):
        disc = STFTSubDiscriminator(
            fft_size=512, hop_size=128, win_size=512, audio_channels=2
        )
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)
        assert logits.shape[0] == 2
        assert logits.shape[1] == 1
        assert len(features) == 5
        assert all(f.shape[0] == 2 for f in features)

    def test_mono_audio(self):
        disc = STFTSubDiscriminator(
            fft_size=256, hop_size=64, win_size=256, audio_channels=1
        )
        x = torch.randn(3, 1, 4096)
        logits, features = disc(x)
        assert logits.shape[0] == 3

    def test_odd_length_input(self):
        disc = STFTSubDiscriminator(
            fft_size=256, hop_size=64, win_size=256, audio_channels=2
        )
        x = torch.randn(1, 2, 4097)
        logits, _ = disc(x)
        assert torch.isfinite(logits).all()

    def test_gradients_flow_to_input(self):
        disc = STFTSubDiscriminator(
            fft_size=256, hop_size=64, win_size=256, audio_channels=2
        )
        x = torch.randn(1, 2, 4096, requires_grad=True)
        logits, _ = disc(x)
        logits.mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestMultiScaleSTFTDiscriminator:
    def test_number_of_scales(self):
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=(512, 256, 128), audio_channels=2, base_filters=8
        )
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)
        assert len(logits) == 3
        assert len(features) == 3
        assert all(len(f) == 5 for f in features)

    def test_default_config(self):
        disc = MultiScaleSTFTDiscriminator(audio_channels=2, base_filters=4)
        x = torch.randn(1, 2, 16384)
        logits, features = disc(x)
        assert len(logits) == 5


class TestDiscriminatorLoss:
    def test_perfect_discrimination_gives_low_loss(self):
        loss_fn = DiscriminatorLoss()
        # Real logits far above +1, fake logits far below -1 → hinge inactive
        logits_real = [torch.full((2, 1, 4, 4), 5.0)]
        logits_fake = [torch.full((2, 1, 4, 4), -5.0)]
        loss = loss_fn(logits_real, logits_fake)
        assert loss.item() < 1e-6

    def test_confused_discriminator_gives_high_loss(self):
        loss_fn = DiscriminatorLoss()
        logits_real = [torch.full((2, 1, 4, 4), -5.0)]
        logits_fake = [torch.full((2, 1, 4, 4), 5.0)]
        loss = loss_fn(logits_real, logits_fake)
        assert loss.item() > 10.0

    def test_multi_scale_averaging(self):
        loss_fn = DiscriminatorLoss()
        logits_real = [torch.zeros(1, 1, 2, 2) for _ in range(4)]
        logits_fake = [torch.zeros(1, 1, 2, 2) for _ in range(4)]
        loss = loss_fn(logits_real, logits_fake)
        # relu(1-0) + relu(1+0) = 2 per scale, averaged over scales
        assert abs(loss.item() - 2.0) < 1e-6


class TestGeneratorAdversarialLoss:
    def test_components_present(self):
        loss_fn = GeneratorAdversarialLoss()
        logits_fake = [torch.randn(2, 1, 4, 4)]
        features_real = [[torch.randn(2, 8, 4, 4) for _ in range(3)]]
        features_fake = [[torch.randn(2, 8, 4, 4) for _ in range(3)]]
        losses = loss_fn(logits_fake, features_real, features_fake)
        assert "adv_loss" in losses
        assert "feature_matching_loss" in losses
        assert torch.isfinite(losses["adv_loss"])
        assert torch.isfinite(losses["feature_matching_loss"])

    def test_feature_matching_zero_for_identical_features(self):
        loss_fn = GeneratorAdversarialLoss()
        feats = [[torch.randn(2, 8, 4, 4) for _ in range(3)]]
        logits_fake = [torch.full((2, 1, 4, 4), 5.0)]
        losses = loss_fn(logits_fake, feats, feats)
        assert losses["feature_matching_loss"].item() < 1e-6
        # Fake logits above +1 → generator hinge satisfied
        assert losses["adv_loss"].item() < 1e-6

    def test_generator_gradient_flow(self):
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=(256, 128), audio_channels=2, base_filters=4
        )
        gen_loss_fn = GeneratorAdversarialLoss()

        fake = torch.randn(1, 2, 4096, requires_grad=True)
        real = torch.randn(1, 2, 4096)

        with torch.no_grad():
            _, features_real = disc(real)
        logits_fake, features_fake = disc(fake)

        losses = gen_loss_fn(logits_fake, features_real, features_fake)
        total = losses["adv_loss"] + losses["feature_matching_loss"]
        total.backward()

        assert fake.grad is not None
        assert torch.isfinite(fake.grad).all()
        assert fake.grad.abs().sum() > 0


class TestAdversarialTrainingStep:
    def test_gan_update_reduces_no_grads_leakage(self):
        """One full G + D update cycle: gradients land where they should."""
        from synthgen.model.vae import AudioVAE

        torch.manual_seed(0)
        vae = AudioVAE(in_channels=2, latent_dim=8, base_channels=4)
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=(256,), audio_channels=2, base_filters=4
        )
        disc_loss_fn = DiscriminatorLoss()
        gen_loss_fn = GeneratorAdversarialLoss()

        audio = torch.randn(1, 2, 4096)
        recon, target, mean, log_var = vae(audio)

        # Generator pass with frozen discriminator
        for p in disc.parameters():
            p.requires_grad = False
        with torch.no_grad():
            _, features_real = disc(target)
        logits_fake, features_fake = disc(recon)
        g_losses = gen_loss_fn(logits_fake, features_real, features_fake)
        (g_losses["adv_loss"] + g_losses["feature_matching_loss"]).backward()

        assert all(p.grad is None for p in disc.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in vae.decoder.parameters()
        )

        # Discriminator pass on detached audio
        for p in disc.parameters():
            p.requires_grad = True
        logits_real, _ = disc(target.detach())
        logits_fake, _ = disc(recon.detach())
        d_loss = disc_loss_fn(logits_real, logits_fake)
        d_loss.backward()

        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in disc.parameters()
        )
