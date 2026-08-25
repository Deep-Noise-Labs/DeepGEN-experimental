"""
Unit tests for the multi-resolution STFT discriminator and adversarial losses.
"""

import torch

from synthgen.model.discriminator import (
    MultiResolutionSTFTDiscriminator,
    STFTSubDiscriminator,
)
from synthgen.model.vae import AudioVAE
from synthgen.training.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


def _small_discriminator() -> MultiResolutionSTFTDiscriminator:
    return MultiResolutionSTFTDiscriminator(
        fft_sizes=(512, 256),
        in_channels=2,
        base_channels=8,
    )


class TestSTFTSubDiscriminator:
    def test_output_shapes(self):
        disc = STFTSubDiscriminator(
            fft_size=512, hop_size=128, win_size=512,
            in_channels=2, base_channels=8,
        )
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)

        assert logits.shape[0] == 2
        assert logits.shape[1] == 1
        assert len(features) == 5
        for f in features:
            assert f.shape[0] == 2

    def test_mono_input(self):
        disc = STFTSubDiscriminator(
            fft_size=256, hop_size=64, win_size=256,
            in_channels=1, base_channels=8,
        )
        x = torch.randn(2, 1, 4096)
        logits, _ = disc(x)
        assert logits.shape[0] == 2

    def test_gradients_flow_to_input(self):
        disc = STFTSubDiscriminator(
            fft_size=256, hop_size=64, win_size=256,
            in_channels=2, base_channels=8,
        )
        x = torch.randn(1, 2, 4096, requires_grad=True)
        logits, _ = disc(x)
        logits.mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestMultiResolutionSTFTDiscriminator:
    def test_one_output_per_resolution(self):
        disc = _small_discriminator()
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)
        assert len(logits) == 2
        assert len(features) == 2

    def test_phase_sensitivity(self):
        """The complex-STFT input must distinguish signals with identical
        magnitude spectra but different phase alignment."""
        disc = _small_discriminator()
        t = torch.arange(8192) / 44100.0
        a = torch.sin(2 * torch.pi * 440.0 * t)
        b = torch.sin(2 * torch.pi * 440.0 * t + 1.5)
        a = a.expand(1, 2, -1).contiguous()
        b = b.expand(1, 2, -1).contiguous()

        spec_a = disc.discriminators[0]._spectrogram(a)
        spec_b = disc.discriminators[0]._spectrogram(b)
        assert not torch.allclose(spec_a, spec_b, atol=1e-3)


class TestAdversarialLosses:
    def test_discriminator_loss_rewards_correct_scores(self):
        confident = discriminator_loss(
            real_logits=[torch.full((1, 1, 4, 4), 2.0)],
            fake_logits=[torch.full((1, 1, 4, 4), -2.0)],
        )
        wrong = discriminator_loss(
            real_logits=[torch.full((1, 1, 4, 4), -2.0)],
            fake_logits=[torch.full((1, 1, 4, 4), 2.0)],
        )
        assert confident.item() == 0.0
        assert wrong.item() > confident.item()

    def test_generator_loss_decreases_as_fake_scores_rise(self):
        low = generator_adversarial_loss([torch.full((1, 1, 4, 4), -1.0)])
        high = generator_adversarial_loss([torch.full((1, 1, 4, 4), 1.0)])
        assert high.item() < low.item()

    def test_feature_matching_zero_for_identical_features(self):
        features = [[torch.randn(1, 8, 4, 4) for _ in range(3)]]
        loss = feature_matching_loss(features, features)
        assert loss.item() < 1e-6

    def test_feature_matching_positive_for_different_features(self):
        real = [[torch.randn(1, 8, 4, 4) for _ in range(3)]]
        fake = [[torch.randn(1, 8, 4, 4) for _ in range(3)]]
        loss = feature_matching_loss(real, fake)
        assert loss.item() > 0

    def test_feature_matching_only_updates_generator_side(self):
        real = [[torch.randn(1, 4, 4, 4, requires_grad=True)]]
        fake = [[torch.randn(1, 4, 4, 4, requires_grad=True)]]
        loss = feature_matching_loss(real, fake)
        loss.backward()
        assert real[0][0].grad is None
        assert fake[0][0].grad is not None


class TestAdversarialTrainingStep:
    def test_vae_and_discriminator_updates_are_isolated(self):
        """One full adversarial step: discriminator gradients must not reach
        the VAE, and generator gradients must not reach the discriminator."""
        torch.manual_seed(0)
        vae = AudioVAE(
            in_channels=2,
            latent_dim=8,
            base_channels=8,
            encoder_channel_multipliers=(1, 2),
            decoder_channel_multipliers=(2, 1),
            strides=(4, 4),
            num_residual_per_block=1,
        )
        disc = MultiResolutionSTFTDiscriminator(
            fft_sizes=(256,), in_channels=2, base_channels=8
        )

        audio = torch.randn(1, 2, 4096)
        reconstruction, target, _, _ = vae(audio)

        # Discriminator update on detached reconstruction
        real_logits, real_features = disc(target.detach())
        fake_logits, _ = disc(reconstruction.detach())
        d_loss = discriminator_loss(real_logits, fake_logits)
        d_loss.backward()

        assert all(p.grad is None for p in vae.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in disc.parameters()
        )

        # Generator update with discriminator frozen
        for p in disc.parameters():
            p.grad = None
            p.requires_grad_(False)
        fake_logits_g, fake_features = disc(reconstruction)
        g_loss = generator_adversarial_loss(
            fake_logits_g
        ) + feature_matching_loss(real_features, fake_features)
        g_loss.backward()

        assert all(p.grad is None for p in disc.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in vae.parameters()
        )
