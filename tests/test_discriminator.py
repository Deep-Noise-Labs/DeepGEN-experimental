"""
Unit tests for the multi-scale STFT discriminator and adversarial losses.

All tests run on CPU with small tensors.
"""

import torch

from synthgen.model.discriminator import (
    MultiScaleSTFTDiscriminator,
    STFTSubDiscriminator,
)
from synthgen.model.vae import AudioVAE
from synthgen.training.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


SMALL_FFT_SIZES = (512, 256)


class TestSTFTSubDiscriminator:
    def test_output_shapes(self):
        disc = STFTSubDiscriminator(fft_size=512, in_channels=2, filters=8, max_filters=32)
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)

        assert logits.dim() == 4
        assert logits.shape[0] == 2
        assert logits.shape[1] == 1
        assert len(features) == len(disc.convs)
        for f in features:
            assert f.shape[0] == 2

    def test_mono_input(self):
        disc = STFTSubDiscriminator(fft_size=256, in_channels=1, filters=8, max_filters=32)
        x = torch.randn(2, 1, 4096)
        logits, _ = disc(x)
        assert logits.shape[0] == 2


class TestMultiScaleSTFTDiscriminator:
    def test_one_output_per_scale(self):
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=SMALL_FFT_SIZES, in_channels=2, filters=8, max_filters=32
        )
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)

        assert len(logits) == len(SMALL_FFT_SIZES)
        assert len(features) == len(SMALL_FFT_SIZES)

    def test_gradients_flow_to_input(self):
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=SMALL_FFT_SIZES, in_channels=2, filters=8, max_filters=32
        )
        x = torch.randn(1, 2, 4096, requires_grad=True)
        logits, _ = disc(x)
        generator_adversarial_loss(logits).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestAdversarialLosses:
    def _make_disc_and_inputs(self):
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=SMALL_FFT_SIZES, in_channels=2, filters=8, max_filters=32
        )
        real = torch.randn(2, 2, 8192)
        fake = torch.randn(2, 2, 8192)
        return disc, real, fake

    def test_discriminator_loss_positive_and_finite(self):
        disc, real, fake = self._make_disc_and_inputs()
        real_logits, _ = disc(real)
        fake_logits, _ = disc(fake)
        loss = discriminator_loss(real_logits, fake_logits)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_generator_loss_finite(self):
        disc, _, fake = self._make_disc_and_inputs()
        fake_logits, _ = disc(fake)
        loss = generator_adversarial_loss(fake_logits)
        assert torch.isfinite(loss)

    def test_feature_matching_zero_for_identical_inputs(self):
        disc, real, _ = self._make_disc_and_inputs()
        _, features_a = disc(real)
        _, features_b = disc(real.clone())
        loss = feature_matching_loss(features_a, features_b)
        assert loss.item() < 1e-6

    def test_feature_matching_positive_for_different_inputs(self):
        disc, real, fake = self._make_disc_and_inputs()
        _, real_features = disc(real)
        _, fake_features = disc(fake)
        loss = feature_matching_loss(fake_features, real_features)
        assert loss.item() > 0

    def test_generator_terms_reach_vae_decoder(self):
        """Adversarial + FM gradients must flow back into the VAE decoder."""
        vae = AudioVAE(
            in_channels=2,
            latent_dim=8,
            base_channels=8,
            encoder_channel_multipliers=(1, 2),
            decoder_channel_multipliers=(2, 1),
            strides=(4, 4),
            num_residual_per_block=1,
        )
        disc = MultiScaleSTFTDiscriminator(
            fft_sizes=SMALL_FFT_SIZES, in_channels=2, filters=8, max_filters=32
        )

        audio = torch.randn(1, 2, 4096)
        reconstruction, target, _, _ = vae(audio)

        fake_logits, fake_features = disc(reconstruction)
        with torch.no_grad():
            _, real_features = disc(target)

        loss = generator_adversarial_loss(fake_logits) + feature_matching_loss(
            fake_features, real_features
        )
        loss.backward()

        decoder_grads = [
            p.grad for p in vae.decoder.parameters() if p.grad is not None
        ]
        assert len(decoder_grads) > 0
        assert any(g.abs().sum() > 0 for g in decoder_grads)
