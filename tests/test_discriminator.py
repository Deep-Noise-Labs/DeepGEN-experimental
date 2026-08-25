"""
Unit tests for the multi-resolution STFT discriminator and adversarial losses.
"""

import torch

from synthgen.model.discriminator import (
    MultiResolutionDiscriminator,
    STFTSubDiscriminator,
)
from synthgen.training.losses import (
    MultiScaleMelSpectrogramLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


class TestSTFTSubDiscriminator:
    def test_output_shapes(self):
        disc = STFTSubDiscriminator(fft_size=512, hop_size=128)
        x = torch.randn(2, 8192)
        logits, features = disc(x)
        assert logits.shape[0] == 2
        assert logits.shape[1] == 1
        assert len(features) == 5

    def test_gradients_flow_to_input(self):
        disc = STFTSubDiscriminator(fft_size=512, hop_size=128)
        x = torch.randn(1, 8192, requires_grad=True)
        logits, _ = disc(x)
        logits.mean().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMultiResolutionDiscriminator:
    def test_stereo_input_folded_to_batch(self):
        disc = MultiResolutionDiscriminator(
            resolutions=((512, 128), (256, 64)), channels=8
        )
        x = torch.randn(2, 2, 8192)
        logits, features = disc(x)
        assert len(logits) == 2
        assert len(features) == 2
        # Stereo channels folded into batch: 2 * 2 = 4
        assert logits[0].shape[0] == 4

    def test_adversarial_losses(self):
        disc = MultiResolutionDiscriminator(
            resolutions=((512, 128),), channels=8
        )
        real = torch.randn(1, 1, 8192)
        fake = torch.randn(1, 1, 8192)

        real_logits, real_features = disc(real)
        fake_logits, fake_features = disc(fake)

        d_loss = discriminator_loss(real_logits, fake_logits)
        g_loss = generator_adversarial_loss(fake_logits)
        fm_loss = feature_matching_loss(real_features, fake_features)

        assert torch.isfinite(d_loss)
        assert torch.isfinite(g_loss)
        assert fm_loss.item() > 0

    def test_feature_matching_zero_for_identical(self):
        disc = MultiResolutionDiscriminator(
            resolutions=((512, 128),), channels=8
        )
        x = torch.randn(1, 1, 8192)
        _, features = disc(x)
        fm_loss = feature_matching_loss(features, features)
        assert fm_loss.item() < 1e-6


class TestMultiScaleMelSpectrogramLoss:
    def test_zero_loss_for_identical_signals(self):
        loss_fn = MultiScaleMelSpectrogramLoss(
            sample_rate=22050,
            fft_sizes=(512, 256),
            n_mels=(40, 20),
        )
        x = torch.randn(2, 1, 8192)
        assert loss_fn(x, x).item() < 1e-5

    def test_positive_loss_for_different_signals(self):
        loss_fn = MultiScaleMelSpectrogramLoss(
            sample_rate=22050,
            fft_sizes=(512, 256),
            n_mels=(40, 20),
        )
        pred = torch.randn(2, 1, 8192)
        target = torch.randn(2, 1, 8192)
        assert loss_fn(pred, target).item() > 0

    def test_gradients_flow(self):
        loss_fn = MultiScaleMelSpectrogramLoss(
            sample_rate=22050, fft_sizes=(256,), n_mels=(20,)
        )
        pred = torch.randn(1, 1, 4096, requires_grad=True)
        target = torch.randn(1, 1, 4096)
        loss_fn(pred, target).backward()
        assert pred.grad is not None
