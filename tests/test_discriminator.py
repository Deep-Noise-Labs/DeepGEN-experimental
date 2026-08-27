"""
Unit tests for the Stage-1 adversarial discriminators.

These are shape/plumbing tests: they verify that every sub-discriminator
produces logits and feature maps, that gradients reach the generator through
the critic, and that awkward input lengths (not divisible by any period, not a
multiple of the STFT hop) do not break the forward pass.
"""

import pytest
import torch

from synthgen.training.discriminator import (
    AudioDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionSTFTDiscriminator,
    PeriodDiscriminator,
)
from synthgen.training.losses import (
    FeatureMatchingLoss,
    discriminator_hinge_loss,
    generator_hinge_loss,
)

# Narrow channel stacks keep these tests fast on CPU.
TINY_PERIOD_CHANNELS = (4, 8, 16)


class TestPeriodDiscriminator:
    def test_returns_logits_and_features(self):
        disc = PeriodDiscriminator(period=3, channels=TINY_PERIOD_CHANNELS)
        logits, features = disc(torch.randn(2, 1, 4096))
        assert logits.dim() == 2
        assert logits.shape[0] == 2
        # One feature map per conv, plus the final logit map.
        assert len(features) == len(TINY_PERIOD_CHANNELS) + 1

    @pytest.mark.parametrize("samples", [4096, 4097, 5000])
    def test_handles_lengths_not_divisible_by_period(self, samples):
        disc = PeriodDiscriminator(period=11, channels=TINY_PERIOD_CHANNELS)
        logits, _ = disc(torch.randn(1, 1, samples))
        assert torch.isfinite(logits).all()


class TestMultiPeriodDiscriminator:
    def test_one_output_per_period(self):
        periods = (2, 3, 5)
        disc = MultiPeriodDiscriminator(periods=periods, channels=TINY_PERIOD_CHANNELS)
        logits, features = disc(torch.randn(2, 2, 8192))
        assert len(logits) == len(periods)
        assert len(features) == len(periods)

    def test_channels_are_folded_into_batch(self):
        disc = MultiPeriodDiscriminator(periods=(3,), channels=TINY_PERIOD_CHANNELS)
        logits, _ = disc(torch.randn(2, 2, 8192))
        # Stereo is critiqued per channel: batch 2 x 2 channels = 4 rows.
        assert logits[0].shape[0] == 4


class TestMultiResolutionSTFTDiscriminator:
    def test_one_output_per_resolution(self):
        resolutions = ((512, 128, 512), (256, 64, 256))
        disc = MultiResolutionSTFTDiscriminator(resolutions=resolutions, channels=4)
        logits, features = disc(torch.randn(2, 1, 8192))
        assert len(logits) == len(resolutions)
        assert all(torch.isfinite(logit).all() for logit in logits)
        # 4 convs per band x 5 bands, plus the joint logit map.
        assert len(features[0]) == 4 * 5 + 1

    def test_gradient_flows_to_input(self):
        disc = MultiResolutionSTFTDiscriminator(
            resolutions=((256, 64, 256),), channels=4
        )
        x = torch.randn(1, 1, 8192, requires_grad=True)
        logits, _ = disc(x)
        logits[0].sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestAudioDiscriminator:
    @pytest.fixture
    def disc(self):
        return AudioDiscriminator(
            periods=(2, 3),
            stft_resolutions=((512, 128, 512), (256, 64, 256)),
            stft_channels=4,
            period_channels=TINY_PERIOD_CHANNELS,
        )

    def test_combines_both_banks(self, disc):
        logits, features = disc(torch.randn(2, 2, 8192))
        assert len(logits) == 4  # 2 periods + 2 STFT resolutions
        assert len(features) == 4

    def test_requires_at_least_one_bank(self):
        with pytest.raises(ValueError):
            AudioDiscriminator(use_period=False, use_stft=False)

    def test_adversarial_losses_are_finite(self, disc):
        real = torch.randn(2, 2, 8192) * 0.1
        fake = torch.randn(2, 2, 8192) * 0.1

        real_logits, real_features = disc(real)
        fake_logits, fake_features = disc(fake)

        d_loss = discriminator_hinge_loss(real_logits, fake_logits)
        g_loss = generator_hinge_loss(fake_logits)
        fm_loss = FeatureMatchingLoss()(real_features, fake_features)

        assert torch.isfinite(d_loss) and d_loss.item() >= 0
        assert torch.isfinite(g_loss)
        assert torch.isfinite(fm_loss) and fm_loss.item() >= 0

    def test_discriminator_hinge_is_zero_when_confident(self):
        # D(real) >= 1 and D(fake) <= -1 sit outside the hinge margin.
        real_logits = [torch.full((2, 4), 2.0)]
        fake_logits = [torch.full((2, 4), -2.0)]
        assert discriminator_hinge_loss(real_logits, fake_logits).item() == 0.0

    def test_feature_matching_zero_for_identical_features(self):
        features = [[torch.randn(2, 4, 8)] for _ in range(3)]
        assert FeatureMatchingLoss()(features, features).item() == 0.0

    def test_gradient_reaches_generator_through_critic(self, disc):
        """The adversarial term must be differentiable w.r.t. the decoder output."""
        fake = torch.randn(1, 1, 8192, requires_grad=True)
        fake_logits, _ = disc(fake)
        generator_hinge_loss(fake_logits).backward()
        assert fake.grad is not None
        assert fake.grad.abs().sum() > 0
