"""
Unit tests for the adversarial discriminators and their losses.
"""

import pytest
import torch

from synthgen.training.discriminators import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionSTFTDiscriminator,
)
from synthgen.training.losses import (
    discriminator_adversarial_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)

# Small banks so the suite stays fast on CPU.
PERIODS = (2, 3)
FFT_SIZES = (512, 256)
HOP_SIZES = (128, 64)


@pytest.fixture
def discriminator():
    return CombinedDiscriminator(
        periods=PERIODS, fft_sizes=FFT_SIZES, hop_sizes=HOP_SIZES, stft_channels=8
    )


class TestMultiPeriodDiscriminator:
    def test_one_output_per_period(self):
        disc = MultiPeriodDiscriminator(periods=PERIODS)
        logits, features = disc(torch.randn(2, 2, 4096))
        assert len(logits) == len(PERIODS)
        assert len(features) == len(PERIODS)

    def test_handles_length_not_divisible_by_period(self):
        disc = MultiPeriodDiscriminator(periods=(7,))
        logits, _ = disc(torch.randn(1, 2, 4095))
        assert torch.isfinite(logits[0]).all()

    def test_mono_and_stereo_both_accepted(self):
        disc = MultiPeriodDiscriminator(periods=PERIODS)
        mono, _ = disc(torch.randn(2, 1, 4096))
        stereo, _ = disc(torch.randn(2, 2, 4096))
        # Channels fold into the batch, so stereo doubles the scored rows.
        assert stereo[0].shape[0] == 2 * mono[0].shape[0]


class TestMultiResolutionSTFTDiscriminator:
    def test_one_output_per_resolution(self):
        disc = MultiResolutionSTFTDiscriminator(
            fft_sizes=FFT_SIZES, hop_sizes=HOP_SIZES, channels=8
        )
        logits, features = disc(torch.randn(2, 2, 8192))
        assert len(logits) == len(FFT_SIZES)
        assert all(torch.isfinite(logit).all() for logit in logits)

    def test_mismatched_resolution_lists_are_rejected(self):
        with pytest.raises(ValueError):
            MultiResolutionSTFTDiscriminator(fft_sizes=(512,), hop_sizes=(128, 64))


class TestCombinedDiscriminator:
    def test_combines_both_banks(self, discriminator):
        logits, features = discriminator(torch.randn(1, 2, 8192))
        assert len(logits) == len(PERIODS) + len(FFT_SIZES)
        assert len(features) == len(logits)

    def test_gradients_flow_to_the_generator_input(self, discriminator):
        audio = torch.randn(1, 2, 8192, requires_grad=True)
        logits, _ = discriminator(audio)
        generator_adversarial_loss(logits).backward()
        assert audio.grad is not None
        assert audio.grad.abs().sum() > 0


class TestAdversarialLosses:
    def test_feature_matching_is_zero_for_identical_features(self, discriminator):
        audio = torch.randn(1, 2, 8192)
        _, features = discriminator(audio)
        assert feature_matching_loss(features, features).item() == pytest.approx(0.0)

    def test_feature_matching_is_positive_for_different_audio(self, discriminator):
        _, real = discriminator(torch.randn(1, 2, 8192))
        _, fake = discriminator(torch.randn(1, 2, 8192))
        assert feature_matching_loss(real, fake).item() > 0

    def test_hinge_discriminator_loss_rewards_correct_separation(self):
        confident = discriminator_adversarial_loss(
            [torch.full((1, 4), 2.0)], [torch.full((1, 4), -2.0)]
        )
        confused = discriminator_adversarial_loss(
            [torch.full((1, 4), -2.0)], [torch.full((1, 4), 2.0)]
        )
        assert confident.item() == pytest.approx(0.0)
        assert confused.item() > confident.item()

    def test_generator_loss_falls_as_the_critic_is_fooled(self):
        fooled = generator_adversarial_loss([torch.full((1, 4), 2.0)])
        caught = generator_adversarial_loss([torch.full((1, 4), -2.0)])
        assert fooled.item() < caught.item()

    def test_empty_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            generator_adversarial_loss([])
        with pytest.raises(ValueError):
            feature_matching_loss([], [])
