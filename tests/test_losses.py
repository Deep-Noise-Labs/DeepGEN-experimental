"""
Unit tests for SynthGen loss functions.
"""

import math

import pytest
import torch

from synthgen.training.losses import (
    FlowMatchingLoss,
    MultiResolutionMelLoss,
    MultiResolutionSTFTLoss,
    VAELoss,
)


class TestMultiResolutionSTFTLoss:
    def test_zero_loss_for_identical_signals(self):
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=(512, 256),
            hop_sizes=(128, 64),
            win_sizes=(512, 256),
        )
        x = torch.randn(2, 2, 4096)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-5

    def test_positive_loss_for_different_signals(self):
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=(512, 256),
            hop_sizes=(128, 64),
            win_sizes=(512, 256),
        )
        pred = torch.randn(2, 2, 4096)
        target = torch.randn(2, 2, 4096)
        loss = loss_fn(pred, target)
        assert loss.item() > 0


class TestVAELoss:
    def test_loss_components(self):
        loss_fn = VAELoss()
        reconstruction = torch.randn(2, 2, 4096)
        target = torch.randn(2, 2, 4096)
        mean = torch.randn(2, 64, 16)
        log_var = torch.randn(2, 64, 16)

        losses = loss_fn(reconstruction, target, mean, log_var)
        assert "loss" in losses
        assert "l1_loss" in losses
        assert "spectral_loss" in losses
        assert "kl_loss" in losses
        assert all(v.item() >= 0 for v in losses.values())

    def test_kl_loss_zero_for_standard_normal(self):
        loss_fn = VAELoss()
        reconstruction = torch.zeros(2, 2, 4096)
        target = torch.zeros(2, 2, 4096)
        mean = torch.zeros(2, 64, 16)
        log_var = torch.zeros(2, 64, 16)

        losses = loss_fn(reconstruction, target, mean, log_var)
        assert losses["kl_loss"].item() < 1e-5


class TestFlowMatchingLoss:
    def test_uniform_weighting(self):
        loss_fn = FlowMatchingLoss(weighting="uniform")
        v_pred = torch.randn(4, 64, 50)
        v_target = torch.randn(4, 64, 50)
        t = torch.rand(4)

        loss = loss_fn(v_pred, v_target, t)
        assert loss.item() > 0

    def test_zero_loss_for_identical_velocities(self):
        loss_fn = FlowMatchingLoss(weighting="uniform")
        v = torch.randn(4, 64, 50)
        loss = loss_fn(v, v)
        assert loss.item() < 1e-6

    def test_min_snr_weighting(self):
        loss_fn = FlowMatchingLoss(weighting="min_snr")
        v_pred = torch.randn(4, 64, 50)
        v_target = torch.randn(4, 64, 50)
        t = torch.rand(4)

        loss = loss_fn(v_pred, v_target, t)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestMultiResolutionMelLoss:
    def test_zero_loss_for_identical_signals(self):
        loss_fn = MultiResolutionMelLoss(
            sample_rate=16000, window_sizes=(512, 128), n_mels=(40, 20)
        )
        x = torch.randn(2, 2, 8192) * 0.1
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different_signals(self):
        loss_fn = MultiResolutionMelLoss(
            sample_rate=16000, window_sizes=(512, 128), n_mels=(40, 20)
        )
        pred = torch.randn(2, 2, 8192) * 0.1
        target = torch.randn(2, 2, 8192) * 0.1
        assert loss_fn(pred, target).item() > 0

    def test_mismatched_resolution_lists_are_rejected(self):
        with pytest.raises(ValueError):
            MultiResolutionMelLoss(window_sizes=(512,), n_mels=(40, 20))

    def test_gradients_reach_the_prediction(self):
        loss_fn = MultiResolutionMelLoss(
            sample_rate=16000, window_sizes=(256,), n_mels=(20,)
        )
        pred = (torch.randn(1, 1, 4096) * 0.1).requires_grad_(True)
        loss_fn(pred, torch.randn(1, 1, 4096) * 0.1).backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0

    def test_weights_low_frequencies_more_than_a_linear_stft_does(self):
        """
        The point of the mel term. Inject the same-amplitude error at 100 Hz and
        at 15 kHz into the same broadband reference, and compare how each
        criterion ranks them. A linear-frequency STFT barely distinguishes the
        two, because it allocates gradient by bin count and half its bins sit in
        the top octave. The mel criterion allocates it the way the ear does.
        """
        sample_rate = 44100
        torch.manual_seed(0)
        t = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        clean = 0.3 * torch.randn(1, 1, sample_rate)

        low_error = clean + 0.02 * torch.sin(2 * math.pi * 100.0 * t).view(1, 1, -1)
        high_error = clean + 0.02 * torch.sin(2 * math.pi * 15000.0 * t).view(1, 1, -1)

        mel = MultiResolutionMelLoss(sample_rate=sample_rate)
        stft = MultiResolutionSTFTLoss()

        mel_ratio = mel(low_error, clean).item() / mel(high_error, clean).item()
        stft_ratio = stft(low_error, clean).item() / stft(high_error, clean).item()

        assert mel_ratio > 5 * stft_ratio, (
            f"mel low/high ratio {mel_ratio:.2f} should dominate the linear "
            f"STFT's {stft_ratio:.2f}"
        )


class TestVAELossMelTerm:
    def test_mel_term_is_reported_and_included(self):
        loss_fn = VAELoss(mel_weight=15.0)
        pred = torch.randn(1, 2, 8192) * 0.1
        target = torch.randn(1, 2, 8192) * 0.1
        mean = torch.zeros(1, 64, 8)
        log_var = torch.zeros(1, 64, 8)

        losses = loss_fn(pred, target, mean, log_var)
        assert "mel_loss" in losses
        assert losses["loss"].item() > losses["spectral_loss"].item()

    def test_mel_term_can_be_disabled(self):
        loss_fn = VAELoss(mel_weight=0.0)
        losses = loss_fn(
            torch.randn(1, 2, 4096),
            torch.randn(1, 2, 4096),
            torch.zeros(1, 64, 4),
            torch.zeros(1, 64, 4),
        )
        assert "mel_loss" not in losses
