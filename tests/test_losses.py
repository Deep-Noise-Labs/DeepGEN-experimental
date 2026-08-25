"""
Unit tests for SynthGen loss functions.
"""

import pytest
import torch

from synthgen.training.losses import (
    FlowMatchingLoss,
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

    def test_mel_loss_component_when_enabled(self):
        loss_fn = VAELoss(mel_weight=1.0, sample_rate=22050)
        reconstruction = torch.randn(2, 2, 4096)
        target = torch.randn(2, 2, 4096)
        mean = torch.randn(2, 64, 16)
        log_var = torch.randn(2, 64, 16)

        losses = loss_fn(reconstruction, target, mean, log_var)
        assert "mel_loss" in losses
        assert losses["mel_loss"].item() > 0

    def test_mel_loss_absent_by_default(self):
        loss_fn = VAELoss()
        losses = loss_fn(
            torch.randn(2, 2, 4096),
            torch.randn(2, 2, 4096),
            torch.randn(2, 64, 16),
            torch.randn(2, 64, 16),
        )
        assert "mel_loss" not in losses

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
