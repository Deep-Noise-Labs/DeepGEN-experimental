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

    def test_kl_loss_zero_for_standard_normal(self):
        loss_fn = VAELoss()
        reconstruction = torch.zeros(2, 2, 4096)
        target = torch.zeros(2, 2, 4096)
        mean = torch.zeros(2, 64, 16)
        log_var = torch.zeros(2, 64, 16)

        losses = loss_fn(reconstruction, target, mean, log_var)
        assert losses["kl_loss"].item() < 1e-5


LEGACY_STFT_KWARGS = dict(
    fft_sizes=(2048, 1024, 512, 256),
    hop_sizes=(512, 256, 128, 64),
    win_sizes=(2048, 1024, 512, 256),
    sum_and_difference=False,
    log_eps=1e-8,
)


def _discrimination(loss_fn, degraded, reference):
    """Loss of the degraded pair relative to an inaudible +0.1 dB control."""
    control = reference * 10 ** (0.1 / 20.0)
    d = loss_fn(degraded, reference).item()
    c = loss_fn(control, reference).item()
    return d / c


class TestMultiResolutionSTFTLossUpgrade:
    def test_default_bank_spans_bass_and_transients(self):
        loss_fn = MultiResolutionSTFTLoss()
        assert max(loss_fn.fft_sizes) >= 8192  # ~5.4 Hz bins at 44.1 kHz
        assert min(loss_fn.fft_sizes) <= 64  # ~1.5 ms time localisation

    def test_zero_loss_for_identical_stereo_signals(self):
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(1, 2, 32768)
        assert loss_fn(x, x).item() < 1e-4

    def test_short_signal_skips_oversized_resolutions(self):
        loss_fn = MultiResolutionSTFTLoss()
        pred = torch.randn(1, 2, 1024)
        target = torch.randn(1, 2, 1024)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_bf16_inputs_are_computed_in_float32(self):
        loss_fn = MultiResolutionSTFTLoss()
        pred = torch.randn(1, 2, 16384).to(torch.bfloat16)
        target = torch.randn(1, 2, 16384).to(torch.bfloat16)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        assert loss.dtype == torch.float32

    def test_detects_stereo_collapse_better_than_legacy(self):
        torch.manual_seed(0)
        t = torch.arange(44100) / 44100.0
        # Decorrelated wide content: same tone, independent noise per channel
        tone = torch.sin(2 * torch.pi * 220.0 * t)
        left = tone + 0.3 * torch.randn(44100)
        right = tone + 0.3 * torch.randn(44100)
        reference = torch.stack([left, right])[None]
        mid = 0.5 * (reference[:, 0] + reference[:, 1])
        collapsed = torch.stack([mid, mid], dim=1)

        legacy = MultiResolutionSTFTLoss(**LEGACY_STFT_KWARGS)
        upgraded = MultiResolutionSTFTLoss()
        assert _discrimination(upgraded, collapsed, reference) > 2 * _discrimination(
            legacy, collapsed, reference
        )

    def test_detects_sub_bass_detune_better_than_legacy(self):
        t = torch.arange(66150) / 44100.0  # 1.5 s
        ref = torch.sin(2 * torch.pi * 41.2 * t)  # E1
        flat = torch.sin(2 * torch.pi * 38.9 * t)  # one semitone flat
        reference = torch.stack([ref, ref])[None]
        degraded = torch.stack([flat, flat])[None]

        legacy = MultiResolutionSTFTLoss(**LEGACY_STFT_KWARGS)
        upgraded = MultiResolutionSTFTLoss()
        assert _discrimination(upgraded, degraded, reference) > 1.5 * _discrimination(
            legacy, degraded, reference
        )


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

    def test_snr_is_squared_coefficient_ratio(self):
        # For x_t = t * x_0 + (1 - t) * noise, SNR(t) = t^2 / (1 - t)^2
        loss_fn = FlowMatchingLoss(weighting="min_snr")
        snr = loss_fn._snr(torch.tensor([0.5, 0.75]))
        assert torch.allclose(snr, torch.tensor([1.0, 9.0]), rtol=1e-3)

    def test_min_snr_finite_at_timestep_boundaries(self):
        # t=0 previously produced 0/0 = NaN weights; t=1 an infinite SNR
        loss_fn = FlowMatchingLoss(weighting="min_snr")
        v_pred = torch.randn(3, 8, 10)
        v_target = torch.randn(3, 8, 10)
        t = torch.tensor([0.0, 0.5, 1.0])
        loss = loss_fn(v_pred, v_target, t)
        assert torch.isfinite(loss)

    def test_snr_weighting_finite_at_timestep_boundaries(self):
        loss_fn = FlowMatchingLoss(weighting="snr")
        v_pred = torch.randn(3, 8, 10)
        v_target = torch.randn(3, 8, 10)
        t = torch.tensor([0.0, 0.5, 1.0])
        loss = loss_fn(v_pred, v_target, t)
        assert torch.isfinite(loss)
