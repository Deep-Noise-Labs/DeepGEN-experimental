"""
Unit tests for SynthGen loss functions.
"""

import pytest
import torch

from synthgen.training.losses import (
    FlowMatchingLoss,
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
    VAELoss,
    mel_filterbank,
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


class TestMelFilterbank:
    def test_shape_and_non_negative(self):
        fb = mel_filterbank(n_freqs=1025, n_mels=80, sample_rate=44100)
        assert fb.shape == (80, 1025)
        assert (fb >= 0).all()

    def test_every_band_has_support(self):
        """No mel band may be empty, or its slice of the loss is silently dead."""
        fb = mel_filterbank(n_freqs=257, n_mels=80, sample_rate=44100)
        assert (fb.sum(dim=1) > 0).all()

    def test_bands_are_ordered_low_to_high(self):
        fb = mel_filterbank(n_freqs=513, n_mels=40, sample_rate=44100)
        centres = fb.argmax(dim=1)
        assert torch.all(centres[1:] >= centres[:-1])


class TestMelSpectrogramLoss:
    def test_zero_loss_for_identical_signals(self):
        loss_fn = MelSpectrogramLoss(sample_rate=44100)
        x = torch.randn(2, 2, 16384) * 0.1
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different_signals(self):
        loss_fn = MelSpectrogramLoss(sample_rate=44100)
        pred = torch.randn(2, 2, 16384) * 0.1
        target = torch.randn(2, 2, 16384) * 0.1
        assert loss_fn(pred, target).item() > 0

    def test_short_signals_skip_oversized_windows(self):
        """Clips shorter than a window must not produce a degenerate STFT."""
        loss_fn = MelSpectrogramLoss(sample_rate=44100)
        pred = torch.randn(1, 1, 512) * 0.1
        loss = loss_fn(pred, torch.randn(1, 1, 512) * 0.1)
        assert torch.isfinite(loss)

    def test_rejects_more_bands_than_bins(self):
        with pytest.raises(ValueError):
            MelSpectrogramLoss(window_lengths=(64,), n_mels=(128,))

    def test_band_allocation_follows_hearing_not_the_fft_grid(self):
        """
        The structural reason this loss exists.

        A linear-frequency STFT spends its measurement units uniformly across
        the spectrum, which at n_fft=2048 / 44.1 kHz means half of them land in
        the top octave and almost none in the region where pitch and timbre are
        resolved. A mel axis reverses that allocation.
        """
        sample_rate = 44100
        n_fft = 2048
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

        fb = mel_filterbank(n_freqs, n_mels=160, sample_rate=sample_rate)
        mel_centres = fft_freqs[fb.argmax(dim=1)]

        fft_low = (fft_freqs < 1500).float().mean()
        mel_low = (mel_centres < 1500).float().mean()
        fft_high = (fft_freqs > 11000).float().mean()
        mel_high = (mel_centres > 11000).float().mean()

        # Far more resolution where hearing is sharp...
        assert mel_low > 4 * fft_low
        # ...and far less spent on the top octave.
        assert mel_high < fft_high / 2

    def test_log_term_is_bounded_at_digital_silence(self):
        """
        Silence must not dominate the gradient.

        With an unclamped ``log(x + 1e-8)`` the distance between a silent target
        and a very quiet prediction is unbounded; the clamp caps it.
        """
        loss_fn = MelSpectrogramLoss(sample_rate=44100, mag_weight=0.0)
        silence = torch.zeros(1, 1, 16384)
        near_silence = torch.full((1, 1, 16384), 1e-12)

        clamped = loss_fn(near_silence, silence)
        unclamped = (
            torch.log(torch.tensor(1e-12) + 1e-8)
            - torch.log(torch.tensor(0.0) + 1e-8)
        ).abs()

        assert clamped.item() == pytest.approx(0.0, abs=1e-6)
        assert clamped.item() < unclamped.item()

    def test_detects_a_removed_top_octave(self):
        """A decoder that drops its top octave must be penalised, monotonically."""
        sample_rate = 44100
        t = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        fundamental = torch.sin(2 * torch.pi * 220.0 * t)

        def with_air(amplitude: float) -> torch.Tensor:
            air = amplitude * torch.sin(2 * torch.pi * 15000.0 * t)
            return (fundamental + air).unsqueeze(0).unsqueeze(0)

        loss_fn = MelSpectrogramLoss(sample_rate=sample_rate)
        reference = with_air(0.02)

        losses = [loss_fn(with_air(a), reference).item() for a in (0.02, 0.01, 0.0)]
        assert losses[0] < losses[1] < losses[2]


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
        assert "mel_loss" in losses
        assert "kl_loss" in losses
        assert all(v.item() >= 0 for v in losses.values())

    def test_adversarial_terms_are_opt_in(self):
        """Without discriminator outputs the loss stays purely reconstructive."""
        loss_fn = VAELoss()
        args = (
            torch.randn(2, 2, 4096),
            torch.randn(2, 2, 4096),
            torch.randn(2, 64, 16),
            torch.randn(2, 64, 16),
        )
        losses = loss_fn(*args)
        assert "adv_loss" not in losses
        assert "fm_loss" not in losses

    def test_adversarial_terms_change_the_total(self):
        loss_fn = VAELoss()
        torch.manual_seed(0)
        args = (
            torch.randn(2, 2, 4096),
            torch.randn(2, 2, 4096),
            torch.randn(2, 64, 16),
            torch.randn(2, 64, 16),
        )
        baseline = loss_fn(*args)["loss"]

        fake_logits = [torch.full((2, 4), 0.5)]
        real_features = [[torch.zeros(2, 4, 8)]]
        fake_features = [[torch.ones(2, 4, 8)]]

        losses = loss_fn(
            *args,
            fake_logits=fake_logits,
            real_features=real_features,
            fake_features=fake_features,
        )
        assert "adv_loss" in losses
        assert "fm_loss" in losses
        # -0.5 (hinge) * 1.0 + 1.0 (feature L1) * 2.0 = +1.5
        assert losses["loss"].item() == pytest.approx(baseline.item() + 1.5, abs=1e-4)

    def test_weights_are_configurable(self):
        zeroed = VAELoss(mel_weight=0.0, spectral_weight=0.0, kl_weight=0.0)
        recon = torch.randn(2, 2, 4096)
        target = torch.randn(2, 2, 4096)
        mean = torch.randn(2, 64, 16)
        log_var = torch.randn(2, 64, 16)
        losses = zeroed(recon, target, mean, log_var)
        expected = 0.1 * torch.nn.functional.l1_loss(recon, target)
        assert losses["loss"].item() == pytest.approx(expected.item(), abs=1e-6)

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
