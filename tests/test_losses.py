"""
Unit tests for SynthGen loss functions.
"""

import math

import torch

from synthgen.model.discriminator import MultiScaleSTFTDiscriminator
from synthgen.training.losses import (
    FlowMatchingLoss,
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
    StereoCoherenceLoss,
    VAELoss,
    build_mel_filterbank,
)


def _sine(freq: float, seconds: float = 0.5, sr: int = 44100, amp: float = 0.5):
    t = torch.arange(int(seconds * sr)) / sr
    return (amp * torch.sin(2 * math.pi * freq * t)).view(1, 1, -1)


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


    def test_default_ladder_resolves_low_frequencies(self):
        """
        The default resolution ladder must reach a window long enough to
        separate bass fundamentals: at 44.1 kHz a 2048-point FFT gives 21.5 Hz
        bins, which is coarser than a semitone anywhere below ~370 Hz.
        """
        loss_fn = MultiResolutionSTFTLoss()
        longest = max(loss_fn.fft_sizes)
        assert 44100 / longest < 10.0

    def test_short_input_skips_oversized_windows(self):
        """A clip shorter than the longest window must not raise."""
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(1, 2, 4096)
        loss = loss_fn(x, x)
        assert torch.isfinite(loss)
        assert loss.item() < 1e-4

    def test_distinguishes_bass_error_the_old_ladder_could_not(self):
        """
        A 40 Hz vs 45 Hz error is a musically enormous mistake (roughly two
        semitones at the bottom of the range). A ladder capped at 2048 puts both
        in adjacent bins; the default ladder must score it far more heavily.
        """
        target = _sine(40.0)
        pred = _sine(45.0)

        coarse = MultiResolutionSTFTLoss(fft_sizes=(2048, 1024, 512, 256))
        fine = MultiResolutionSTFTLoss()

        assert fine(pred, target).item() > coarse(pred, target).item()

    def test_per_item_normalisation_protects_quiet_sounds(self):
        """
        With batch-global normalisation a single loud item dominates the
        spectral-convergence term, so an error on a quiet item is nearly free.
        Per-item normalisation must close that gap.
        """
        loud = torch.randn(1, 1, 16384)
        quiet = torch.randn(1, 1, 16384) * 1e-3
        target = torch.cat([loud, quiet], dim=0)

        # Ruin the quiet item only.
        broken_quiet = torch.cat([loud, torch.randn(1, 1, 16384) * 1e-3], dim=0)
        # Ruin the loud item only.
        broken_loud = torch.cat([torch.randn(1, 1, 16384), quiet], dim=0)

        def convergence_ratio(per_item: bool) -> float:
            loss_fn = MultiResolutionSTFTLoss(
                fft_sizes=(1024, 256), per_item=per_item, mag_floor=1e-12
            )
            return (
                loss_fn(broken_quiet, target).item()
                / loss_fn(broken_loud, target).item()
            )

        assert convergence_ratio(per_item=True) > convergence_ratio(per_item=False)

    def test_magnitude_floor_bounds_silence(self):
        """
        Numerical noise on digital silence must not produce an enormous loss.
        Without a floor, both the log term and the spectral-convergence
        denominator degenerate, and a silent item — a release tail, a gap
        between hits — dominates the whole batch.
        """
        silence = torch.zeros(1, 1, 16384)
        dither = torch.randn(1, 1, 16384) * 1e-9

        unfloored = MultiResolutionSTFTLoss(fft_sizes=(1024,), mag_floor=1e-12)
        floored = MultiResolutionSTFTLoss(fft_sizes=(1024,), mag_floor=1e-5)

        assert unfloored(dither, silence).item() > 1.0
        assert floored(dither, silence).item() < 0.01


class TestMelSpectrogramLoss:
    def test_filterbank_shape_and_positivity(self):
        fb = build_mel_filterbank(n_fft=1024, n_mels=64, sample_rate=44100)
        assert fb.shape == (64, 513)
        assert (fb >= 0).all()
        assert fb.sum() > 0

    def test_zero_for_identical_signals(self):
        loss_fn = MelSpectrogramLoss(fft_sizes=(512,), n_mels=(32,))
        x = torch.randn(2, 2, 16384)
        assert loss_fn(x, x).item() < 1e-6

    def test_allocates_more_capacity_below_1khz_than_a_linear_stft(self):
        """
        The mechanism behind the mel term. In a 2048-point linear STFT at
        44.1 kHz only ~4.5% of bins sit below 1 kHz, so a bin-uniform loss
        spends ~95% of its capacity above the range where musical fundamentals
        and formants live. Mel spacing must rebalance that by a wide margin.
        """
        n_fft, n_mels, sr = 2048, 128, 44100
        fb = build_mel_filterbank(n_fft, n_mels, sr)

        bin_hz = torch.arange(n_fft // 2 + 1) * sr / n_fft
        # Energy-weighted centre frequency of each triangular band.
        centres = (fb * bin_hz).sum(dim=1) / fb.sum(dim=1).clamp_min(1e-8)

        mel_share = (centres < 1000.0).float().mean().item()
        linear_share = (bin_hz < 1000.0).float().mean().item()

        assert mel_share > 4 * linear_share

    def test_shifts_weight_towards_the_bottom_octave(self):
        """
        Empirical form of the same claim, on broadband material. A 6 dB shelf
        on the 40–80 Hz octave and the same shelf on the 10–20 kHz octave are
        equally audible mistakes to a sound designer, but a linear-frequency
        loss sees ~250x more bins in the top octave. Mel spacing must move the
        balance measurably back down — it lands at roughly 2.6x here.
        """
        torch.manual_seed(0)
        target = torch.randn(1, 1, 44100) * 0.1

        def shelf(x: torch.Tensor, low: float, high: float, gain_db: float):
            spectrum = torch.fft.rfft(x, dim=-1)
            freqs = torch.fft.rfftfreq(x.shape[-1], d=1 / 44100)
            band = (freqs >= low) & (freqs < high)
            spectrum = spectrum.clone()
            spectrum[..., band] *= 10 ** (gain_db / 20.0)
            return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)

        low_error = shelf(target, 40.0, 80.0, 6.0)
        high_error = shelf(target, 10000.0, 20000.0, 6.0)

        mel = MelSpectrogramLoss(fft_sizes=(2048,), n_mels=(128,))
        linear = MultiResolutionSTFTLoss(fft_sizes=(2048,))

        mel_ratio = mel(low_error, target).item() / mel(high_error, target).item()
        linear_ratio = (
            linear(low_error, target).item() / linear(high_error, target).item()
        )
        assert mel_ratio > 2 * linear_ratio


class TestStereoCoherenceLoss:
    def test_mono_input_is_a_noop(self):
        loss_fn = StereoCoherenceLoss(fft_sizes=(512,))
        assert loss_fn(torch.randn(2, 1, 16384), torch.randn(2, 1, 16384)).item() == 0.0

    def test_penalises_stereo_collapse(self):
        """
        A decoder that collapses a wide stereo image to mono is the classic
        failure of a per-channel objective. The mid/side term must see it.
        """
        left = torch.randn(1, 1, 16384)
        right = torch.randn(1, 1, 16384)
        target = torch.cat([left, right], dim=1)

        mid = (left + right) * 0.5
        collapsed = torch.cat([mid, mid], dim=1)

        loss_fn = StereoCoherenceLoss(fft_sizes=(1024, 512))
        assert loss_fn(collapsed, target).item() > 0.1
        assert loss_fn(target, target).item() < 1e-6


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

    def test_reports_perceptual_components(self):
        loss_fn = VAELoss()
        audio = torch.randn(1, 2, 16384)
        losses = loss_fn(
            audio, audio, torch.zeros(1, 64, 16), torch.zeros(1, 64, 16)
        )
        assert "mel_loss" in losses
        assert "stereo_loss" in losses

    def test_works_without_a_discriminator(self):
        """Pure-reconstruction mode must stay available for smoke runs and CI."""
        loss_fn = VAELoss()
        losses = loss_fn(
            torch.randn(1, 2, 16384),
            torch.randn(1, 2, 16384),
            torch.zeros(1, 64, 16),
            torch.zeros(1, 64, 16),
        )
        assert "adversarial_loss" not in losses
        assert torch.isfinite(losses["loss"])

    def test_adversarial_terms_are_added_and_differentiable(self):
        loss_fn = VAELoss()
        disc = MultiScaleSTFTDiscriminator(
            n_ffts=(256, 128), channels=8, num_layers=2, max_channels=32
        )
        reconstruction = torch.randn(1, 2, 16384, requires_grad=True)
        target = torch.randn(1, 2, 16384)

        losses = loss_fn(
            reconstruction,
            target,
            torch.zeros(1, 64, 16),
            torch.zeros(1, 64, 16),
            discriminator=disc,
        )

        assert "adversarial_loss" in losses
        assert "feature_matching_loss" in losses

        losses["loss"].backward()
        assert reconstruction.grad is not None
        assert torch.isfinite(reconstruction.grad).all()


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
