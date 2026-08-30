"""
Tests for the phase- and stereo-aware terms in MultiResolutionSTFTLoss.

Each test builds a signal pair that a magnitude-only objective scores as
identical -- loss exactly 0.0 -- but that sounds obviously different. These
are the reconstructions the previous objective was free to produce.
"""

import math

import pytest
import torch

from synthgen.training.losses import MultiResolutionSTFTLoss, VAELoss


FFT_KWARGS = dict(
    fft_sizes=(1024, 512, 256),
    hop_sizes=(256, 128, 64),
    win_sizes=(1024, 512, 256),
)


@pytest.fixture
def magnitude_only() -> MultiResolutionSTFTLoss:
    """The objective as it was before this change."""
    return MultiResolutionSTFTLoss(phase_weight=0.0, stereo_weight=0.0, **FFT_KWARGS)


@pytest.fixture
def perceptual() -> MultiResolutionSTFTLoss:
    """The objective with the phase and stereo terms enabled."""
    return MultiResolutionSTFTLoss(phase_weight=1.0, stereo_weight=1.0, **FFT_KWARGS)


def _stereo_music(seconds: float = 0.5, sample_rate: int = 44100) -> torch.Tensor:
    """A deterministic harmonic stereo signal with a real stereo image."""
    n = int(sample_rate * seconds)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    left = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate([220, 440, 660], 1))
    right = sum(torch.cos(2 * math.pi * f * t) / k for k, f in enumerate([221, 442, 663], 1))
    return torch.stack([left, right]).unsqueeze(0) * 0.3


class TestPhaseBlindness:
    def test_polarity_inversion_is_free_under_magnitude_only(
        self, magnitude_only, perceptual
    ):
        """
        Inverting the polarity of one channel leaves both channels' magnitude
        spectrograms bit-identical, so the magnitude-only objective scores it
        as a perfect reconstruction. It is not one: the stereo image inverts
        and the sound largely cancels on mono fold-down.
        """
        target = _stereo_music()
        flipped = target.clone()
        flipped[:, 1] *= -1

        assert magnitude_only(flipped, target).item() == pytest.approx(0.0, abs=1e-6)
        assert perceptual(flipped, target).item() > 1.0

    def test_global_polarity_inversion_is_free_under_magnitude_only(
        self, magnitude_only, perceptual
    ):
        target = _stereo_music()
        assert magnitude_only(-target, target).item() == pytest.approx(0.0, abs=1e-6)
        assert perceptual(-target, target).item() > 1.0

    def test_phase_dispersion_is_nearly_free_under_magnitude_only(
        self, magnitude_only, perceptual
    ):
        """
        An all-pass filter leaves the magnitude spectrum essentially intact
        while smearing transients in time -- the classic "underwater" artefact
        of a latent audio autoencoder. The magnitude-only objective barely
        registers it; the phase term does.
        """
        torch.manual_seed(0)
        target = _stereo_music()

        # Random-phase all-pass: unit magnitude at every frequency.
        spectrum = torch.fft.rfft(target, dim=-1)
        phase = torch.rand(spectrum.shape[-1]) * 2 * math.pi
        all_pass = torch.polar(torch.ones_like(phase), phase)
        dispersed = torch.fft.irfft(spectrum * all_pass, n=target.shape[-1], dim=-1)

        magnitude_score = magnitude_only(dispersed, target).item()
        perceptual_score = perceptual(dispersed, target).item()
        assert perceptual_score > 2 * magnitude_score


class TestStereoImage:
    def test_channel_swap_is_free_under_magnitude_only(
        self, magnitude_only, perceptual
    ):
        """
        The magnitude terms flatten (B, C, T) to (B*C, T) and score channels
        independently, so a decoder that swaps left and right pays nothing.
        """
        target = _stereo_music()
        swapped = target.flip(dims=[1])

        # Channels are scored as a set, so a swap is invisible to the
        # per-channel magnitude terms but not to mid/side.
        assert perceptual(swapped, target).item() > magnitude_only(
            swapped, target
        ).item()

    def test_mono_collapse_penalised(self, magnitude_only, perceptual):
        """A decoder that outputs a mono sum loses the image; say so."""
        target = _stereo_music()
        mono = target.mean(dim=1, keepdim=True).expand_as(target).contiguous()
        assert perceptual(mono, target).item() > magnitude_only(mono, target).item()


class TestBackwardCompatibility:
    def test_identical_signals_still_score_zero(self, perceptual):
        x = _stereo_music()
        assert perceptual(x, x).item() < 1e-5

    def test_different_signals_still_score_positive(self, perceptual):
        torch.manual_seed(0)
        assert perceptual(torch.randn(2, 2, 8192), torch.randn(2, 2, 8192)).item() > 0

    def test_weights_recover_magnitude_only_behaviour(self):
        """The previous objective remains reachable via configuration."""
        loss = MultiResolutionSTFTLoss(phase_weight=0.0, stereo_weight=0.0, **FFT_KWARGS)
        target = _stereo_music()
        assert loss(-target, target).item() == pytest.approx(0.0, abs=1e-6)

    def test_mono_input_skips_the_stereo_term(self, perceptual):
        mono = _stereo_music()[:, :1]
        components = perceptual.components(mono, mono)
        assert components["stereo"].item() == 0.0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_runs_under_reduced_precision(self, dtype):
        """
        The VAE stage trains in bf16 by default, and the FFT backends reject
        bf16 and fp16 tensors outright. The loss promotes to float32 rather
        than crashing on the first step.
        """
        loss = MultiResolutionSTFTLoss(**FFT_KWARGS)
        target = _stereo_music(seconds=0.2)
        value = loss(target.to(dtype) * 0.9, target.to(dtype))
        assert torch.isfinite(value)
        assert value.item() > 0

    def test_gradients_flow_through_every_term(self):
        loss = MultiResolutionSTFTLoss(**FFT_KWARGS)
        pred = _stereo_music().requires_grad_(True)
        loss(pred, _stereo_music() * 0.9).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestVAELossReporting:
    def test_reports_phase_and_stereo_components(self):
        loss_fn = VAELoss()
        target = _stereo_music()
        losses = loss_fn(
            -target, target, torch.randn(1, 16, 32), torch.randn(1, 16, 32)
        )
        for key in ("spectral_sc", "spectral_log_mag", "spectral_phase", "spectral_stereo"):
            assert key in losses
        # A polarity inversion is invisible to the magnitude terms...
        assert losses["spectral_sc"].item() == pytest.approx(0.0, abs=1e-6)
        assert losses["spectral_log_mag"].item() == pytest.approx(0.0, abs=1e-6)
        # ...and obvious to the new ones.
        assert losses["spectral_phase"].item() > 1.0
        assert losses["spectral_stereo"].item() > 1.0
