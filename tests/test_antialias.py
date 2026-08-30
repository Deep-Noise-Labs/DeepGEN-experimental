"""
Tests for the alias-free activation stack in the Audio VAE.

The point of these tests is the aliasing measurement: a pointwise
nonlinearity applied at the native rate folds its own harmonics back into
the audible band as inharmonic partials. Running it at 2x rate between
matched low-pass resamplers should remove most of that energy.
"""

import math

import pytest
import torch

from synthgen.model.vae import (
    AliasFreeSnake,
    AudioVAE,
    DownSample1d,
    Snake,
    SnakeBeta,
    UpSample1d,
    kaiser_sinc_filter1d,
    make_activation,
)


SAMPLE_RATE = 44100


def _tone(freq: float, seconds: float = 0.25, amplitude: float = 3.0) -> torch.Tensor:
    """A single loud sine, shaped (1, 1, samples)."""
    n = int(SAMPLE_RATE * seconds)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    return (amplitude * torch.sin(2 * math.pi * freq * t)).view(1, 1, n)


def _energy_at(signal: torch.Tensor, freq: float, bandwidth: float = 60.0) -> float:
    """Spectral energy of a (1, 1, N) signal in a narrow band around freq."""
    x = signal.detach().reshape(-1)
    window = torch.hann_window(x.numel())
    spectrum = torch.fft.rfft(x * window).abs()
    freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / SAMPLE_RATE)
    band = (freqs >= freq - bandwidth) & (freqs <= freq + bandwidth)
    return float(spectrum[band].pow(2).sum())


class TestResamplers:
    @pytest.mark.parametrize("length", [1000, 4097, 44100])
    def test_round_trip_preserves_length(self, length: int):
        x = torch.randn(2, 5, length)
        y = DownSample1d(2)(UpSample1d(2)(x))
        assert y.shape == x.shape

    def test_round_trip_preserves_band_limited_signal(self):
        # A tone well inside the passband should survive up/down resampling
        # essentially unchanged.
        x = _tone(1000.0, seconds=0.2, amplitude=1.0)
        y = DownSample1d(2)(UpSample1d(2)(x))
        # Ignore filter transients at the edges.
        error = (y[..., 512:-512] - x[..., 512:-512]).abs().max()
        assert error < 0.02

    def test_filter_has_unit_dc_gain(self):
        kernel = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)
        assert kernel.shape == (1, 1, 12)
        assert abs(float(kernel.sum()) - 1.0) < 1e-5


class TestAliasFreeSnake:
    def test_reduces_aliased_energy(self):
        """
        A 9 kHz tone through a Snake nonlinearity produces a 4th harmonic at
        36 kHz, above Nyquist (22.05 kHz), which folds back to
        44100 - 36000 = 8100 Hz. That partial is pure aliasing: it sits
        *below* the fundamental, is inharmonic with respect to it, and cannot
        be removed after the fact. Oversampling the nonlinearity suppresses it.
        """
        fundamental = 9000.0
        x = _tone(fundamental)

        plain = SnakeBeta(1)
        aliased = plain(x)

        alias_free = AliasFreeSnake(1)
        # Same nonlinearity parameters, so the only difference is oversampling.
        alias_free.activation.load_state_dict(plain.state_dict())
        clean = alias_free(x)

        alias_freq = SAMPLE_RATE - 4 * fundamental  # 8100 Hz
        aliased_energy = _energy_at(aliased, alias_freq)
        clean_energy = _energy_at(clean, alias_freq)

        # The fundamental must survive: this is not just a low-pass filter.
        assert _energy_at(clean, fundamental) > 0.9 * _energy_at(aliased, fundamental)
        # The aliased partial should be attenuated by well over 20 dB.
        assert clean_energy < 0.01 * aliased_energy

    def test_shape_and_gradient(self):
        act = AliasFreeSnake(8)
        x = torch.randn(2, 8, 2048, requires_grad=True)
        y = act(x)
        y.sum().backward()
        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestSnakeBeta:
    def test_legacy_snake_has_a_pole_at_negative_epsilon(self):
        """
        The original Snake computes ``1 / (alpha + 1e-8)``. Nothing constrains
        alpha to stay positive, and the denominator hits exactly zero when a
        training step lands alpha on -1e-8, producing a non-finite activation.
        """
        legacy = Snake(4)
        with torch.no_grad():
            legacy.alpha.fill_(-1e-8)
        assert not torch.isfinite(legacy(torch.ones(1, 4, 8))).all()

    def test_snake_beta_is_finite_across_the_parameter_range(self):
        """
        SnakeBeta stores both parameters in log space, so alpha and beta are
        strictly positive for every possible parameter value and the
        activation keeps its intended shape however training moves them.
        """
        act = SnakeBeta(4)
        x = torch.randn(1, 4, 256)
        for value in (-30.0, -5.0, 0.0, 5.0):
            with torch.no_grad():
                act.log_alpha.fill_(value)
                act.log_beta.fill_(value)
            assert torch.isfinite(act(x)).all()

    def test_matches_snake_shape(self):
        x = torch.randn(2, 6, 512)
        assert SnakeBeta(6)(x).shape == x.shape


class TestVAEIntegration:
    @pytest.mark.parametrize("antialias", [False, True])
    def test_forward_round_trip(self, antialias: bool):
        vae = AudioVAE(
            in_channels=2,
            latent_dim=16,
            base_channels=8,
            strides=(4, 4, 4, 4),
            antialias=antialias,
        )
        audio = torch.randn(1, 2, 8192)
        reconstruction, target, mean, log_var = vae(audio)
        assert reconstruction.shape == target.shape == audio.shape
        assert mean.shape == log_var.shape == (1, 16, 8192 // 256)

    def test_make_activation_selects_implementation(self):
        assert isinstance(make_activation(4, antialias=True), AliasFreeSnake)
        assert isinstance(make_activation(4, antialias=False), Snake)

    def test_antialias_adds_few_parameters(self):
        """Oversampling costs compute, not weights: the filters are buffers."""
        kwargs = dict(in_channels=2, latent_dim=16, base_channels=8, strides=(4, 4, 4, 4))
        plain = sum(p.numel() for p in AudioVAE(**kwargs, antialias=False).parameters())
        alias_free = sum(p.numel() for p in AudioVAE(**kwargs, antialias=True).parameters())
        # SnakeBeta adds one beta parameter per channel alongside alpha.
        assert alias_free < plain * 1.02
