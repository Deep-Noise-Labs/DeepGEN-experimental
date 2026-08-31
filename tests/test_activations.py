"""
Unit tests for the anti-aliased activation stack.

The interesting test here is ``test_aliasing_is_suppressed``: it is the one that
checks the change actually does what it exists to do, rather than merely
producing tensors of the right shape.
"""

import math

import pytest
import torch

from synthgen.model.activations import (
    AntiAliasedActivation,
    DownSample1d,
    LowPassFilter1d,
    Snake,
    SnakeBeta,
    UpSample1d,
    build_activation,
    kaiser_sinc_filter1d,
)

SR = 44100


def _tone(freq: float, seconds: float = 0.5, sample_rate: int = SR) -> torch.Tensor:
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float32) / sample_rate
    return torch.sin(2 * math.pi * freq * t).view(1, 1, -1)


def _magnitude_db(x: torch.Tensor, freq: float, sample_rate: int = SR) -> float:
    """Windowed magnitude in dB at one frequency."""
    y = x[0, 0]
    spectrum = torch.fft.rfft(y * torch.hann_window(y.numel()))
    freqs = torch.fft.rfftfreq(y.numel(), 1.0 / sample_rate)
    index = int(torch.argmin((freqs - freq).abs()))
    return 20 * math.log10(spectrum[index].abs().item() + 1e-12)


class TestSnakeBeta:
    def test_output_shape(self):
        act = SnakeBeta(channels=64)
        x = torch.randn(2, 64, 100)
        assert act(x).shape == x.shape

    def test_parameters_stay_positive_under_log_scale(self):
        act = SnakeBeta(channels=8, alpha_init=1.0)
        act.alpha.data.fill_(-20.0)  # would be negative without the exp
        out = act(torch.randn(1, 8, 32))
        assert torch.isfinite(out).all()

    def test_gradients_reach_alpha_and_beta(self):
        act = SnakeBeta(channels=4)
        act(torch.randn(1, 4, 64)).pow(2).sum().backward()
        assert act.alpha.grad is not None and act.alpha.grad.abs().sum() > 0
        assert act.beta.grad is not None and act.beta.grad.abs().sum() > 0

    def test_matches_snake_when_alpha_equals_beta(self):
        snake = Snake(channels=4, alpha_init=1.5)
        snake_beta = SnakeBeta(channels=4, alpha_init=1.5, beta_init=1.5)
        x = torch.randn(1, 4, 128)
        torch.testing.assert_close(snake(x), snake_beta(x), rtol=1e-5, atol=1e-5)


class TestResampling:
    @pytest.mark.parametrize("length", [100, 101, 4096, 44100])
    def test_up_then_down_preserves_length(self, length):
        x = torch.randn(2, 8, length)
        assert UpSample1d(2)(x).shape[-1] == 2 * length
        assert DownSample1d(2)(UpSample1d(2)(x)).shape[-1] == length

    def test_lowpass_preserves_length_at_stride_one(self):
        x = torch.randn(1, 3, 999)
        assert LowPassFilter1d(stride=1)(x).shape == x.shape

    def test_round_trip_preserves_band_limited_signal(self):
        signal = _tone(440.0) + 0.5 * _tone(3000.0)
        recovered = DownSample1d(2)(UpSample1d(2)(signal))
        # Interior only: the first and last kernel-length of samples carry the
        # replicate-padding boundary, which is not what transparency means here.
        error = (recovered - signal)[..., 1024:-1024]
        error_db = 20 * math.log10(error.abs().max().item() + 1e-12)
        assert error_db < -60

    @pytest.mark.parametrize("freq", [1000.0, 10000.0, 16000.0, 20000.0])
    def test_round_trip_is_flat_across_the_audible_band(self, freq):
        """
        The reason this bank is not BigVGAN's 12 taps. The decoder stacks eight
        of these at full rate, so even a 1 dB droop per pass compounds into an
        audibly dull top end.
        """
        tone = _tone(freq)
        recovered = DownSample1d(2)(UpSample1d(2)(tone))
        droop = _magnitude_db(recovered, freq) - _magnitude_db(tone, freq)
        assert abs(droop) < 0.25, f"{droop:+.2f} dB at {freq:.0f} Hz"

    def test_filter_has_unit_dc_gain(self):
        taps = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)
        assert taps.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_cutoff_above_nyquist_is_rejected(self):
        with pytest.raises(ValueError):
            LowPassFilter1d(cutoff=0.75)


class TestAntiAliasedActivation:
    def test_shape_is_unchanged(self):
        act = AntiAliasedActivation(SnakeBeta(16), ratio=2)
        x = torch.randn(2, 16, 1024)
        assert act(x).shape == x.shape

    def test_ratio_one_is_a_passthrough_wrapper(self):
        inner = SnakeBeta(4)
        wrapped = AntiAliasedActivation(inner, ratio=1)
        x = torch.randn(1, 4, 64)
        torch.testing.assert_close(wrapped(x), inner(x))

    def test_aliasing_is_suppressed(self):
        """
        An 8 kHz tone through Snake generates a 4th harmonic at 32 kHz, which is
        above the 22.05 kHz Nyquist frequency and folds back to 12.1 kHz as an
        inharmonic partial. That fold is what the sandwich exists to remove.
        """
        tone = _tone(8000.0)

        naive = Snake(1)
        naive.alpha.data.fill_(2.0)

        guarded = AntiAliasedActivation(Snake(1), ratio=2)
        guarded.activation.alpha.data.fill_(2.0)

        alias_naive = _magnitude_db(naive(tone), 12100.0)
        alias_guarded = _magnitude_db(guarded(tone), 12100.0)

        assert alias_naive - alias_guarded > 40, (
            f"expected >40 dB alias suppression, got {alias_naive - alias_guarded:.1f} dB"
        )

    def test_legitimate_harmonics_are_preserved(self):
        """Suppressing the fold must not also flatten in-band harmonics."""
        tone = _tone(2000.0)

        naive = Snake(1)
        naive.alpha.data.fill_(2.0)
        guarded = AntiAliasedActivation(Snake(1), ratio=2)
        guarded.activation.alpha.data.fill_(2.0)

        # 2nd harmonic at 4 kHz is real content, well inside the band.
        assert abs(
            _magnitude_db(naive(tone), 4000.0) - _magnitude_db(guarded(tone), 4000.0)
        ) < 1


class TestBuildActivation:
    def test_builds_plain_activation(self):
        act = build_activation(8, kind="snake", anti_aliased=False)
        assert isinstance(act, Snake)

    def test_builds_anti_aliased_snakebeta_by_default(self):
        act = build_activation(8)
        assert isinstance(act, AntiAliasedActivation)
        assert isinstance(act.activation, SnakeBeta)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            build_activation(8, kind="relu")
