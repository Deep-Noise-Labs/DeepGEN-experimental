"""
Tests for PerceptualSampleLoss.

The behavioural tests below are the point of the class: they assert that the
loss actually *moves* for degradations the legacy loss is close to blind to.
"""

import math

import pytest
import torch

from synthgen.model.vae import Snake
from synthgen.training.losses import (
    MultiResolutionSTFTLoss,
    PerceptualSampleLoss,
    VAELoss,
)

SR = 44100
# Short windows keep the tests fast; the behaviour under test is unchanged.
FAST_FFTS = (4096, 1024, 256)


def loss_fn() -> PerceptualSampleLoss:
    return PerceptualSampleLoss(fft_sizes=FAST_FFTS, sample_rate=SR)


def stereo_noise(seed: int = 0, samples: int = SR // 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 2, samples, generator=g) * 0.1


def band_filter(x: torch.Tensor, sr: int, low: float, high: float) -> torch.Tensor:
    """Zero out everything outside ``[low, high)`` Hz."""
    spectrum = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / sr)
    mask = (freqs >= low) & (freqs < high)
    return torch.fft.irfft(spectrum * mask, n=x.shape[-1], dim=-1)


class TestBasics:
    def test_identical_signals_give_zero_loss(self):
        x = stereo_noise()
        assert loss_fn()(x, x).item() < 1e-5

    def test_different_signals_give_positive_loss(self):
        assert loss_fn()(stereo_noise(0), stereo_noise(1)).item() > 0

    def test_gradients_flow_to_the_prediction(self):
        target = stereo_noise(0)
        pred = stereo_noise(1).requires_grad_(True)
        loss_fn()(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0

    def test_handles_mono_input(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 1, SR // 2, generator=g) * 0.1
        assert torch.isfinite(loss_fn()(x, x * 0.5)).all()

    def test_handles_signal_shorter_than_longest_window(self):
        x = stereo_noise(samples=1024)
        assert torch.isfinite(loss_fn()(x, x * 0.5)).all()

    def test_rejects_unknown_spectral_loss_name(self):
        with pytest.raises(ValueError, match="Unknown spectral_loss"):
            VAELoss(spectral_loss="nope")


class TestSensitivity:
    """The degradations that separate a usable sample from a commercial one."""

    def test_penalises_removal_of_the_air_band(self):
        target = stereo_noise()
        dulled = band_filter(target, SR, 0.0, 10000.0)
        assert loss_fn()(dulled, target).item() > 0.1

    def test_is_near_blind_to_inaudible_noise(self):
        """
        The null test, and the headline result of docs/EVALUATION.md.

        Noise at -90 dBFS cannot be heard on any playback system, so a loss
        that reacts to it is spending gradient budget that should have gone to
        audible content. Measured against each loss's own response to a real,
        audible defect, so the two scales are comparable.

        Note what is deliberately *not* asserted here: that the new loss weighs
        the air band more heavily than the legacy one. That was the original
        hypothesis and measurement did not support it -- the air-band share
        barely moves (0.89x on real audio). The wins are here, on stereo, and
        on sub-bass. See docs/EVALUATION.md section 3.
        """
        # The target must roll off, as all real audio does. On flat white
        # noise every bin sits far above the dither and both losses ignore it;
        # the difference only appears where the spectrum goes quiet, which is
        # most of a real instrument sample.
        target = band_filter(stereo_noise(), SR, 0.0, 4000.0)
        audible = band_filter(target, SR, 0.0, 2000.0)
        rng = torch.Generator().manual_seed(7)
        inaudible = target + torch.randn(
            target.shape, generator=rng
        ) * (10 ** (-90.0 / 20.0))

        legacy = MultiResolutionSTFTLoss(
            fft_sizes=FAST_FFTS,
            hop_sizes=tuple(n // 4 for n in FAST_FFTS),
            win_sizes=FAST_FFTS,
        )
        new = loss_fn()

        legacy_waste = (
            legacy(inaudible, target).item() / legacy(audible, target).item()
        )
        new_waste = new(inaudible, target).item() / new(audible, target).item()

        # The legacy loss scores above 1.0 here: it responds to noise nobody
        # can hear *more* than to a 2 kHz band genuinely going missing.
        assert legacy_waste > 1.0
        assert new_waste < legacy_waste / 100

    def test_penalises_stereo_collapse_where_legacy_loss_cannot(self):
        """
        Build a target whose two channels have identical magnitude spectra but
        opposite phase. Collapsing it to mono leaves every per-channel
        magnitude untouched, so a channel-independent loss cannot see the
        change at all -- but the image is destroyed.
        """
        base = stereo_noise()
        target = torch.stack([base[:, 0], -base[:, 0]], dim=1)
        collapsed = torch.stack([base[:, 0], base[:, 0]], dim=1)

        legacy = MultiResolutionSTFTLoss(
            fft_sizes=FAST_FFTS,
            hop_sizes=tuple(n // 4 for n in FAST_FFTS),
            win_sizes=FAST_FFTS,
        )
        # Magnitude-only and per-channel: the polarity flip is invisible.
        assert legacy(collapsed, target).item() < 1e-5
        # The mid/side term sees it.
        assert loss_fn()(collapsed, target).item() > 0.1

    def test_penalises_transient_smearing(self):
        """A click, and the same click smeared over 10 ms."""
        n = SR // 2
        click = torch.zeros(1, 2, n)
        click[..., n // 2] = 1.0
        width = int(SR * 0.01)
        kernel = torch.ones(2, 1, width) / width
        smeared = torch.nn.functional.conv1d(
            click, kernel, padding=width // 2, groups=2
        )[..., :n]
        assert loss_fn()(smeared, click).item() > 0.1


class TestSnakeStability:
    def test_alpha_stays_positive_under_a_large_negative_update(self):
        snake = Snake(channels=4)
        with torch.no_grad():
            snake.log_alpha -= 20.0  # would drive a raw alpha far past zero
        assert (snake.alpha > 0).all()
        out = snake(torch.randn(1, 4, 128))
        assert torch.isfinite(out).all()

    def test_matches_the_reference_formula_at_alpha_one(self):
        snake = Snake(channels=1, alpha_init=1.0)
        x = torch.linspace(-3, 3, 64).view(1, 1, -1)
        expected = x + torch.sin(x) ** 2
        assert torch.allclose(snake(x), expected, atol=1e-6)

    def test_loads_a_legacy_alpha_checkpoint(self):
        snake = Snake(channels=3, alpha_init=1.0)
        legacy_state = {"alpha": torch.full((1, 3, 1), 2.0)}
        snake.load_state_dict(legacy_state)
        assert torch.allclose(snake.alpha, torch.full((1, 3, 1), 2.0), atol=1e-5)

    def test_rejects_non_positive_alpha_init(self):
        with pytest.raises(ValueError):
            Snake(channels=2, alpha_init=0.0)
