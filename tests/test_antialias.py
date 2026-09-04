"""Tests for the alias-free synthesis primitives."""

import numpy as np
import pytest
import torch

from synthgen.eval.metrics import harmonic_analysis
from synthgen.eval.signals import bandlimited_saw, pure_tone
from synthgen.model.antialias import (
    AntiAliasedSnake,
    DownSample1d,
    LowPassFilter1d,
    Snake,
    UpSample1d,
    kaiser_sinc_filter1d,
    make_activation,
)

SR = 44100


def test_filter_has_unit_dc_gain():
    """A level-preserving filter must sum to 1, or every block changes gain."""
    for kernel_size in (8, 12, 16, 24):
        filt = kaiser_sinc_filter1d(0.25, 0.3, kernel_size)
        assert filt.shape == (1, 1, kernel_size)
        assert float(filt.sum()) == pytest.approx(1.0, abs=1e-6)


def test_lowpass_rejects_above_cutoff():
    """Content above the cutoff must be attenuated, content below preserved."""
    t = np.arange(SR) / SR
    low = torch.from_numpy(np.sin(2 * np.pi * 1000 * t)).float().view(1, 1, -1)
    high = torch.from_numpy(np.sin(2 * np.pi * 18000 * t)).float().view(1, 1, -1)
    lpf = LowPassFilter1d(cutoff=0.25, half_width=0.05, kernel_size=64)

    passed = float(lpf(low)[..., 500:-500].std())
    stopped = float(lpf(high)[..., 500:-500].std())
    assert passed > 0.6
    assert stopped < 0.05


def test_upsample_downsample_round_trip_is_near_identity():
    """A band-limited signal must survive 2x up then 2x down intact."""
    t = np.arange(8192) / SR
    x = torch.from_numpy(np.sin(2 * np.pi * 1000 * t)).float().view(1, 1, -1)
    y = DownSample1d(2)(UpSample1d(2)(x))
    assert y.shape == x.shape
    assert float((y - x)[..., 128:-128].abs().max()) < 5e-3


def test_antialiased_activation_preserves_length():
    for ratio in (2, 3, 4):
        x = torch.randn(2, 5, 4096)
        assert AntiAliasedSnake(5, ratio=ratio)(x).shape == x.shape


def test_antialiasing_reduces_alias_energy():
    """
    The claim this whole module exists to make.

    A band-limited sawtooth has no inharmonic energy. After a pointwise
    nonlinearity at the base rate it does, because harmonics above Nyquist
    fold back. Oversampling the nonlinearity must measurably reduce that.
    """
    f0 = 2090.1
    stimulus = bandlimited_saw(f0, 0.5, SR, 0.5)
    x = torch.from_numpy(stimulus).view(1, 1, -1)

    torch.manual_seed(0)
    with torch.no_grad():
        plain = Snake(1)(x).view(-1).numpy()
        guarded = AntiAliasedSnake(1, ratio=2)(x).view(-1).numpy()

    plain_asr = harmonic_analysis(plain, f0, SR).alias_to_signal_db
    guarded_asr = harmonic_analysis(guarded, f0, SR).alias_to_signal_db

    assert guarded_asr < plain_asr - 3.0, (
        f"expected a clear reduction, got {plain_asr:.2f} -> {guarded_asr:.2f} dB"
    )


def test_higher_ratio_reduces_alias_further():
    """More oversampling headroom must not make aliasing worse."""
    f0 = 2090.1
    x = torch.from_numpy(bandlimited_saw(f0, 0.5, SR, 0.5)).view(1, 1, -1)
    torch.manual_seed(0)
    with torch.no_grad():
        r2 = harmonic_analysis(
            AntiAliasedSnake(1, ratio=2)(x).view(-1).numpy(), f0, SR
        ).alias_to_signal_db
        r4 = harmonic_analysis(
            AntiAliasedSnake(1, ratio=4)(x).view(-1).numpy(), f0, SR
        ).alias_to_signal_db
    assert r4 <= r2 + 0.5


def test_pure_tone_stays_clean_through_the_filters():
    """The fix must not itself add junk to an already-clean signal."""
    f0 = 903.7
    x = torch.from_numpy(pure_tone(f0, 0.5, SR, 0.5)).view(1, 1, -1)
    with torch.no_grad():
        y = DownSample1d(2)(UpSample1d(2)(x)).view(-1).numpy()
    assert harmonic_analysis(y, f0, SR).alias_to_signal_db < -60.0


def test_make_activation_switches_implementation():
    assert isinstance(make_activation(8, antialias=False), Snake)
    assert isinstance(make_activation(8, antialias=True), AntiAliasedSnake)


def test_antialias_adds_no_parameters():
    """The filters are fixed buffers - the fix must be free at inference."""
    plain = sum(p.numel() for p in make_activation(32, antialias=False).parameters())
    guarded = sum(p.numel() for p in make_activation(32, antialias=True).parameters())
    assert plain == guarded


def test_alpha_is_reachable_on_the_wrapper():
    """Checkpoint remapping depends on alpha staying addressable."""
    act = AntiAliasedSnake(16)
    assert act.alpha.shape == (1, 16, 1)


def test_filters_stay_out_of_the_state_dict():
    """
    Non-persistent buffers keep checkpoints interchangeable between the two
    activations - only ``alpha`` is stored, so a Snake checkpoint remains
    remappable and parameter counts stay comparable.
    """
    keys = list(AntiAliasedSnake(4).state_dict().keys())
    assert keys == ["act.activation.alpha"]


def test_short_sequences_do_not_crash():
    """
    Replicate padding refuses pads wider than the input, so very short
    latents are a real failure mode for a filtered activation.
    """
    for n in (1, 2, 4, 5, 8, 64):
        x = torch.randn(1, 3, n)
        assert AntiAliasedSnake(3, ratio=2)(x).shape == x.shape


def test_gradient_reaches_alpha():
    act = AntiAliasedSnake(4)
    act(torch.randn(1, 4, 256)).sum().backward()
    assert act.alpha.grad is not None
    assert torch.isfinite(act.alpha.grad).all()


def test_runs_under_bf16_autocast():
    """Training config defaults to bf16; the filters must survive it."""
    act = AntiAliasedSnake(4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = act(torch.randn(1, 4, 512))
    assert torch.isfinite(out.float()).all()
