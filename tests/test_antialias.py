"""Tests for the alias-free activation path."""

import numpy as np
import pytest
import torch

from synthgen.eval.metrics import alias_to_signal_ratio, total_harmonic_distortion
from synthgen.eval.probes import saw_note, sine
from synthgen.model.antialias import (
    AntiAliasedActivation,
    DownSample1d,
    UpSample1d,
    kaiser_sinc_filter1d,
)
from synthgen.model.vae import AudioVAE, Snake, make_activation, remap_state_dict

SR = 44100


def _apply(module, x):
    with torch.no_grad():
        return module(torch.from_numpy(np.asarray(x, np.float32))[None, None, :])[0, 0].numpy()


# ---------------------------------------------------------------- filters


def test_filter_has_unit_dc_gain():
    """A resampling filter must not change the level of a DC/low-frequency signal."""
    f = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=32)
    assert f.shape == (1, 1, 32)
    assert float(f.sum()) == pytest.approx(1.0, abs=1e-6)


def test_filter_is_symmetric():
    """Symmetry is what makes the filter linear-phase, so up/down stays aligned."""
    f = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=32)[0, 0]
    assert torch.allclose(f, f.flip(0), atol=1e-6)


# ---------------------------------------------------------------- resamplers


@pytest.mark.parametrize("ratio", [2, 4])
def test_resampler_shapes(ratio):
    x = torch.randn(2, 5, 4096)
    up = UpSample1d(ratio)(x)
    assert up.shape == (2, 5, 4096 * ratio)
    assert DownSample1d(ratio)(up).shape == x.shape


@pytest.mark.parametrize("freq", [100.0, 1000.0, 5000.0])
def test_resample_roundtrip_preserves_band_limited_signal(freq):
    """Up then down must be near-transparent for content well below Nyquist."""
    x = sine(freq, 0.25, SR)
    y = _apply(torch.nn.Sequential(UpSample1d(2), DownSample1d(2)), x)
    n = min(len(x), len(y))
    err = y[:n] - x[:n]
    snr = 10 * np.log10(np.sum(x[:n] ** 2) / (np.sum(err**2) + 1e-30))
    assert snr > 40.0, f"round-trip SNR {snr:.1f} dB at {freq} Hz"


# ---------------------------------------------------------------- activation


def test_antialiased_activation_preserves_shape():
    """Drop-in requirement: identical shape contract to the bare activation."""
    x = torch.randn(3, 8, 2048)
    assert AntiAliasedActivation(Snake(8))(x).shape == Snake(8)(x).shape


def test_antialiasing_reduces_alias_energy_at_high_frequency():
    """The headline claim, asserted as a regression guard."""
    f0 = 8000.0
    probe = sine(f0, 1.0, SR)
    before = alias_to_signal_ratio(
        _apply(make_activation(1, antialias=False, alpha_init=2.0), probe), f0, SR
    )
    after = alias_to_signal_ratio(
        _apply(make_activation(1, antialias=True, alpha_init=2.0), probe), f0, SR
    )
    assert after < before - 30.0, f"expected >30 dB improvement, got {before:.1f} -> {after:.1f}"


def test_antialiasing_preserves_intended_harmonic_character():
    """
    Removing aliasing must not also remove the harmonics Snake is *meant* to add,
    otherwise the fix would just be a low-pass filter.
    """
    f0 = 8000.0
    probe = sine(f0, 1.0, SR)
    thd_before = total_harmonic_distortion(
        _apply(make_activation(1, antialias=False, alpha_init=2.0), probe), f0, SR
    )
    thd_after = total_harmonic_distortion(
        _apply(make_activation(1, antialias=True, alpha_init=2.0), probe), f0, SR
    )
    assert abs(thd_after - thd_before) < 3.0


def test_low_frequency_content_is_unaffected():
    """Below the aliasing regime the two paths should agree closely."""
    probe = saw_note(110.0, 0.5, SR)
    before = _apply(make_activation(1, antialias=False, alpha_init=1.0), probe)
    after = _apply(make_activation(1, antialias=True, alpha_init=1.0), probe)
    n = min(len(before), len(after))
    rel = np.sqrt(np.mean((before[:n] - after[:n]) ** 2)) / (
        np.sqrt(np.mean(before[:n] ** 2)) + 1e-30
    )
    assert rel < 0.05


# ---------------------------------------------------------------- VAE wiring


@pytest.mark.parametrize("antialias", [False, True])
def test_vae_forward_shapes(antialias):
    model = AudioVAE(antialias=antialias).eval()
    x = torch.randn(1, 2, 8192)
    with torch.no_grad():
        recon, target, mean, log_var = model(x)
    assert mean.shape[1] == model.latent_dim
    assert recon.shape == target.shape
    assert recon.shape[1] == 2


def test_antialiasing_adds_no_parameters():
    """The wrapper holds fixed FIR buffers only, so capacity is unchanged."""
    plain = sum(p.numel() for p in AudioVAE(antialias=False).parameters())
    anti = sum(p.numel() for p in AudioVAE(antialias=True).parameters())
    assert plain == anti


def test_state_dict_remap_roundtrip():
    """A checkpoint must move between the two layouts without losing weights."""
    anti = AudioVAE(antialias=True)
    plain = AudioVAE(antialias=False)
    plain.load_state_dict(remap_state_dict(anti.state_dict(), to_antialias=False), strict=True)

    restored = AudioVAE(antialias=True)
    restored.load_state_dict(remap_state_dict(plain.state_dict(), to_antialias=True), strict=True)

    for (key, before), (_, after) in zip(
        anti.state_dict().items(), restored.state_dict().items()
    ):
        assert torch.equal(before, after), key


def test_compression_ratio_matches_strides():
    """Guards the docstring claim that drifted from the code once already."""
    model = AudioVAE(strides=(4, 4, 8, 8))
    assert model.compression_ratio == 1024
