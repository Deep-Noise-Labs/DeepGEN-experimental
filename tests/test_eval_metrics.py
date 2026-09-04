"""Tests for the synth-quality metrics.

Each test pins a metric against a signal whose correct answer is known
analytically, so a regression in the measurement cannot silently flatter a
model change.
"""

import numpy as np
import pytest

from synthgen.eval import metrics as M
from synthgen.eval.signals import (
    DEFAULT_TEST_FREQS,
    SWEEP_TEST_FREQS,
    alias_visibility_hz,
    bandlimited_saw,
    bandlimited_square,
    impulse,
    log_sweep,
    pure_tone,
)
from synthgen.eval.suite import SUITE, evaluate_reconstruction, evaluate_synthesis

SR = 44100


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------


def test_default_test_frequencies_can_actually_reveal_aliasing():
    """
    Guard against the trap documented in signals.py: at a badly chosen f0
    every alias lands on top of a harmonic and the metric reads clean no
    matter how bad the model is. 220.5 Hz at 44.1 kHz is exactly such a
    frequency, and it looks entirely reasonable.
    """
    # 0.5 s at 44.1 kHz -> 2 Hz bins, tolerance +/-8 bins -> +/-16 Hz.
    required = 2 * 8 * (SR / 22050)
    for f0 in set(DEFAULT_TEST_FREQS) | set(SWEEP_TEST_FREQS):
        assert alias_visibility_hz(f0, SR) > required, f0
    # and the trap itself must be correctly flagged as unusable
    assert alias_visibility_hz(220.5, SR) == pytest.approx(0.0, abs=1e-9)
    assert alias_visibility_hz(4409.1, SR) < required


def test_bandlimited_saw_has_no_content_above_nyquist_by_construction():
    """Every partial must be a harmonic below Nyquist, or the probe is invalid."""
    f0 = 903.7
    x = bandlimited_saw(f0, 1.0, SR, 0.5)
    report = M.harmonic_analysis(x, f0, SR)
    assert report.alias_to_signal_db < -80.0


def test_square_contains_only_odd_harmonics():
    f0 = 903.7
    x = bandlimited_square(f0, 1.0, SR, 0.5)
    mag, freqs_norm = M._spectrum(x)
    freqs = freqs_norm * SR
    third = mag[np.argmin(np.abs(freqs - 3 * f0))]
    second = mag[np.argmin(np.abs(freqs - 2 * f0))]
    assert third > second * 100


def test_stimuli_are_finite_and_bounded():
    for x in (
        pure_tone(453.1, 0.2, SR),
        bandlimited_saw(453.1, 0.2, SR),
        log_sweep(20, 20000, 0.2, SR),
        impulse(0.2, SR),
    ):
        assert np.all(np.isfinite(x))
        assert np.max(np.abs(x)) <= 1.0


# ---------------------------------------------------------------------------
# Alias metrics
# ---------------------------------------------------------------------------


def test_blackman_harris_sidelobes_beat_numpy_blackman():
    """The measurement floor depends on this; guard it."""
    n = 4096
    for window, limit in ((M.blackman_harris(n), -85.0), (np.blackman(n), -55.0)):
        spec = np.abs(np.fft.rfft(window))
        spec = spec / spec.max()
        sidelobe = 20 * np.log10(spec[20:].max() + 1e-15)
        assert sidelobe < limit


def test_added_inharmonic_tone_is_detected_at_the_right_level():
    """An injected spur 40 dB down must read as roughly -40 dB."""
    f0 = 1000.7
    t = np.arange(SR) / SR
    clean = np.sin(2 * np.pi * f0 * t)
    spur = 0.01 * np.sin(2 * np.pi * 3123.4 * t)  # -40 dB, off the harmonic grid
    report = M.harmonic_analysis(clean + spur, f0, SR)
    assert -44.0 < report.alias_to_signal_db < -36.0
    assert report.worst_spur_hz == pytest.approx(3123.4, abs=5.0)


def test_sub_fundamental_only_counts_energy_below_f0():
    f0 = 4000.7
    t = np.arange(SR) / SR
    clean = np.sin(2 * np.pi * f0 * t)
    below = clean + 0.01 * np.sin(2 * np.pi * 501.3 * t)
    above = clean + 0.01 * np.sin(2 * np.pi * 9501.3 * t)
    assert M.sub_fundamental_alias_db(below, f0, SR) > -50.0
    assert M.sub_fundamental_alias_db(above, f0, SR) < -80.0


def test_sfdr_matches_an_injected_spur():
    f0 = 1000.7
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * f0 * t) + 0.001 * np.sin(2 * np.pi * 3123.4 * t)
    assert M.spurious_free_dynamic_range_db(x, f0, SR) == pytest.approx(60.0, abs=3.0)


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------


def test_identical_signals_score_perfectly():
    x = bandlimited_saw(453.1, 0.5, SR, 0.5)
    stereo = np.stack([x, np.roll(x, 13)])
    assert M.si_sdr_db(x, x) > 100
    assert M.multires_stft_distance(x, x) == pytest.approx(0.0, abs=1e-6)
    assert M.high_frequency_retention_db(x, x, SR) == pytest.approx(0.0, abs=1e-6)
    assert M.stereo_width_error(stereo, stereo) == pytest.approx(0.0, abs=1e-9)
    assert M.transient_error_ms(x, x, SR) == pytest.approx(0.0, abs=1e-9)


def test_lowpass_shows_up_as_lost_air():
    """Removing the top octave must register as negative HF retention."""
    x = bandlimited_saw(453.1, 0.5, SR, 0.5)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec[freqs > 8000] = 0
    dulled = np.fft.irfft(spec, n=len(x))
    assert M.high_frequency_retention_db(dulled, x, SR) < -20.0


def test_mono_collapse_shows_up_as_stereo_error():
    # A rolled *periodic* signal is still correlated with itself, so the wide
    # reference has to be genuinely decorrelated - as a real wide pad is.
    rng = np.random.default_rng(0)
    x = bandlimited_saw(453.1, 0.5, SR, 0.5)
    wide = np.stack([x, rng.normal(scale=0.3, size=len(x)).astype(np.float32)])
    collapsed = np.stack([x, x])
    assert M.stereo_correlation(collapsed) == pytest.approx(1.0, abs=1e-6)
    assert abs(M.stereo_width_error(collapsed, wide)) > 0.2


def test_smeared_attack_shows_up_as_transient_error():
    x = impulse(0.3, SR, 0.9, position=0.3)
    smeared = np.convolve(x, np.hanning(441) / np.hanning(441).sum(), mode="same")
    assert M.transient_error_ms(smeared, x, SR) > 1.0


def test_noise_floor_tracks_added_noise():
    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 440 * t)
    tone[SR // 2 :] = 0.0  # silence in the second half
    quiet = M.noise_floor_db(tone, SR)
    noisy = M.noise_floor_db(tone + 0.01 * np.random.default_rng(0).normal(size=SR), SR)
    assert noisy > quiet


# ---------------------------------------------------------------------------
# Suite plumbing
# ---------------------------------------------------------------------------


def test_every_gate_has_a_rationale_and_a_stretch_beyond_its_target():
    for gate in SUITE:
        assert gate.rationale.strip()
        assert gate.passes(gate.stretch), f"{gate.key}: stretch must pass its own gate"


def test_identity_process_passes_the_synthesis_gates():
    results = evaluate_synthesis(lambda x: x, SR, freqs=(903.7, 2090.1), duration=0.5)
    assert all(r.passed for r in results.values()), {
        k: r.value for k, r in results.items()
    }


def test_identity_reconstruction_passes_the_reference_gates():
    x = bandlimited_saw(453.1, 0.5, SR, 0.5)
    stereo = np.stack([x, np.roll(x, 13)])
    results = evaluate_reconstruction(stereo, stereo, SR)
    for key in ("hf_retention_db", "transient_error_ms", "si_sdr_db", "multires_stft"):
        assert results[key].passed, f"{key} = {results[key].value}"
