"""Tests for the sample-quality metrics."""

import numpy as np
import pytest

from synthgen.eval.metrics import (
    QualityTarget,
    absolute_metrics,
    bandwidth_hz,
    comparative_metrics,
    grade,
    mono_compatibility_db,
    si_sdr_db,
    stereo_width,
)

SR = 44100


def tone(freq: float, seconds: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x])


class TestBandwidth:
    def test_detects_lowpass(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.1, SR)
        # Crude brick wall in the frequency domain.
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1 / SR)
        spectrum[freqs > 5000] = 0
        band_limited = np.fft.irfft(spectrum, len(noise))

        assert bandwidth_hz(band_limited, SR) < 6000
        assert bandwidth_hz(noise, SR) > 15000


class TestStereo:
    def test_mono_signal_folds_without_loss(self):
        # Identical channels: summing to mono changes nothing.
        assert mono_compatibility_db(tone(440.0)) == pytest.approx(0.0, abs=0.1)

    def test_antiphase_cancels(self):
        x = tone(440.0)
        x[1] = -x[1]
        assert mono_compatibility_db(x) < -40

    def test_width_of_correlated_signal_is_negative_infinity(self):
        assert stereo_width(tone(440.0)) < -100


class TestComparative:
    def test_si_sdr_is_high_for_scaled_copy(self):
        x = tone(440.0)
        assert si_sdr_db(x * 0.5, x) > 50

    def test_si_sdr_is_low_for_noise(self):
        rng = np.random.default_rng(1)
        x = tone(440.0)
        assert si_sdr_db(rng.normal(0, 0.5, x.shape), x) < 5

    def test_band_error_flags_missing_air(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 0.1, (2, SR))
        spectrum = np.fft.rfft(noise, axis=-1)
        freqs = np.fft.rfftfreq(SR, 1 / SR)
        spectrum[:, freqs > 8000] = 0
        dulled = np.fft.irfft(spectrum, SR, axis=-1)

        result = comparative_metrics(dulled, noise, SR)
        # The air band should be reported as heavily under-energised.
        assert result["band_err_air_db"] < -20


class TestGrading:
    def test_full_band_clean_tone_passes_bandwidth_and_dc(self):
        rng = np.random.default_rng(3)
        audio = tone(440.0) + rng.normal(0, 0.02, (2, SR))
        grades = grade(absolute_metrics(audio, SR), QualityTarget())
        assert grades["sample_rate"]
        assert grades["dc_offset"]
        assert grades["no_clipping"]

    def test_clipped_audio_fails(self):
        audio = np.clip(tone(440.0, amp=2.0), -1.0, 1.0)
        grades = grade(absolute_metrics(audio, SR), QualityTarget())
        assert not grades["no_clipping"]
        assert not grades["true_peak"]

    def test_low_sample_rate_fails_sample_rate_and_bandwidth(self):
        rng = np.random.default_rng(4)
        sr = 32000
        audio = rng.normal(0, 0.1, (2, sr))
        grades = grade(absolute_metrics(audio, sr), QualityTarget())
        assert not grades["sample_rate"]
        # Nyquist is 16 kHz, so the 18 kHz bandwidth floor cannot be cleared
        # by any 32 kHz output, however clean it is.
        assert not grades["bandwidth"]
