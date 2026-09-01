"""Tests for the sample-grade preprocessing pipeline.

The guarantees asserted here are the ones the model depends on: no clipping, no
DC, no edge clicks, a preserved attack transient and a consistent loudness.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthgen.data.preprocessing import (
    clipped_fraction,
    crest_factor_db,
    db_to_linear,
    detect_onset,
    fade_edges,
    is_continuous,
    loudness_normalize,
    onset_anchored_crop,
    peak_db,
    prepare_sample,
    prepare_sample_legacy,
    remove_dc_offset,
    rms_db,
    soft_limit,
    trim_silence,
)

SR = 44100


def _pluck(sample_rate: int = SR, seconds: float = 2.0, freq: float = 220.0) -> np.ndarray:
    """A percussive one-shot: instant attack, exponential decay."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = np.exp(-6.0 * t)
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 3 * t)
    return np.stack([tone * envelope, tone * envelope]).astype(np.float32) * 0.5


def _with_leading_silence(audio: np.ndarray, seconds: float, sample_rate: int = SR):
    pad = np.zeros((audio.shape[0], int(sample_rate * seconds)), dtype=np.float32)
    return np.concatenate([pad, audio], axis=-1)


class TestLevelHelpers:
    def test_rms_and_peak_of_full_scale_sine(self):
        t = np.arange(SR) / SR
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)[np.newaxis, :]
        assert peak_db(sine) == pytest.approx(0.0, abs=0.01)
        # A sine's RMS sits 3.01 dB below its peak.
        assert rms_db(sine) == pytest.approx(-3.01, abs=0.05)
        assert crest_factor_db(sine) == pytest.approx(3.01, abs=0.05)

    def test_clipped_fraction_detects_hard_clipping(self):
        x = np.clip(np.linspace(-2.0, 2.0, 1000, dtype=np.float32), -1.0, 1.0)
        assert clipped_fraction(x[np.newaxis, :]) > 0.4


class TestDCOffset:
    def test_removes_per_channel_offset(self):
        audio = _pluck()
        audio[0] += 0.2
        audio[1] -= 0.1
        out = remove_dc_offset(audio)
        assert abs(float(np.mean(out[0]))) < 1e-6
        assert abs(float(np.mean(out[1]))) < 1e-6

    def test_handles_mono_1d(self):
        out = remove_dc_offset(np.full(1000, 0.5, dtype=np.float32))
        assert abs(float(np.mean(out))) < 1e-6


class TestOnsetAndTrim:
    def test_detect_onset_finds_the_attack(self):
        audio = _with_leading_silence(_pluck(), seconds=0.5)
        onset = detect_onset(audio, SR)
        assert onset is not None
        # Within 20 ms of the true onset at 0.5 s.
        assert abs(onset - int(SR * 0.5)) < int(SR * 0.02)

    def test_detect_onset_returns_none_for_silence(self):
        assert detect_onset(np.zeros((2, SR), dtype=np.float32), SR) is None

    def test_trim_removes_leading_silence(self):
        audio = _with_leading_silence(_pluck(), seconds=1.0)
        trimmed = trim_silence(audio, SR)
        assert trimmed.shape[-1] < audio.shape[-1]
        # The attack survives near the very start of the trimmed clip.
        assert detect_onset(trimmed, SR) < int(SR * 0.02)

    def test_continuous_material_is_flagged(self):
        t = np.arange(SR * 3) / SR
        drone = np.stack([np.sin(2 * np.pi * 110 * t)] * 2).astype(np.float32) * 0.4
        assert is_continuous(drone, SR) is True
        assert is_continuous(_with_leading_silence(_pluck(), 0.5), SR) is False


class TestOnsetAnchoredCrop:
    def test_attack_is_preserved_when_cropping_a_long_file(self):
        # 8 s source, attack at 3 s, cropped to 2 s. A random crop would keep
        # the onset only ~6% of the time; anchoring always does.
        source = np.concatenate(
            [
                np.zeros((2, SR * 3), dtype=np.float32),
                _pluck(seconds=5.0),
            ],
            axis=-1,
        )
        cropped = onset_anchored_crop(source, SR * 2, SR, anchor_prob=1.0)
        assert cropped.shape[-1] == SR * 2
        onset = detect_onset(cropped, SR)
        assert onset is not None
        # The attack lands inside the pre-roll at the head of the crop.
        assert onset < int(SR * 0.05)

    def test_short_input_is_padded_to_target(self):
        out = onset_anchored_crop(_pluck(seconds=0.5), SR * 2, SR)
        assert out.shape == (2, SR * 2)

    def test_long_sources_still_get_random_windows(self):
        """anchor_prob keeps augmentation diversity on long recordings."""
        t = np.arange(SR * 6) / SR
        drone = np.stack([np.sin(2 * np.pi * 110 * t)] * 2).astype(np.float32) * 0.4
        rng = np.random.default_rng(0)
        starts = {
            float(onset_anchored_crop(drone, SR, SR, anchor_prob=0.5, rng=rng)[0, 0])
            for _ in range(16)
        }
        # Random starts give different first samples; anchoring alone would not.
        assert len(starts) > 1

    def test_short_sources_are_always_anchored(self):
        """Under 2x the window, a random crop buys no diversity worth an attack."""
        source = np.concatenate(
            [np.zeros((2, SR // 2), dtype=np.float32), _pluck(seconds=2.0)], axis=-1
        )
        rng = np.random.default_rng(0)
        crops = [
            onset_anchored_crop(source, SR * 2, SR, anchor_prob=0.0, rng=rng)
            for _ in range(8)
        ]
        assert all(np.array_equal(c, crops[0]) for c in crops)
        assert detect_onset(crops[0], SR) < int(SR * 0.05)


class TestFadesAndLimiting:
    def test_fade_edges_removes_the_boundary_step(self):
        block = np.full((2, SR), 0.8, dtype=np.float32)
        faded = fade_edges(block, SR)
        assert abs(float(faded[0, 0])) < 0.01
        assert abs(float(faded[0, -1])) < 0.01
        # The body is untouched.
        assert float(faded[0, SR // 2]) == pytest.approx(0.8, abs=1e-6)

    def test_soft_limit_bounds_without_hard_clipping(self):
        loud = np.linspace(-2.0, 2.0, 4096, dtype=np.float32)[np.newaxis, :]
        out = soft_limit(loud, ceiling_db=-1.0)
        assert peak_db(out) <= -1.0 + 1e-3
        assert clipped_fraction(out) == 0.0
        # Strictly monotonic: no flat top, which is what generates the harmonics.
        diffs = np.diff(out[0])
        assert float(np.min(diffs)) > 0.0

    def test_soft_limit_leaves_quiet_signal_untouched(self):
        quiet = (_pluck() * 0.05).astype(np.float32)
        assert np.allclose(soft_limit(quiet, ceiling_db=-1.0), quiet)


class TestLoudnessNormalize:
    def test_hits_the_rms_target_for_moderate_crest_material(self):
        t = np.arange(SR) / SR
        sine = np.stack([np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32) * 0.01
        out = loudness_normalize(sine, target_rms_db=-20.0, peak_ceiling_db=-1.0)
        assert rms_db(out) == pytest.approx(-20.0, abs=0.5)

    def test_never_exceeds_the_ceiling(self):
        for scale in (0.001, 0.1, 1.0, 8.0):
            out = loudness_normalize(_pluck() * scale, peak_ceiling_db=-1.0)
            assert peak_db(out) <= -1.0 + 1e-3
            assert clipped_fraction(out) == 0.0

    def test_matches_loudness_across_crest_factors(self):
        """A pluck and a pad end up at the same perceived level; peak
        normalisation would leave them far apart."""
        t = np.arange(SR * 2) / SR
        pad = np.stack([np.sin(2 * np.pi * 220 * t) * 0.7] * 2).astype(np.float32)
        pluck = _pluck(seconds=2.0)

        norm_pad = loudness_normalize(pad)
        norm_pluck = loudness_normalize(pluck)
        assert abs(rms_db(norm_pad) - rms_db(norm_pluck)) < 1.5

        # The legacy behaviour: same peak, very different loudness.
        peak_pad = pad * (0.95 / np.max(np.abs(pad)))
        peak_pluck = pluck * (0.95 / np.max(np.abs(pluck)))
        assert abs(rms_db(peak_pad) - rms_db(peak_pluck)) > 5.0

    def test_silence_is_left_alone(self):
        silence = np.zeros((2, 1000), dtype=np.float32)
        assert np.array_equal(loudness_normalize(silence), silence)


class TestPrepareSample:
    @pytest.mark.parametrize("seed", range(12))
    def test_never_clips_with_augmentation_on(self, seed):
        rng = np.random.default_rng(seed)
        out = prepare_sample(
            _with_leading_silence(_pluck(), 0.3),
            sample_rate=SR,
            target_samples=SR,
            augment=True,
            rng=rng,
        )
        assert clipped_fraction(out) == 0.0
        assert peak_db(out) <= -1.0 + 1e-3

    def test_output_shape_and_dtype(self):
        out = prepare_sample(_pluck(), SR, SR * 3)
        assert out.shape == (2, SR * 3)
        assert out.dtype == np.float32

    def test_accepts_mono_input(self):
        mono = _pluck()[0]
        out = prepare_sample(mono, SR, SR)
        assert out.shape == (1, SR)

    def test_silent_input_does_not_crash(self):
        out = prepare_sample(np.zeros((2, SR), dtype=np.float32), SR, SR)
        assert out.shape == (2, SR)
        assert np.all(np.isfinite(out))

    def test_attack_survives_the_pipeline(self):
        source = np.concatenate(
            [np.zeros((2, SR * 3), dtype=np.float32), _pluck(seconds=5.0)], axis=-1
        )
        out = prepare_sample(source, SR, SR * 2, augment=False)
        assert detect_onset(out, SR) < int(SR * 0.05)


class TestLegacyBaseline:
    """Characterisation of the pipeline this change replaces."""

    def test_legacy_clips_on_roughly_43_percent_of_draws(self):
        """0.95 peak plus a U(-3, +3) dB gain clips whenever the draw exceeds
        +0.446 dB, i.e. (3 - 0.446) / 6 = 42.6% of the time."""
        rng = np.random.default_rng(0)
        audio = _pluck()
        clipped = sum(
            clipped_fraction(prepare_sample_legacy(audio, SR, augment=True, rng=rng)) > 0
            for _ in range(500)
        )
        assert 0.35 < clipped / 500 < 0.50

    def test_new_pipeline_clips_on_none_of_them(self):
        rng = np.random.default_rng(0)
        audio = _pluck()
        clipped = sum(
            clipped_fraction(
                prepare_sample(audio, SR, SR, augment=True, rng=rng)
            ) > 0
            for _ in range(500)
        )
        assert clipped == 0

    def test_clipping_threshold_is_where_the_arithmetic_says(self):
        # 0.95 * 10^(0.446/20) = 1.0
        assert 0.95 * db_to_linear(0.446) == pytest.approx(1.0, abs=1e-3)
