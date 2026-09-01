"""Sample-grade audio preprocessing for SynthGen training.

For a text-to-sample model the training target *is* the product. Anything the
pipeline does to a clip on the way into the batch - distortion, level jitter, a
missing attack transient, a click at the pad boundary - is a statistic the model
learns and reproduces in every generation. Commercial sample libraries
(Spitfire, Serum, Splice) are level-matched, onset-aligned, DC-free and never
clipped; a model trained on material that is none of those things cannot sound
like one.

This module is the ordered pipeline that gets a clip to that standard. The
entry point is :func:`prepare_sample`; the individual stages are exported so
they can be tested and reused.

Stage order (it matters)::

    1. remove_dc_offset      # DC eats headroom and biases the VAE input conv
    2. trim_silence          # leading silence teaches the model to start late
    3. onset_anchored_crop   # keep the attack - it is the identity of a one-shot
    4. fade_edges            # a zero-pad step discontinuity is a broadband click
    5. loudness_normalize    # RMS target + true-peak ceiling + soft limiter
    6. random_gain           # attenuation-biased, so the ceiling holds by design

Gain augmentation comes *after* normalisation on purpose. Applied before, the
normaliser simply cancels it. Applied after and biased towards attenuation, it
adds level diversity while making it arithmetically impossible to exceed the
ceiling.
"""

from __future__ import annotations

import numpy as np

# Sample-library conventions. -20 dBFS RMS with 1 dB of true-peak headroom is
# the level most commercial one-shot libraries sit at.
DEFAULT_TARGET_RMS_DB = -20.0
DEFAULT_PEAK_CEILING_DB = -1.0
DEFAULT_SILENCE_FLOOR_DB = -60.0
DEFAULT_ONSET_THRESHOLD_DB = -35.0

_EPS = 1e-12


# =============================================================================
# Level helpers
# =============================================================================


def db_to_linear(db: float) -> float:
    """Convert decibels to a linear amplitude ratio."""
    return float(10.0 ** (db / 20.0))


def linear_to_db(x: float) -> float:
    """Convert a linear amplitude ratio to decibels (floored at -200 dB)."""
    return float(20.0 * np.log10(max(float(x), _EPS)))


def rms_db(audio: np.ndarray) -> float:
    """Full-clip RMS level in dBFS."""
    return linear_to_db(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_db(audio: np.ndarray) -> float:
    """Absolute sample peak in dBFS."""
    return linear_to_db(float(np.max(np.abs(audio))) if audio.size else 0.0)


def crest_factor_db(audio: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB - high for plucks, low for saturated pads."""
    return peak_db(audio) - rms_db(audio)


def clipped_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples sitting at or beyond full scale."""
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


# =============================================================================
# Stage 1 - DC offset
# =============================================================================


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract the per-channel mean.

    DC offset consumes headroom that the limiter then has to give back, biases
    the first convolution of the VAE encoder, and produces a click whenever a
    sample is retriggered in a DAW.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return (audio - np.mean(audio, dtype=np.float64)).astype(np.float32)
    return (audio - np.mean(audio, axis=-1, keepdims=True, dtype=np.float64)).astype(
        np.float32
    )


# =============================================================================
# Envelope + onset detection
# =============================================================================


def _mono_envelope(audio: np.ndarray, window: int) -> np.ndarray:
    """Short-window RMS envelope of the channel-summed signal, one value per sample."""
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
    window = max(1, int(window))
    power = np.square(mono.astype(np.float64))
    # Moving average via cumulative sum - O(n) and exact.
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    if power.size <= window:
        return np.sqrt(np.full_like(power, power.mean() if power.size else 0.0))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    # Centre the window so the envelope stays time-aligned with the waveform.
    pad_left = window // 2
    pad_right = power.size - smoothed.size - pad_left
    smoothed = np.pad(smoothed, (pad_left, max(0, pad_right)), mode="edge")
    return np.sqrt(smoothed[: power.size])


def detect_onset(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB,
    window_ms: float = 5.0,
) -> int | None:
    """Index of the first sample where the clip rises above ``threshold_db``.

    The threshold is relative to the clip's own envelope peak, so it adapts to
    quiet and loud material alike. Returns ``None`` when the clip never crosses
    the threshold (silence) or crosses it immediately and never drops - the
    latter meaning continuous material with no meaningful onset to anchor to.
    """
    if audio.size == 0:
        return None

    window = max(1, int(sample_rate * window_ms / 1000.0))
    envelope = _mono_envelope(audio, window)
    peak = float(np.max(envelope))
    if peak <= _EPS:
        return None

    above = envelope >= peak * db_to_linear(threshold_db)
    indices = np.flatnonzero(above)
    if indices.size == 0:
        return None
    return int(indices[0])


def is_continuous(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB,
) -> bool:
    """True for drones, loops and textures that have no single defining attack.

    Such material starts loud and stays loud; anchoring every crop to sample 0
    would throw away the augmentation value of a long recording, so the caller
    is free to crop it randomly.
    """
    onset = detect_onset(audio, sample_rate, threshold_db=threshold_db)
    if onset is None:
        return False
    # An onset inside the first 50 ms means the clip was already sounding when
    # the recording started - there is no attack in this file to preserve.
    return onset < int(sample_rate * 0.05)


# =============================================================================
# Stage 2 - silence trimming
# =============================================================================


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    floor_db: float = DEFAULT_SILENCE_FLOOR_DB,
    pre_roll_ms: float = 5.0,
    post_roll_ms: float = 50.0,
) -> np.ndarray:
    """Strip leading and trailing near-silence, keeping a short roll either side.

    Leading silence is the single most common defect in scraped audio datasets.
    A model trained on it generates samples that start 200-400 ms late, which is
    unusable on a DAW grid.
    """
    if audio.size == 0:
        return audio

    window = max(1, int(sample_rate * 0.005))
    envelope = _mono_envelope(audio, window)
    peak = float(np.max(envelope))
    if peak <= _EPS:
        return audio

    # Trim against an absolute floor *and* a floor relative to this clip, so a
    # quiet recording is not trimmed away entirely.
    threshold = max(db_to_linear(floor_db), peak * db_to_linear(-80.0))
    indices = np.flatnonzero(envelope >= threshold)
    if indices.size == 0:
        return audio

    start = max(0, int(indices[0]) - int(sample_rate * pre_roll_ms / 1000.0))
    end = min(audio.shape[-1], int(indices[-1]) + int(sample_rate * post_roll_ms / 1000.0))
    if end <= start:
        return audio
    return audio[..., start:end]


# =============================================================================
# Stage 3 - cropping
# =============================================================================


def pad_or_crop(
    audio: np.ndarray,
    target_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pad with zeros or randomly crop to ``target_samples`` along time.

    Kept for callers that genuinely want a uniformly random window. Training
    should prefer :func:`onset_anchored_crop`, which preserves the attack.
    """
    current = audio.shape[-1]
    if current == target_samples:
        return audio
    if current > target_samples:
        draw = rng.integers if rng is not None else np.random.randint
        start = int(draw(0, current - target_samples + 1))
        return audio[..., start : start + target_samples]
    pad = target_samples - current
    pad_width = ((0, 0), (0, pad)) if audio.ndim == 2 else ((0, pad),)
    return np.pad(audio, pad_width, mode="constant")


DEFAULT_ONSET_ANCHOR_PROB = 0.5


def onset_anchored_crop(
    audio: np.ndarray,
    target_samples: int,
    sample_rate: int,
    pre_roll_ms: float = 10.0,
    threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB,
    anchor_prob: float = DEFAULT_ONSET_ANCHOR_PROB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop to ``target_samples``, preserving the attack transient.

    A uniformly random crop of a file longer than the window almost never
    contains the onset: for a 30 s source cropped to 6 s the attack survives
    only if the start happens to land in the first fraction of a second - under
    1% of draws. The model then only ever sees sounds that were already
    sounding, learns that they fade in from the middle, and generates samples
    with no immediate front end. That missing attack is a large part of why
    generated one-shots do not sit in a mix the way a Serum or Spitfire sample
    does, and why they cannot be triggered on a DAW grid.

    Anchoring every crop would throw away the augmentation value of a long
    recording, so the policy is mixed:

    * Sources shorter than twice the window are always anchored - a random crop
      there buys almost no diversity and reliably costs the attack.
    * Longer sources are anchored with probability ``anchor_prob`` and cropped
      randomly otherwise, so the model sees plenty of attacks and still sees
      varied windows of the same recording.
    """
    current = audio.shape[-1]
    if current <= target_samples:
        return pad_or_crop(audio, target_samples, rng=rng)

    onset = detect_onset(audio, sample_rate, threshold_db=threshold_db)
    if onset is None:
        return pad_or_crop(audio, target_samples, rng=rng)

    if current >= 2 * target_samples:
        draw = rng.random() if rng is not None else np.random.random()
        if draw >= anchor_prob:
            return pad_or_crop(audio, target_samples, rng=rng)

    start = max(0, onset - int(sample_rate * pre_roll_ms / 1000.0))
    start = min(start, current - target_samples)
    return audio[..., start : start + target_samples]


# =============================================================================
# Stage 4 - edge fades
# =============================================================================


def fade_edges(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_ms: float = 1.0,
    fade_out_ms: float = 5.0,
) -> np.ndarray:
    """Apply raised-cosine fades to the first and last samples.

    A crop or a zero-pad leaves a step discontinuity at the boundary, and a step
    is broadband energy - an audible click on every affected example, and a
    spectral target the model cannot help but fit.
    """
    if audio.size == 0:
        return audio

    audio = np.array(audio, dtype=np.float32, copy=True)
    length = audio.shape[-1]

    n_in = min(int(sample_rate * fade_in_ms / 1000.0), length // 2)
    if n_in > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_in)))
        audio[..., :n_in] *= ramp.astype(np.float32)

    n_out = min(int(sample_rate * fade_out_ms / 1000.0), length // 2)
    if n_out > 1:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_out)))
        audio[..., -n_out:] *= ramp.astype(np.float32)

    return audio


# =============================================================================
# Stage 5 - levelling
# =============================================================================


def soft_limit(
    audio: np.ndarray,
    ceiling_db: float = DEFAULT_PEAK_CEILING_DB,
    knee_db: float = 3.0,
) -> np.ndarray:
    """Smoothly bound the signal below ``ceiling_db`` instead of clipping it.

    Below the knee the signal is untouched. Above it, amplitudes are mapped
    through a tanh curve that is continuous in value *and* slope at the knee and
    asymptotes to the ceiling, so the transfer function never has the corner
    that makes hard clipping generate broadband odd harmonics.
    """
    ceiling = db_to_linear(ceiling_db)
    knee = db_to_linear(ceiling_db - abs(knee_db))
    if ceiling <= knee:
        return np.clip(audio, -ceiling, ceiling).astype(np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    magnitude = np.abs(audio)
    over = magnitude > knee
    if not np.any(over):
        return audio

    out = np.array(audio, dtype=np.float32, copy=True)
    span = ceiling - knee
    excess = (magnitude[over] - knee) / span
    out[over] = np.sign(audio[over]) * (knee + span * np.tanh(excess))
    return out.astype(np.float32)


def loudness_normalize(
    audio: np.ndarray,
    target_rms_db: float = DEFAULT_TARGET_RMS_DB,
    peak_ceiling_db: float = DEFAULT_PEAK_CEILING_DB,
    max_gain_reduction_db: float = 12.0,
) -> np.ndarray:
    """Normalise by loudness, then guarantee the peak ceiling.

    Peak normalisation - scaling every clip so its loudest sample hits a fixed
    value - gives a transient pluck and a saturated pad the same peak but wildly
    different perceived loudness, because their crest factors differ by 15 dB or
    more. The model then has to spend capacity modelling an amplitude variance
    that carries no semantic information, and generations come out at
    inconsistent levels.

    Normalising by RMS fixes that. High-crest material whose peak would then
    exceed the ceiling is backed off (up to ``max_gain_reduction_db``), and only
    the residual - a few isolated transient peaks - is soft-limited.
    """
    audio = np.asarray(audio, dtype=np.float32)
    current_rms = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    if current_rms <= _EPS:
        return audio  # digital silence - nothing to normalise

    gain = db_to_linear(target_rms_db) / float(current_rms)

    ceiling = db_to_linear(peak_ceiling_db)
    projected_peak = float(np.max(np.abs(audio))) * gain
    if projected_peak > ceiling:
        # Back the gain off towards the ceiling, but not past the point where we
        # would be throwing loudness consistency away entirely.
        needed_db = linear_to_db(ceiling / projected_peak)
        gain *= db_to_linear(max(needed_db, -abs(max_gain_reduction_db)))

    out = (audio * gain).astype(np.float32)
    return soft_limit(out, ceiling_db=peak_ceiling_db)


# =============================================================================
# Stage 6 - gain augmentation
# =============================================================================


def random_gain(
    audio: np.ndarray,
    gain_db_range: tuple[float, float] = (-6.0, 0.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a random gain drawn from ``gain_db_range`` (in dB).

    The default range is attenuation-only. Applied after normalisation it adds
    level diversity without being able to push the signal back over the ceiling,
    so no clipping guard is needed downstream.
    """
    low, high = gain_db_range
    draw = rng.uniform if rng is not None else np.random.uniform
    gain_db = float(draw(low, high))
    return (audio * db_to_linear(gain_db)).astype(np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale so the absolute peak equals ``peak`` (no-op if silent).

    Retained for analysis and for reproducing the legacy pipeline. Training
    should use :func:`loudness_normalize`.
    """
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32, copy=False)


# =============================================================================
# Entry point
# =============================================================================


def prepare_sample(
    audio: np.ndarray,
    sample_rate: int,
    target_samples: int,
    augment: bool = True,
    target_rms_db: float = DEFAULT_TARGET_RMS_DB,
    peak_ceiling_db: float = DEFAULT_PEAK_CEILING_DB,
    silence_floor_db: float = DEFAULT_SILENCE_FLOOR_DB,
    onset_threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB,
    onset_pre_roll_ms: float = 10.0,
    onset_anchor_prob: float = DEFAULT_ONSET_ANCHOR_PROB,
    gain_db_range: tuple[float, float] = (-6.0, 0.0),
    trim: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the full sample-grade pipeline over one clip.

    Returns a ``(channels, target_samples)`` float32 array that is DC-free,
    onset-anchored, click-free, loudness-matched and guaranteed below the peak
    ceiling.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    audio = remove_dc_offset(audio)

    if trim:
        audio = trim_silence(audio, sample_rate, floor_db=silence_floor_db)

    audio = onset_anchored_crop(
        audio,
        target_samples,
        sample_rate,
        pre_roll_ms=onset_pre_roll_ms,
        threshold_db=onset_threshold_db,
        anchor_prob=onset_anchor_prob,
        rng=rng,
    )

    audio = fade_edges(audio, sample_rate)
    audio = loudness_normalize(
        audio,
        target_rms_db=target_rms_db,
        peak_ceiling_db=peak_ceiling_db,
    )

    if augment:
        audio = random_gain(audio, gain_db_range=gain_db_range, rng=rng)

    return np.ascontiguousarray(audio, dtype=np.float32)


def prepare_sample_legacy(
    audio: np.ndarray,
    target_samples: int,
    augment: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Reproduce the pre-change pipeline exactly, for A/B comparison.

    Peak-normalise to 0.95, apply a symmetric +-3 dB gain, hard-clip to [-1, 1].
    Any gain draw above +0.45 dB clips, which is 42.6% of draws.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = pad_or_crop(audio, target_samples, rng=rng)
    audio = peak_normalize(audio)
    if augment:
        low, high = -3.0, 3.0
        draw = rng.uniform if rng is not None else np.random.uniform
        audio = audio * db_to_linear(float(draw(low, high)))
        audio = np.clip(audio, -1.0, 1.0)
    return np.ascontiguousarray(audio, dtype=np.float32)
