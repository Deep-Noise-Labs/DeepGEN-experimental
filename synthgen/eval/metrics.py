"""
Sample-quality metrics for SynthGen / DeepGEN.

These metrics encode the *job to be done*: a generated clip is only useful if a
producer can drop it straight into a session next to a Spitfire, Serum or Splice
sample without reaching for an EQ, a limiter or a gain stage first.

Two families are provided:

``absolute_metrics``
    Reference-free. Grade a single clip on its own merits. These are the ones
    that decide whether output is *sellable*, and they can be run against
    production audio where no ground truth exists.

``comparative_metrics``
    Reference-based. Compare a candidate against a target, used for
    autoencoder-reconstruction and regression testing.

Everything here is NumPy/SciPy only so evaluation never needs a GPU or torch.

Conventions
-----------
Audio is ``(channels, samples)`` float in [-1, 1]. Sample rate is explicit.
All dB values are decibels; ``dBFS`` is relative to full scale (1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

EPS = 1e-12

# Frequency bands that matter for judging a synthesizer sample. The band edges
# are the ones sound designers actually talk in: sub, bass, low-mid, mid,
# presence, air.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 1000.0),
    "mid": (1000.0, 4000.0),
    "presence": (4000.0, 10000.0),
    "air": (10000.0, 20000.0),
}


# =============================================================================
# Helpers
# =============================================================================


def as_2d(audio: np.ndarray) -> np.ndarray:
    """Coerce audio to ``(channels, samples)`` float64."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Sum to mono by averaging channels."""
    return as_2d(audio).mean(axis=0)


def db(x: float | np.ndarray) -> np.ndarray:
    """Amplitude ratio to decibels."""
    return 20.0 * np.log10(np.abs(x) + EPS)


def power_db(x: float | np.ndarray) -> np.ndarray:
    """Power ratio to decibels."""
    return 10.0 * np.log10(np.abs(x) + EPS)


def _welch_psd(x: np.ndarray, sr: int, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a mono signal."""
    nperseg = int(min(nperseg, len(x)))
    freqs, psd = signal.welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def band_energy(x: np.ndarray, sr: int, low: float, high: float) -> float:
    """Total power in ``[low, high)`` Hz, via Welch PSD integration."""
    freqs, psd = _welch_psd(x, sr)
    mask = (freqs >= low) & (freqs < min(high, sr / 2.0))
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


# =============================================================================
# Absolute (reference-free) metrics
# =============================================================================


def bandwidth_hz(x: np.ndarray, sr: int, floor_db: float = -60.0) -> float:
    """
    Highest frequency still carrying real content.

    Defined as the top edge of the highest PSD bin within ``floor_db`` of the
    spectral peak. A 32 kHz model can never exceed 16000; a sample library
    reaches ~20000.
    """
    freqs, psd = _welch_psd(x, sr)
    if psd.max() <= 0:
        return 0.0
    rel = power_db(psd / psd.max())
    above = np.where(rel > floor_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def band_balance_db(x: np.ndarray, sr: int) -> dict[str, float]:
    """Energy per named band, in dB relative to the clip's total energy."""
    total = band_energy(x, sr, 0.0, sr / 2.0)
    out: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        out[name] = float(power_db(band_energy(x, sr, low, high) / (total + EPS)))
    return out


def true_peak_dbfs(audio: np.ndarray, oversample: int = 4) -> float:
    """
    Inter-sample (true) peak in dBFS, via polyphase oversampling.

    A clip that measures 0.0 dBFS sample-peak can still overshoot after
    conversion; anything above -1.0 dBTP will distort on a consumer DAC.
    """
    a = as_2d(audio)
    up = signal.resample_poly(a, oversample, 1, axis=-1)
    return float(db(np.max(np.abs(up))))


def clip_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples pinned at (or beyond) full scale."""
    a = as_2d(audio)
    return float(np.mean(np.abs(a) >= threshold))


def rms_dbfs(audio: np.ndarray) -> float:
    """Whole-clip RMS level in dBFS."""
    a = as_2d(audio)
    return float(db(np.sqrt(np.mean(a**2))))


def crest_factor_db(audio: np.ndarray) -> float:
    """Peak-to-RMS ratio. Low values mean squashed, lifeless dynamics."""
    a = as_2d(audio)
    return float(db(np.max(np.abs(a))) - db(np.sqrt(np.mean(a**2))))


def noise_floor_dbfs(audio: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """
    Noise floor, taken as the 10th percentile of short-frame RMS.

    Sample libraries sit below -70 dBFS. Anything above -55 dBFS is audible
    hiss between notes.
    """
    x = to_mono(audio)
    n = max(1, int(sr * frame_ms / 1000.0))
    frames = len(x) // n
    if frames < 4:
        return rms_dbfs(audio)
    rms = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1))
    return float(db(np.percentile(rms, 10)))


def dc_offset(audio: np.ndarray) -> float:
    """Largest per-channel DC offset. Non-zero DC wastes headroom."""
    a = as_2d(audio)
    return float(np.max(np.abs(a.mean(axis=-1))))


def mono_compatibility_db(audio: np.ndarray) -> float:
    """
    Level lost when the clip is summed to mono.

    0 dB means the mono sum is as loud as the stereo original. Large negative
    values mean the stereo image is built from decorrelated (or anti-phase)
    channels that partially cancel on a mono club system or phone speaker.
    Returns 0.0 for mono input.
    """
    a = as_2d(audio)
    if a.shape[0] < 2:
        return 0.0
    stereo_rms = np.sqrt(np.mean(a**2))
    mono_rms = np.sqrt(np.mean(a.mean(axis=0) ** 2))
    return float(db(mono_rms) - db(stereo_rms))


def stereo_width(audio: np.ndarray) -> float:
    """
    Side-to-mid energy ratio in dB.

    Around -12 dB is a natural, usable width; above 0 dB the side signal
    dominates and the patch has no stable centre.
    """
    a = as_2d(audio)
    if a.shape[0] < 2:
        return float("-inf")
    mid = (a[0] + a[1]) / 2.0
    side = (a[0] - a[1]) / 2.0
    return float(power_db(np.mean(side**2) / (np.mean(mid**2) + EPS)))


def attack_time_ms(audio: np.ndarray, sr: int) -> float:
    """
    Time from onset (-40 dB of peak) to the envelope peak, in milliseconds.

    A plucked or percussive patch should land under ~30 ms; a smeared decoder
    stretches this out and the sample loses its bite.
    """
    x = np.abs(to_mono(audio))
    win = max(1, int(sr * 0.002))
    env = np.convolve(x, np.ones(win) / win, mode="same")
    if env.max() <= 0:
        return 0.0
    peak_idx = int(np.argmax(env))
    threshold = env.max() * (10 ** (-40.0 / 20.0))
    above = np.where(env[: peak_idx + 1] >= threshold)[0]
    onset = int(above[0]) if len(above) else 0
    return float((peak_idx - onset) / sr * 1000.0)


def loop_discontinuity_db(audio: np.ndarray, sr: int, window_ms: float = 5.0) -> float:
    """
    Level of the step between the clip's end and its start, relative to RMS.

    Measures whether the sample can be looped without an audible click.
    Lower (more negative) is better.
    """
    a = as_2d(audio)
    n = max(1, int(sr * window_ms / 1000.0))
    step = np.abs(a[:, 0] - a[:, -1]).max()
    edge_rms = np.sqrt(np.mean(np.concatenate([a[:, :n], a[:, -n:]], axis=-1) ** 2))
    return float(db(step) - db(edge_rms + EPS))


def harmonic_to_noise_db(audio: np.ndarray, sr: int) -> float:
    """
    Harmonic-to-noise ratio via cepstral-free spectral peak picking.

    Compares energy concentrated in spectral peaks against the surrounding
    spectral floor. High values mean clean, stable partials; low values mean
    the tone is smeared into noise -- the classic latent-decoder artefact.
    """
    x = to_mono(audio)
    freqs, psd = _welch_psd(x, sr)
    if psd.max() <= 0:
        return 0.0
    # Median-filtered PSD approximates the noise floor under the partials.
    floor = signal.medfilt(psd, kernel_size=31)
    peaks = np.maximum(psd - floor, 0.0)
    return float(power_db(peaks.sum() / (floor.sum() + EPS)))


def absolute_metrics(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Full reference-free quality profile for one clip."""
    a = as_2d(audio)
    mono = to_mono(a)
    out: dict[str, float] = {
        "sample_rate": float(sr),
        "channels": float(a.shape[0]),
        "duration_s": float(a.shape[-1] / sr),
        "bandwidth_hz": bandwidth_hz(mono, sr),
        "true_peak_dbfs": true_peak_dbfs(a),
        "clip_ratio": clip_ratio(a),
        "rms_dbfs": rms_dbfs(a),
        "crest_factor_db": crest_factor_db(a),
        "noise_floor_dbfs": noise_floor_dbfs(a, sr),
        "dc_offset": dc_offset(a),
        "mono_compat_db": mono_compatibility_db(a),
        "stereo_width_db": stereo_width(a),
        "attack_time_ms": attack_time_ms(a, sr),
        "loop_discontinuity_db": loop_discontinuity_db(a, sr),
        "hnr_db": harmonic_to_noise_db(a, sr),
    }
    for name, value in band_balance_db(mono, sr).items():
        out[f"band_{name}_db"] = value
    return out


# =============================================================================
# Comparative (reference-based) metrics
# =============================================================================


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = as_2d(a), as_2d(b)
    n = min(a.shape[-1], b.shape[-1])
    c = min(a.shape[0], b.shape[0])
    return a[:c, :n], b[:c, :n]


def si_sdr_db(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio, in dB."""
    est, ref = _align(estimate, target)
    est, ref = est.reshape(-1), ref.reshape(-1)
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    proj = alpha * ref
    return float(power_db(np.sum(proj**2) / (np.sum((est - proj) ** 2) + EPS)))


def log_spectral_distance_db(
    estimate: np.ndarray,
    target: np.ndarray,
    sr: int,
    n_fft: int = 2048,
) -> float:
    """Mean absolute log-magnitude STFT distance, in dB."""
    est, ref = _align(estimate, target)
    total, count = 0.0, 0
    for ch in range(est.shape[0]):
        _, _, e = signal.stft(est[ch], fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        _, _, r = signal.stft(ref[ch], fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        total += float(np.mean(np.abs(db(np.abs(e)) - db(np.abs(r)))))
        count += 1
    return total / max(count, 1)


def band_error_db(estimate: np.ndarray, target: np.ndarray, sr: int) -> dict[str, float]:
    """Per-band energy error in dB (estimate minus target). 0 is perfect."""
    est, ref = _align(estimate, target)
    e_mono, r_mono = to_mono(est), to_mono(ref)
    out: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        if low >= sr / 2.0:
            out[name] = float("nan")
            continue
        e = band_energy(e_mono, sr, low, high)
        r = band_energy(r_mono, sr, low, high)
        out[name] = float(power_db(e + EPS) - power_db(r + EPS))
    return out


def envelope_correlation(estimate: np.ndarray, target: np.ndarray, sr: int) -> float:
    """
    Pearson correlation of the two amplitude envelopes.

    Catches transient smearing that spectral distance alone will miss.
    """
    est, ref = _align(estimate, target)
    win = max(1, int(sr * 0.005))
    kernel = np.ones(win) / win
    e = np.convolve(np.abs(to_mono(est)), kernel, mode="same")
    r = np.convolve(np.abs(to_mono(ref)), kernel, mode="same")
    if e.std() < EPS or r.std() < EPS:
        return 0.0
    return float(np.corrcoef(e, r)[0, 1])


def stereo_image_error(estimate: np.ndarray, target: np.ndarray) -> float:
    """Absolute difference in side/mid width, in dB. 0 means image preserved."""
    est, ref = _align(estimate, target)
    if est.shape[0] < 2:
        return 0.0
    return float(abs(stereo_width(est) - stereo_width(ref)))


def comparative_metrics(estimate: np.ndarray, target: np.ndarray, sr: int) -> dict[str, float]:
    """Full reference-based comparison of a candidate against a target."""
    out: dict[str, float] = {
        "si_sdr_db": si_sdr_db(estimate, target),
        "lsd_db": log_spectral_distance_db(estimate, target, sr),
        "envelope_corr": envelope_correlation(estimate, target, sr),
        "stereo_image_error_db": stereo_image_error(estimate, target),
    }
    for name, value in band_error_db(estimate, target, sr).items():
        out[f"band_err_{name}_db"] = value
    return out


# =============================================================================
# Grading against a production-quality target spec
# =============================================================================


@dataclass(frozen=True)
class QualityTarget:
    """
    The bar a clip must clear to count as commercially usable.

    Defaults are set from what commercial sample libraries ship: full-band
    44.1 kHz content, consistent level with real headroom, no clipping, a
    stereo image that survives a mono fold-down, and a quiet noise floor.
    """

    min_sample_rate: float = 44100.0
    # Above 16 kHz on purpose: 16 kHz is exactly Nyquist for a 32 kHz model, so
    # a floor at 16 kHz would be cleared by output that has no air band at all.
    min_bandwidth_hz: float = 18000.0
    max_true_peak_dbfs: float = -1.0
    max_clip_ratio: float = 0.0
    rms_dbfs_range: tuple[float, float] = (-24.0, -12.0)
    min_crest_factor_db: float = 6.0
    max_noise_floor_dbfs: float = -60.0
    max_dc_offset: float = 0.001
    min_mono_compat_db: float = -3.0
    max_loop_discontinuity_db: float = -20.0
    min_hnr_db: float = 3.0


def grade(metrics: dict[str, float], target: QualityTarget | None = None) -> dict[str, bool]:
    """Check one clip's metrics against the target spec, criterion by criterion."""
    t = target or QualityTarget()
    lo, hi = t.rms_dbfs_range
    return {
        "sample_rate": metrics["sample_rate"] >= t.min_sample_rate,
        "bandwidth": metrics["bandwidth_hz"] >= t.min_bandwidth_hz,
        "true_peak": metrics["true_peak_dbfs"] <= t.max_true_peak_dbfs,
        "no_clipping": metrics["clip_ratio"] <= t.max_clip_ratio,
        "level_consistency": lo <= metrics["rms_dbfs"] <= hi,
        "dynamics": metrics["crest_factor_db"] >= t.min_crest_factor_db,
        "noise_floor": metrics["noise_floor_dbfs"] <= t.max_noise_floor_dbfs,
        "dc_offset": metrics["dc_offset"] <= t.max_dc_offset,
        "mono_compatible": metrics["mono_compat_db"] >= t.min_mono_compat_db,
        "loopable": metrics["loop_discontinuity_db"] <= t.max_loop_discontinuity_db,
        "harmonic_clarity": metrics["hnr_db"] >= t.min_hnr_db,
    }


def pass_rate(grades: dict[str, bool]) -> float:
    """Fraction of criteria a clip passes."""
    return float(sum(grades.values()) / max(len(grades), 1))
