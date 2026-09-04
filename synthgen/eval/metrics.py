"""
Audio-quality metrics for SynthGen, chosen for *synthesiser and sampler*
output rather than for general music generation.

The usual text-to-audio yardsticks (FAD, CLAP score, KL over an audio
classifier) answer "does this sound broadly like the prompt". They are
blind to the things that decide whether a sound is usable in a session:
whether the oscillator is clean, whether the attack survived, whether the
top octave is there, whether the stereo image held. Those are what this
module measures.

Every function takes plain ``numpy`` arrays so it can be pointed at any
WAV on disk, not only at model output. Mono is ``(n,)``; stereo is
``(2, n)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "to_mono",
    "match_length",
    "harmonic_analysis",
    "alias_to_signal_ratio_db",
    "spurious_free_dynamic_range_db",
    "sub_fundamental_alias_db",
    "blackman_harris",
    "thd_n_percent",
    "band_energy_db",
    "high_frequency_retention_db",
    "spectral_centroid_hz",
    "attack_time_ms",
    "transient_error_ms",
    "stereo_correlation",
    "stereo_width_error",
    "multires_stft_distance",
    "si_sdr_db",
    "noise_floor_db",
    "HarmonicReport",
]

EPS = 1e-12


# =============================================================================
# Helpers
# =============================================================================


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse ``(c, n)`` to ``(n,)``; pass ``(n,)`` through."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=0)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")


def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both signals to their common length along the last axis."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def blackman_harris(n: int) -> np.ndarray:
    """
    4-term Blackman-Harris window (~-92 dB peak sidelobe).

    ``numpy.blackman`` is the 3-term variant and only reaches ~-58 dB,
    which puts a hard floor under every alias measurement here. Aliasing
    that matters lives well below that, so the extra term is not optional.
    """
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    k = np.arange(n)
    return (
        a[0]
        - a[1] * np.cos(2 * np.pi * k / (n - 1))
        + a[2] * np.cos(4 * np.pi * k / (n - 1))
        - a[3] * np.cos(6 * np.pi * k / (n - 1))
    )


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Blackman-Harris-windowed magnitude spectrum.

    Blackman-Harris has ~-92 dB sidelobes, so leakage from the fundamental
    cannot masquerade as an alias spur even at the low levels we care about.
    """
    x = to_mono(x)
    spec = np.fft.rfft(x * blackman_harris(len(x)))
    return np.abs(spec), np.fft.rfftfreq(len(x), 1.0)


# =============================================================================
# Aliasing / oscillator purity
# =============================================================================


@dataclass
class HarmonicReport:
    """Result of splitting a spectrum into harmonic and inharmonic parts."""

    harmonic_power: float
    inharmonic_power: float
    fundamental_db: float
    worst_spur_db: float
    worst_spur_hz: float
    alias_to_signal_db: float
    sfdr_db: float
    sub_fundamental_db: float


def harmonic_analysis(
    x: np.ndarray,
    f0: float,
    sample_rate: int = 44100,
    tolerance_bins: int = 8,
    hp_cutoff_hz: float = 20.0,
) -> HarmonicReport:
    """
    Split a spectrum into energy that belongs to the note and energy that
    does not.

    Given a signal built from a single fundamental ``f0``, every partial a
    well-behaved synthesiser may produce sits at an integer multiple of
    ``f0``. Anything else is either aliasing (a harmonic that folded back
    off Nyquist) or added noise. Both are defects, and both are audible in
    the same way: as content that will not fuse with the note.

    Args:
        x: Signal under test, mono or stereo.
        f0: Fundamental of the stimulus in Hz.
        sample_rate: Sample rate in Hz.
        tolerance_bins: Half-width, in FFT bins, of the band claimed by
            each harmonic. Covers the window's main lobe.
        hp_cutoff_hz: Ignore everything below this (DC drift, window
            skirt around DC) so it cannot inflate the inharmonic total.

    Returns:
        A :class:`HarmonicReport`.
    """
    mag, freqs_norm = _spectrum(x)
    freqs = freqs_norm * sample_rate
    power = mag**2

    bin_hz = sample_rate / (2 * (len(mag) - 1)) if len(mag) > 1 else sample_rate
    nyquist = sample_rate / 2

    harmonic_mask = np.zeros(len(mag), dtype=bool)
    k = 1
    while k * f0 < nyquist:
        centre = int(round(k * f0 / bin_hz))
        lo = max(0, centre - tolerance_bins)
        hi = min(len(mag), centre + tolerance_bins + 1)
        harmonic_mask[lo:hi] = True
        k += 1

    valid = freqs >= hp_cutoff_hz
    inharmonic_mask = valid & ~harmonic_mask

    harmonic_power = float(power[valid & harmonic_mask].sum())
    inharmonic_power = float(power[inharmonic_mask].sum())

    # Fundamental level
    fund_centre = int(round(f0 / bin_hz))
    fund_lo = max(0, fund_centre - tolerance_bins)
    fund_hi = min(len(mag), fund_centre + tolerance_bins + 1)
    fundamental = float(mag[fund_lo:fund_hi].max()) if fund_hi > fund_lo else EPS

    if inharmonic_mask.any():
        spur_idx = int(np.argmax(np.where(inharmonic_mask, mag, 0.0)))
        worst_spur = float(mag[spur_idx])
        worst_spur_hz = float(freqs[spur_idx])
    else:
        worst_spur, worst_spur_hz = EPS, 0.0

    # Energy below the fundamental. No harmonic of f0 can live here, and
    # nothing above can mask it, so this band is pure defect and is the
    # part a listener notices first as "grit" under the note.
    sub_mask = inharmonic_mask & (freqs < f0)
    sub_power = float(power[sub_mask].sum())

    return HarmonicReport(
        harmonic_power=harmonic_power,
        inharmonic_power=inharmonic_power,
        fundamental_db=20 * np.log10(fundamental + EPS),
        worst_spur_db=20 * np.log10(worst_spur + EPS),
        worst_spur_hz=worst_spur_hz,
        alias_to_signal_db=10
        * np.log10((inharmonic_power + EPS) / (harmonic_power + EPS)),
        sfdr_db=20 * np.log10((fundamental + EPS) / (worst_spur + EPS)),
        sub_fundamental_db=10
        * np.log10((sub_power + EPS) / (harmonic_power + EPS)),
    )


def sub_fundamental_alias_db(
    x: np.ndarray, f0: float, sample_rate: int = 44100
) -> float:
    """
    Inharmonic energy *below the fundamental*, relative to harmonic energy.

    The most perceptually damning alias measure. Folded partials that land
    under the note cannot be masked by anything - there is no louder
    harmonic beneath them - so they are heard directly as grit, buzz or a
    metallic shimmer that does not track the pitch.
    """
    return harmonic_analysis(x, f0, sample_rate).sub_fundamental_db


def alias_to_signal_ratio_db(
    x: np.ndarray, f0: float, sample_rate: int = 44100
) -> float:
    """
    Total inharmonic energy relative to total harmonic energy, in dB.

    **The headline metric.** Lower is better. ``-60 dB`` means the folded
    junk sits 60 dB under the note and is inaudible in a mix; ``-20 dB``
    means it is a clearly audible metallic layer.
    """
    return harmonic_analysis(x, f0, sample_rate).alias_to_signal_db


def spurious_free_dynamic_range_db(
    x: np.ndarray, f0: float, sample_rate: int = 44100
) -> float:
    """
    Level of the fundamental above the single worst inharmonic spur, in dB.

    The classic converter/oscillator spec. Higher is better. Complements
    the alias-to-signal ratio: ASR catches a broad haze of many small
    spurs, SFDR catches one loud whistle.
    """
    return harmonic_analysis(x, f0, sample_rate).sfdr_db


def thd_n_percent(x: np.ndarray, f0: float, sample_rate: int = 44100) -> float:
    """
    Total harmonic distortion plus noise, as a percentage.

    Everything that is not the fundamental, over the total. Reported for
    continuity with hardware measurement practice; for a *synthesiser*
    the alias ratio matters more, because harmonics are wanted here and
    THD+N counts them as error.
    """
    mag, freqs_norm = _spectrum(x)
    freqs = freqs_norm * sample_rate
    power = mag**2
    bin_hz = sample_rate / (2 * (len(mag) - 1)) if len(mag) > 1 else sample_rate

    centre = int(round(f0 / bin_hz))
    lo, hi = max(0, centre - 4), min(len(mag), centre + 5)
    fundamental_power = float(power[lo:hi].sum())
    total_power = float(power[freqs >= 20.0].sum())
    rest = max(total_power - fundamental_power, 0.0)
    return 100.0 * float(np.sqrt(rest / (total_power + EPS)))


# =============================================================================
# Spectral balance / "air"
# =============================================================================


def band_energy_db(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate: int = 44100,
) -> float:
    """Energy inside a frequency band, in dB."""
    mag, freqs_norm = _spectrum(x)
    freqs = freqs_norm * sample_rate
    band = (freqs >= low_hz) & (freqs < high_hz)
    return 10 * np.log10(float((mag[band] ** 2).sum()) + EPS)


def high_frequency_retention_db(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
    low_hz: float = 10000.0,
    high_hz: float = 20000.0,
) -> float:
    """
    Energy in the "air" band of ``pred`` relative to ``target``, in dB.

    ``0`` is perfect. Negative means the top octave was lost - the single
    most common reason a neural codec output sounds dull and "small"
    against a Spitfire or Splice reference. Positive means energy was
    *added* up there, which for a codec means noise or aliasing, not
    detail.
    """
    pred, target = match_length(to_mono(pred), to_mono(target))
    return band_energy_db(pred, low_hz, high_hz, sample_rate) - band_energy_db(
        target, low_hz, high_hz, sample_rate
    )


def spectral_centroid_hz(x: np.ndarray, sample_rate: int = 44100) -> float:
    """Brightness proxy: the energy-weighted mean frequency."""
    mag, freqs_norm = _spectrum(x)
    freqs = freqs_norm * sample_rate
    return float((freqs * mag).sum() / (mag.sum() + EPS))


# =============================================================================
# Transients
# =============================================================================


def _envelope(x: np.ndarray, sample_rate: int, window_ms: float = 1.0) -> np.ndarray:
    x = np.abs(to_mono(x))
    win = max(1, int(round(window_ms * sample_rate / 1000.0)))
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def attack_time_ms(
    x: np.ndarray,
    sample_rate: int = 44100,
    low_pct: float = 0.10,
    high_pct: float = 0.90,
) -> float:
    """
    Time for the envelope to rise from ``low_pct`` to ``high_pct`` of its
    peak, in milliseconds.

    A plucked or struck sample lives or dies on this number. Smear a 2 ms
    piano attack out to 8 ms and it stops reading as a piano.
    """
    env = _envelope(x, sample_rate)
    peak = float(env.max())
    if peak <= EPS:
        return 0.0
    peak_idx = int(np.argmax(env))
    lo_target, hi_target = low_pct * peak, high_pct * peak

    lo_idx = 0
    for i in range(peak_idx, -1, -1):
        if env[i] <= lo_target:
            lo_idx = i
            break
    hi_idx = peak_idx
    for i in range(lo_idx, peak_idx + 1):
        if env[i] >= hi_target:
            hi_idx = i
            break
    return 1000.0 * (hi_idx - lo_idx) / sample_rate


def transient_error_ms(
    pred: np.ndarray, target: np.ndarray, sample_rate: int = 44100
) -> float:
    """
    Attack-time difference between ``pred`` and ``target``, in ms.

    Positive means the model smeared the attack; negative means it made it
    unnaturally abrupt. ``0`` is perfect.
    """
    return attack_time_ms(pred, sample_rate) - attack_time_ms(target, sample_rate)


# =============================================================================
# Stereo
# =============================================================================


def stereo_correlation(x: np.ndarray) -> float:
    """
    Inter-channel correlation in ``[-1, 1]``.

    ``1`` is mono, ``0`` is fully decorrelated, negative means the image
    will partly cancel when a listener folds to mono. Wide-but-mono-safe
    pads live around ``0.3-0.7``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 2:
        return 1.0
    left, right = x[0] - x[0].mean(), x[1] - x[1].mean()
    denom = np.sqrt((left**2).sum() * (right**2).sum())
    if denom <= EPS:
        return 1.0
    return float((left * right).sum() / denom)


def stereo_width_error(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Difference in inter-channel correlation between ``pred`` and ``target``.

    Codecs that were trained on a mono-heavy diet tend to pull the image
    inwards; this catches that collapse. ``0`` is perfect.
    """
    return stereo_correlation(pred) - stereo_correlation(target)


# =============================================================================
# Reconstruction fidelity
# =============================================================================


def multires_stft_distance(
    pred: np.ndarray,
    target: np.ndarray,
    fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256),
) -> float:
    """
    Mean spectral-convergence distance over several FFT resolutions.

    Scale-aware and phase-blind, so it tracks perceived timbre difference
    far better than a waveform L1. Lower is better.
    """
    pred, target = match_length(to_mono(pred), to_mono(target))
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        if len(pred) < n_fft:
            continue
        window = np.hanning(n_fft)
        frames = 1 + (len(pred) - n_fft) // hop
        p = np.stack(
            [
                np.abs(np.fft.rfft(pred[i * hop : i * hop + n_fft] * window))
                for i in range(frames)
            ]
        )
        t = np.stack(
            [
                np.abs(np.fft.rfft(target[i * hop : i * hop + n_fft] * window))
                for i in range(frames)
            ]
        )
        total += float(np.linalg.norm(t - p) / (np.linalg.norm(t) + EPS))
    return total / max(len(fft_sizes), 1)


def si_sdr_db(pred: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio, in dB. Higher is better."""
    pred, target = match_length(to_mono(pred), to_mono(target))
    pred = pred - pred.mean()
    target = target - target.mean()
    alpha = (pred @ target) / ((target @ target) + EPS)
    projection = alpha * target
    noise = pred - projection
    return 10 * np.log10(
        (float(projection @ projection) + EPS) / (float(noise @ noise) + EPS)
    )


def noise_floor_db(
    x: np.ndarray, sample_rate: int = 44100, percentile: float = 10.0
) -> float:
    """
    Level of the quietest frames relative to the peak, in dB.

    A sampler instrument with a -50 dB floor hisses audibly under a
    sustained pad. Commercial libraries sit below -80 dB.
    """
    mono = to_mono(x)
    frame = max(1, int(0.010 * sample_rate))
    n_frames = len(mono) // frame
    if n_frames < 2:
        return 0.0
    rms = np.array(
        [
            np.sqrt(np.mean(mono[i * frame : (i + 1) * frame] ** 2))
            for i in range(n_frames)
        ]
    )
    peak = float(rms.max())
    if peak <= EPS:
        return 0.0
    return 20 * np.log10((float(np.percentile(rms, percentile)) + EPS) / peak)
