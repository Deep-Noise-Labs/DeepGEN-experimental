"""
Objective metrics for judging synthesiser-grade audio quality.

These are deliberately *signal* metrics rather than embedding metrics such as
FAD or CLAP. Embedding metrics answer "does this sound roughly like the target
distribution"; they are close to blind to the defects that separate a usable
sample library from an unusable one. A patch with -20 dB of aliasing and a
patch with -70 dB can land in the same FAD bucket while only one of them is
sellable. The metrics here target the specific failure modes that a Spitfire /
Serum / Splice-grade sound must not have.

All functions take mono ``float`` arrays unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

# Default half-width, in Hz, of the window treated as "on" a harmonic peak.
# Wide enough to absorb FFT leakage from a Hann window at typical analysis
# lengths, narrow enough not to swallow nearby alias products.
_PEAK_HALF_WIDTH_HZ = 8.0


def _spectrum(x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, power spectrum) of a Hann-windowed signal."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    return freqs, spec


def _harmonic_mask(
    freqs: np.ndarray,
    f0: float,
    sample_rate: int,
    half_width_hz: float = _PEAK_HALF_WIDTH_HZ,
) -> np.ndarray:
    """Boolean mask selecting bins lying on an integer multiple of ``f0``."""
    mask = np.zeros_like(freqs, dtype=bool)
    nyquist = sample_rate / 2.0
    order = 1
    while order * f0 < nyquist:
        mask |= np.abs(freqs - order * f0) < half_width_hz
        order += 1
    return mask


def alias_to_signal_ratio(
    x: np.ndarray,
    f0: float,
    sample_rate: int,
    half_width_hz: float = _PEAK_HALF_WIDTH_HZ,
) -> float:
    """
    Alias-to-Signal Ratio (ASR) in dB. **The headline metric.**

    Drive the system with a single sine at ``f0``. Any memoryless non-linearity
    can only legitimately produce energy at integer multiples of ``f0``. Every
    other spectral bin is an alias product -- a partial folded back from above
    Nyquist, landing at a frequency that bears no harmonic relation to the note.

    That inharmonicity is why aliasing is so damaging for instruments: it is
    perceived as metallic roughness or grit that moves in the *wrong direction*
    as you play up the keyboard, which is precisely the artefact absent from
    professional libraries.

    Returns:
        ``10*log10(alias_power / harmonic_power)``. Lower is better.
        Analogue-modelling synthesisers target roughly -60 dB or below.
    """
    freqs, spec = _spectrum(x, sample_rate)
    harmonic = _harmonic_mask(freqs, f0, sample_rate, half_width_hz)
    signal_power = spec[harmonic].sum()
    alias_power = spec[~harmonic].sum()
    return float(10.0 * np.log10(alias_power / (signal_power + 1e-30) + 1e-30))


def total_harmonic_distortion(
    x: np.ndarray,
    f0: float,
    sample_rate: int,
    max_order: int = 20,
) -> float:
    """
    Total harmonic distortion in dB relative to the fundamental.

    Unlike ASR this counts *legitimate* harmonics, so it measures how much
    colour the non-linearity adds rather than how much of it is illegal.
    Reported alongside ASR so that a fix can be shown to remove aliasing
    without also flattening the intended harmonic character.
    """
    freqs, spec = _spectrum(x, sample_rate)
    nyquist = sample_rate / 2.0

    def peak(order: int) -> float:
        target = order * f0
        if target >= nyquist:
            return 0.0
        band = np.abs(freqs - target) < _PEAK_HALF_WIDTH_HZ
        return float(spec[band].sum())

    fundamental = peak(1)
    harmonics = sum(peak(k) for k in range(2, max_order + 1))
    return float(10.0 * np.log10(harmonics / (fundamental + 1e-30) + 1e-30))


def harmonic_partial_levels(
    x: np.ndarray,
    f0: float,
    sample_rate: int,
    max_order: int = 12,
) -> dict[int, float]:
    """Level of each harmonic partial in dB relative to the fundamental."""
    freqs, spec = _spectrum(x, sample_rate)
    nyquist = sample_rate / 2.0
    levels: dict[int, float] = {}

    fundamental_band = np.abs(freqs - f0) < _PEAK_HALF_WIDTH_HZ
    fundamental = float(spec[fundamental_band].sum()) + 1e-30

    for order in range(1, max_order + 1):
        target = order * f0
        if target >= nyquist:
            break
        band = np.abs(freqs - target) < _PEAK_HALF_WIDTH_HZ
        levels[order] = float(10.0 * np.log10(float(spec[band].sum()) / fundamental + 1e-30))
    return levels


# Bands chosen for what they mean musically, not for equal spacing.
# The 8-16 kHz and 16-20 kHz bands are the "air" that makes a library sound
# expensive, and are the first thing a lossy or aliasing signal path damages.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 120.0),
    ("low", 120.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 8000.0),
    ("presence", 8000.0, 16000.0),
    ("air", 16000.0, 20000.0),
)


def band_energy_error_db(
    reference: np.ndarray,
    test: np.ndarray,
    sample_rate: int,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS,
) -> dict[str, float]:
    """
    Per-band energy error in dB (``test`` relative to ``reference``).

    Positive means the system added energy in that band, negative means it lost
    it. Reported per band because a single broadband number hides the failure
    that matters: a signal path can look fine overall while dumping alias
    energy into the presence band and eating the air band.
    """
    length = min(len(reference), len(test))
    freqs, ref_spec = _spectrum(reference[:length], sample_rate)
    _, test_spec = _spectrum(test[:length], sample_rate)

    errors: dict[str, float] = {}
    for name, low, high in bands:
        band = (freqs >= low) & (freqs < high)
        if not band.any():
            continue
        ref_energy = float(ref_spec[band].sum()) + 1e-30
        test_energy = float(test_spec[band].sum()) + 1e-30
        errors[name] = float(10.0 * np.log10(test_energy / ref_energy))
    return errors


def log_spectral_distance(
    reference: np.ndarray,
    test: np.ndarray,
    sample_rate: int,
    fft_size: int = 2048,
    hop: int = 512,
) -> float:
    """
    Mean log-spectral distance in dB between two signals.

    A single broadband fidelity number, useful for tracking regressions.
    Lower is better; 0 means the two signals are spectrally identical.
    """
    length = min(len(reference), len(test))
    reference = np.asarray(reference[:length], dtype=np.float64)
    test = np.asarray(test[:length], dtype=np.float64)

    window = np.hanning(fft_size)
    distances = []
    for start in range(0, length - fft_size, hop):
        ref_frame = np.abs(np.fft.rfft(reference[start : start + fft_size] * window))
        test_frame = np.abs(np.fft.rfft(test[start : start + fft_size] * window))
        diff = 20.0 * np.log10(test_frame + 1e-10) - 20.0 * np.log10(ref_frame + 1e-10)
        distances.append(float(np.sqrt(np.mean(diff**2))))
    return float(np.mean(distances)) if distances else 0.0


def crest_factor_db(x: np.ndarray) -> float:
    """
    Peak-to-RMS ratio in dB -- a proxy for transient preservation.

    Plucks, mallets and drum hits live or die on their attack. A signal path
    that smears transients pulls the crest factor down, and the sample stops
    cutting through a mix.
    """
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    return float(20.0 * np.log10(peak / (rms + 1e-30) + 1e-30))


def stereo_correlation(stereo: np.ndarray) -> float:
    """
    Inter-channel correlation of a ``(2, samples)`` signal.

    Width is a first-class feature of pads and cinematic patches. A path that
    collapses or randomises the stereo field is unusable for that material
    even when both channels measure well in isolation.
    """
    stereo = np.asarray(stereo, dtype=np.float64)
    if stereo.ndim != 2 or stereo.shape[0] != 2:
        raise ValueError(f"expected shape (2, samples), got {stereo.shape}")
    left, right = stereo[0] - stereo[0].mean(), stereo[1] - stereo[1].mean()
    denominator = np.sqrt((left**2).sum() * (right**2).sum())
    if denominator == 0:
        return 0.0
    return float((left * right).sum() / denominator)
