"""
Measurement stimuli for the SynthGen audio-quality evals.

These are *test signals*, not training data. Each one is constructed to be
strictly band-limited, so it contains no aliasing of its own. Any
inharmonic content measured at the output of a module under test was
therefore created by that module - which is what makes the alias metrics
in :mod:`synthgen.eval.metrics` attributable.

Choosing f0
-----------
A folded harmonic lands at ``|k*f0 - n*fs|``. That coincides with a real
harmonic ``m*f0`` exactly when ``f0 = n*fs / (k -/+ m)`` - i.e. when f0 is a
simple rational fraction of the sample rate. Round numbers are close enough
to such ratios that aliases hide under the harmonic grid and the
measurement under-reports - at 44.1 kHz, an innocent-looking 220.5 Hz "A3"
gives exactly ``fs / f0 = 200`` and reports *no aliasing whatsoever*, no
matter how badly the module under test is actually behaving.

So the defaults here are not chosen by ear. :func:`alias_visibility_hz`
scores a candidate by how far its folded products land from the nearest
true harmonic, and every entry in :data:`DEFAULT_TEST_FREQS` was picked by
maximising that score near a musical pitch.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_TEST_FREQS",
    "SWEEP_TEST_FREQS",
    "alias_visibility_hz",
    "pure_tone",
    "bandlimited_saw",
    "bandlimited_square",
    "log_sweep",
    "impulse",
    "STIMULI",
    "build_stimulus",
]


def alias_visibility_hz(
    f0: float,
    sample_rate: int = 44100,
    max_harmonic: int = 200,
) -> float:
    """
    How far, in Hz, the *closest* folded alias lands from the nearest true
    harmonic of ``f0``.

    This is the validity check for a test frequency. If the answer is near
    zero, every alias hides underneath a harmonic of the note, the
    harmonic/inharmonic split cannot separate them, and any alias metric
    measured at that ``f0`` will read far too clean.

    The failure is not hypothetical. At 44.1 kHz, ``f0 = 220.5 Hz`` gives
    exactly ``sample_rate / f0 = 200`` and scores 0.0 here: a perfectly
    plausible-looking "A3" that silently reports no aliasing at all. Pick
    test frequencies with this function, never by ear.

    Returns:
        Minimum separation in Hz. Should comfortably exceed the width of
        the harmonic tolerance band used by
        :func:`synthgen.eval.metrics.harmonic_analysis` (``tolerance_bins``
        times the FFT bin width).
    """
    nyquist = sample_rate / 2
    first_folding = int(np.ceil(nyquist / f0)) + 1
    ks = np.arange(first_folding, max_harmonic + 1)
    if len(ks) == 0:
        return 0.0

    folded = ks * f0
    folded = np.abs(folded - np.round(folded / sample_rate) * sample_rate)

    harmonics = np.arange(1, int(nyquist / f0) + 1) * f0
    if len(harmonics) == 0:
        return 0.0

    distance = np.abs(folded[:, None] - harmonics[None, :]).min(axis=1)
    # Distance to DC counts too - an alias at ~0 Hz is masked by the
    # high-pass cutoff rather than by a harmonic.
    distance = np.minimum(distance, folded)
    return float(distance.min())


# Fundamentals spanning the range a synth lead actually plays, each one
# selected by maximising :func:`alias_visibility_hz` in a +/-3% window around
# a musical pitch (A2, A4, A5, C7, A7). Every entry clears 50 Hz of
# separation; the obvious round choices near them (110.0, 220.5, 4409.1)
# score 0-9 Hz and would have made the model look clean when it was not.
DEFAULT_TEST_FREQS: tuple[float, ...] = (111.5, 453.1, 903.7, 2090.1, 3604.3)

# The wider grid used for the alias-vs-pitch figure.
SWEEP_TEST_FREQS: tuple[float, ...] = (
    111.5,
    223.3,
    453.1,
    903.7,
    1804.1,
    2090.1,
    3604.3,
    4260.9,
)


def _t(duration: float, sample_rate: int) -> np.ndarray:
    return np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate


def pure_tone(
    f0: float = 903.7,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A single sinusoid. The cleanest possible alias probe."""
    return (amplitude * np.sin(2 * np.pi * f0 * _t(duration, sample_rate))).astype(
        np.float32
    )


def bandlimited_saw(
    f0: float = 903.7,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Additive sawtooth containing only harmonics below Nyquist.

    This is the signal that matters commercially: a saw is the backbone of
    nearly every supersaw, lead and bass patch, and its ``1/k`` harmonic
    series puts real energy right up against Nyquist - exactly the content
    that folds when a nonlinearity is evaluated without oversampling.
    """
    t = _t(duration, sample_rate)
    nyquist = sample_rate / 2
    out = np.zeros_like(t)
    k = 1
    while k * f0 < nyquist:
        out += np.sin(2 * np.pi * k * f0 * t) / k
        k += 1
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    return (amplitude * out).astype(np.float32)


def bandlimited_square(
    f0: float = 903.7,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Additive square wave (odd harmonics only), band-limited to Nyquist."""
    t = _t(duration, sample_rate)
    nyquist = sample_rate / 2
    out = np.zeros_like(t)
    k = 1
    while k * f0 < nyquist:
        out += np.sin(2 * np.pi * k * f0 * t) / k
        k += 2
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    return (amplitude * out).astype(np.float32)


def log_sweep(
    f_start: float = 20.0,
    f_end: float = 20000.0,
    duration: float = 2.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Exponential sine sweep.

    Plotted as a spectrogram, a sweep makes aliasing *visible*: the
    fundamental rises left-to-right, while folded harmonics appear as
    extra lines that reflect off the Nyquist ceiling and travel downwards.
    """
    t = _t(duration, sample_rate)
    T = t[-1] if len(t) > 1 else duration
    k = np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * T / k * (np.exp(k * t / T) - 1.0)
    return (amplitude * np.sin(phase)).astype(np.float32)


def impulse(
    duration: float = 0.5,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    position: float = 0.25,
) -> np.ndarray:
    """Single-sample impulse - probes transient smearing and ringing."""
    n = int(round(duration * sample_rate))
    out = np.zeros(n, dtype=np.float32)
    out[int(position * n)] = amplitude
    return out


STIMULI = {
    "pure_tone": pure_tone,
    "bandlimited_saw": bandlimited_saw,
    "bandlimited_square": bandlimited_square,
    "log_sweep": log_sweep,
    "impulse": impulse,
}


def build_stimulus(name: str, **kwargs) -> np.ndarray:
    """Look up and build a stimulus by name."""
    if name not in STIMULI:
        raise KeyError(f"unknown stimulus {name!r}; choose from {sorted(STIMULI)}")
    return STIMULI[name](**kwargs)
