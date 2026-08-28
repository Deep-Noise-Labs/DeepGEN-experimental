"""
Controlled, audible degradations used to probe what a training objective can
and cannot hear.

Every degradation is applied in the frequency domain with numpy only, so the
probe has no dependency beyond what SynthGen already requires. Each one
corresponds to a failure mode a sound designer would reject on the spot:

- ``bass_detune``       — the bass goes ~2 semitones out of tune, nothing else
                          moves. Tests analysis *resolution* rather than level.
- ``bass_rolloff``      — the patch loses its weight and sub.
- ``transient_smear``   — phase-only dispersion; the magnitude spectrum is
                          preserved almost exactly, so a magnitude loss is
                          nearly blind to it, but plucks turn into swells.
- ``hf_phase_scramble`` — randomised phase above 5 kHz; the classic "watery",
                          metallic neural-codec top end.
- ``stereo_collapse``   — the stereo image folds to mono.
- ``sample_shift``      — a one-sample delay. Completely inaudible. Included as
                          a control, because waveform L1 punishes it heavily.
- ``gain_1db``          — a 1 dB level error. Also included as a control: it is
                          the calibration unit the probe reports scores in.

Plus a matched pair used for the mean-seeking demonstration:

- ``texture_redraw``    — identical magnitude spectrum, all phases re-drawn. On
                          noise-like material this is perceptually the same
                          sound; a regression objective still scores it badly.
- ``spectral_blur``     — magnitude smoothed in frequency, phase kept. Audibly
                          duller, and what "predict the average" sounds like.
"""

from collections.abc import Callable

import numpy as np

__all__ = ["DEGRADATIONS", "MEAN_SEEKING_PAIR", "apply_degradation"]


def _rfft(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(audio, axis=-1)
    freqs = np.fft.rfftfreq(audio.shape[-1], d=1.0 / 44100.0)
    return spectrum, freqs


def _irfft(spectrum: np.ndarray, n: int) -> np.ndarray:
    return np.fft.irfft(spectrum, n=n, axis=-1).astype(np.float32)


def bass_rolloff(audio: np.ndarray, cutoff: float = 80.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass: removes the sub and low fundamentals."""
    spectrum, freqs = _rfft(audio)
    with np.errstate(divide="ignore"):
        response = 1.0 / np.sqrt(1.0 + (cutoff / np.maximum(freqs, 1e-6)) ** (2 * order))
    return _irfft(spectrum * response, audio.shape[-1])


def bass_detune(
    audio: np.ndarray, shift_hz: float = 6.0, band_max: float = 200.0
) -> np.ndarray:
    """
    Shift everything below ``band_max`` up by ``shift_hz``, leaving the rest.

    6 Hz is nothing at 5 kHz and roughly two semitones at 40 Hz, so this is a
    badly out-of-tune bass with an otherwise untouched spectrum. It is the
    honest test of analysis resolution: a 2048-point window at 44.1 kHz has
    21.5 Hz bins, so the whole error fits inside a third of one bin and the
    objective can barely see it.
    """
    spectrum, freqs = _rfft(audio)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    bins = max(int(round(shift_hz / df)), 1)

    band = freqs < band_max
    shifted = spectrum.copy()
    low = np.zeros_like(spectrum)
    low[..., band] = spectrum[..., band]
    shifted[..., band] = 0.0
    shifted += np.roll(low, bins, axis=-1)
    return _irfft(shifted, audio.shape[-1])


def texture_redraw(audio: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    Re-draw the texture: a random all-pass, keeping every magnitude exactly.

    On noise-like material this produces a *different but statistically
    identical* realisation of the same texture — a listener cannot reliably
    tell it from the original. Used to show that a regression objective still
    scores it as a large error, which is the mean-seeking problem.

    One random phase curve is applied to both channels, so the operation is a
    single all-pass filter on the stereo pair: each channel's magnitude
    spectrum and the mid/side relationship are both preserved exactly. Without
    that the probe would be measuring stereo collapse instead of texture.
    """
    rng = np.random.default_rng(seed)
    spectrum, freqs = _rfft(audio)
    phase = rng.uniform(-np.pi, np.pi, size=freqs.shape[0])
    return _irfft(spectrum * np.exp(1j * phase), audio.shape[-1])


def spectral_blur(audio: np.ndarray, freq_bins: int = 65) -> np.ndarray:
    """
    Smooth the magnitude spectrum in frequency, keeping phase.

    This is what "predict the conditional mean" sounds like: the fine structure
    that gives a texture its character is replaced by its local average. Audibly
    duller than the original.
    """
    spectrum, _ = _rfft(audio)
    magnitude = np.abs(spectrum)
    kernel = np.ones(freq_bins) / freq_bins
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), -1, magnitude
    )
    phase = np.angle(spectrum)
    return _irfft(smoothed * np.exp(1j * phase), audio.shape[-1])


def transient_smear(
    audio: np.ndarray, max_group_delay_ms: float = 25.0, seed: int = 0
) -> np.ndarray:
    """
    Phase-only (all-pass) dispersion.

    A smooth random group delay is integrated into a phase curve and applied to
    the spectrum. Long-term magnitude is preserved to within numerical error, so
    this is the cleanest possible test of whether an objective hears phase.
    """
    rng = np.random.default_rng(seed)
    spectrum, freqs = _rfft(audio)
    n_freqs = freqs.shape[0]

    # Smooth random group delay in seconds, band-limited by a moving average.
    raw = rng.standard_normal(n_freqs)
    kernel = np.ones(max(n_freqs // 64, 3))
    smooth = np.convolve(raw, kernel / kernel.sum(), mode="same")
    smooth = smooth / (np.abs(smooth).max() + 1e-9)
    group_delay = 0.5 * (smooth + 1.0) * (max_group_delay_ms / 1000.0)

    df = freqs[1] - freqs[0] if n_freqs > 1 else 1.0
    phase = -2.0 * np.pi * np.cumsum(group_delay) * df
    phase = phase - phase[0]

    return _irfft(spectrum * np.exp(1j * phase), audio.shape[-1])


def hf_phase_scramble(
    audio: np.ndarray, cutoff: float = 5000.0, seed: int = 0
) -> np.ndarray:
    """Randomise phase above ``cutoff``; magnitudes are untouched."""
    rng = np.random.default_rng(seed)
    spectrum, freqs = _rfft(audio)

    phase = rng.uniform(-np.pi, np.pi, size=freqs.shape[0])
    # Crossfade the effect in over an octave so there is no hard discontinuity.
    ramp = np.clip((freqs - cutoff) / cutoff, 0.0, 1.0)
    return _irfft(spectrum * np.exp(1j * phase * ramp), audio.shape[-1])


def stereo_collapse(audio: np.ndarray) -> np.ndarray:
    """Fold the stereo image to mono, preserving the mid channel."""
    if audio.ndim < 2 or audio.shape[0] != 2:
        return audio.copy()
    mid = audio.mean(axis=0, keepdims=True)
    return np.repeat(mid, 2, axis=0)


def sample_shift(audio: np.ndarray, samples: int = 1) -> np.ndarray:
    """Delay by a single sample. Inaudible; a control, not a degradation."""
    return np.roll(audio, samples, axis=-1)


def gain_1db(audio: np.ndarray, gain_db: float = 1.0) -> np.ndarray:
    """A 1 dB broadband level error. The calibration unit for the probe."""
    return (audio * 10.0 ** (gain_db / 20.0)).astype(np.float32)


DEGRADATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bass_detune": bass_detune,
    "bass_rolloff": bass_rolloff,
    "transient_smear": transient_smear,
    "hf_phase_scramble": hf_phase_scramble,
    "stereo_collapse": stereo_collapse,
    "sample_shift": sample_shift,
    "gain_1db": gain_1db,
}

# Not degradations in the same sense: a matched pair used to demonstrate that a
# regression objective prefers a dull average to a plausible alternative.
MEAN_SEEKING_PAIR: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "texture_redraw": texture_redraw,
    "spectral_blur": spectral_blur,
}


def apply_degradation(name: str, audio: np.ndarray) -> np.ndarray:
    transform = DEGRADATIONS.get(name) or MEAN_SEEKING_PAIR.get(name)
    if transform is None:
        raise KeyError(f"Unknown degradation: {name}")
    return transform(audio)
