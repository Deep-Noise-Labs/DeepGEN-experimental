"""
Controlled, audible degradations used to probe what a training loss can see.

Each function takes and returns ``(channels, samples)`` float audio. They are
deliberately simple and exact (brick-wall filters in the frequency domain,
plain moving averages) so that the *only* thing being measured is the loss's
response, not the artefacts of the degradation itself.
"""

from __future__ import annotations

import numpy as np


def _spectral_mask(audio: np.ndarray, sr: int, keep: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(audio, axis=-1)
    return np.fft.irfft(spectrum * keep, n=audio.shape[-1], axis=-1)


def lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Remove everything above ``cutoff`` Hz. Kills 'air' and brightness."""
    freqs = np.fft.rfftfreq(audio.shape[-1], 1 / sr)
    return _spectral_mask(audio, sr, freqs <= cutoff)


def highpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Remove everything below ``cutoff`` Hz. Kills weight and sub."""
    freqs = np.fft.rfftfreq(audio.shape[-1], 1 / sr)
    return _spectral_mask(audio, sr, freqs >= cutoff)


def mono_collapse(audio: np.ndarray, sr: int) -> np.ndarray:
    """Replace both channels with their average. Destroys the stereo image."""
    if audio.shape[0] < 2:
        return audio.copy()
    mono = audio.mean(axis=0)
    return np.stack([mono] * audio.shape[0])


def transient_smear(audio: np.ndarray, sr: int, window_ms: float = 10.0) -> np.ndarray:
    """Moving-average the waveform. Softens attacks; the sample loses its bite."""
    n = max(2, int(sr * window_ms / 1000.0))
    kernel = np.ones(n) / n
    return np.stack([np.convolve(ch, kernel, mode="same") for ch in audio])


def quantize(audio: np.ndarray, sr: int, bits: int = 8) -> np.ndarray:
    """Requantise to ``bits``. Raises the noise floor audibly."""
    levels = 2 ** (bits - 1)
    return np.round(audio * levels) / levels


def gain(audio: np.ndarray, sr: int, db: float = -6.0) -> np.ndarray:
    """Plain level change. The control: every loss should see this clearly."""
    return audio * (10 ** (db / 20.0))


def shelf(audio: np.ndarray, sr: int, corner: float, db: float, high: bool = True) -> np.ndarray:
    """
    Attenuate (or boost) one side of ``corner`` by ``db``.

    Unlike a brick wall this never produces exactly-zero bins, so it measures a
    loss's response to *audibility* rather than to a log-of-zero cliff. This is
    also what a real decoder does when it dulls: it attenuates, it does not
    delete.
    """
    freqs = np.fft.rfftfreq(audio.shape[-1], 1 / sr)
    scale = np.ones_like(freqs)
    side = freqs >= corner if high else freqs <= corner
    scale[side] = 10 ** (db / 20.0)
    return _spectral_mask(audio, sr, scale)


def hf_noise_substitution(audio: np.ndarray, sr: int, corner: float = 10000.0) -> np.ndarray:
    """
    Replace content above ``corner`` with noise of the same band energy.

    This is the characteristic latent-decoder failure: the high band is not
    missing, it is *wrong* -- structured harmonics and cymbal detail replaced
    by hiss at the right level. Band energy is preserved, so any metric that
    only looks at band totals will score it as perfect.
    """
    freqs = np.fft.rfftfreq(audio.shape[-1], 1 / sr)
    band = freqs >= corner
    spectrum = np.fft.rfft(audio, axis=-1)
    rng = np.random.default_rng(0)
    for ch in range(spectrum.shape[0]):
        energy = np.sqrt(np.mean(np.abs(spectrum[ch, band]) ** 2))
        phase = rng.uniform(0, 2 * np.pi, band.sum())
        spectrum[ch, band] = energy * np.exp(1j * phase)
    return np.fft.irfft(spectrum, n=audio.shape[-1], axis=-1)


def inaudible_dither(audio: np.ndarray, sr: int, level_dbfs: float = -90.0) -> np.ndarray:
    """
    Add noise at ``level_dbfs``. Completely inaudible on any playback system.

    This is the null test. A loss that reacts strongly here is spending its
    gradient budget on content no listener will ever hear, at the expense of
    content they will.
    """
    rng = np.random.default_rng(1)
    return audio + rng.normal(0, 10 ** (level_dbfs / 20.0), audio.shape)


#: Name -> (callable, human-readable description of what a listener hears).
#: These are the realistic degradations: attenuation and substitution rather
#: than deletion, so no measurement is an artefact of a log-of-zero cliff.
DEGRADATIONS: dict[str, tuple] = {
    "air_dulled_12db": (
        lambda a, sr: shelf(a, sr, 8000.0, -12.0, high=True),
        "Everything above 8 kHz pulled down 12 dB - veiled, no sparkle",
    ),
    "hf_noise_substitution": (
        hf_noise_substitution,
        "High band replaced by hiss at the same energy - the decoder artefact",
    ),
    "sub_dulled_12db": (
        lambda a, sr: shelf(a, sr, 80.0, -12.0, high=False),
        "Everything below 80 Hz pulled down 12 dB - thin, no weight",
    ),
    "mono_collapse": (
        mono_collapse,
        "Stereo image flattened to a single point",
    ),
    "transient_smear": (
        lambda a, sr: transient_smear(a, sr, 10.0),
        "Attacks smeared over 10 ms - soft, lifeless",
    ),
    "inaudible_dither": (
        inaudible_dither,
        "Noise added at -90 dBFS - the null test; nobody can hear this",
    ),
    "gain_minus_6db": (
        gain,
        "6 dB quieter - the control; every loss must see this",
    ),
}

