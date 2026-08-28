"""
Evaluation metrics for the VAE objective ablation.

Deliberately *not* the training losses. Comparing two objectives by scoring the
result with one of them is circular, so everything here is a standard signal
measure that neither arm optimises directly.
"""

import numpy as np

__all__ = [
    "si_sdr_db",
    "log_spectral_distance_db",
    "stereo_width",
    "envelope_error_db",
    "band_energy_error_db",
]

_EPS = 1e-12


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    count = 1 + max(0, (len(x) - n_fft) // hop)
    if count <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return np.abs(np.fft.rfft(x[idx] * window, axis=-1))


def si_sdr_db(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Scale-invariant signal-to-distortion ratio, in dB. Higher is better.

    Scale invariance matters here because neither objective constrains absolute
    output gain especially tightly, and a pure level offset is not what is being
    compared.
    """
    p, t = _to_mono(pred).astype(np.float64), _to_mono(target).astype(np.float64)
    n = min(len(p), len(t))
    p, t = p[:n] - p[:n].mean(), t[:n] - t[:n].mean()

    alpha = np.dot(p, t) / (np.dot(t, t) + _EPS)
    projection = alpha * t
    noise = p - projection
    return float(
        10 * np.log10((np.sum(projection**2) + _EPS) / (np.sum(noise**2) + _EPS))
    )


def log_spectral_distance_db(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    floor_db: float = -80.0,
) -> float:
    """
    RMS log-spectral distance in dB over an optional frequency band.

    Lower is better. Restricting the band is how the low-end claim is measured:
    ``fmax=200`` scores only what a bass patch lives on.
    """
    p, t = _to_mono(pred), _to_mono(target)
    n = min(len(p), len(t))
    P, T = _frames(p[:n], n_fft, hop), _frames(t[:n], n_fft, hop)
    if P.shape[0] == 0:
        return float("nan")

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band = freqs >= fmin
    if fmax is not None:
        band &= freqs < fmax
    if not band.any():
        return float("nan")

    # Normalise both spectrograms so the metric measures spectral *shape*,
    # then floor them so silent bins cannot dominate.
    def to_db(mag: np.ndarray) -> np.ndarray:
        scaled = mag / (mag.max() + _EPS)
        return np.maximum(20 * np.log10(scaled + _EPS), floor_db)

    diff = to_db(P)[:, band] - to_db(T)[:, band]
    return float(np.sqrt((diff**2).mean()))


def stereo_width(audio: np.ndarray) -> float:
    """RMS(side) / RMS(mid). 0.0 is mono; higher is wider."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 0.0
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    return float(np.sqrt((side**2).mean()) / (np.sqrt((mid**2).mean()) + _EPS))


def envelope_error_db(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
    window_ms: float = 5.0,
    floor_db: float = -60.0,
) -> float:
    """
    Mean absolute error between short-window log RMS envelopes, in dB.

    This is the transient measure: a smeared attack tracks the target's
    envelope badly even when its average spectrum is perfect.
    """
    p, t = _to_mono(pred), _to_mono(target)
    n = min(len(p), len(t))
    window = max(int(sample_rate * window_ms / 1000.0), 1)
    count = n // window
    if count == 0:
        return float("nan")

    def envelope(x: np.ndarray) -> np.ndarray:
        blocks = x[: count * window].reshape(count, window)
        rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
        db = 20 * np.log10(rms / (rms.max() + _EPS) + _EPS)
        return np.maximum(db, floor_db)

    return float(np.abs(envelope(p) - envelope(t)).mean())


def band_energy_error_db(
    pred: np.ndarray,
    target: np.ndarray,
    fmin: float,
    fmax: float,
    sample_rate: int = 44100,
) -> float:
    """
    Signed error in the *proportion* of total energy a band carries, in dB.

    Negative means the band is under-represented relative to the target — which
    is exactly what "the bass went missing" looks like numerically.
    """
    p, t = _to_mono(pred), _to_mono(target)
    n = min(len(p), len(t))

    def band_share(x: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        mask = (freqs >= fmin) & (freqs < fmax)
        return float(spectrum[mask].sum() / (spectrum.sum() + _EPS))

    return float(10 * np.log10((band_share(p) + _EPS) / (band_share(t) + _EPS)))
