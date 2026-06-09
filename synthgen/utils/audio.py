"""
Audio I/O and processing utilities for SynthGen.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 44100,
    channels: int = 2,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """
    Load an audio file and return as numpy array.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate.
        channels: Target number of channels (1=mono, 2=stereo).
        offset: Start time in seconds.
        duration: Duration to load in seconds. None means load all.

    Returns:
        Audio array of shape (channels, samples).
    """
    path = str(path)

    if sf is not None:
        info = sf.info(path)
        orig_sr = info.samplerate

        start_frame = int(offset * orig_sr)
        frames = int(duration * orig_sr) if duration else -1

        audio, file_sr = sf.read(
            path,
            start=start_frame,
            stop=start_frame + frames if frames > 0 else None,
            dtype="float32",
            always_2d=True,
        )
        # audio shape: (samples, channels)
        audio = audio.T  # -> (channels, samples)

    elif librosa is not None:
        audio, file_sr = librosa.load(
            path,
            sr=None,
            mono=False,
            offset=offset,
            duration=duration,
        )
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # -> (1, samples)
    else:
        raise ImportError(
            "Either soundfile or librosa must be installed for audio loading. "
            "Install with: pip install soundfile librosa"
        )

    # Resample if necessary
    if file_sr != sample_rate:
        audio = resample_audio(audio, file_sr, sample_rate)

    # Handle channel conversion
    current_channels = audio.shape[0]
    if current_channels != channels:
        if channels == 1:
            # Downmix to mono
            audio = np.mean(audio, axis=0, keepdims=True)
        elif channels == 2:
            if current_channels == 1:
                # Duplicate mono to stereo
                audio = np.repeat(audio, 2, axis=0)
            else:
                # Take first two channels
                audio = audio[:2]

    return audio


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 44100,
) -> None:
    """
    Save audio array to a file.

    Args:
        path: Output file path.
        audio: Audio array of shape (channels, samples) or (samples,).
        sample_rate: Sample rate of the audio.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim == 2:
        audio = audio.T  # (channels, samples) -> (samples, channels)

    if sf is not None:
        sf.write(str(path), audio, sample_rate)
    else:
        raise ImportError("soundfile is required for saving audio files.")


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a target sample rate.

    Args:
        audio: Audio array of shape (channels, samples).
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio

    if librosa is not None:
        # Resample each channel independently
        resampled = np.stack([
            librosa.resample(audio[ch], orig_sr=orig_sr, target_sr=target_sr)
            for ch in range(audio.shape[0])
        ])
        return resampled
    else:
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(audio.shape[-1] * ratio)
        indices = np.linspace(0, audio.shape[-1] - 1, new_length)
        resampled = np.stack([
            np.interp(indices, np.arange(audio.shape[-1]), audio[ch])
            for ch in range(audio.shape[0])
        ])
        return resampled


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to a target peak level in dB.

    Args:
        audio: Audio array of shape (channels, samples).
        target_db: Target peak level in dB.

    Returns:
        Normalized audio array.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_linear = 10 ** (target_db / 20.0)
        audio = audio * (target_linear / peak)
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.

    Args:
        audio: Audio array of shape (channels, samples) or (samples,).
        sample_rate: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel bands.
        fmin: Minimum frequency.
        fmax: Maximum frequency. None means sr/2.

    Returns:
        Mel spectrogram of shape (channels, n_mels, time_frames).
    """
    if librosa is None:
        raise ImportError("librosa is required for mel spectrogram computation.")

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    if fmax is None:
        fmax = sample_rate / 2.0

    mel_specs = []
    for ch in range(audio.shape[0]):
        mel = librosa.feature.melspectrogram(
            y=audio[ch],
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_specs.append(mel_db)

    return np.stack(mel_specs)
