"""Lightweight audio preprocessing helpers for training."""

from __future__ import annotations

import numpy as np


def pad_or_crop(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or randomly crop to ``target_samples`` along time."""
    current = audio.shape[-1]
    if current == target_samples:
        return audio
    if current > target_samples:
        start = np.random.randint(0, current - target_samples + 1)
        return audio[..., start : start + target_samples]
    pad = target_samples - current
    return np.pad(audio, ((0, 0), (0, pad)), mode="constant")


def random_gain(audio: np.ndarray, gain_db_range: float = 3.0) -> np.ndarray:
    """Apply a random gain in ±gain_db_range dB."""
    gain_db = np.random.uniform(-gain_db_range, gain_db_range)
    return audio * (10 ** (gain_db / 20.0))


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale so the absolute peak equals ``peak`` (no-op if silent)."""
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32, copy=False)
