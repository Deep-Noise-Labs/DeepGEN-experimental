"""
PyTorch dataset and collator for captioned audio clips.

Expected on-disk layout (written by ``synthgen-download``):

    {data_dir}/
      metadata.jsonl   # {"file_name": "audio/000000.wav", "caption": "..."}
      audio/*.wav

If ``data_dir`` itself has no ``metadata.jsonl`` but contains an ``audiocaps``
subdirectory that does, that subdirectory is used automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from synthgen.data.preprocessing import pad_or_crop, peak_normalize, random_gain
from synthgen.utils.audio import load_audio

logger = logging.getLogger(__name__)


def resolve_dataset_root(data_dir: Path) -> Path:
    """Resolve a dataset root that contains ``metadata.jsonl``."""
    data_dir = Path(data_dir)
    if (data_dir / "metadata.jsonl").exists():
        return data_dir
    nested = data_dir / "audiocaps"
    if (nested / "metadata.jsonl").exists():
        return nested
    raise FileNotFoundError(
        f"No metadata.jsonl under {data_dir} or {nested}. "
        "Run: uv run synthgen-download --dataset audiocaps --output-dir ./data"
    )


def load_metadata(dataset_root: Path, max_samples: int | None = None) -> list[dict]:
    """Load metadata rows, optionally truncated to ``max_samples``."""
    path = dataset_root / "metadata.jsonl"
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"Empty metadata file: {path}")
    return records


class AudioTextDataset(Dataset):
    """Captioned audio clips for VAE / DiT training."""

    def __init__(
        self,
        data_dir: str | Path,
        sample_rate: int = 44100,
        duration: float = 15.0,
        channels: int = 2,
        augment: bool = True,
        max_samples: int | None = None,
    ):
        self.dataset_root = resolve_dataset_root(Path(data_dir))
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.augment = augment
        self.target_samples = int(sample_rate * duration)
        self.records = load_metadata(self.dataset_root, max_samples=max_samples)
        logger.info(
            "AudioTextDataset: %d clips from %s",
            len(self.records),
            self.dataset_root,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        path = self.dataset_root / row["file_name"]
        audio = load_audio(
            path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        original_duration = audio.shape[-1] / float(self.sample_rate)
        audio = pad_or_crop(audio, self.target_samples)
        audio = peak_normalize(audio)
        if self.augment:
            audio = random_gain(audio)
            audio = np.clip(audio, -1.0, 1.0)

        return {
            "audio": torch.from_numpy(audio.copy()),
            "caption": str(row["caption"]),
            "duration": min(original_duration, self.duration),
        }


@dataclass
class SynthGenCollator:
    """Pad a batch of variable-length clips to a common length."""

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item["audio"].shape[-1] for item in batch)
        audios = []
        for item in batch:
            audio = item["audio"]
            if audio.shape[-1] < max_len:
                pad = max_len - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad))
            audios.append(audio)

        return {
            "audio": torch.stack(audios, dim=0),  # (B, C, T)
            "captions": [item["caption"] for item in batch],
            "durations": torch.tensor(
                [item["duration"] for item in batch], dtype=torch.float32
            ),
        }
