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

from synthgen.data.preprocessing import (
    DEFAULT_ONSET_ANCHOR_PROB,
    DEFAULT_ONSET_THRESHOLD_DB,
    DEFAULT_PEAK_CEILING_DB,
    DEFAULT_SILENCE_FLOOR_DB,
    DEFAULT_TARGET_RMS_DB,
    prepare_sample,
    remove_dc_offset,
)
from synthgen.data.preprocessing import trim_silence as _trim_silence
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
        target_rms_db: float = DEFAULT_TARGET_RMS_DB,
        peak_ceiling_db: float = DEFAULT_PEAK_CEILING_DB,
        silence_floor_db: float = DEFAULT_SILENCE_FLOOR_DB,
        onset_threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB,
        onset_pre_roll_ms: float = 10.0,
        onset_anchor_prob: float = DEFAULT_ONSET_ANCHOR_PROB,
        gain_db_range: tuple[float, float] = (-6.0, 0.0),
        trim_silence: bool = True,
    ):
        self.dataset_root = resolve_dataset_root(Path(data_dir))
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.augment = augment
        self.target_samples = int(sample_rate * duration)
        self.target_rms_db = target_rms_db
        self.peak_ceiling_db = peak_ceiling_db
        self.silence_floor_db = silence_floor_db
        self.onset_threshold_db = onset_threshold_db
        self.onset_pre_roll_ms = onset_pre_roll_ms
        self.onset_anchor_prob = onset_anchor_prob
        self.gain_db_range = tuple(gain_db_range)
        self.trim_silence = trim_silence
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

        # Derive augmentation randomness from the worker seed rather than the
        # global numpy state. DataLoader seeds torch per worker but not numpy,
        # so a global-state pipeline hands every worker the same gain sequence.
        rng = np.random.default_rng((torch.initial_seed() + index) % (2**32))

        # Trim before measuring, so the duration conditioning describes the
        # sounding content and not whatever silence the source file carried.
        audio = remove_dc_offset(audio)
        if self.trim_silence:
            audio = _trim_silence(
                audio, self.sample_rate, floor_db=self.silence_floor_db
            )
        content_duration = audio.shape[-1] / float(self.sample_rate)

        audio = prepare_sample(
            audio,
            sample_rate=self.sample_rate,
            target_samples=self.target_samples,
            augment=self.augment,
            target_rms_db=self.target_rms_db,
            peak_ceiling_db=self.peak_ceiling_db,
            onset_threshold_db=self.onset_threshold_db,
            onset_pre_roll_ms=self.onset_pre_roll_ms,
            onset_anchor_prob=self.onset_anchor_prob,
            gain_db_range=self.gain_db_range,
            trim=False,  # already trimmed above
            rng=rng,
        )

        return {
            "audio": torch.from_numpy(audio.copy()),
            "caption": str(row["caption"]),
            "duration": max(0.0, min(content_duration, self.duration)),
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
