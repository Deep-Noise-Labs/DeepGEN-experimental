"""Tests for AudioTextDataset and SynthGenCollator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from synthgen.data.dataset import AudioTextDataset, SynthGenCollator, resolve_dataset_root
from synthgen.data.preprocessing import pad_or_crop, peak_normalize


def _write_fixture(root: Path, n: int = 4, sr: int = 16000, seconds: float = 0.5) -> Path:
    dataset_root = root / "audiocaps"
    audio_dir = dataset_root / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    for i in range(n):
        # Vary length slightly
        samples = int(sr * (seconds + 0.05 * i))
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        rel = f"audio/{i:06d}.wav"
        sf.write(str(dataset_root / rel), audio, sr)
        records.append({"file_name": rel, "caption": f"test sound {i}"})
    with open(dataset_root / "metadata.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    return dataset_root


class TestPreprocessing:
    def test_pad_or_crop_pad(self):
        x = np.zeros((2, 100), dtype=np.float32)
        out = pad_or_crop(x, 150)
        assert out.shape == (2, 150)

    def test_pad_or_crop_crop(self):
        x = np.ones((2, 200), dtype=np.float32)
        out = pad_or_crop(x, 80)
        assert out.shape == (2, 80)

    def test_peak_normalize(self):
        x = np.array([[0.0, 0.5]], dtype=np.float32)
        out = peak_normalize(x, peak=1.0)
        assert np.isclose(np.max(np.abs(out)), 1.0)


class TestAudioTextDataset:
    def test_resolve_nested(self, tmp_path: Path):
        dataset_root = _write_fixture(tmp_path)
        assert resolve_dataset_root(tmp_path) == dataset_root
        assert resolve_dataset_root(dataset_root) == dataset_root

    def test_len_and_max_samples(self, tmp_path: Path):
        _write_fixture(tmp_path, n=5)
        ds = AudioTextDataset(tmp_path, duration=1.0, augment=False, max_samples=3)
        assert len(ds) == 3

    def test_item_stereo_and_duration(self, tmp_path: Path):
        _write_fixture(tmp_path, n=2, sr=22050, seconds=0.25)
        ds = AudioTextDataset(
            tmp_path,
            sample_rate=22050,
            duration=1.0,
            channels=2,
            augment=False,
        )
        item = ds[0]
        assert item["audio"].shape == (2, 22050)
        assert isinstance(item["caption"], str)
        assert 0 < item["duration"] <= 1.0

    def test_collator_batch(self, tmp_path: Path):
        _write_fixture(tmp_path, n=4)
        ds = AudioTextDataset(tmp_path, duration=0.5, augment=False)
        loader = DataLoader(ds, batch_size=2, collate_fn=SynthGenCollator())
        batch = next(iter(loader))
        assert batch["audio"].shape[0] == 2
        assert batch["audio"].ndim == 3
        assert len(batch["captions"]) == 2
        assert batch["durations"].shape == (2,)
        assert batch["audio"].dtype == torch.float32

    def test_missing_metadata_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            AudioTextDataset(tmp_path)
