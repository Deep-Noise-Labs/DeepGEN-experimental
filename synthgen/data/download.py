"""
Download and materialize audio-text datasets for SynthGen.

AudioCaps is exported from Hugging Face (OpenSound/AudioCaps) into a local
layout under ``{output_dir}/audiocaps/``:

    audiocaps/
      metadata.jsonl   # {"file_name": "audio/<id>.wav", "caption": "..."}
      audio/*.wav
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from synthgen.utils.audio import save_audio

logger = logging.getLogger(__name__)

AUDIOCAPS_HF_ID = "OpenSound/AudioCaps"
AUDIOCAPS_RESOLVE = (
    f"https://huggingface.co/datasets/{AUDIOCAPS_HF_ID}/resolve/main"
)
SUPPORTED_DATASETS = ("audiocaps",)


def _http_download(url: str, dest: Path, attempts: int = 8) -> None:
    """
    Download ``url`` to ``dest`` via curl.

    The Hugging Face Python clients often 403 on the Xet CDN from this host;
    ``curl -L`` follows redirects more reliably. Transient 403s are retried
    with backoff.
    """
    import time

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err = ""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-A",
                "synthgen-download/0.1",
                "-o",
                str(tmp),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return
        last_err = result.stderr.strip() or result.stdout.strip()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        sleep_s = min(60, 2 ** attempt)
        logger.warning(
            "Download attempt %d/%d failed for %s; retrying in %ss",
            attempt,
            attempts,
            Path(url).name,
            sleep_s,
        )
        time.sleep(sleep_s)
    raise RuntimeError(f"curl failed for {url} after {attempts} attempts: {last_err}")


def _list_train_shards() -> list[str]:
    """Return sorted train parquet paths under the AudioCaps HF repo."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required (via datasets). "
            "Install with: uv sync --extra download"
        ) from exc

    files = list_repo_files(AUDIOCAPS_HF_ID, repo_type="dataset")
    shards = sorted(
        f
        for f in files
        if f.startswith("data/train-") and f.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"No train parquet shards found in {AUDIOCAPS_HF_ID}")
    return shards


def _audio_from_parquet_row(audio_struct: dict) -> tuple[np.ndarray, int]:
    raw = audio_struct["bytes"]
    array, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    # soundfile -> (samples, channels); we store (channels, samples)
    array = array.T.astype(np.float32, copy=False)
    return array, int(sample_rate)


def download_audiocaps(
    output_dir: Path,
    max_samples: int | None = None,
    split: str = "train",
) -> Path:
    """
    Download AudioCaps and write wav files + metadata.jsonl.

    Fetches parquet shards over HTTPS (urllib) rather than the HF Xet CDN
    client, which has been observed to 403 on some hosts.

    Returns:
        Path to the dataset root (``.../audiocaps``).
    """
    if split != "train":
        raise ValueError("Only the train split is currently supported")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for AudioCaps download. "
            "Install with: uv sync --extra download"
        ) from exc

    dataset_root = output_dir / "audiocaps"
    audio_dir = dataset_root / "audio"
    shard_cache = dataset_root / ".shards"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shard_cache.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "metadata.jsonl"

    existing: dict[str, str] = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                existing[row["file_name"]] = row["caption"]

    records: list[dict[str, str]] = []
    for file_name, caption in existing.items():
        if (dataset_root / file_name).exists():
            records.append({"file_name": file_name, "caption": caption})

    if max_samples is not None and len(records) >= max_samples:
        records = records[:max_samples]
        with open(metadata_path, "w") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "Already have %d clips (>= max_samples=%d); skipping download",
            len(records),
            max_samples,
        )
        return dataset_root

    n_new = 0
    skip_rows = len(records)
    rows_seen = 0

    shards = _list_train_shards()
    logger.info("Found %d AudioCaps train shards", len(shards))

    progress = tqdm(
        desc="Exporting AudioCaps",
        total=max_samples,
        unit="clip",
        initial=len(records),
    )
    try:
        for shard_rel in shards:
            if max_samples is not None and len(records) >= max_samples:
                break

            shard_name = Path(shard_rel).name
            shard_path = shard_cache / shard_name
            if not shard_path.exists():
                url = f"{AUDIOCAPS_RESOLVE}/{shard_rel}"
                logger.info("Downloading shard %s", shard_name)
                try:
                    _http_download(url, shard_path)
                except RuntimeError as exc:
                    logger.warning("Skipping shard %s (%s)", shard_name, exc)
                    continue

            table = pq.read_table(shard_path, columns=["caption", "audio"])
            for i in range(table.num_rows):
                if rows_seen < skip_rows:
                    rows_seen += 1
                    continue
                if max_samples is not None and len(records) >= max_samples:
                    break

                file_name = f"audio/{rows_seen:06d}.wav"
                wav_path = dataset_root / file_name
                caption = str(table.column("caption")[i].as_py())
                audio_struct = table.column("audio")[i].as_py()
                array, sample_rate = _audio_from_parquet_row(audio_struct)
                save_audio(wav_path, array, sample_rate=sample_rate)
                records.append({"file_name": file_name, "caption": caption})
                n_new += 1
                rows_seen += 1
                progress.update(1)
    finally:
        progress.close()

    records.sort(key=lambda r: r["file_name"])
    if max_samples is not None:
        records = records[:max_samples]

    with open(metadata_path, "w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "AudioCaps ready at %s (%d clips, %d newly written)",
        dataset_root,
        len(records),
        n_new,
    )
    return dataset_root


def download_dataset(
    name: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> Path:
    name = name.lower()
    if name in ("audiocaps", "all"):
        return download_audiocaps(output_dir, max_samples=max_samples)
    raise ValueError(
        f"Unsupported dataset '{name}'. "
        f"Currently supported: {', '.join(SUPPORTED_DATASETS)}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download SynthGen datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (audiocaps) or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Root directory for datasets (default: ./data)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of clips (useful for smoke tests)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(args.dataset, output_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
