#!/usr/bin/env python3
"""Render a before/after comparison of the training-data preprocessing pipeline.

For every source clip this writes two WAVs - one through the legacy pipeline
(peak-normalise, symmetric random gain, hard clip, uniformly random crop) and
one through the sample-grade pipeline - plus a JSON of the measured differences.

What you are listening to is *the training target*, not a model generation: the
audio the VAE and the DiT are asked to reproduce. Defects here are learned.

Sources may be local paths or ``s3://bucket/key`` URIs. Reading Deep Noise
sound buckets for QA and comparison is pre-approved; nothing is ever written
back to S3.

Usage::

    python scripts/ab_preprocess_demo.py \
        --source assets/pad.wav --source s3://dnl-core-sounds-s3-prod/<uid>/<gid>/C1.wav \
        --out-dir ./ab_demo --duration 6.0 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from synthgen.data.preprocessing import (
    clipped_fraction,
    crest_factor_db,
    detect_onset,
    peak_db,
    prepare_sample,
    prepare_sample_legacy,
    rms_db,
)
from synthgen.utils.audio import load_audio

logger = logging.getLogger(__name__)


def _fetch(source: str, cache_dir: Path) -> Path:
    """Resolve a local path, or download an ``s3://`` URI into ``cache_dir``."""
    if not source.startswith("s3://"):
        return Path(source)

    import boto3

    bucket, _, key = source[len("s3://") :].partition("/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / key.replace("/", "_")
    if not local.exists():
        logger.info("Downloading %s", source)
        boto3.client("s3").download_file(bucket, key, str(local))
    return local


def _measure(audio: np.ndarray, sample_rate: int) -> dict:
    onset = detect_onset(audio, sample_rate)
    return {
        "peak_dbfs": round(peak_db(audio), 2),
        "rms_dbfs": round(rms_db(audio), 2),
        "crest_factor_db": round(crest_factor_db(audio), 2),
        "clipped_sample_pct": round(100.0 * clipped_fraction(audio), 3),
        "dc_offset": round(float(np.max(np.abs(np.mean(audio, axis=-1)))), 6),
        "onset_ms": None if onset is None else round(1000.0 * onset / sample_rate, 1),
        "first_sample_abs": round(float(np.max(np.abs(audio[..., 0]))), 4),
    }


def render(
    sources: list[str],
    out_dir: Path,
    sample_rate: int = 44100,
    duration: float = 6.0,
    seed: int = 0,
    channels: int = 2,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_samples = int(sample_rate * duration)
    report: list[dict] = []

    for index, source in enumerate(sources):
        path = _fetch(source, out_dir / "_cache")
        audio = load_audio(path, sample_rate=sample_rate, channels=channels)

        # Same seed for both branches, so the only difference heard is the
        # pipeline itself and not a different random gain or crop position.
        legacy = prepare_sample_legacy(
            audio, target_samples, augment=True, rng=np.random.default_rng(seed + index)
        )
        improved = prepare_sample(
            audio,
            sample_rate=sample_rate,
            target_samples=target_samples,
            augment=True,
            rng=np.random.default_rng(seed + index),
        )

        stem = f"{index:02d}_{Path(path).stem}"
        sf.write(str(out_dir / f"{stem}_before.wav"), legacy.T, sample_rate)
        sf.write(str(out_dir / f"{stem}_after.wav"), improved.T, sample_rate)

        report.append(
            {
                "source": source,
                "stem": stem,
                "source_duration_s": round(audio.shape[-1] / sample_rate, 2),
                "before": _measure(legacy, sample_rate),
                "after": _measure(improved, sample_rate),
            }
        )
        logger.info(
            "%s  clipped %.2f%% -> %.2f%%",
            stem,
            report[-1]["before"]["clipped_sample_pct"],
            report[-1]["after"]["clipped_sample_pct"],
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True,
        help="Local path or s3:// URI. Repeat for multiple clips.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./ab_demo"))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render(
        sources=args.source,
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        seed=args.seed,
        channels=args.channels,
    )


if __name__ == "__main__":
    main()
