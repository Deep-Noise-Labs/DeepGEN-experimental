"""
Sample Quality Bench -- run the quality metrics over a folder of audio.

Usage::

    uv run synthgen-eval --input ./audio/candidates
    uv run synthgen-eval --input ./audio/candidates --reference ./audio/targets
    uv run synthgen-eval --input ./out --json report.json

With ``--reference``, files are paired by filename and the comparative metrics
are reported alongside the reference-free ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from synthgen.eval.metrics import (
    QualityTarget,
    absolute_metrics,
    comparative_metrics,
    grade,
    pass_rate,
)

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file as ``(channels, samples)`` float64."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data.T, int(sr)


def find_audio(root: Path) -> list[Path]:
    """All audio files under ``root``, sorted by name."""
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def evaluate_folder(
    input_dir: Path,
    reference_dir: Path | None = None,
    target: QualityTarget | None = None,
) -> dict:
    """Evaluate every clip under ``input_dir``, optionally against references."""
    files = find_audio(input_dir)
    if not files:
        raise FileNotFoundError(f"No audio files found under {input_dir}")

    clips: list[dict] = []
    for path in files:
        audio, sr = read_audio(path)
        metrics = absolute_metrics(audio, sr)
        grades = grade(metrics, target)
        entry = {
            "file": str(path),
            "metrics": metrics,
            "grades": grades,
            "pass_rate": pass_rate(grades),
        }

        if reference_dir is not None:
            ref_path = reference_dir / path.name
            if ref_path.exists():
                ref_audio, ref_sr = read_audio(ref_path)
                if ref_sr != sr:
                    raise ValueError(
                        f"Sample-rate mismatch for {path.name}: {sr} vs {ref_sr}"
                    )
                entry["comparative"] = comparative_metrics(audio, ref_audio, sr)

        clips.append(entry)

    return {"clips": clips, "summary": summarise(clips)}


def summarise(clips: list[dict]) -> dict:
    """Aggregate per-clip results into means and per-criterion pass counts."""
    if not clips:
        return {}
    keys = clips[0]["metrics"].keys()
    means = {
        k: float(np.nanmean([c["metrics"][k] for c in clips]))
        for k in keys
    }
    criteria = clips[0]["grades"].keys()
    passes = {
        k: int(sum(bool(c["grades"][k]) for c in clips)) for k in criteria
    }
    return {
        "n_clips": len(clips),
        "mean_metrics": means,
        "criteria_passed": passes,
        "mean_pass_rate": float(np.mean([c["pass_rate"] for c in clips])),
    }


def format_report(result: dict) -> str:
    """Render a plain-text summary suitable for a terminal or a CI log."""
    summary = result["summary"]
    n = summary["n_clips"]
    lines = [
        "SynthGen Sample Quality Bench",
        "=" * 60,
        f"clips evaluated: {n}",
        f"mean pass rate:  {summary['mean_pass_rate'] * 100:.1f}%",
        "",
        "Criteria (clips passing / total)",
        "-" * 60,
    ]
    for name, count in summary["criteria_passed"].items():
        mark = "PASS" if count == n else "FAIL"
        lines.append(f"  {name:<20} {count:>3}/{n:<3}  {mark}")

    lines += ["", "Mean metrics", "-" * 60]
    for name, value in summary["mean_metrics"].items():
        lines.append(f"  {name:<26} {value:>12.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SynthGen Sample Quality Bench")
    parser.add_argument("--input", required=True, type=Path, help="Audio file or folder")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional folder of reference audio, paired by filename",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write full results as JSON")
    args = parser.parse_args()

    result = evaluate_folder(args.input, args.reference)
    print(format_report(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=float))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
