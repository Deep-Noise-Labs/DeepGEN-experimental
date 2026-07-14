"""Metadata-only ClearML dataset registration.

Scans a local data directory and uploads a compact JSON manifest only.
Never uploads audio/media binaries to the ClearML fileserver.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Skip hashing file bodies beyond this size; still record size + path.
_HASH_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    if size > _HASH_SIZE_LIMIT_BYTES:
        # Hash size + mtime as a cheap fingerprint for huge files
        fingerprint = f"{path.name}:{size}:{path.stat().st_mtime_ns}".encode()
        return hashlib.sha256(fingerprint).hexdigest()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-serializable manifest for all files under data_dir."""
    root = Path(data_dir)
    files: list[dict[str, Any]] = []
    total_bytes = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "relative_path": rel,
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )

    manifest: dict[str, Any] = {
        "data_dir": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return manifest


def register_dataset_metadata(
    data_dir: str | Path,
    dataset_name: str,
    dataset_project: str,
    task: Any = None,
    dataset_cls: Any = None,
    dataset_version: str | None = None,
) -> dict[str, Any] | None:
    """
    Register dataset lineage on ClearML without uploading media.

    Creates a ClearML Dataset (metadata/versioning) and attaches only the JSON
    manifest as a task artifact. Does not call add_files / sync_folder.
    """
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        logger.warning(
            "Dataset directory %s is missing; skipping ClearML dataset registration.",
            root,
        )
        return None

    manifest = build_dataset_manifest(root)
    version = dataset_version or manifest["manifest_sha256"][:12]

    if dataset_cls is None:
        try:
            from clearml import Dataset as dataset_cls  # type: ignore
        except ImportError:
            logger.warning("clearml not installed; skipping dataset registration.")
            if task is not None:
                task.upload_artifact(name="dataset_manifest", artifact_object=manifest)
            return {"dataset_id": None, "manifest": manifest, "version": version}

    dataset_id: str | None = None
    try:
        dataset = dataset_cls.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_version=version,
            description=(
                "SynthGen metadata-only dataset registry. "
                "Audio files remain on local/external storage; "
                "only this JSON manifest is uploaded to ClearML."
            ),
        )
        dataset_id = getattr(dataset, "id", None)
        # Finalize without adding binary files — version record only.
        if hasattr(dataset, "finalize"):
            try:
                dataset.finalize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ClearML Dataset.finalize failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClearML Dataset.create failed: %s", exc)

    if task is not None:
        try:
            task.upload_artifact(name="dataset_manifest", artifact_object=manifest)
            if dataset_id is not None:
                task.set_parameter(f"Datasets/{dataset_name}", dataset_id)
            task.set_parameter(f"Datasets/{dataset_name}_version", version)
            task.set_parameter(f"Datasets/{dataset_name}_file_count", manifest["file_count"])
            task.set_parameter(f"Datasets/{dataset_name}_total_bytes", manifest["total_bytes"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to attach dataset manifest to task: %s", exc)

    return {
        "dataset_id": dataset_id,
        "manifest": manifest,
        "version": version,
    }
