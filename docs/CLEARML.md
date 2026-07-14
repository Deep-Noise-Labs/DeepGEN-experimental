# ClearML Experiment Tracking

ClearML is the **primary** experiment tracking backend for SynthGen. Weights & Biases remains available as an optional secondary logger.

## What is logged

| Item | Uploaded to ClearML fileserver? |
|------|----------------------------------|
| Hyperparameters (`Task.connect`) | No (metadata only) |
| Scalars (loss, LR, component losses, throughput) | No (metrics API) |
| Training config JSON artifact | Yes (small JSON) |
| Dataset manifest JSON | Yes (paths, sizes, hashes — **not** audio) |
| Checkpoint local path + step | No (parameters only by default) |
| Checkpoint `.pt` binaries | **No** unless `clearml_upload_checkpoints: true` |
| Generated audio / dataset media | **Never** |

Audio and other media stay on local or external storage. Dataset registration is metadata-only so the ClearML server does not grow with corpus size.

## Setup

```bash
# Install training extras (includes clearml)
uv sync --extra train

# Authenticate against your ClearML server (hosted or self-hosted)
clearml-init
```

Alternatively set environment variables:

```bash
export CLEARML_API_HOST="https://api.clear.ml"   # or your server
export CLEARML_WEB_HOST="https://app.clear.ml"
export CLEARML_FILES_HOST="https://files.clear.ml"
export CLEARML_API_ACCESS_KEY="..."
export CLEARML_API_SECRET_KEY="..."
```

Do **not** commit `~/clearml.conf` or API keys to the repository.

## Training

Enable ClearML via YAML:

```yaml
use_clearml: true
clearml_project: "synthgen-vae"
clearml_task_name: null          # defaults to synthgen-{stage}
clearml_tags: ["vae"]
clearml_dataset_name: "synthgen-data"
clearml_register_dataset: true
clearml_upload_checkpoints: false
```

Or via CLI:

```bash
uv run synthgen-train --config configs/default.yaml --clearml
uv run synthgen-train --config configs/default.yaml --clearml --clearml-project synthgen-dit

# Secondary WandB alongside ClearML
uv run synthgen-train --config configs/default.yaml --clearml --wandb
```

### Distributed (DDP)

Only **rank 0** creates a ClearML Task and reports metrics. Launch as usual:

```bash
uv run torchrun --nproc_per_node=4 -m synthgen.training.trainer \
  --config configs/default.yaml --clearml
```

### PyTorch auto-upload

Tasks are created with `auto_connect_frameworks={"pytorch": False}` so `torch.save` does not silently upload large checkpoints to the fileserver.

## Inference

```bash
uv run synthgen-generate \
  --prompt "warm analog pad" \
  --checkpoint ./checkpoints/dit/checkpoint-10000.pt \
  --output ./out.wav \
  --clearml
```

This logs prompt, sampling params, checkpoint path, and the **local** output path. Audio bytes are not uploaded.

## Dataset registry

When `clearml_register_dataset` is true, training scans `data_dir`, builds a JSON manifest (`relative_path`, `size_bytes`, `sha256`), and attaches it as the `dataset_manifest` artifact. A ClearML Dataset version may be created for lineage, but **audio files are never added via `add_files` / `sync_folder`**.

If `data_dir` is missing, a warning is logged and training continues.

## Troubleshooting

- **ClearML not configured**: training continues; a warning is logged and tracking becomes a no-op.
- **`clearml` not installed**: run `uv sync --extra train`.
- **Need checkpoint binaries in ClearML**: set `clearml_upload_checkpoints: true` (ask first in shared clusters — this uses fileserver space).
