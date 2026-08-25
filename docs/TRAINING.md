# SynthGen Training Guidelines

This document provides comprehensive instructions for training the SynthGen text-to-audio model. The training process is divided into two distinct stages: training the Audio Variational Autoencoder (VAE), and training the Diffusion Transformer (DiT) using Conditional Flow Matching.

## Hardware Requirements

Training SynthGen requires significant computational resources. The recommended hardware specifications are:

| Component | Minimum Requirement | Recommended Specification |
|-----------|---------------------|---------------------------|
| **GPU** | 1x NVIDIA RTX 3090 (24GB VRAM) | 4x or 8x NVIDIA A100/H100 (80GB VRAM) |
| **RAM** | 64GB System RAM | 256GB System RAM |
| **Storage** | 2TB NVMe SSD | 10TB+ NVMe SSD |
| **CPU** | 16-core Processor | 64-core Processor |

For the recommended multi-GPU setup, the repository fully supports PyTorch Distributed Data Parallel (DDP).

## Dataset Preparation

Before beginning the training process, you must acquire and prepare the datasets. SynthGen is designed to train on a mixture of datasets to ensure diverse instrument and synthesizer generation capabilities.

1. **Download Datasets**: Use the provided download utility to acquire the datasets. We recommend starting with **AudioCaps** before scaling up to the full corpus.

   ```bash
   # Smoke download (few dozen clips)
   uv run synthgen-download --dataset audiocaps --output-dir ./data --max-samples 64

   # Full AudioCaps export from Hugging Face (OpenSound/AudioCaps → wav + metadata.jsonl)
   uv run synthgen-download --dataset audiocaps --output-dir ./data
   ```

   Layout:

   ```text
   data/audiocaps/
     metadata.jsonl   # {"file_name": "audio/000000.wav", "caption": "..."}
     audio/*.wav
   ```

2. **Verify Data**: The dataset loader automatically handles resampling to 44.1 kHz, stereo conversion, and amplitude normalization during the training loop. No manual pre-processing is required.

## Stage 1: Training the Audio VAE

The first stage involves training the Audio VAE to compress raw audio waveforms into a compact latent space. This step is critical, as the quality of the VAE dictates the maximum possible audio quality of the final generation.

The VAE compresses 44.1 kHz stereo audio by a factor of 2048x, mapping it to a 64-dimensional latent space. It is trained using a combination of L1 reconstruction loss, Multi-resolution STFT loss, and KL divergence loss.

### Configuration

Ready-made AudioCaps validation configs live in the repo:

- [`configs/vae_audiocaps.yaml`](../configs/vae_audiocaps.yaml) — Stage 1 (short budget)
- [`configs/dit_audiocaps.yaml`](../configs/dit_audiocaps.yaml) — Stage 2 (loads `vae_checkpoint`)

### Execution

```bash
# GPU install (CUDA 12.1)
uv sync --python 3.12 --extra train --extra download

# Smoke (single GPU)
uv run synthgen-train --config configs/vae_audiocaps.yaml --max-samples 64 --max-steps 50 --batch-size 2

# Multi-GPU AudioCaps VAE — only rank 0 reports to ClearML
uv run torchrun --nproc_per_node=4 -m synthgen.training.trainer \
  --config configs/vae_audiocaps.yaml --clearml
```

### Evaluation

Monitor training metrics in ClearML (see [CLEARML.md](CLEARML.md)). The key metric to watch is `spectral_loss`. Once the loss plateaus (typically around 300k-500k steps), the VAE is ready. You can test the reconstruction quality by encoding and decoding test audio files.

## Stage 2: Training the Diffusion Transformer (DiT)

Once the VAE is trained, you can proceed to train the DiT. The DiT learns to generate the latent representations conditioned on text prompts and timing information. We use Conditional Flow Matching, which enables faster inference than traditional DDPM.

During this stage, the VAE encoder and the T5 text encoder are frozen. Only the DiT parameters are updated.

### Configuration / Execution

Pass a Stage-1 checkpoint via `vae_checkpoint` (YAML) or `--vae-checkpoint`. Optimizer state from the VAE run is not loaded — only VAE weights.

```bash
# Use the highest available Stage-1 checkpoint (numeric retention keeps the last 3)
uv run torchrun --nproc_per_node=4 -m synthgen.training.trainer \
  --config configs/dit_audiocaps.yaml \
  --vae-checkpoint ./checkpoints/vae_audiocaps/checkpoint-9000.pt \
  --clearml

# Optional: generate a local wav (audio is never uploaded to ClearML)
uv run synthgen-generate --prompt "a dog barking" --duration 5.0 \
  --checkpoint ./checkpoints/dit_audiocaps/checkpoint-10000.pt
```

### Classifier-Free Guidance (CFG) Dropout

The training loop implements CFG dropout by randomly replacing the text conditioning embeddings with zeros for 10% of the batches. This enables the model to learn both conditional and unconditional generation, which is necessary for classifier-free guidance during inference.

### Timestep Sampling (Flow Matching)

Training timesteps are drawn from a **logit-normal** distribution by default (`timestep_sampling: "logit_normal"`, with `timestep_logit_mean: 0.0` and `timestep_logit_std: 1.0`): `t = sigmoid(n)` with `n ~ N(mean, std)`. Compared to uniform sampling, this concentrates roughly 73% of training batches on the intermediate timesteps (`0.25 < t < 0.75`) where the velocity target is hardest to predict and where sample quality is decided, instead of spending a large share of steps on the trivially easy near-noise and near-data extremes. This follows the rectified-flow training recipe validated in Stable Diffusion 3 ([arXiv:2403.03206](https://arxiv.org/abs/2403.03206)), where logit-normal sampling outperformed uniform sampling across all evaluated benchmarks. Set `timestep_sampling: "uniform"` to reproduce the legacy behaviour; the sampled distribution only affects training - inference integrates the ODE on a fixed step grid and is unchanged.

## Monitoring and Troubleshooting

Experiment tracking defaults to **ClearML** (`--clearml` or `use_clearml: true`). Optional Weights & Biases can run as a secondary backend with `--wandb`. Full setup, space policy (no audio uploads), and env vars are documented in [CLEARML.md](CLEARML.md).

Logged scalars include `loss`, component losses when present (e.g. `spectral_loss`, `l1_loss`, `kl_loss`), `learning_rate`, `steps_per_second`, and `epoch`.

### Common Issues

1. **Out of Memory (OOM) Errors**: If you encounter CUDA OOM errors, reduce the `batch_size` and proportionally increase the `gradient_accumulation_steps` to maintain the same effective batch size.
2. **Loss Spikes or NaNs**: This is typically caused by unstable gradients. Ensure you are using `bf16` rather than `fp16` if your hardware supports it. Alternatively, reduce the learning rate or increase the warmup steps.
3. **Slow Data Loading**: If GPU utilization is low, increase the `num_workers` parameter in the configuration file to accelerate data loading.

### Integrity Checks

We strongly recommend implementing automated integrity checks to ensure the data pipeline and model architectures remain functional during development. You can run the test suite to verify the components:

```bash
uv run pytest tests/
```

This will validate the tensor shapes, loss computations, and dataset loading mechanisms.
