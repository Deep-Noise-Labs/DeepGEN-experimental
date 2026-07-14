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

1. **Download Datasets**: Use the provided download utility to acquire the datasets. We recommend starting with smaller datasets for testing before scaling up to the full corpus.
   
   ```bash
   # Download specific datasets
   uv run synthgen-download --dataset audiocaps --output-dir ./data
   uv run synthgen-download --dataset nsynth --output-dir ./data
   
   # Or download all available datasets (requires significant time and storage)
   uv run synthgen-download --dataset all --output-dir ./data
   ```

2. **Verify Data**: The dataset loader automatically handles resampling to 44.1 kHz, stereo conversion, and amplitude normalization during the training loop. No manual pre-processing is required.

## Stage 1: Training the Audio VAE

The first stage involves training the Audio VAE to compress raw audio waveforms into a compact latent space. This step is critical, as the quality of the VAE dictates the maximum possible audio quality of the final generation.

The VAE compresses 44.1 kHz stereo audio by a factor of 2048x, mapping it to a 64-dimensional latent space. It is trained using a combination of L1 reconstruction loss, Multi-resolution STFT loss, and KL divergence loss.

### Configuration

Create a configuration file `configs/vae_train.yaml`:

```yaml
stage: "vae"
batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 1e-4
max_steps: 500000
warmup_steps: 5000
mixed_precision: "bf16"
checkpoint_dir: "./checkpoints/vae"
use_clearml: true
clearml_project: "synthgen-vae"
clearml_upload_checkpoints: false
```

### Execution

Launch the training script:

```bash
# Single GPU (ClearML primary tracking)
uv run synthgen-train --config configs/vae_train.yaml --clearml

# Multi-GPU (e.g., 4 GPUs) — only rank 0 reports to ClearML
uv run torchrun --nproc_per_node=4 -m synthgen.training.trainer --config configs/vae_train.yaml --clearml
```

### Evaluation

Monitor training metrics in ClearML (see [CLEARML.md](CLEARML.md)). The key metric to watch is `spectral_loss`. Once the loss plateaus (typically around 300k-500k steps), the VAE is ready. You can test the reconstruction quality by encoding and decoding test audio files.

## Stage 2: Training the Diffusion Transformer (DiT)

Once the VAE is trained, you can proceed to train the DiT. The DiT learns to generate the latent representations conditioned on text prompts and timing information. We use Conditional Flow Matching, which enables faster inference than traditional DDPM.

During this stage, the VAE encoder and the T5 text encoder are frozen. Only the DiT parameters are updated.

### Configuration

Create a configuration file `configs/dit_train.yaml`:

```yaml
stage: "dit"
batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 1e-4
max_steps: 1000000
warmup_steps: 10000
mixed_precision: "bf16"
checkpoint_dir: "./checkpoints/dit"
use_clearml: true
clearml_project: "synthgen-dit"
clearml_upload_checkpoints: false
```

### Execution

Launch the training script:

```bash
# Single GPU (ClearML primary tracking)
uv run synthgen-train --config configs/dit_train.yaml --clearml

# Multi-GPU (e.g., 8 GPUs) — only rank 0 reports to ClearML
uv run torchrun --nproc_per_node=8 -m synthgen.training.trainer --config configs/dit_train.yaml --clearml
```

### Classifier-Free Guidance (CFG) Dropout

The training loop implements CFG dropout by randomly replacing the text conditioning embeddings with zeros for 10% of the batches. This enables the model to learn both conditional and unconditional generation, which is necessary for classifier-free guidance during inference.

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
