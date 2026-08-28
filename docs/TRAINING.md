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

The first stage involves training the Audio VAE to compress raw audio waveforms into a compact latent space. This step is critical, as the quality of the VAE dictates the maximum possible audio quality of the final generation. The DiT only ever produces latents, so anything the decoder cannot render is unrecoverable no matter how good Stage 2 becomes.

The VAE compresses 44.1 kHz stereo audio by a factor of 1024x (strides `4, 4, 8, 8`), mapping it to a 64-dimensional latent space.

### The Stage-1 objective

The VAE is trained with a five-term objective:

| Term | Config key | Default | What it is for |
|------|-----------|---------|----------------|
| Waveform L1 | `vae_l1_weight` | 0.1 | Absolute amplitude and DC correctness |
| Multi-resolution STFT | `vae_spectral_weight` | 1.0 | Linear-frequency spectral magnitude fidelity |
| Multi-scale log-mel | `vae_mel_weight` | 15.0 | Perceptual weighting: log-frequency band pooling with a bounded log floor |
| KL divergence | `vae_kl_weight` | 1e-4 | Keeps the latent close to a unit Gaussian so the DiT has a well-conditioned target |
| Hinge adversarial + feature matching | `vae_adv_weight`, `vae_fm_weight` | 1.0, 2.0 | Removes the conditional-mean blur that no regression loss can avoid |

The adversarial terms matter more than their weights suggest. A purely reconstructive objective is a conditional-mean estimator: many waveforms share a magnitude spectrum, and an L-p distance between them is minimised by their *average*. Averaging over phase is what smears transients, hollows the stereo image, and gives regression-trained audio autoencoders their characteristic "under water" quality. Every production neural codec (SoundStream, EnCodec, DAC, the Stable Audio autoencoder) is trained adversarially for exactly this reason.

The critic is a multi-period discriminator (HiFi-GAN) plus a multi-resolution complex-STFT discriminator with per-frequency-band sub-critics (EnCodec/DAC). See `synthgen/training/discriminator.py`.

Adversarial training is on by default. It is engaged only after `disc_start_step` optimizer steps, so the decoder gets a purely reconstructive warm-up first - an adversarial gradient against a decoder that still outputs noise is destructive rather than informative. To ablate it:

```bash
uv run synthgen-train --config configs/vae_audiocaps.yaml --no-adversarial
```

`experiments/vae_objective_ab.py` runs the two objectives head to head on the same data, seed and step count and writes the reconstructions out as audio.

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

Monitor training metrics in ClearML (see [CLEARML.md](CLEARML.md)). Watch `mel_loss` as the headline reconstruction number - it tracks perceived quality more closely than `spectral_loss`, which is dominated by bins far below the threshold of hearing. Once it plateaus (typically around 300k-500k steps), the VAE is ready. Test reconstruction quality by encoding and decoding held-out audio and listening to it; no scalar substitutes for that.

Once the critic is engaged, `disc_loss` should settle near 1.0-1.5 and stay there. Two failure modes to watch for:

- `disc_loss` collapsing towards 0 means the critic has won outright and the generator gradient has gone flat. Lower `disc_learning_rate` or raise `disc_start_step`.
- `adv_loss` diverging while `mel_loss` climbs means the adversarial term is overpowering reconstruction. Lower `vae_adv_weight`; `vae_fm_weight` can usually be raised in its place, since feature matching supplies the same perceptual signal with a far better conditioned gradient.

### A step is an optimizer step

`max_steps`, `warmup_steps`, `save_every_steps` and `disc_start_step` all count **optimizer** steps. One optimizer step consumes `gradient_accumulation_steps` micro-batches, so the config above sees `10000 x 8 = 80000` clips. (Before this was aligned, `global_step` counted micro-batches while the LR scheduler was stepped once per optimizer step, so a cosine schedule configured for `max_steps` only traversed `1/gradient_accumulation_steps` of its curve and the learning rate never annealed. If you are resuming a run started before that change, its step counter means something different.)

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
