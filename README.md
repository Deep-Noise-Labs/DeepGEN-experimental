# SynthGen-Experimental

A text-to-sample generation model for synthesizing short audio sounds (3–15 seconds) from natural language descriptions. SynthGen generates both classical synthesizer timbres (pads, leads, bass, bells, and other synthetic textures) and samples of real instruments (guitar, piano, drums, strings, etc.).

## Architecture Overview

SynthGen employs a **Latent Diffusion Transformer (DiT)** architecture with **Conditional Flow Matching** for efficient and high-quality audio generation. The system consists of three main components:

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Audio VAE** | Variational autoencoder that compresses raw waveforms into a compact latent space | ~80M |
| **Text Encoder** | Frozen T5-base encoder that produces text conditioning embeddings | ~110M |
| **DiT (Flow Matching)** | Transformer-based generative model operating in latent space | ~350M |

The model generates **44.1 kHz stereo audio** of variable length (3–15 seconds) conditioned on text prompts describing the desired sound.

## Architecture Rationale

The architecture draws from state-of-the-art research in text-to-audio generation:

**Why Latent Diffusion over Autoregressive?** Autoregressive models (e.g., MusicGen) generate tokens sequentially, making them slow for high-sample-rate audio. Latent diffusion operates in a compressed space, enabling parallel generation of the entire audio sequence.

**Why DiT over U-Net?** Diffusion Transformers scale better with compute and data compared to U-Net architectures. They also handle variable-length sequences more naturally through attention mechanisms.

**Why Flow Matching over DDPM?** Conditional Flow Matching learns straighter transport paths from noise to data, enabling fewer sampling steps at inference time (as few as 10–25 steps vs. 50–200 for DDPM) while maintaining quality.

**Why T5-base?** T5-base provides a strong text understanding backbone that has been validated across multiple audio generation systems (Stable Audio, AudioLDM). It balances quality with computational efficiency.

**Why the VAE objective matters most.** The decoder is a hard ceiling on the whole system: whatever it cannot reproduce, the model cannot generate, however good the DiT gets. The VAE is trained against a multi-scale log-mel loss computed on mid/side, plus adversarial and feature-matching terms against multi-period and complex-STFT discriminators. A magnitude-only objective is phase-blind - it rates a 41 ms transient smear as roughly a fifth as costly as a mild 9 kHz roll-off - and smeared transients are most of what separates model output from a Spitfire or Splice sample. See [docs/VAE_OBJECTIVE.md](docs/VAE_OBJECTIVE.md) for the measurement and the A/B.

## Key Features

- Text-conditioned generation of short audio samples (3–15 seconds)
- 44.1 kHz stereo output for production-quality audio
- Support for both synthetic and acoustic instrument timbres
- Variable-length generation with timing conditioning
- Efficient inference via flow matching (10–25 steps)
- Triton Inference Server deployment support

## Repository Structure

```
synthgen-experimental/
├── synthgen/
│   ├── model/           # Neural network architecture
│   │   ├── vae.py       # Audio VAE (encoder + decoder)
│   │   ├── discriminator.py # Multi-period + complex-STFT discriminators (VAE stage)
│   │   ├── dit.py       # Diffusion Transformer
│   │   ├── text_encoder.py  # T5-based text conditioning
│   │   └── synthgen.py  # Full model assembly
│   ├── data/            # Dataset handling
│   │   ├── download.py  # Dataset download scripts
│   │   ├── dataset.py   # PyTorch dataset classes
│   │   └── preprocessing.py  # Audio preprocessing
│   ├── training/        # Training logic
│   │   ├── trainer.py   # Training loop
│   │   ├── losses.py    # Loss functions
│   │   └── scheduler.py # Learning rate schedulers
│   ├── inference/       # Inference utilities
│   │   └── generate.py  # Generation pipeline
│   ├── tracking/        # ClearML (primary) / WandB experiment tracking
│   └── utils/           # Shared utilities
│       └── audio.py     # Audio I/O utilities
├── tests/               # Unit and integration tests
├── docs/                # Documentation
│   ├── TRAINING.md      # Training guidelines
│   ├── VAE_OBJECTIVE.md # Why the VAE loss is what it is (with measurements)
│   ├── CLEARML.md       # ClearML setup and upload policy
│   └── TRITON_INFERENCE.md  # Triton deployment guide
├── scripts/             # Reproducible experiments
│   ├── vae_objective_probe.py  # What can the objective actually hear?
│   └── vae_objective_ab.py     # Controlled A/B of the two objectives
├── configs/             # Configuration files
├── uv.lock              # Reproducible dependency lockfile
├── pyproject.toml       # Project metadata and uv configuration
```

## Quick Start

This project uses [`uv`](https://github.com/astral-sh/uv) as the primary Python package and project manager.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies and setup environment
# For CPU-only environments (e.g., CI/CD, local testing)
uv sync --all-extras

# For GPU environments (CUDA 12.1)
uv sync --all-extras --index "https://download.pytorch.org/whl/cu121"

# 3. Download datasets
uv run synthgen-download --dataset all --output-dir ./data

# 4. Train the model with ClearML tracking (see docs/TRAINING.md and docs/CLEARML.md)
clearml-init   # once per machine
uv run synthgen-train --config configs/default.yaml --clearml

# 5. Generate audio (optional ClearML param logging; audio stays local)
uv run synthgen-generate --prompt "warm analog pad with slow attack and reverb" --duration 8.0 --checkpoint ./checkpoints/dit/checkpoint-10000.pt
```

Experiment tracking uses **ClearML** as the primary destination. Dataset lineage is registered as a lightweight JSON manifest only — audio is never uploaded to the ClearML fileserver. See [docs/CLEARML.md](docs/CLEARML.md).

## Datasets

SynthGen is trained on a diverse mixture of audio-text datasets. See `synthgen/data/download.py` for automated download scripts.

| Dataset | Size | Description |
|---------|------|-------------|
| AudioCaps | ~50K clips | Human-annotated captions for AudioSet clips |
| WavCaps | ~400K clips | ChatGPT-assisted weakly-labelled audio captions |
| NSynth | ~306K notes | Musical notes with pitch, timbre, and envelope labels |
| MusicCaps | ~5.5K clips | Expert-written music descriptions |
| FSD50K | ~51K clips | Freesound clips with AudioSet ontology labels |
| LAION-Audio-630K | ~633K pairs | Large-scale audio-text pairs |
| Clotho | ~7K clips | Audio captioning with 5 captions per clip |

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 12.0+ (for training)
- 24GB+ VRAM recommended for training
- 8GB+ VRAM for inference

## License

This project is released under the MIT License. Individual datasets may have their own licenses — please check each dataset's terms before use.

## References

- [Stable Audio Open](https://arxiv.org/abs/2407.14358) — Latent diffusion for audio generation
- [Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) — Structured text-to-sample generation
- [FlashAudio](https://arxiv.org/abs/2410.12266) — Rectified flows for text-to-audio
- [AudioLDM 2](https://arxiv.org/abs/2308.05734) — Holistic audio generation framework
- [MusicGen](https://arxiv.org/abs/2306.05284) — Autoregressive music generation
