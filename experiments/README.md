# Experiments

Self-contained, reproducible ablations. Each script is runnable on a laptop or a
CPU-only container and writes both audio and machine-readable metrics, so a
claim about audio quality can be listened to rather than argued about.

These are **not** training runs. They are sized to isolate one variable at a
fraction of production compute; read them as relative comparisons under an
identical budget.

## `vae_objective_ab.py` - Stage-1 objective A/B

Trains two Audio VAEs that are identical in architecture, data, data order,
seed, optimizer and step count, and differ only in the training objective:

| Arm | Objective |
|-----|-----------|
| `before` | L1 + multi-resolution STFT + KL |
| `after` | the above + multi-scale log-mel + hinge adversarial + feature matching |

It then reconstructs held-out audio through both and writes the results.

```bash
# Any directory of audio files works. Real instrument samples - one note per
# file, with the attack intact - show the difference most clearly.
uv run python experiments/vae_objective_ab.py \
    --audio-dir ./data/instrument_samples \
    --output-dir ./experiments/out \
    --steps 800

# Re-run a single arm (e.g. after changing only the adversarial settings)
uv run python experiments/vae_objective_ab.py \
    --audio-dir ./data/instrument_samples --arms after --steps 800
```

Outputs, under `--output-dir`:

- `audio/{clip}__original.wav`, `audio/{clip}__before.wav`, `audio/{clip}__after.wav`
  Sample-aligned, same length, so they can be dropped straight onto three DAW
  tracks and A/B'd.
- `results.json` - per-clip and aggregate metrics: mel distance, log-spectral
  distance, time-domain SNR, per-band dB error (0-1k / 1-4k / 4-11k / 11-22k),
  and onset-envelope correlation.
- `train_before.jsonl`, `train_after.jsonl` - per-step loss components.

### What to listen for

The interesting differences are not loudness or overall tone, they are:

- **Attack** - does the pick, hammer or mallet arrive as an event or as a swell?
- **Decay tail** - does the note ring out or dissolve into a wash?
- **Top end** - is there air above ~8 kHz, or does it stop dead?

The `transient_envelope_corr` and `band_db_error_air_11k_22k` metrics are the
numeric proxies for the first and third of those.

### Scale caveat

The default configuration is roughly 1/1000th of the compute of a real Stage-1
run: mono, 1-second crops, a 24-base-channel VAE, a few hundred optimizer steps.
The latent bitrate is matched to production (44100 / 256 x 32 dims mono is the
same 5512 floats/s as 44100 x 2 / 1024 x 64 dims stereo), but neither arm is
anywhere near converged. Both will sound like an early-training autoencoder.
The comparison between them is the result; the absolute quality is not.
