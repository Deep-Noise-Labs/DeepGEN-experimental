# Stage-1 fidelity: what decides how good the model can ever sound

Everything the model emits passes through the VAE decoder. The DiT can produce a
perfect latent and the output will still be whatever that decoder can render, so
the decoder sets a hard ceiling on the whole system. This page describes the
three changes that raise it, why each is needed, and what they cost.

The bar we are aiming at is commercial sample-library and synthesiser quality
(Spitfire, Serum, Splice). That bar is not about "sounding like music" -- it is
about clean transients, coherent phase, harmonics that belong to the fundamental,
and no artefacts that a producer would hear on a good pair of monitors.

## 1. Anti-aliased nonlinearities in the decoder

**The problem.** `Snake` is `x + (1/alpha) * sin^2(alpha * x)`. Like any pointwise
nonlinearity it generates harmonics of whatever goes into it, and it generates a
lot of them. Harmonics above the Nyquist frequency of the sample grid they are
created on do not vanish; they fold back down into the audible band at
`sample_rate - f`, landing at frequencies that have no harmonic relationship to
the fundamental.

This matters more here than in most audio models because of what we generate.
Inharmonic partials hide inside dense, noisy material -- but on a sustained synth
pad, a bell, a bowed string, a clean lead, they are exposed, and the ear picks
them out instantly as metallic grit or a "digital" edge. The decoder applies
Snake at roughly thirty sites, at every rate from the latent grid up to 44.1 kHz,
and each one contributes.

No amount of training removes this. It is a property of running a
harmonic-generating function on a grid too coarse for its own output bandwidth.

**The fix.** BigVGAN's anti-aliased multi-periodicity block: upsample 2x with a
Kaiser-windowed sinc, apply the nonlinearity there, then low-pass and decimate.
The harmonics are created on a grid that can hold them and filtered out before
they can fold. Implemented in `synthgen/model/activations.py`.

**Measured.** An 8 kHz tone through Snake produces a 4th harmonic at 32 kHz that
folds to 12.1 kHz. That fold drops by **36 dB** with the sandwich in place, while
the legitimate 2nd harmonic at 16 kHz is untouched (within 1 dB). See
`tests/test_activations.py::test_aliasing_is_suppressed`.

**Cost.** Each wrapped activation runs on 2x the samples plus two short grouped
FIR convolutions. Applied to the decoder only -- the encoder feeds a learned
latent rather than a loudspeaker, so the same compute buys far less there. Turn
either on or off with `vae_anti_aliased_decoder` / `vae_anti_aliased_encoder`.

## 2. `SnakeBeta`

Plain Snake ties the frequency of its periodic component to its magnitude
through the same `alpha`. A channel cannot ask for a fast periodic component
without also asking for a quiet one, which is a strange constraint to impose on
a decoder whose job includes bright synthetic timbres. `SnakeBeta` splits them:
`x + (1/beta) * sin^2(alpha * x)`.

Both parameters are stored in log space, which also removes the
`1/(alpha + 1e-8)` blow-up plain Snake suffers if `alpha` is driven towards zero.

Cost: one extra parameter per channel (+0.01M over the whole VAE).

## 3. A perceptual objective, and a critic

**The problem.** The Stage-1 objective was L1 on the waveform plus a
magnitude multi-resolution STFT term. Two things follow from that.

First, it is weighted wrong. A linear-frequency STFT allocates gradient by bin
count, and half of a linear spectrum's bins sit in the top octave, 11--22 kHz,
where the ear has almost no frequency resolution. The two or three bins covering
20--200 Hz -- where the fundamental of a bass patch or a piano's bottom register
lives -- contribute almost nothing. Injecting the same-amplitude error at 100 Hz
and at 15 kHz into the same signal, the linear STFT ranks the low-frequency error
as **1.2x** worse; the mel criterion ranks it **11.3x** worse
(`tests/test_losses.py::test_weights_low_frequencies_more_than_a_linear_stft_does`).

Second, and more fundamentally: both terms are magnitude criteria averaged over a
spectrogram. Neither can tell a crisp transient from a smeared one, or coherent
phase from incoherent phase. The optimum they point at is the conditional mean of
the data, which is the blurry, phasey, slightly underwater sound that every
regression-only audio autoencoder converges to. You cannot reach a commercial
quality bar with an objective that is blind to the difference.

**The fix.** Two parts.

- `MultiResolutionMelLoss` (`synthgen/training/losses.py`): L1 on log-mel at five
  time/frequency resolutions, following the Descript Audio Codec recipe. The
  short windows (64--256 samples) resolve transients that a 2048-sample window
  smears across 46 ms; the long windows resolve steady-state partials. Weighted
  at 15x, it becomes the dominant reconstruction term.

- Adversarial training (`synthgen/training/discriminators.py`). Two critic
  families, because they see different things:
  - **Multi-period** (HiFi-GAN): reshapes the waveform by coprime periods and
    convolves, so it sees per-cycle waveshape and phase coherence within a
    cycle. This is what makes a saw sound like a saw rather than a band-limited
    approximation of one.
  - **Multi-resolution complex STFT** (EnCodec/DAC): convolves over real and
    imaginary parts rather than magnitude, so it can penalise phase incoherence
    directly -- exactly the thing the magnitude loss cannot see.

  Plus a feature-matching term, weighted 2x the adversarial term. Feature
  matching is the dense, well-conditioned half of the signal; the scalar
  adversarial term on its own is happy to be satisfied by artefacts that fool the
  critic without resembling the target.

**Cost.** The critic bank is ~41M parameters and roughly doubles Stage-1 step
time. It is discarded after training -- it adds nothing to inference.

## Running it

```bash
uv run synthgen-train --config configs/vae_gan.yaml --clearml
```

Key settings in that config:

| Setting | Default | Why |
|---|---|---|
| `disc_start_step` | 20000 | Reconstruction-only warm-up. Introducing critics against a randomly initialised decoder just teaches them to detect noise. |
| `feature_matching_weight` | 2.0 | Higher than `adv_weight`; it is what keeps the run from diverging. |
| `disc_update_every` | 1 | 1:1 critic/generator updates per micro-batch. Raise towards `gradient_accumulation_steps` if `disc_loss` collapses to ~0. |
| `mel_weight` | 15.0 | Dominant reconstruction term. |
| `max_duration` | 5.0 | Adversarial training is memory-hungry and critics care about local structure. 15s clips are for Stage 2. |

## What to watch

- `disc_loss` pinned near 0 means the critic has won and the generator is getting
  no usable gradient. Raise `disc_update_every` or lower `disc_learning_rate`.
- `adv_loss` diverging upward with `mel_loss` flat means the generator is chasing
  the critic at the expense of reconstruction. Lower `adv_weight`.
- `mel_loss` is the number that tracks perceived quality. Watch it, not `loss`.

## Checkpoint compatibility

This changes the Stage-1 parameter set: `SnakeBeta` adds a `beta` per channel and
the anti-aliased wrapper nests the activation one level deeper in the module
tree. Checkpoints from before this change will not load cleanly into the new
default architecture.

To load one, construct with the old settings:

```yaml
vae_activation: "snake"
vae_anti_aliased_decoder: false
```

The Stage-1 checkpoints that exist today are from pipeline-validation runs
(10k steps on AudioCaps), so in practice this costs nothing. Stage-2 configs
must use the same two settings as the Stage-1 checkpoint they load.
