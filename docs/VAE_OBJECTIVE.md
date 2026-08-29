# The VAE objective: why it was changed

## Summary

SynthGen is a latent diffusion system. The DiT generates latents; the VAE
decoder turns latents into audio. **Whatever the decoder cannot reproduce, the
system cannot generate** - no amount of DiT capacity, data or sampling steps
recovers detail the decoder throws away. The VAE's reconstruction quality is a
hard ceiling on the product.

The VAE was trained against:

```
0.1 * L1(waveform) + 1.0 * MRSTFT(linear magnitude) + 1e-4 * KL
```

Every term in that objective is either phase-blind or nearly so. This document
records the measurement that showed the problem, and what replaced it.

## The measurement

Real audio, four controlled degradations, each RMS-matched to the reference so
gross level is never the variable being scored. Each degradation mimics a
specific way a magnitude-trained audio decoder is known to fail. Source: a real
48 kHz / 24-bit stereo acoustic guitar recording, resampled to 44.1 kHz.

Penalties are shown as a multiple of each objective's own penalty for the
control degradation (a mild 4th-order low-pass at 9 kHz), because the two
objectives are on different absolute scales and only their *rankings* are
comparable.

| Degradation | What it sounds like | Magnitude spectrum moved | Legacy objective | Proposed (deterministic half) |
|---|---|---|---|---|
| HF roll-off @ 9 kHz (control) | slightly dull | 67 dB | **1.00** | **1.00** |
| All-pass dispersion, ~41 ms | pluck becomes a smeared "boing" | **0.56 dB** | **0.22** | 0.39 |
| All-pass dispersion, ~14 ms | softened pick attack | 0.27 dB | 0.16 | 0.25 |
| Stereo side x0.15 | image collapses to near-mono | 5.2 dB | 0.11 | 0.92 |
| Attack slowed to ~30 ms | no snap | 15.0 dB | 0.11 | 0.39 |

The guitar recording is nearly mono to begin with (side/mid ~= 0.002), so it
understates the stereo case. On wide programme material - a produced stereo
track, side/mid ~= 1.09 - the same degradations give:

| Degradation | Legacy objective | Proposed |
|---|---|---|
| HF roll-off @ 9 kHz (control) | **1.00** | **1.00** |
| All-pass dispersion, ~41 ms | 0.34 | 0.46 |
| Stereo side x0.15 | 0.48 | **2.82** |
| Attack slowed to ~30 ms | 0.40 | **1.47** |

Reproduce both with `scripts/vae_objective_probe.py`. All figures are from the
shipped loss classes, RMS-matched, 2.0 s excerpts at 44.1 kHz.

### Reading the table

The dispersion row is the important one. It is a **pure all-pass filter**:
`|H(f)| = 1` at every frequency, so the magnitude spectrum is unchanged by
construction (the residual 0.56 dB is windowing, not signal). Only phase moves -
by up to 41 ms of frequency-dependent group delay, which is the difference
between a pick hitting a string and a swell.

The legacy objective rates that as **22% as costly** as a low-pass most
listeners would call the milder of the two defects. An objective is a statement
about what the model is allowed to do, and gradient descent will take anything
priced that cheaply. Smeared transients, a collapsed image and a dull top end
are not accidents of undertraining - they are inside the old optimum.

The proposed objective does not fully close the phase gap and is not claimed to:
0.22 to 0.39 is a ~1.8x relative reweighting, and no magnitude-domain loss can
do better in principle, because the magnitude spectrum genuinely has not moved.
Phase is what the discriminator is for. Where the deterministic half *does*
close the gap is stereo (0.11 to 0.92 on the guitar, 0.48 to 2.82 on the track)
and envelope shape (0.11 to 0.39, and 0.40 to 1.47).

That is the gap between model output and a Spitfire or Splice sample.

## What replaced it

```
0.1 * L1(waveform)
+ 0.25 * MRSTFT(linear magnitude)          # broadband anchor, weight reduced
+ 15  * multi-scale log-mel(mid/side)      # new
+ 1e-4 * KL
+ 1.0 * adversarial(hinge)                 # new, after warmup
+ 2.0 * feature-matching                   # new, after warmup
```

Four changes, each aimed at one blind spot.

**1. Multi-scale log-mel, replacing linear magnitude as the main term.**
A spectral-convergence term over linear magnitudes is a Frobenius norm, so it is
dominated by whichever bins carry the most energy - in practice the bottom two
octaves. Mel bands spread the loss budget the way hearing does. Five scales
(2048 down to 32-sample windows, 160 down to 5 bands) mean short windows
constrain transients while coarse band counts supervise overall spectral
balance. This is the DAC recipe, and it is what stops the decoder trading away
the top end to buy bass accuracy.

Each scale contributes both a linear-magnitude and a log-magnitude L1 term
(DAC's `mag_weight` / `log_weight`, both 1.0). Keeping the linear term was
checked rather than assumed: dropping it (`mag_weight=0`) *reduces* relative
sensitivity on every degradation measured - on the guitar, dispersion 0.39 to
0.31 and envelope 0.39 to 0.16; on the track, stereo 2.82 to 1.30 and envelope
1.47 to 0.61. The two terms are complementary, so both stay.

**2. Mid/side instead of L/R.** Measured independently on L and R, collapsing
the stereo image is nearly free: both channels move towards each other and each
stays close to its target. Measured on mid/side, the side channel becomes a
first-class target. Synth and sampled-instrument content is wide, and width is
the first thing a per-channel loss lets go.

**3. Adversarial + feature-matching terms.** This is the only part of the
objective that sees phase at all. Two discriminator banks, following
HiFi-GAN / EnCodec / DAC:

- `MultiPeriodDiscriminator` folds the waveform to 2D by period (2, 3, 5, 7, 11)
  so each sub-discriminator specialises in a different periodic structure. This
  is what recovers pitch stability and harmonic detail.
- `MultiResolutionSTFTDiscriminator` operates on the **complex** STFT - real and
  imaginary as channels, not magnitude - at five resolutions. Feeding magnitude
  here would reproduce the exact blindness the module exists to fix.

Both fold L/R to mid/side before folding channels into the batch, so the side
channel gets adversarial pressure of its own.

Feature matching (L1 between the discriminator's intermediate activations for
real and reconstructed audio) is what keeps this stable: it gives the generator
a dense signal pointing at the real distribution, rather than only the scalar
"fooled / not fooled" gradient.

**4. Stabilised Snake activation.** The previous Snake kept a single raw `alpha`
and computed `x + (1/(alpha + 1e-8)) * sin^2(alpha*x)`. Nothing constrained
`alpha` to stay positive, so a decoder whose alpha drifted through zero hit a
~1e8 gain and took the run out with a NaN. `alpha` and `beta` are now stored in
log space (BigVGAN's SnakeBeta), which makes both strictly positive by
construction and lets the network scale the periodic term without also changing
its frequency. Pre-existing checkpoints are upgraded transparently on load.

## Does it actually sound different?

A controlled A/B was run on CPU: same ~1M-parameter model, same real 44.1 kHz
stereo clips, same seed, same optimiser, same 4000 steps, each objective
rescaled to 1.0 at initialisation so the shared learning rate and gradient clip
mean the same thing for both. Only the loss differs. Reproduce with
`scripts/vae_objective_ab.py`.

This is a reconstruction-*ceiling* test at CPU scale: both runs sound rough, and
the result to read is the difference in character, not absolute fidelity.

Means over the four clips that converged (guitar pluck, guitar in a stereo room,
a stereo pair of measured reverb impulses, a spring impulse):

| Metric | Before | After | Note |
|---|---|---|---|
| Mid-channel correlation | **0.74** | 0.64 | near-neutral metric |
| Side-channel correlation | **0.50** | 0.29 | near-neutral metric |
| Log-spectral distance | **4.25 dB** | 6.04 dB | favours legacy by construction |
| 2-8 kHz energy error | -10.18 dB | **-4.92 dB** | nearer zero better |
| 8-16 kHz energy error | 2.88 dB | **2.37 dB** | nearer zero better |

The clearest single result is the stereo reverb pair, reference width 1.00:

| | Side correlation | Width | 2-8 kHz |
|---|---|---|---|
| Before | 0.93 | **0.26** | **-17.1 dB** |
| After | 0.90 | **1.04** | **-2.0 dB** |

The legacy objective reproduces the side channel's *shape* accurately - 0.93
correlation - but at **a quarter of its level**, and drops 17 dB of midrange. The
new objective holds essentially the same side correlation at the correct level.
That is the mid/side change doing exactly what it was added for, and it is
audible as image width.

**Read the whole table, though.** Averaged over the converged clips the legacy
objective reconstructs the waveform *more* accurately (mid correlation 0.74
against 0.64). The new objective buys spectral balance and stereo image with
some waveform precision. At CPU scale that trade is the honest summary, and
whether it holds at real scale is not established here.

Two caveats on the metrics themselves. Log-spectral distance is a
linear-magnitude distance, i.e. close to what the legacy objective optimises, so
the baseline is expected to win on it and it is not a neutral adjudicator.
Correlation is the closest thing to neutral available here, since both
objectives are spectral and neither optimises time-domain correlation directly.

Two earlier versions of this A/B produced wrong answers, both caught by
inspection rather than by the harness, and both now guarded in the script:

1. A large apparent stereo win that was actually **non-convergence**. On dense
   polyphonic material neither model reconstructed the clip at all (correlation
   ~0.03 for both), yet the new objective's output still scored a plausible
   stereo width - decorrelated content with roughly the right energy ratio.
   Width alone cannot distinguish that from a faithful image; width plus side
   correlation can. The script now reports correlation first and flags clips
   where both runs stay under 0.3.
2. A **handicap on the new objective**. Its loss is ~15x larger (mel_weight
   alone is 15) and both runs clipped gradients at the same threshold, so it was
   clipped on essentially every step and trained at a smaller effective step
   size. Hence the loss normalisation described above.

The adversarial half is deliberately **not** part of that A/B. A GAN needs far
more than a CPU-scale budget to become useful, and switching it on there would
measure the budget rather than the idea. It is implemented, unit-tested and
gated behind `adv_start_step`, but **it is not yet validated at scale** - that
needs a real stage-1 run on GPU. Treat the adversarial terms as the motivated,
untested half of this change until that run exists.

## Configuration

```yaml
# Reconstruction
vae_l1_weight: 0.1
vae_spectral_weight: 0.25      # legacy linear MRSTFT, kept as a broadband anchor
vae_mel_weight: 15.0           # multi-scale log-mel, the main term
vae_kl_weight: 1.0e-4
vae_mid_side: true

# Adversarial
adv_enabled: true
adv_start_step: 20000          # reconstruction-only warmup first
adv_weight: 1.0
fm_weight: 2.0
adv_mode: "hinge"              # or "lsgan"
disc_learning_rate: 1.0e-4
```

`legacy_vae_objective: true` restores the previous objective exactly, and
disables the discriminator. That is what the A/B compares against, and it is the
rollback switch if adversarial training misbehaves on a real run.

### Operational notes

- **Warm up first.** `adv_start_step` should sit after the reconstruction loss
  has flattened. Starting adversarial training against a decoder that still
  outputs mush wastes the discriminator's capacity on easy negatives.
- **Resuming.** The discriminator and its optimizer are saved in the checkpoint.
  Resuming a *pre-adversarial* checkpoint past `adv_start_step` would pit a
  trained generator against a fresh critic; the trainer warns when it detects
  this. Raise `adv_start_step` above the resumed step in that case.
- **Cost.** The discriminator adds roughly 55M parameters and a second
  forward/backward over the audio per step. Expect stage-1 steps to be ~2-2.5x
  slower once adversarial training kicks in. It only affects stage 1 - the DiT
  stage and inference are untouched.

## What this change does not address

Two further defects were found while measuring this, both real and both out of
scope here:

1. **Latents are not normalised before the DiT.** `SynthGen.compute_loss` feeds
   raw VAE means straight into `x_t = t*x_0 + (1-t)*noise`. Flow matching
   assumes data at roughly unit scale; if the VAE's latent std is far from 1 the
   noise schedule and CFG are both miscalibrated. Stable Diffusion's `0.18215`
   exists for exactly this reason. Fix is a running latent-std estimate stored
   on the VAE and applied on encode/decode.
2. **T5 padding pollutes cross-attention.** `T5TextEncoder.forward` computes
   `attention_mask` and then returns only `last_hidden_state`, so the DiT's
   cross-attention attends to all 256 padded positions. This directly costs
   prompt adherence. Fix is to return the mask and pass it into
   `MultiHeadAttention` as an additive attention bias.

Both are cheap and independently testable, and neither belongs in a change about
the reconstruction ceiling.
