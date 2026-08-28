# The Stage-1 VAE objective

## Why this file exists

SynthGen is a two-stage system. The DiT generates latents; the VAE decoder
turns them into audio. **The decoder is a hard ceiling on final quality** — no
amount of DiT training, data or sampling steps can render detail the decoder
cannot produce. If the target is Spitfire/Serum/Splice-grade output, the
Stage-1 objective is the single highest-leverage thing in this repository.

The objective the repository started with was:

```
0.1 · L1(waveform) + 1.0 · MRSTFT(2048, 1024, 512, 256) + 1e-4 · KL
```

This document records what was wrong with it, what replaced it, and — because
several of the obvious-sounding arguments turned out not to survive
measurement — what the evidence actually shows.

Everything below is reproducible:

```bash
python -m experiments.vae_objective_ablation probe --out runs/probe
python -m experiments.vae_objective_ablation train --out runs/train --steps 900
```

## What was wrong

**1. The analysis windows were too short for 44.1 kHz.**
A 2048-point FFT gives 21.5 Hz bins. A 41 Hz bass fundamental lands in bin 2,
and a two-semitone error down there fits inside a third of one bin. The
objective could barely see bass *tuning* at all. The ladder now starts at 8192
(5.4 Hz bins) and still ends at 256 (5.8 ms) so transients stay sharp.

**2. Linear frequency weighting.**
Of the 1025 bins in a 2048-point FFT at 44.1 kHz, half sit above 11 kHz. The
loss spent most of its capacity on the top octave and almost none on
100 Hz – 1 kHz, where musical fundamentals and formants live. A multi-scale
log-mel term rebalances this; measured on broadband material it moves the
low-octave/high-octave balance by about 2.6x relative to the linear loss.

**3. Batch-global normalisation.**
`torch.norm(target - pred, 'fro') / torch.norm(target, 'fro')` was computed over
the whole flattened batch, so one loud item dominated the spectral-convergence
term and a quiet item — a soft pad, a release tail — contributed almost
nothing. Normalisation is now per item.

**4. No magnitude floor.**
`log(mag + 1e-8)` maps digital silence to -18.4, so numerical noise in
inaudible bins produced enormous log errors. Worse, the spectral-convergence
denominator was unbounded: a silent item divides by ~0. Measured: a real clip
with added noise scores 5.3 under the old loss; a silent target against a
near-silent prediction scores **915 251** — roughly 170 000x higher. Every clip
containing a gap or a decayed release tail is a training spike waiting to
happen. Magnitudes are now window-normalised (so one floor means the same thing
at every FFT size) and clamped, and the convergence denominator is floored too;
the same silent case now scores 0.004.

**5. Stereo was never constrained.**
Channels were folded into the batch dimension, so L and R were scored
independently and the *relationship* between them — the stereo image — was
free. The cheapest way to reduce a per-channel loss is to make both channels
more alike. A mid/side term now makes width part of the objective.

**6. No adversarial term.**
The docstring promised "optional adversarial loss"; none existed. See below —
this is the one problem no amount of reweighting can fix.

## What replaced it

| Term | Weight | What it buys |
|---|---|---|
| `l1_loss` | 0.1 | waveform alignment |
| `spectral_loss` | 1.0 | multi-resolution STFT, 8192 → 256, per-item, floored |
| `mel_loss` | 1.0 | perceptual (mel) frequency weighting |
| `stereo_loss` | 0.25 | mid/side coherence |
| `kl_loss` | 1e-4 | latent regularisation |
| `adversarial_loss` | 0.1 | plausible detail, via a complex-STFT critic |
| `feature_matching_loss` | 2.0 | dense gradient from the critic |

The critic (`synthgen/model/discriminator.py`) is a bank of 2-D convolutional
discriminators over the complex STFT at five resolutions, trained with a hinge
loss. It is **training-only**: never exported to Triton, never used at
inference. It switches on at `adv_start_step` (default 20 000) because an
untrained decoder gives the critic a trivially easy job and the resulting
gradients are noise.

Passing no `discriminator=` to `VAELoss.forward` keeps the objective purely
reconstructive, so CPU smoke runs and CI are unaffected.

## What the evidence actually shows

### The reweighting: measured sensitivity reallocation

`experiments/vae_objective_ablation.py probe` applies controlled, audible
degradations to five real clips and scores each under both objectives. Absolute
scores are not comparable across two objectives with different term counts, so
what is reported is each objective's **share of sensitivity** — where it spends
its attention.

Mean share across the five sources, spectral terms only:

| Degradation | Old objective | New objective | Change |
|---|---|---|---|
| Bass detuned ~2 semitones | 5.9% | 7.1% | **+21%** |
| Bass rolled off below 80 Hz | 6.1% | 7.4% | **+21%** |
| Transients smeared | 28.8% | 30.6% | +6% |
| Stereo image collapsed to mono | 19.2% | 27.9% | **+46%** |
| Top-octave phase randomised | 44.4% | 35.7% | **-20%** |

That is the intended trade: attention moves out of the top octave and into bass
tuning, bass weight and the stereo image.

**An honest note on what this is not.** Normalised against a common unit — each
objective's own response to a 1 dB broadband level error — most individual
degradations score *lower* under the new objective, because a broadband level
change moves every one of its terms and so inflates that reference. The claim
supported by the data is reallocation of sensitivity, not a uniform increase in
it.

### The critic: why reweighting cannot be enough

The probe includes a matched pair scored against the same noise-like target:

- **`texture_redraw`** — a random all-pass. Every channel's magnitude spectrum
  and the mid/side relationship are preserved *exactly*; only phase changes. On
  noise-like material this is a different realisation of the same texture, and
  a listener cannot reliably tell it from the original. (Phase randomisation
  raises crest factor, so the exported group is scaled by one common factor to
  keep it out of the clamp — a uniform gain leaves both scores unchanged.)
- **`spectral_blur`** — magnitudes smoothed in frequency, phase kept. Audibly
  duller. This is what "predict the conditional mean" sounds like.

| Candidate | Old objective | New objective |
|---|---|---|
| `texture_redraw` (sounds the same) | 1.36 | 2.07 |
| `spectral_blur` (audibly duller) | **0.72** | **1.14** |

Both objectives score the dull candidate **1.8–1.9x better** than the
perceptually equivalent one. This is the mean-seeking problem, and it is
structural: a regression loss is minimised by the average of all plausible
outputs, and the average of many plausible textures is a dull one. No
reweighting of a regression objective can fix it. A critic can, because it
scores *plausibility* rather than proximity to an average — which is why every
production audio autoencoder (EnCodec, DAC, Stable Audio) has one.

### Reduced-scale training ablation

`... train` trains the same small AudioVAE twice — identical architecture,
seed, data, step count and optimiser — changing only the objective, then
reports third-party metrics (SI-SDR, band-limited log-spectral distance,
envelope error, stereo width) that neither arm optimises directly.

The run used for the numbers below:

```bash
python -m experiments.vae_objective_ablation train --out runs/train \
  --sources 3 --seconds 1.0 --crops-per-source 1 --batch-size 3 \
  --steps 2000 --lr 2e-3 --strides 2,2,4,4
```

Read it for what it is: **an objective ablation at reduced scale, in the
overfit regime** (reconstruction is measured on the clips it trained on,
because the question is what the objective preserves, not how the model
generalises). The bottleneck is 64x, not the production VAE's 1024x, because
that is what converges on a CPU in minutes. It is not a quality claim about the
production model, and the adversarial terms are **not** exercised at this
scale, since a GAN needs far more steps to say anything.

### The learning-rate control

The obvious objection: the new objective has more terms and so a larger raw
gradient at equal reconstruction quality, so running both arms at one shared
learning rate might compare step sizes rather than objectives. Clipping at norm
1.0 absorbs much of that, but not all of it. `experiments/lr_control.py` sweeps
both objectives instead of arguing about it — 400 steps per point, SI-SDR after:

| Learning rate | Current objective | New objective |
|---|---|---|
| 5e-4 | -27.0 dB | -9.5 dB |
| 1e-3 | -19.5 dB | -2.5 dB |
| **2e-3** | **-16.0 dB** | **+3.5 dB** |
| 4e-3 | diverged | diverged |

The new objective is ahead at every rate, and at its *worst* rate (-9.5 dB) it
is still 6.5 dB ahead of the old objective's *best* (-16.0 dB). Both peak at
2e-3, so the head-to-head runs there and the step-size confound is controlled
rather than assumed away.

Note what this does **not** show: both objectives diverge to NaN at 4e-3. The
floors added to the new objective bound the silent-input case measured above;
they do not make it stable at an arbitrary learning rate, and nothing here says
they do.

## Practical notes

- `vae_adversarial: false` (or `--no-adversarial`) reverts to pure
  reconstruction.
- `adv_start_step` should be roughly the point where reconstruction loss
  flattens. On the short `configs/vae_audiocaps.yaml` budget it is set to 4000.
- The critic is checkpointed alongside the model. Resuming without it would
  restart adversarial training from scratch against an already-converged
  decoder, which is worse than not resuming at all.
- The longer FFT ladder costs real time: six resolutions up to 8192 against
  four up to 2048. On the 15 s clips in `configs/vae_audiocaps.yaml` this is the
  dominant per-step cost after the model itself.
