# Evaluation

How DeepGEN decides whether a generated sound is good enough to ship, and the
measurements behind the current training objective.

Everything in this document was measured on 2026-09-03. Where a hypothesis did
not survive measurement it is written up as a negative result rather than
removed, because the negative results are the part a future contributor is most
likely to waste time re-deriving.

## 1. The job to be done

DeepGEN is not competing with general text-to-audio models. It is competing
with Spitfire Audio, Serum and Splice: a producer auditions a sound, drags it
into a session, and either uses it or does not. The bar is therefore not "does
this sound plausible" but **"can this be used without being fixed first"**.

That decomposes into properties that can be measured directly, without a
listening panel and without a reference recording. `synthgen/eval/metrics.py`
implements them, and `QualityTarget` encodes the bar:

| Criterion | Target | Why a producer cares |
|---|---|---|
| `sample_rate` | >= 44.1 kHz | 32 kHz cannot carry an air band at all |
| `bandwidth` | >= 18 kHz of real content | Above 16 kHz on purpose: 16 kHz *is* Nyquist at 32 kHz, so a 16 kHz floor would be cleared by output with no air band |
| `true_peak` | <= -1.0 dBTP | Inter-sample peaks distort on conversion |
| `no_clipping` | 0 samples at full scale | Clipping is unrecoverable |
| `level_consistency` | -24 to -12 dBFS RMS | A library whose clips jump 25 dB forces a gain stage on every audition |
| `dynamics` | >= 6 dB crest factor | Squashed output has no life |
| `noise_floor` | <= -60 dBFS | Audible hiss between notes |
| `dc_offset` | <= 0.001 | Wastes headroom, thumps on edit |
| `mono_compatible` | >= -3 dB on fold-down | Club systems and phone speakers are mono |
| `loopable` | <= -20 dB end-to-start step | A click at the loop point makes a sample unusable as a pad |
| `harmonic_clarity` | >= 3 dB HNR | Separates a tone from smeared noise |

Reference-based metrics (`comparative_metrics`) cover autoencoder
reconstruction and regression testing: SI-SDR, log-spectral distance, per-band
energy error, envelope correlation and stereo-image error.

Run it:

```bash
uv run synthgen-eval --input ./audio/candidates --json report.json
uv run synthgen-eval --input ./recon --reference ./originals   # A/B mode
```

## 2. Baseline: what the shipping model actually produces

Twelve generations were pulled from `dnl-core-sounds-s3-prod` (one user prefix,
`C1` take of twelve different generations) and run through the bench.

**Format, measured and identical across every object in the bucket:** 32 kHz,
16-bit PCM, stereo, exactly 8.000 s, 1,024,078 bytes. That is MusicGen's native
output format.

**Mean pass rate: 53.8%.** Per criterion, clips passing out of 12:

```
sample_rate         0/12     bandwidth           0/12     loopable            0/12
level_consistency   6/12     noise_floor         6/12     true_peak           8/12
dc_offset           9/12     harmonic_clarity    9/12     no_clipping        10/12
mono_compatible    11/12     dynamics           12/12
```

The three zeros are structural rather than incidental:

- **Sample rate and bandwidth.** At 32 kHz, Nyquist is 16 kHz. Mean measured
  bandwidth is 12.7 kHz, and two clips stop below 2.3 kHz. The air band that
  makes a Spitfire library sound expensive is not degraded here; it cannot
  exist.
- **Loopability.** Mean end-to-start step is **+2.2 dB relative to edge RMS** --
  the discontinuity is louder than the audio around it. None of these clips can
  be looped without an audible click.

Four clips also exceed 0 dBTP (true peaks up to +1.06 dBFS), and clip RMS spans
-40.7 to -15.0 dBFS, a 25.7 dB spread.

Reproduce with `synthgen-eval`. Raw numbers: `results/production_baseline.json`.

## 3. The training objective is the ceiling

A decoder can only learn to preserve what its loss responds to. Whatever the
DiT does, output quality is bounded above by what the autoencoder reconstructs,
and that is set by the reconstruction loss. So the loss is where the leverage
is.

`MultiResolutionSTFTLoss` -- the objective this repository inherited, and the
one most latent audio autoencoders use -- was measured against the same twelve
real clips using a set of controlled, audible degradations
(`experiments/degradations.py`). Degradations attenuate or substitute; none of
them zeroes a spectral bin, because a brick-wall filter produces exactly-zero
magnitudes and `log(0 + 1e-8)` then measures a numerical cliff rather than
audibility. (That was a real false start: the first version of this experiment
used brick walls and produced misleading numbers in both directions.)

Two scale-free measures, both computed within a single loss so they compare
across losses:

**Wasted sensitivity** -- response to noise at -90 dBFS (inaudible on any
system), relative to the mean response to audible defects:

| Loss | Wasted sensitivity |
|---|---|
| legacy MRSTFT | 0.1853 |
| PerceptualSampleLoss | 0.0052 (**35x less**) |

The legacy loss spends roughly a fifth of an average audible defect's worth of
gradient on content no listener can hear.

It gets worse the more band-limited the material is, and real instrument
samples are very band-limited. On a target rolled off at 4 kHz, the legacy
loss's response to -90 dBFS dither is **1.8x its response to an entire 2 kHz
band going missing** -- it ranks the inaudible defect as the more serious of
the two. `PerceptualSampleLoss` scores 0.0019 on the same comparison. Locked in
by `tests/test_perceptual_loss.py::test_is_near_blind_to_inaudible_noise`.

**Allocation share** -- each audible defect's share of the total response.
Sums to 1.00 per loss, so a rise in one share is a fall in another; gradient
budget is finite.

| Defect | legacy | perceptual | change |
|---|---|---|---|
| air dulled 12 dB > 8 kHz | 0.080 | 0.071 | 0.89x |
| HF replaced by noise | 0.124 | 0.128 | 1.03x |
| sub dulled 12 dB < 80 Hz | **0.026** | **0.084** | **3.18x** |
| stereo collapsed to mono | **0.117** | **0.253** | **2.17x** |
| attacks smeared 10 ms | 0.653 | 0.464 | 0.71x |

Read honestly: the rebalancing moves budget out of transients -- which at 65%
were crowding everything else out -- into sub-bass and stereo image. The two
high-frequency defects are essentially unchanged. The headline result is the
35x cut in wasted sensitivity, then stereo, then sub.

Reproduce: `python experiments/run_loss_sensitivity.py --input <audio> --out <dir>`

## 4. What changed in the loss, and why

`PerceptualSampleLoss` makes three changes to the inherited objective.

**Relative log floor.** `log(mag + 1e-8)` gives every bin 160 dB of range to be
wrong in. The floor is now `dynamic_range_db` (default 60) below *that scale's
own peak target magnitude*. An absolute floor does not work: STFT magnitudes
scale with both window length and clip level, so a constant means a different
number of dB on every clip. This was measured -- an earlier version used a
fixed `1e-5` and made wasted sensitivity *worse* than the legacy loss
(0.459 vs 0.299 on the normalised scale then in use). The relative floor is
what produces the 35x improvement.

**Band weighting.** Linear-frequency bins are counted uniformly, which hands
most of the loss to the crowded top octaves. `DEFAULT_BAND_WEIGHTS` lifts sub
(2.0) and air (2.0) explicitly.

**Mid/side term.** The legacy loss flattens `(B, C, T)` to `(B*C, T)` and
compares channels independently, so *any* output with correct per-channel
magnitudes scores identically regardless of what it does to the stereo image.
`tests/test_perceptual_loss.py::test_penalises_stereo_collapse_where_legacy_loss_cannot`
constructs the worst case -- two channels with identical magnitude spectra and
opposite phase -- and shows the legacy loss returns < 1e-5 for a completely
destroyed image.

A spectral-flux term is also included, penalising smeared attacks directly.

## 5. Negative results

### 5.1 Longer FFT windows buy nothing

The obvious change is to extend the window set upward: at 44.1 kHz a
2048-sample window resolves only 21.5 Hz per bin, which cannot separate
adjacent bass partials. This was implemented, measured, and **dropped**.

Ablating `(8192, 4096, 2048, 1024, 512, 256, 128)` against
`(2048, 1024, 512, 256, 128)` on the same twelve real clips changed every
allocation share by **less than 0.01**:

| Defect | max 2048 | with 8192 |
|---|---|---|
| air dulled | 0.071 | 0.070 |
| HF noise substitution | 0.128 | 0.120 |
| sub dulled | 0.084 | 0.087 |
| mono collapse | 0.253 | 0.257 |
| transient smear | 0.464 | 0.465 |
| wasted sensitivity | 0.0052 | 0.0047 |

An 8192-point STFT is the most expensive term in the loss, and the shipped
config file already warns that multi-resolution STFT at 44.1 kHz forces a
micro-batch of 2. Paying that for a change below measurement noise is a bad
trade, so the long windows were dropped.

### 5.2 No window size gives a pitch gradient

The deeper reason 5.1 fails. Sweeping a detune away from E1 (41.20 Hz):

| detune | 0 | 5c | 10c | 25c | 50c | 100c | 200c |
|---|---|---|---|---|---|---|---|
| max window 2048 | 0.000 | 1.082 | 1.119 | 1.102 | 1.190 | 1.206 | 1.175 |
| max window 8192 | 0.000 | 0.814 | 0.770 | 0.843 | 0.960 | 1.061 | 1.172 |

Both saturate at **5 cents -- a 0.12 Hz shift**. A semitone error (a wrong bass
note) scores barely higher than an imperceptible one. This family of losses
encodes "different" but not "how different", and spectral leakage is what it
actually responds to, not resolved partials. **Bass tuning accuracy needs an
explicitly pitch-aware term** -- an f0 tracker, a differentiable harmonic
comb, or an autocorrelation loss. A bigger FFT will not deliver it.

Reproduce: `python experiments/run_resolution_probe.py --out <dir>`

### 5.3 The Snake activation does not blow up

`Snake` computed `1 / (alpha + 1e-8)` on an unconstrained learnable `alpha`,
which looks like a singularity waiting to happen. Measured across alpha from
1.0 down through 0 to -0.1, **nothing blows up**: `sin^2(alpha*x)` vanishes
quadratically while the denominator vanishes linearly, so the ratio tends to
`alpha * x^2` and both output and gradient stay bounded. A -20 update to the
raw parameter leaves the activation finite.

Two narrower problems are real, and are what the log-space change fixes:

1. `alpha == -1e-8` is exactly singular -- output and gradient both `inf`.
2. Negative alpha computes a **different function**, not a mis-scaled one.
   `sin^2` is even, so `f(-a, x) == -f(a, -x)` exactly (measured difference:
   0.00e+00). Half the parameter space silently implements a mirrored
   activation.

Storing `log_alpha` makes both unreachable. Pre-existing checkpoints are
converted on load. This is a correctness guard, **not** a fix for observed
training instability, and should not be described as one.

Reproduce: `python experiments/run_snake_stability.py --out <dir>`

## 6. What is still unproven

The measurements above are all properties of the *objective*. They do not show
that a model trained with it sounds better, because no model has been trained
with it -- this work was done on CPU with no GPU available.

`experiments/run_loss_inversion.py` is the closest available proxy: it
optimises a damaged real clip back toward the original under each loss, with no
network in the way, so any difference is attributable to the objective alone.
It shows the direction each loss pulls in. It is not a substitute for a
training run.

**The open question is therefore a Stage-1 VAE run comparing
`spectral_loss="legacy"` against `spectral_loss="perceptual"` on identical data
and schedule, scored with `synthgen-eval` in A/B mode.** That is the experiment
that would confirm or refute all of this, and it needs a GPU.
