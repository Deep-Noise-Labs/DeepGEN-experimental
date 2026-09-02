# Evaluation spec - what "good enough to ship" means

The target for DeepGEN is not "sounds plausibly like audio". It is **sample-library
grade**: material a producer would drop into a session next to Spitfire, Serum or
Splice content without apologising for it. That is a much narrower target than the
usual text-to-audio benchmark suite measures, so this document defines the job in
terms that can actually fail.

## Why the usual metrics are not enough

FAD and CLAP score answer "does this land in roughly the right distribution".
They are close to blind to the defects that decide whether a sound is usable. A
patch with -20 dB of aliasing and a patch with -70 dB can sit in the same FAD
bucket while only one of them is sellable, because FAD is computed on embeddings
from a classifier that was never trained to care about inharmonic grit.

So the suite is split in two. Tier 1 is signal-level and can fail a checkpoint on
its own. Tier 2 is perceptual and distributional, and is used for ranking rather
than gating.

## Tier 1 - signal integrity (gating)

These run on deterministic probe signals where the correct answer is known
analytically, so a failure is unambiguous. Implemented in `synthgen/eval/`.

| # | Metric | Function | Gate | Why it decides usability |
|---|---|---|---|---|
| 1 | **Alias-to-signal ratio (ASR)** | `alias_to_signal_ratio` | < -60 dB across 110 Hz - 9 kHz | Inharmonic energy is heard as metallic grit that moves the *wrong way* up the keyboard. The single clearest tell of a cheap synth. |
| 2 | **Total harmonic distortion** | `total_harmonic_distortion` | within 3 dB of reference | Guards the fix: aliasing must be removed without flattening the harmonic character the model is supposed to have. |
| 3 | **Band energy error** | `band_energy_error_db` | within 1 dB per band, presence and air especially | The 8-20 kHz air band is what makes a library sound expensive, and is the first thing a lossy path damages. |
| 4 | **Crest factor** | `crest_factor_db` | within 1.5 dB of source | Plucks, mallets and drums live on their attack. A path that smears transients produces samples that will not cut through a mix. |
| 5 | **Stereo correlation** | `stereo_correlation` | within 0.05 of source | Width is a first-class feature of pads and cinematic patches. |
| 6 | **Log-spectral distance** | `log_spectral_distance` | tracked, no gate | Broadband regression tracker. |

The -60 dB figure in gate 1 is the level commercial analogue-modelling
synthesisers design to; below it, alias products sit under the noise floor of a
normal mix.

### Running it

```bash
uv run python -m synthgen.eval.alias_bench --out results/
```

## Tier 2 - perceptual and distributional (ranking, not gating)

Not yet implemented; these need a trained checkpoint and a reference corpus.

| Metric | What it answers |
|---|---|
| FAD (VGGish / CLAP embeddings) | Is the output distribution close to the reference library? |
| CLAP text-audio similarity | Does the sound match the prompt? |
| Pitch accuracy (CREPE) vs requested note | Can it be played as an instrument at all? |
| Loop-point continuity | Can the sample be looped without a click? |
| Note-to-note timbral consistency | Does a scale played across the keyboard sound like one instrument? |
| MUSHRA / ABX against Splice and Spitfire references | The only measure that ultimately matters. |

The last one is the real definition of done. Everything above it is a cheap proxy
that can be run automatically on every commit.

## What Tier 1 can and cannot tell you before a checkpoint exists

This matters, because DeepGEN currently has no trained weights.

**Valid without training.** The activation's aliasing behaviour. Snake is a
*pointwise* non-linearity: how much energy it folds back over Nyquist is fixed by
the architecture and the working sample rate, not by any learned parameter.
Measuring it on an untrained model is therefore a real measurement, and it is what
`bench_activation` and `bench_feature_rates` do.

**Not valid without training.** Anything about reconstruction. On an untrained
network the output is uncorrelated with the input - measured r = 0.0001 on a 1 kHz
sine - so its spectrum describes the random weights and nothing else. `bench_vae`
is retained as a diagnostic and prints a warning to that effect; its numbers must
not be quoted as quality results until a checkpoint exists.

There is also a second alias source the Tier 1 activation benchmark does *not*
cover. The encoder's strided convolutions are decimators and the decoder's
transposed convolutions are interpolators. Both are time-varying, so unlike
ordinary convolution they manufacture new frequencies of their own. Wrapping the
activation does not address them. Quantifying that path needs a trained model, and
is the next open question.

## Design notes recorded from the first run

Two results worth not re-deriving:

- **Oversampling ratio 2 beats 4 and 8.** With a fixed 32-tap kernel, a higher
  ratio forces a proportionally sharper cutoff than the filter can realise, so
  quality degrades while cost rises linearly. Measured at 8 kHz, alpha=2:
  -65 dB at r=2, -55 dB at r=4, -42 dB at r=8.
- **Filter length is nearly free.** Cost is memory-bound rather than tap-bound.
  Going from BigVGAN's reference 12 taps to 32 buys roughly 13 dB of alias
  suppression and 15 dB of air-band accuracy for single-digit percent runtime,
  which is why `DEFAULT_FILTER_TAPS` is 32.
