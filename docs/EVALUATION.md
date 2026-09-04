# Evaluating DeepGEN: what "done" means

DeepGEN is not competing to produce plausible three-second audio clips. It
is competing to produce **sounds a professional will load into a session**.
Those are different jobs and they need different measurements.

This document defines the job. `synthgen/eval/` implements it.

## Why the usual benchmarks are not enough

Frechet Audio Distance, CLAP score, and KL over an audio classifier all
answer variants of *"does this resemble real audio, or the prompt?"* They
are useful, and DeepGEN should track them. But a sound can score well on
every one of them and still be unusable:

- A lead can be spectrally plausible while aliasing audibly, because
  aliasing is a small fraction of total energy and FAD is not sensitive to
  where energy sits relative to the harmonic grid.
- A plucked sample can match the prompt while its 2 ms attack has been
  smeared to 9 ms, at which point it no longer reads as a pluck.
- A pad can be indistinguishable in mono while its stereo image has
  collapsed - destroying the one property it was chosen for.

None of these are exotic. All three are the standard failure modes of a
neural audio codec, and none of them is caught by the standard metrics.

## The gates

Run `synthgen-eval gates` to print these with their full rationale.

| Gate | Unit | Target | Stretch | Catches |
|---|---|---|---|---|
| Alias-to-signal ratio | dB | ≤ -60 | ≤ -80 | Metallic inharmonic layer on sustained notes |
| Sub-fundamental alias energy | dB | ≤ -70 | ≤ -90 | Unmaskable grit under the note |
| Spurious-free dynamic range | dB | ≥ 60 | ≥ 80 | A single loud whistling artefact |
| Air-band retention (10-20 kHz) | dB | \|·\| ≤ 1.5 | ≤ 0.5 | Dull, "small" output vs the source |
| Attack-time error | ms | \|·\| ≤ 1.0 | ≤ 0.3 | Smeared transients on plucks and percussion |
| Stereo image error | corr | \|·\| ≤ 0.15 | ≤ 0.05 | Width collapsing towards mono |
| Noise floor | dB | ≤ -70 | ≤ -85 | Hiss that stacks across layered voices |
| Scale-invariant SDR | dB | ≥ 12 | ≥ 20 | General reconstruction regressions |
| Multi-resolution STFT distance | ratio | ≤ 0.35 | ≤ 0.15 | Timbre drift |

**These thresholds are proposed engineering targets, not measured
specifications of any third-party product.** Where a number is a judgement
call, the gate's `rationale` field says so. They should be revised as real
listening data arrives; the point of writing them down is that a change can
now be argued about with numbers instead of adjectives.

## Two families of check

**Synthesis gates** (`evaluate_synthesis`) are reference-free. They feed a
known band-limited stimulus through the model and measure what comes out.
Because the stimulus provably contains no inharmonic energy, anything
inharmonic at the output was created by the model.

**Reconstruction gates** (`evaluate_reconstruction`) compare output against
a reference recording. These are the ones to use for VAE/codec work.

## The trap in alias measurement

A folded harmonic lands at `|k·f0 - n·fs|`. When `f0` is a simple rational
fraction of the sample rate, those folded products land *exactly on top of*
real harmonics of `f0`, where the harmonic/inharmonic split cannot see
them.

This is not a corner case. At 44.1 kHz:

| Candidate f0 | fs / f0 | Alias separation | Usable? |
|---|---|---|---|
| 220.5 Hz | 200.000 | **0.0 Hz** | No - reports zero aliasing always |
| 110.3 Hz | 399.819 | **0.0 Hz** | No |
| 4409.1 Hz | 10.002 | 9.0 Hz | No - inside the tolerance band |
| 903.7 Hz | 48.799 | 178.5 Hz | Yes |
| 2090.1 Hz | 21.099 | 207.9 Hz | Yes |

An engineer picking "220.5 Hz, that's about A3" would measure a *perfectly
clean* result from an arbitrarily bad model. `alias_visibility_hz()` scores
a candidate frequency, `DEFAULT_TEST_FREQS` holds a validated set, and
`test_default_test_frequencies_can_actually_reveal_aliasing` fails the
build if anyone adds a bad one.

The measurement floor also matters: `numpy.blackman` is the 3-term window
with ~-58 dB sidelobes, which silently caps every alias reading at -58 dB.
`metrics.blackman_harris` is the 4-term variant (~-92 dB) and the suite
measures a floor around **-93 dB**, leaving ~70 dB of real headroom.

## Usage

```bash
# What the gates are and why
synthgen-eval gates

# Reference-free: a WAV holding one sustained note
synthgen-eval synthesis --audio lead.wav --f0 903.7

# Reference-based: reconstruction vs source
synthgen-eval reconstruction --pred out.wav --target in.wav --json
```

```python
from synthgen.eval import evaluate_synthesis, evaluate_reconstruction

results = evaluate_synthesis(my_decoder_fn)          # no reference needed
results = evaluate_reconstruction(recon, source)     # against a reference
```

## What is not covered yet

Honest gaps, in rough priority order:

1. **No perceptual listening data.** Every threshold is reasoned, not
   fitted to human ratings. A small MUSHRA-style panel would let us
   calibrate them properly.
2. **No prompt-adherence metric.** CLAP score or an equivalent still needs
   wiring in; these gates measure *sound quality*, not *did it follow the
   text*.
3. **No polyphonic alias measurement.** The harmonic/inharmonic split
   assumes a single fundamental. Two simultaneous notes need a different
   decomposition.
4. **No loop-point or release-tail checks**, which matter for sampler
   instruments specifically.
