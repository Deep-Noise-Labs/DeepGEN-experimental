# Alias-free synthesis in the DeepGEN decoder

## The problem

The decoder is built from Snake activations:

```
snake(x) = x + (1/a) · sin²(a·x)
```

Snake is a good choice for audio - its periodic inductive bias helps a
network learn oscillatory signals, which is why BigVGAN and Stable Audio
both use it. But it is a **memoryless nonlinearity**, and `sin²` generates
harmonics without bound.

Evaluated on a discrete-time signal at 44.1 kHz, every harmonic Snake
produces above Nyquist does not vanish. It **folds back** into the audible
band at

```
f_alias = | k·f0 − n·fs |
```

Those folded partials are almost never at integer multiples of `f0`, so
they are *inharmonic*. They do not fuse with the note. The ear hears them
as a separate metallic layer that does not track pitch - the exact defect
that separates a cheap oscillator from Serum, and the reason every serious
software synth oversamples its waveshapers.

The current decoder does this **~30 times in series**, at full audio rate,
with no band-limiting anywhere.

## The fix

Evaluate the nonlinearity at a higher rate, band-limit, then decimate:

```
x → upsample(×2) → snake → low-pass → downsample(×2)
```

This is the standard alias-free construction from Alias-Free GAN (Karras et
al., 2021) and BigVGAN (Lee et al., ICLR 2023). `synthgen/model/antialias.py`
implements it with Kaiser-windowed sinc filters.

It does **not** eliminate aliasing - harmonics above `ratio × Nyquist` still
fold. It buys a large, measurable reduction. The numbers below are what it
actually buys, measured, not estimated.

**Cost: zero parameters.** The filters are fixed, non-persistent buffers.
Encoder and decoder have byte-identical parameter counts with the flag on
or off. The cost is compute: roughly 2× wall-clock per activation on CPU.

## What was measured

All numbers come from `experiments/generate_proofs.py`, which runs this
repository's own modules. Re-run it to reproduce them.

Measurement method and its pitfalls are documented in
[EVALUATION.md](EVALUATION.md). In short: the stimulus is a band-limited
sawtooth that provably contains no inharmonic energy, so anything
inharmonic at the output was created by the module under test. Test
frequencies are selected by `alias_visibility_hz()` rather than by ear -
several obvious-looking choices (220.5 Hz, 110.3 Hz) report *zero* aliasing
regardless of how bad the model is, and would have made this change look
worthless.

### 1. The repository's own `ResidualBlock`, at audio rate

12 `ResidualBlock`s from `synthgen/model/vae.py`, **identical weights in
both arms** (same seed, same construction order). Only the activation
differs.

| Note f0 (Hz) | Before (dB) | After (dB) | Improvement |
|---|---|---|---|
| 111.5 | −33.76 | −53.02 | **19.3 dB** |
| 223.3 | −31.43 | −50.39 | **19.0 dB** |
| 453.1 | −30.49 | −48.60 | **18.1 dB** |
| 903.7 | −25.13 | −44.92 | **19.8 dB** |
| 1804.1 | −27.17 | −43.73 | **16.6 dB** |
| 2090.1 | −24.44 | −46.37 | **21.9 dB** |
| 3604.3 | −22.83 | −39.45 | **16.6 dB** |
| 4260.9 | −21.23 | −41.85 | **20.6 dB** |

**Mean improvement: 19.0 dB** (min 16.6, max 21.9). That is roughly a
**9× reduction in alias amplitude**.

Note the baseline trend: aliasing gets steadily worse as the note gets
higher (−33.8 dB at 111 Hz → −21.2 dB at 4261 Hz), because more of the
harmonic series sits above Nyquist. This is exactly what theory predicts,
and it means the defect is worst in the register where leads and bells live.

### 2. Aliasing compounds with depth

Bare activation cascade, band-limited saw at 2090.1 Hz:

| Activations in series | Before (dB) | After (dB) |
|---|---|---|
| 1 | −22.74 | −29.08 |
| 4 | −15.34 | −31.20 |
| 8 | −16.97 | −32.16 |
| 16 | −15.82 | −29.19 |
| 32 | −14.77 | −23.22 |

A single Snake is not catastrophic. The problem is that a deep stack keeps
folding already-folded content, and the baseline degrades from −22.7 dB to
about −15 dB as depth grows, while the anti-aliased version holds near
−30 dB. **The deeper the decoder, the more this matters** - and production
decoders are deep.

### 3. Single note, spectrum (f0 = 2090.1 Hz, 8 activations)

| Metric | Before | After | Change |
|---|---|---|---|
| Alias-to-signal ratio | −16.97 dB | −32.16 dB | **−15.2 dB** |
| Sub-fundamental alias | −29.67 dB | −46.07 dB | **−16.4 dB** |
| Spurious-free dynamic range | 23.94 dB | 38.78 dB | **+14.8 dB** |

Sub-fundamental alias energy is the one to watch: those products land
*below* the note, where nothing can mask them.

### 4. Real audio

Real recordings from Deep Noise repositories (`aisynth-vst/assets/guitar.wav`,
`audiocraft/assets/electronic.mp3`, `audiocraft/dataset/example/electro_1.mp3`)
passed through the same 12-block stack in both arms. The **difference
signal** - level-matched, so it isolates the artefact rather than a gain
change - sits at:

| Source | Isolated artefact level |
|---|---|
| guitar.wav | −52.3 dB |
| electronic.mp3 | −27.8 dB |
| electro_1.mp3 (synth pad) | −36.1 dB |

−27.8 dB of inharmonic residue on synth-heavy material is plainly audible.
The rendered WAVs are in `proofs/audio/`.

## Honest limits of these measurements

- The A/B on module stacks uses **randomly initialised weights**. That is
  valid for this specific claim, because aliasing from a memoryless
  nonlinearity is a property of the *function*, not of the learned weights,
  and both arms share identical weights. It is *not* evidence about final
  model quality.
- An untrained **full VAE** round-trip was tried and **discarded**: at
  random init the encoder's 1024× decimation destroys the signal, output is
  broadband noise, and the alias metric is meaningless there. It is not
  reported as a proof.
- 2× oversampling is a compromise. 4× measures better still (see
  `test_higher_ratio_reduces_alias_further`) at higher compute cost.
- The remaining ~19 dB gap to the −60 dB gate is not closed by this change
  alone. See below.

## Not done yet

1. **Transposed-convolution artefacts.** `DecoderBlock` upsamples with
   `ConvTranspose1d(kernel=2·stride, stride=stride)`, the classic
   checkerboard-artefact configuration (Pons et al., 2021). BigVGAN keeps
   transposed convs and relies on alias-free activations, so this was left
   alone deliberately rather than changed on a hunch - but it is the next
   thing to measure.
2. **Encoder decimation is unfiltered.** The strided convolutions decimate
   by up to 1024× with no explicit anti-alias filter. The convolution can
   *learn* one, but nothing makes it.
3. **Higher oversampling ratios**, and per-stage ratios (more where the
   signal is at full rate, less deep in the network where it is not).
4. **Retrofitting existing checkpoints.** `AntiAliasedSnake.alpha` stays
   addressable, so a `Snake` checkpoint can be remapped by renaming
   `<prefix>.alpha` → `<prefix>.act.activation.alpha`. Untested against a
   real checkpoint - there are none yet.
