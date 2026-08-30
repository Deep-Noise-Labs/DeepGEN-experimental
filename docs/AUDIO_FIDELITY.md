# Audio fidelity: the Stage-1 ceiling

DeepGEN generates audio in two stages. The DiT produces latents; the VAE
decoder turns them into a waveform. Whatever the decoder cannot reconstruct,
the system cannot generate - no amount of DiT scale, data or sampling steps
recovers it. The Stage-1 autoencoder is therefore a hard ceiling on output
quality, and it is the right place to spend effort first when the target is
Serum, Spitfire or Splice-grade sound.

This page records two defects found in that stage and what was done about
them. Both are measured, and both are reachable from config so the previous
behaviour can be restored for comparison.

## 1. The nonlinearity was aliasing

Snake is a pointwise nonlinearity: `x + (1/alpha) * sin^2(alpha * x)`. Like
any nonlinearity it manufactures harmonics, and it was being applied at the
native sample rate. Harmonics above Nyquist have nowhere to go, so they fold
back down into the audible band as partials that are inharmonic with respect
to the signal that produced them.

Measured on a 9 kHz tone at 44.1 kHz, through one activation:

| Band | Native rate | Oversampled | Change |
|---|---|---|---|
| Fundamental, 9 kHz | 86.1 dB | 86.2 dB | +0.0 dB |
| 2nd harmonic, 18 kHz | 64.3 dB | 62.1 dB | -2.2 dB |
| **4th harmonic folding 36 kHz -> 8.1 kHz** | **67.7 dB** | **20.7 dB** | **-46.9 dB** |
| 3rd harmonic folding 27 kHz -> 17.1 kHz | -1.6 dB | -5.7 dB | -4.1 dB |

The 8.1 kHz partial is the one that matters. It sits *below* the fundamental
at only 18 dB down, it is inharmonic, and it moves in the opposite direction
to the note when the pitch changes - the signature of aliasing and the reason
cheap digital synths sound cheap. Nothing downstream can remove it, because
by the time it exists it is indistinguishable from real signal.

`AliasFreeSnake` runs the nonlinearity at 2x rate between a matched pair of
Kaiser-windowed-sinc resamplers, following Karras et al. (Alias-Free GAN) and
BigVGAN. The harmonics it generates above the original Nyquist frequency are
removed by the decimation filter instead of folding.

The cost is honest and twofold: the activation runs on twice as many samples,
and the resampling filters roll off slightly below Nyquist, which is the
-2.2 dB at 18 kHz above. In exchange the aliased partial drops by ~47 dB.
Set `vae_antialias: false` to compare.

### It reduces aliasing; it does not eliminate it

Worth stating plainly, because a spectrogram of a sweep through several
chained activations still shows folded traces after the change. Oversampling
at 2x only catches harmonics falling between Nyquist and twice Nyquist.
Anything above *that* still folds, and a stack of chained Snake activations
generates plenty of it.

The effect also depends on pitch, exactly as the physics says it should.
Energy below the fundamental - which a nonlinearity cannot produce except by
folding - measured in 0.4 s windows as a 300 Hz to 5.5 kHz sweep rises
through four chained activations:

| Fundamental | Harmonics below Nyquist | Before | After | Change |
|---|---|---|---|---|
| 813 Hz | 27 | 63.3 dB | 62.8 dB | -0.5 dB |
| 1232 Hz | 18 | 40.3 dB | 39.8 dB | -0.5 dB |
| 1867 Hz | 12 | 0.3 dB | -0.1 dB | -0.4 dB |
| 2829 Hz | 8 | 9.5 dB | 7.2 dB | -2.3 dB |
| 4286 Hz | 5 | 27.1 dB | 12.0 dB | -15.1 dB |

While the harmonics still fit under Nyquist there is no folding to remove and
the change does nothing. The benefit arrives as the pitch climbs - which is
where synth patches and sample libraries live. `AliasFreeSnake(ratio=4)` is
available if the residual matters more than the compute.

### Also: the old Snake had a pole

`1 / (alpha + 1e-8)` with `alpha` an unconstrained learnable parameter goes
non-finite when a training step lands alpha on exactly `-1e-8`, and changes
the shape of the activation for any negative alpha. `SnakeBeta` stores both
alpha (frequency) and beta (magnitude) in log space, so they are strictly
positive for every parameter value, and decouples the two the way BigVGAN
does.

## 2. The reconstruction objective was blind to phase and stereo

Spectral convergence and log-magnitude both depend on `|STFT(x)|` alone. That
makes them exactly invariant to phase - and phase is where a large part of
audible quality lives.

The cleanest demonstration: take a target, invert the polarity of its right
channel, and score it.

| Reconstruction | Magnitude-only | With phase + mid/side |
|---|---|---|
| Right channel polarity inverted | **0.0000** | 4.0485 |
| Phase dispersed (all-pass) | 1.5699 | 6.2864 |
| Left/right swapped | 1.2456 | 3.2711 |

A score of exactly zero means the objective regards a polarity-inverted
channel as a *perfect* reconstruction. It is not: the stereo image inverts,
and the sound largely cancels the moment anything sums to mono - a club PA, a
phone speaker, a mono aux send. For a sample library that is a fatal defect,
and the old objective had no way to express it.

The same blind spot covers transient smear. An all-pass filter leaves every
frequency at its exact level and moves only phase, which is precisely the
"underwater", soft-attack character typical of latent audio autoencoders.
Attack transients are most of what makes a sampled instrument sound real.

Two terms were added to `MultiResolutionSTFTLoss`:

- **Complex term.** `|STFT(pred) - STFT(target)|`, normalised by mean target
  magnitude so it is scale-invariant and stays O(1) alongside the existing
  terms. This is what makes phase visible to the gradient.
- **Mid/side term.** The magnitude terms flatten `(B, C, T)` to `(B*C, T)`
  and score channels independently, so nothing constrains the relationship
  between them. Scoring the mid/side decomposition constrains the image
  directly.

Both default to on. `vae_phase_weight: 0.0` and `vae_stereo_weight: 0.0`
restore the previous objective exactly.

The log-magnitude floor was also changed from an additive `1e-8` to a clamp
at `1e-5`. With the additive floor an empty bin contributes `log(1e-8) =
-18.4`, so silence dominated the gradient in exactly the quiet passages where
detail matters most.

## What this does not fix

Worth being explicit, because these are the next ceilings, roughly in order
of expected impact:

1. **No adversarial term.** Every objective here is a distance to a target,
   and distance losses average. Averaging over plausible high-frequency
   detail produces the smoothed, slightly dull result characteristic of
   non-adversarial autoencoders. A multi-period / multi-scale discriminator
   (HiFi-GAN, BigVGAN, DAC) is the standard fix and is the single largest
   remaining Stage-1 win.
2. **Latents are not normalised.** `SynthGen.compute_loss` feeds raw VAE
   latents into a flow-matching objective whose noise is `N(0, I)`. If the
   latents are not close to unit variance the interpolation path is badly
   scaled and training is much slower than it should be. Stable Audio and
   Stable Diffusion both apply a fixed scale factor calibrated on the
   training set.
3. **Uniform timestep sampling.** `sample_timestep` draws `t ~ U[0, 1]`.
   SD3 and Stable Audio use a logit-normal schedule, which concentrates
   effort on the middle timesteps where the velocity field is hardest.
4. **1024x compression is aggressive** for 3-15 second one-shots. Sample
   quality may be cheaper to buy with a shorter compression ratio than with
   a larger DiT.
5. **`FlowMatchingLoss` is dead code.** `compute_loss` calls
   `F.mse_loss` directly and never uses the SNR weighting implemented in
   `losses.py`.

## Reproducing the measurements

Both tables:

```bash
uv run python scripts/measure_fidelity.py \
    --audio path/to/stereo.wav --offset 2.0 --duration 3.5
```

Table 1 is independent of the input. Table 2 was produced from a real Deep
Noise studio sample; omit `--audio` to score a synthetic stereo signal
instead, which gives the same ordering with different absolute values.

The behaviour behind both tables - aliasing suppressed by more than 20 dB
with the fundamental intact, a polarity-inverted channel scoring exactly zero
under the magnitude-only objective and non-zero under the new one - is pinned
by tests:

```bash
uv run pytest tests/test_antialias.py tests/test_perceptual_losses.py -v
```
