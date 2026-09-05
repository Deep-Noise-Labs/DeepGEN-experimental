# Band-limited resampling in the audio VAE

What the defect is, how it was measured, what the fix buys, and what is not
proven. Every number here comes from code in `experiments/`, run on real Deep
Noise renders or on labelled measurement stimuli. Nothing is estimated.

## The defect

The VAE crosses a 1024x sample-rate change twice. The encoder decimates by
4, 4, 8, 8 with strided convolutions; the decoder puts it back with transposed
convolutions of the same strides. Both are written as a single convolution of
kernel length `2 * stride`.

A transposed convolution with stride `s` and kernel `2s` is exactly a polyphase
interpolator with `s` phases of **two taps each**. Rate conversion creates
spectral images - mirrored copies of the signal around multiples of the input
rate - and the only thing that can suppress them is the interpolation kernel
itself. Two taps per phase is not enough. The encoder has the mirror-image
defect: it decimates with no anti-alias filter, so content above the new
Nyquist folds back down instead of being removed.

Both artefacts are **inharmonic**. They do not sit on the harmonic series of
the note being played, which is what makes them audible as grit and metallic
roughness rather than as timbre. For a model whose job is sustained,
harmonically dense synthesiser material, this is the defect class that most
separates a sampler-instrument sound from a cheap digital one.

## The fix

`synthgen/model/resample.py` adds a fixed, non-learnable Kaiser-windowed sinc
filter on the rate-changing step - low-pass after zero-stuffing on the way up,
before decimating on the way down. This is the Alias-Free GAN / BigVGAN
treatment applied to the resamplers rather than to the activations.

`BandlimitedUpsample1d` and `BandlimitedDownsample1d` are drop-in replacements,
default-on via the `bandlimited` flag on `EncoderBlock`, `DecoderBlock`,
`AudioEncoder`, `AudioDecoder` and `AudioVAE`.

**Zero added parameters.** A 291,580-parameter test VAE has exactly 291,580
parameters with the change on or off. The learnable weight keeps
`ConvTranspose1d`'s `(in, out, kernel)` layout, so checkpoints move between the
two forms.

### Why the comparison is trustworthy

When the fixed filter is reduced to a unit impulse, both replacements are
**bit-for-bit identical** to the operators they replace.
`tests/test_resample.py` asserts this for strides 2, 4 and 8. Every A/B below
uses the same weight tensor in both arms, so a measured difference is the
filter's doing and nothing else's - not a change of weights, initialisation,
capacity or level. The filters are normalised to unit DC gain, which a unit
impulse also has, so the pass-band level matches too.

## What was measured

Corpus: the six raw production renders committed under
`docs/qc-eval/2026-09-02-bakeoff/audio/` in `dnl-inference-backend` - model
0.8.1L2, benchmark tester account, 3 s stereo 32 kHz, generated 2026-07-10,
OGG Vorbis transcodes of the `C4.wav` S3 objects. Vorbis does not move
spectral envelope or pitch, so they are valid for these claims; they are not
valid for noise-floor or codec-artefact claims.

Measurement stimuli (sines) are labelled as stimuli, never as results.

### 1. Image rejection across the pitch range (`p1_image_vs_pitch.png`)

Twelve scored probe frequencies, both arms carrying an identical
linear-interpolation kernel - the best a 2-tap-per-phase kernel does smoothly,
so the baseline is measured in its strongest form rather than its weakest.

| stride | current | band-limited | difference |
|---|---|---|---|
| 4 | -26.6 dB | -102.1 dB | **75.5 dB** |
| 8 | -27.6 dB | -102.8 dB | **75.2 dB** |

### 2. The architectural ceiling (`p2_architectural_ceiling.png`)

The important one, because it is what makes this a design defect rather than a
training problem. A transposed convolution's anti-imaging filter *is* its
kernel, so the best rejection it can ever reach is the best any FIR of that
length can reach. Parks-McClellan gives that optimum exactly:

| taps per phase | best achievable |
|---|---|
| **2 (current)** | **-21.6 dB (stride 4), -22.1 dB (stride 8)** |
| 4 | -38.4 dB |
| 8 | -71.5 dB |
| 16 (shipped filter) | -72.2 dB measured |

No amount of training moves the first row. The empirical linear-interpolation
baseline in experiment 5 lands at -22.0 dB, which is that ceiling, reached
independently.

### 3. Filter design, chosen from data (`p3_filter_ablation.png`)

Sweeping length against transition width on the six renders, splitting the
result into the two things that can change - fidelity *inside* the band the
signal occupies, and invented content *outside* it:

| filter | out-of-band (invented) | in-band fidelity | pass-band droop |
|---|---|---|---|
| none (current) | -34.2 dB | 35.6 dB | - |
| **65 taps, shipped default** | **-113.7 dB** | **35.6 dB** | -0.13 dB |
| 257 taps | -131.7 dB | 35.6 dB | -0.00 dB |

In-band fidelity is **identical to the unfiltered baseline**. The filter costs
nothing in the band that carries the signal; it only removes content that was
never supposed to be there. Beyond 65 taps the extra rejection is already far
below the 16-bit noise floor, so the default stops there.

### 4. Encoder folding (`p4_encoder_fold.png`)

Tones placed above the post-decimation Nyquist, measured relative to an in-band
reference tone through the same weights:

| stride | current | band-limited | rejection |
|---|---|---|---|
| 4 | **+0.2 dB** | -53.3 dB | 53.5 dB |
| 8 | **+4.1 dB** | -53.0 dB | 57.1 dB |

Read the current column carefully: out-of-band content comes back **at or above
the level of the wanted signal**. The encoder is not attenuating it, it is
relocating it.

### 5. Real audio (`p5_spectrograms.png`, `proofs/upsampler/audio/`)

Each render is band-limited to what the decimated representation can carry,
decimated by 4, then reconstructed through both arms with identical weights.
Anything above the band edge is content the upsampler invented, because the
reference has none there.

| render | prompt | current | band-limited | difference |
|---|---|---|---|---|
| 1265755b | Texture | -13.7 dB | -90.8 dB | 77.0 dB |
| 2619cb69 | Synth | -41.2 dB | -122.5 dB | 81.3 dB |
| 7b82e3fd | ambient granular texture | -40.6 dB | -118.0 dB | 77.4 dB |
| 94e0f32f | Texture | -17.6 dB | -96.6 dB | 79.0 dB |
| e32cbb3f | Electric Piano | -41.5 dB | -122.2 dB | 80.7 dB |
| ed46ca9e | Sequence | -41.7 dB | -123.1 dB | 81.4 dB |
| **mean** | | **-32.7 dB** | **-112.2 dB** | **79.5 dB** |

Reconstruction SNR against the reference improves on all six as well (18.6 ->
24.9 dB worst, 38.1 -> 45.8 dB best), so the filtered arm is closer to the
truth as well as cleaner.

Each clip is written out as seven OGG files: the reference, both
reconstructions, each arm's isolated artefact at true level, and each arm's
artefact amplified so it can be heard on its own.

**Attributability guard.** An artefact number measured through a broken
reconstruction says nothing. Every clip carries the SNR of its own
reconstruction and is marked `attributable: false` below 10 dB. All six pass.

### 6. Depth (`p6_stage_compounding.png`)

Through all four decoder stages with competent per-stage interpolators, the
current design sits flat at -22.0 dB and the band-limited one at -95.8 dB.

**Imaging does not compound with depth.** It plateaus at the single-stage
ceiling. This corrects a plausible-sounding expectation - the anti-aliasing
work found that *aliasing* through activations does compound - and it is
recorded here so nobody argues the compounding case for this defect.

## What is not proven

- **Nothing about a trained model.** There is still no DeepGEN checkpoint. The
  measurements are properties of the operators, taken with matched weights.
  Do not describe this as "the model sounds better".
- **No listening test.** The claim is that inharmonic content 80 dB down is
  better than inharmonic content at signal level. That is well-founded but it
  is not a measured preference.
- **No comparison against any commercial product.** Nothing here was measured
  against Serum, Spitfire or Splice, and no such claim should be made without
  doing the work.
- **Compute cost is not benchmarked.** The filter is one depthwise FIR of 65
  taps (stride 4) or 129 taps (stride 8) per rate change. That is real work,
  and it has not been timed against a training step.

## Reproducing

The measured JSON and the figures and audio they produce are regenerated by the
commands below rather than committed, because this environment cannot push
binaries (see the retrospective's environment notes).

```bash
pip install torch torchaudio numpy scipy soundfile librosa matplotlib pytest pyyaml
mkdir -p work/corpus && cp <dnl-inference-backend>/docs/qc-eval/2026-09-02-bakeoff/audio/*_raw_C4.ogg work/corpus/

PYTHONPATH=. python experiments/upsampler_bench.py --out proofs/upsampler
PYTHONPATH=. python experiments/upsampler_audio_proof.py --input work/corpus --out proofs/upsampler
PYTHONPATH=. python experiments/upsampler_filter_ablation.py --input work/corpus --out proofs/upsampler
PYTHONPATH=. python experiments/upsampler_figures.py --out proofs/upsampler
python -m pytest tests/ -q
```
