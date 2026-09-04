# AGENTS.md — DeepGEN-experimental

Cross-tool contract for any agent working in this repository, following the
[agents.md](https://agents.md) convention. Read it in full before writing
anything. It is written to stop the next agent repeating work that has
already been done, and to stop it repeating mistakes that have already been
made and corrected.

---

## What this repository is

DeepGEN (package name `synthgen`) is Deep Noise's own text-to-sample model.
It is **not** a MusicGen fork. MusicGen's architecture was the starting
point of the thinking; the repository landed on a different design:

| Component | Choice | Why |
|---|---|---|
| Generative core | Latent Diffusion Transformer (DiT) | Parallel generation; MusicGen's autoregressive token stack is slow at high sample rates |
| Objective | Conditional Flow Matching | Straighter transport paths, 10-25 sampling steps instead of 50-200 |
| Audio codec | Audio VAE, ~1024× compression | Diffusion needs a compact continuous latent |
| Text | Frozen T5-base | Validated across Stable Audio / AudioLDM |
| Output | 44.1 kHz stereo, 3-15 s | Sampler-instrument territory, not song generation |

The target is **synthesiser and sampler-instrument quality**, benchmarked
against what a professional expects from a commercial sample library or a
software synth. That target drives everything below.

## Ground rules

- **Never fabricate a measurement.** Every number in a doc, report or PR
  must be reproducible by running code in this repository. If something was
  not measured, say it was not measured.
- **Separate FACTS from INFERENCES**, and label proposed targets as
  proposed. `docs/EVALUATION.md` names its thresholds as engineering
  targets, not as measured specifications of any third-party product. Keep
  it that way — inventing a competitor's spec is a factual claim about
  another company.
- British English. Plain hyphens, never em-dashes, in human-facing copy.
- Branch and PR for anything non-trivial. Do not push to `main`.
- Audio used as evidence must come from a real source (AWS S3, a Deep Noise
  repository, or the AI Synthesizer API) — never synthesised to look like a
  result. Measurement *stimuli* (sine, sweep, band-limited saw) are a
  different thing and are fine, but label them as stimuli.

## Read order

1. This file.
2. `README.md` — architecture overview.
3. `docs/EVALUATION.md` — **what "done" means**. Read before proposing any
   quality change; it is the scoring function.
4. `docs/ANTIALIASING.md` — the first substantive quality change and its
   measured results.
5. `docs/TRAINING.md`, `docs/CLEARML.md`, `docs/TRITON_INFERENCE.md`.

---

# Retrospective: the anti-aliasing work (2026-09-04)

Written by the agent that did it, for the agent that comes next.

## What I set out to do

Find the single highest-leverage quality change for a model whose job is
professional synthesiser sounds, prove it with real measurements, and leave
behind an evaluation harness that makes the *next* such argument
quantitative rather than aesthetic.

## What I found

The repository was a well-structured scaffold with **no evaluation code at
all** and no trained checkpoints. That is the actual bottleneck: without a
scoring function, every architecture debate is opinion. So the work split
in two — build the evals, then use them to justify one change.

The change: **the decoder's Snake activations were generating aliasing and
nothing was band-limiting it.** Snake is a memoryless nonlinearity that
produces unbounded harmonics; at 44.1 kHz everything above Nyquist folds
back as inharmonic content, and the decoder applies it ~30 times in series
with no filtering. For sustained, harmonically-rich synth material this is
the most audible defect class there is.

Fix: the BigVGAN / Alias-Free-GAN sandwich — upsample ×2, apply Snake,
low-pass, decimate. **Measured mean improvement: 19.0 dB** alias reduction
across the pitch range, on the repository's own `ResidualBlock`, with
identical weights in both arms. Zero added parameters. ~2× compute per
activation.

## What I got wrong, and how it was caught

Four mistakes. All four would have produced a confident, wrong report.

**1. I chose test frequencies by eye.** I picked "110.3, 220.5, 440.7,
2371.3, 4409.1 Hz" — they look irregular, so they look safe. They are not.
A folded alias lands at `|k·f0 − n·fs|`, which sits *exactly on top of* a
real harmonic when `f0` is a simple rational fraction of `fs`. At 44.1 kHz,
`220.5 Hz` gives `fs/f0 = 200.000` **exactly**, and reports zero aliasing no
matter how bad the model is. `110.3 Hz` scores 0.0 too. `4409.1 Hz` scores
9 Hz, inside the measurement tolerance.

I caught this because the alias-vs-pitch curve had a physically impossible
dip. The fix is `alias_visibility_hz()` plus a test that fails the build if
anyone adds a bad frequency. **If you add a test frequency, score it first.**

**2. I used `numpy.blackman` for the analysis window.** That is the 3-term
Blackman with ~-58 dB sidelobes, which silently caps every alias reading at
-58 dB — right in the range the interesting numbers live. Switching to the
4-term Blackman-Harris moved the measurement floor to about **-93 dB**.
Guarded by `test_blackman_harris_sidelobes_beat_numpy_blackman`.

**3. I nearly reported an invalid experiment.** I reasoned that a conv net
with pointwise nonlinearities preserves periodicity, so an untrained full
`AudioVAE` round-trip should still yield a valid alias measurement. I ran
it: output alias-to-signal was **+12 to +48 dB**, i.e. the output is noise,
not a reconstruction. The argument fails because the encoder decimates
1024× without filtering and destroys the signal at random init. **The
untrained full-VAE A/B is not a valid test bed. Do not resurrect it.** The
valid substitute is the audio-rate `ResidualBlock` stack, where residual
connections keep the signal intact and identical weights isolate the
activation.

**4. A unit test of mine failed for the right reason.** I asserted that a
sawtooth rolled by 200 samples is "wide" stereo — but the saw's period is
~100 samples, so the rolled copy is nearly in phase and correlation stays
~1. The metric was right; my test signal was wrong.

## What is proven, and what is not

**Proven, reproducibly:**
- 19.0 dB mean alias reduction in the repository's audio-rate residual
  stack, identical weights, across 111-4261 Hz.
- Aliasing compounds with stack depth in the baseline and does not in the
  fixed version.
- Zero parameter cost; ~2× compute per activation.
- An audible, level-matched artefact residual on real audio (-27.8 dB on
  synth-heavy material).

**Not proven:**
- Anything about final trained model quality. There are still no production
  checkpoints. The small CPU A/B in `experiments/` is a controlled
  comparison at tiny scale, not evidence of shippable quality.
- Any comparison against a named commercial product. Nothing here was
  measured against Serum, Spitfire or Splice, and no such claim should be
  made without doing the work.

## Where I would go next, in order

1. **Measure the transposed-convolution upsampler.** `DecoderBlock` uses
   `ConvTranspose1d(kernel=2·stride, stride=stride)` — the textbook
   checkerboard-artefact setup. I deliberately did *not* change it, because
   BigVGAN keeps transposed convs and I had no measurement to justify
   touching it. Get one. This is the most likely next win.
2. **Filter the encoder's decimation.** Strided convs decimate up to 1024×
   with no explicit anti-alias filter. The network can learn one; nothing
   requires it to.
3. **Train a real checkpoint and run the full gate suite**, including the
   reconstruction gates that need a reference. Everything is wired; it needs
   GPU time.
4. **Calibrate the gate thresholds against human listening.** Every
   threshold in `docs/EVALUATION.md` is reasoned, not fitted. A small
   MUSHRA-style panel would convert them from defensible to correct.
5. **Add prompt-adherence** (CLAP or equivalent). The current gates measure
   sound quality only.

## Environment notes that cost me time

- **No GPU** in this sandbox; 4 CPUs, 15 GB RAM.
- `download.pytorch.org` is **blocked** by egress policy. `pypi.org` is
  reachable directly — `pip install torch` from PyPI works.
- **AWS was unavailable**: the MCP connector's token was expired and the
  `AWS_ACCESS_KEY_ID` in the environment is stale (`InvalidAccessKeyId`).
  So no S3 sound pulls. `api.deepnoise.ai` is also egress-blocked.
  Real audio therefore came from Deep Noise GitHub repositories. If you
  need S3, get the connector re-authorised first — do not burn an hour on
  it as I did.
- `deepnoise-web-assets/player/track{1..5}.mp3` are **five identical files**
  (same md5) — placeholders, not five different tracks. Do not use them as
  a varied corpus.
