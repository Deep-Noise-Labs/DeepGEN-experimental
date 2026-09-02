# AGENTS.md - DeepGEN experimental

Cross-tool contract and running handover for agents working in this repo. Read it
in full before touching anything. It follows the [agents.md](https://agents.md)
convention and is read unchanged by Claude Code, Codex, Cursor and OpenClaw.

Its second job is to stop the next agent repeating work that is already done or
already known to be a dead end. **Append to the log at the bottom when you finish
a session.**

## What this repo is

An experimental text-to-sample model for short audio (3-15 s) from natural
language: synthesiser timbres and sampled instruments. Latent diffusion
transformer with conditional flow matching, 44.1 kHz stereo. The lineage is
MusicGen by inspiration only - the architecture here is latent-diffusion, not
autoregressive, and the point of the project is that it becomes Deep Noise IP
rather than a fork.

The quality bar is not "recognisably audio". It is **sample-library grade**:
material that sits next to Spitfire, Serum or Splice content in a session without
apology. `docs/EVALS.md` turns that into gates that can fail.

## Ground truth, stated plainly

**There is no trained checkpoint.** Not in this repo, not referenced from it. The
repository is architecture, training scaffolding and now an eval suite. Every
number anywhere in this repo is a measurement of the *signal path*, never of a
trained model's output.

This is the single most important thing to understand before writing a report or
a claim. If you are asked for "before and after audio from the model", the honest
answer is that the model cannot generate anything yet, and you must say so rather
than producing something that implies otherwise.

## Setup

```bash
uv sync --all-extras                     # CPU
uv sync --all-extras --index "https://download.pytorch.org/whl/cu121"   # GPU
uv run pytest tests/ -q                  # 71 tests, all should pass
uv run python -m synthgen.eval.alias_bench --out results/
```

## Conventions

- British English in docs and comments. Plain hyphens `-` in human-facing copy,
  never em-dashes or en-dashes. This is a standing preference across Deep Noise
  repos, not a style opinion.
- Branch and PR, never push to `main`.
- Anything claimed as a measurement must be reproducible by a command in the
  repo. If it cannot be re-run, it does not go in a report.

## Session log

### 2026-09-02 - alias-free activations in the VAE (PR #14)

**What was done.** Identified and fixed the aliasing defect in the VAE's Snake
activation; added `synthgen/eval/` and `docs/EVALS.md`. Branch
`claude/sleepy-archimedes-4t2mzc`.

**Why this and not something else.** In a latent-diffusion audio model the VAE
sets the hard quality ceiling - the DiT can never produce anything the decoder
cannot render. Within the VAE, the activation was the defect that mattered most
for *this* product, because Snake is periodic and therefore manufactures
harmonics above the feature map's Nyquist rate, which fold back as inharmonic
partials. Inharmonic grit is precisely what makes a synth sound cheap. It was
also the one defect that could be *proved* with no GPU and no checkpoint, because
a pointwise non-linearity's aliasing is fixed by architecture, not by weights.

**Headline measurements** (reproduce with `synthgen-alias-bench`):

| probe | before | after |
|---|---|---|
| 8 kHz tone, alpha=2 | -17.1 dB | -65.3 dB |
| 8 kHz tone, alpha=1 | -29.9 dB | -92.9 dB |
| 4 kHz tone, alpha=2 | -32.4 dB | -43.3 dB |
| THD, 8 kHz, alpha=2 | -14.8 dB | -15.1 dB (preserved) |

Cost 2.4x on a CPU forward pass. No parameters added.

**Things already settled - do not re-derive these.**

1. **Oversampling ratio 2 beats 4 and 8.** Counterintuitive but measured: with a
   fixed 32-tap kernel, a higher ratio forces a sharper cutoff than the filter
   can realise, so quality *degrades* while cost rises linearly. 8 kHz/alpha=2:
   -65 dB at r=2, -55 at r=4, -42 at r=8.
2. **Filter taps are nearly free** because the operation is memory-bound. 32 taps
   over BigVGAN's reference 12 buys ~13 dB alias suppression and ~15 dB air-band
   accuracy for single-digit percent runtime. Hence `DEFAULT_FILTER_TAPS = 32`.
3. **Pushing a probe through the untrained full VAE proves nothing.** Output is
   uncorrelated with input (measured r = 0.0001), so the spectrum describes the
   random weights. `bench_vae` is retained *only* so this is not rediscovered as
   though it were a result. It reports positive ASR and shows the alias-free
   build as marginally worse; that is noise, not a regression.
4. The VAE docstring claimed 2048x compression; the strides `(4,4,8,8)` give
   **1024x**. Fixed, and now guarded by a test.

**Known limitations, in priority order for whoever is next.**

1. **The resampling stack is a second, unaddressed alias source.** The encoder's
   strided convolutions are decimators and the decoder's transposed convolutions
   are interpolators. Both are time-varying, so unlike ordinary convolution they
   manufacture new frequencies. Wrapping the activation does nothing for this.
   Literature (BigVGAN, DAC, Stable Audio) leaves learned resamplers alone on the
   assumption that training partially compensates, but that assumption is
   untested here. **Quantifying it needs a trained checkpoint.**
2. **2x oversampling stops being enough at depth.** At the 2756 Hz feature rate a
   probe at 0.29 of the working rate still measures -25 dB after the fix, against
   a -60 dB gate. Deeper blocks may need a higher ratio *with a correspondingly
   longer kernel* - note finding (1) above before trying it.
3. The Tier 2 perceptual evals in `docs/EVALS.md` (FAD, CLAP, pitch accuracy,
   loop continuity, MUSHRA vs commercial references) are specified but not
   implemented; they all need a checkpoint.

**Environment gotchas that cost time.**

- `download.pytorch.org` and `api.deepnoise.ai` are blocked by the sandbox
  network policy (proxy answers 403 to CONNECT). PyPI *is* reachable, so
  `pip install torch` works where the pytorch index URL does not.
- The AWS MCP connector's token was expired, and the `AWS_ACCESS_KEY_ID` in the
  container was stale (`InvalidAccessKeyId`). No S3 access was available.
- **Audio sources in these repos are mostly not what they look like.**
  `aisynthesizer-web/public/demo-audio/take-*.wav` are hand-written JS
  placeholders, not model output - the generator script says so. The five
  `deepnoise-web-assets/player/track*.mp3` files are labelled as five different
  instruments on the live site but are **byte-identical** (same MD5), 128 kbps,
  and heavily band-limited; they are unusable as an audio probe and the
  duplication looks like a genuine content bug worth reporting to the web team.
  The only full-bandwidth real audio found was `aisynth-vst/assets/whitenoise.wav`
  and, at lower bandwidth, `aisynth-vst/assets/guitar.wav`.

**Next most valuable work, in my judgement.** Train even a small VAE on a narrow
subset. Almost every open question above - the resampling alias path, whether
alpha actually drifts upward, real reconstruction quality, the entire Tier 2
suite - is blocked on having weights, and none of it can be resolved by more
static analysis. The eval harness is now in place to score that run the moment it
exists.
