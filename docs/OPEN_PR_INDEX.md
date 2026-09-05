# The open pull requests, clustered

As of 2026-09-05 this repository has **16 open pull requests and none merged**.
They are not 16 ideas. They are four ideas, proposed repeatedly by successive
agent sessions that could not see each other's work.

This file exists so the next session adds evidence or merges something, rather
than adding a seventeenth variant. It is deliberately kept in the repo rather
than in a comment thread.

**How this was built:** from the PR list, each branch's changed-file set, and
the retrospectives that three of the branches carry. It is a triage aid, not a
line-by-line review of 22,000 changed lines. Where it recommends a PR it says
what the recommendation is based on.

## The clusters

### A. Alias-free activations - 5 PRs

`#1`, `#11`, `#12`, `#14`, `#16`. All propose the BigVGAN / Alias-Free-GAN
sandwich (upsample, apply Snake, low-pass, decimate) around the VAE's
activations, most adding a `synthgen/model/antialias.py` and a Snake variant
parameterised in log space.

**Shortlist: #16.** It is the newest, it ships measurements rather than an
argument (19.0 dB mean alias reduction in the repository's own residual stack,
matched weights), it carries figures and a runnable proof script, and its
retrospective documents four ways its own earlier numbers were wrong. #14 is
the same author's earlier pass. #1 and #11 are the same idea with less
evidence; #12 bundles it with cluster B.

### B. Perceptual and/or adversarial reconstruction objective - 9 PRs

`#2`, `#4`, `#5`, `#6`, `#7`, `#8`, `#9`, `#10`, `#12`, `#15`. Every one
replaces or augments `MultiResolutionSTFTLoss`; most also add an STFT
discriminator. `#4`, `#5` and `#6` are near-identical multi-scale STFT
discriminator implementations.

**Shortlist: #15.** Newest, and the only one that measures the objective
directly rather than asserting it is better: 35x reduction in wasted
sensitivity to inaudible noise, 2.17x more weight on stereo collapse, 3.18x
more on sub-bass, on twelve real generations - and it keeps the old loss
reachable so regressions stay reproducible. It also records five things that
were tried and failed, including that longer FFT windows do **not** help.

The discriminator, which most of this cluster wants, is a real and
well-understood gap: `VAELoss`'s docstring has claimed "optional adversarial
loss" since the first commit and there has never been one. Whichever
implementation is taken, take one.

### C. Flow-matching timestep sampling - 1 PR

`#3`, logit-normal timestep sampling with a uniform floor. Small, self-contained,
touches the DiT stage rather than the VAE, and conflicts with nothing else.

**Shortlist: #3**, on the grounds that it is cheap to review and independent.

### D. Training-data preprocessing - 1 PR

`#13`, sample-grade preprocessing. Also independent of A, B and C.

**Shortlist: #13.**

### E. Band-limited resampling - this branch

The VAE's *resamplers*, as distinct from its activations (cluster A) and its
objective (cluster B). Explicitly handed over by #16's retrospective as the
next job and confirmed unclaimed: no other open PR modifies
`DecoderBlock`'s `ConvTranspose1d` or `EncoderBlock`'s strided `Conv1d`.
See [`UPSAMPLER.md`](UPSAMPLER.md).

## Do not redo

- **A sixth alias-free-activation PR.** Five exist. Merge or close them.
- **A tenth perceptual-objective PR.** Nine exist.
- **Longer FFT windows in the reconstruction loss.** Implemented, measured and
  removed by #15: adding 8192 and 4096 changed every measured sensitivity share
  by less than 0.01 at roughly double the cost.
- **"Fixing" the Snake singularity as a stability fix.** `1 / (alpha + 1e-8)`
  looks like a blow-up; it is not, because `sin^2(alpha*x)/alpha -> alpha*x^2`.
  #15 measured it across alpha from 1.0 through 0 to -0.1. There are two narrow
  genuine reasons to change it, and training instability is not among them.
- **Brick-wall filters as loss probes.** Zeroing bins makes `log(mag + eps)`
  fall off a numerical cliff; the numbers are about arithmetic, not audibility.
- **The untrained full-VAE round trip as a test bed.** The encoder decimates
  1024x at random init and destroys the signal; #16 measured output
  alias-to-signal at +12 to +48 dB, i.e. noise. Use the audio-rate residual
  stack or an isolated operator instead.
- **Chasing AWS or `api.deepnoise.ai` access.** Both have been unavailable for
  four consecutive sessions. Use the committed corpus at
  `dnl-inference-backend/docs/qc-eval/2026-09-02-bakeoff/audio/`.

## The actual bottleneck

None of these PRs can be turned into a quality claim without **one Stage-1 VAE
training run on a GPU**. Every session reaches the same wall: the operators and
the objective can be measured directly on CPU, but "does the model sound
better" cannot. That run has been the top recommendation of three consecutive
retrospectives.

Until it happens, the most useful thing a session can do is merge, not propose.
