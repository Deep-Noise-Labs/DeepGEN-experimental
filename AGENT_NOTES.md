# AGENT_NOTES.md

Notes from agents who have worked on DeepGEN, for the agents who come next.
**Read this before starting work.** It exists so nobody repeats an experiment
that has already been run, and nobody re-derives a conclusion that has already
been falsified.

Append a new session block at the bottom. Do not rewrite earlier blocks -- a
wrong conclusion that was later corrected is more useful to the next agent than
a tidy history.

---

## Session 2026-09-03 -- evaluation suite and the reconstruction objective

### What I was asked

Improve DeepGEN toward Spitfire / Serum / Splice-grade sample quality, propose
evals that define the job, execute, and prove it with real audio.

### State of the repo when I started

A complete but **untrained** scaffold. Latent diffusion DiT with conditional
flow matching, 44.1 kHz stereo target, ~80M VAE + T5-base + ~350M DiT. No
checkpoints exist anywhere in the repo. `main` and the working branch were
identical; the four commits of history are all scaffolding.

### Environment (this matters -- it shaped everything)

**No GPU.** 4 CPUs, 15 GB RAM, ~29 GB disk. Training is not possible here, and
neither is inference from a trained checkpoint, because there is no checkpoint.

`pip install torch` from PyPI works. `download.pytorch.org` is **blocked** by
the egress proxy (403 on CONNECT) -- do not waste time on `--index-url
https://download.pytorch.org/whl/cpu`, just use the default index.

The shell environment's `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` **cannot
read S3** (403 on HeadObject). The AWS MCP server's credentials can. To get
bytes onto local disk: call `aws___get_presigned_url` via MCP, then `curl`. The
presigned URLs for a batch share every query parameter except
`X-Amz-Signature`, so you can reconstruct all of them from one URL plus the
per-object signatures instead of pasting each in full. They expire in 900 s.

### What I built

- `synthgen/eval/` -- `metrics.py` (reference-free + reference-based sample
  quality metrics, `QualityTarget`, `grade`) and `bench.py` (the
  `synthgen-eval` CLI). NumPy/SciPy only, no torch, no GPU.
- `synthgen/training/losses.py` -- `PerceptualSampleLoss`, now the default in
  `VAELoss`. The old `MultiResolutionSTFTLoss` is retained and reachable via
  `VAELoss(spectral_loss="legacy")` so regressions stay reproducible.
- `synthgen/model/vae.py` -- `Snake` stores `log_alpha`; legacy checkpoints are
  converted by a load hook.
- `experiments/` -- four runnable scripts, all of which produced the numbers in
  `docs/EVALUATION.md`.

### The one thing to understand about this codebase

**The reconstruction loss is the ceiling on everything.** However good the DiT
gets, output quality is bounded by what the autoencoder can reconstruct, and
that is bounded by what the loss responds to. If you are looking for leverage,
start there, not at the DiT.

### Facts established about the production model (measured, not assumed)

Twelve real generations from `dnl-core-sounds-s3-prod`:

- **32 kHz, 16-bit PCM, stereo, exactly 8.000 s, 1,024,078 bytes.** Every
  object in the bucket is that same byte size. This is MusicGen's native
  format. Nyquist is 16 kHz, so the air band cannot exist at all.
- Mean pass rate against the commercial-sample spec: **53.8%**.
- 0/12 pass sample rate, 0/12 pass bandwidth, **0/12 are loopable** (mean
  end-to-start step is +2.2 dB relative to edge RMS -- louder than the audio
  around it).
- Clip RMS spans -40.7 to -15.0 dBFS. Four clips exceed 0 dBTP.

The DeepGEN scaffold already targets 44.1 kHz stereo, so the sample-rate and
bandwidth gaps are closed by architecture, not by anything I did.

### Do not repeat these -- they were tried and they failed

**1. Longer FFT windows in the loss.** The reasoning is seductive (2048 samples
at 44.1 kHz resolves only 21.5 Hz, so bass partials merge) and it is wrong in
practice. Ablating `8192, 4096` into the window set changed every measured
sensitivity share by **less than 0.01**, at roughly double the cost. They were
implemented, measured, and removed. Do not add them back without new evidence.

**2. Any hope that a multi-resolution STFT loss will fix pitch accuracy.** A
detune sweep at E1 (41.20 Hz) showed both a 2048-max and an 8192-max loss
saturate at **5 cents = 0.12 Hz**. A semitone error scores barely above an
imperceptible one. These losses respond to spectral leakage, not to resolved
partials. If bass tuning matters -- and for a synth model it does -- it needs an
explicitly pitch-aware term: an f0 tracker, a differentiable harmonic comb, or
an autocorrelation loss. **This is the most valuable open thread I found.**

**3. The Snake activation "singularity".** `1 / (alpha + 1e-8)` on an
unconstrained alpha looks like a blow-up waiting to happen. It is not:
`sin^2(alpha*x)/alpha -> alpha*x^2` as alpha approaches zero, so numerator and
denominator cancel and everything stays bounded. Measured across alpha from 1.0
through 0 to -0.1 -- nothing explodes. I shipped the log-space change anyway,
but only for two narrow and genuine reasons (an exact `inf` at `alpha ==
-1e-8`, and negative alpha computing a *mirrored* function rather than a scaled
one). **Do not describe it as a fix for training instability.** There is no
measured instability.

**4. Brick-wall filters as loss probes.** My first sensitivity experiment used
brick-wall lowpass/highpass degradations. Zeroing bins makes `log(mag + 1e-8)`
fall off a numerical cliff, and the resulting numbers are about arithmetic, not
audibility -- they were misleading in both directions. Use attenuation
(`shelf`) or substitution (`hf_noise_substitution`) instead. Both are in
`experiments/degradations.py`.

**5. An absolute magnitude floor in the log term.** I first clamped at a fixed
`1e-5`. STFT magnitudes scale with window length and clip level, so a constant
floor means a different number of dB on every clip, and it made wasted
sensitivity *worse than the legacy loss*. The fix is a floor relative to each
scale's own peak target magnitude (`dynamic_range_db`, default 60).

### What actually held up

Measured on the twelve real clips:

- **35x reduction in wasted sensitivity** (response to inaudible -90 dBFS
  noise, relative to mean audible defect: 0.1853 -> 0.0052). This is the
  headline result and it comes entirely from the relative log floor.
- **2.17x more weight on stereo collapse**, from the mid/side term. The legacy
  loss scores a completely destroyed stereo image at < 1e-5 in the worst case,
  because it compares channels independently.
- **3.18x more weight on sub-bass** (share 0.026 -> 0.084), from band weighting.
- The two high-frequency defects were essentially unchanged (0.89x and 1.03x).
  I had expected an improvement there and did not get one.

### What is NOT proven

**No model has been trained with this loss.** Everything above is a property of
the objective, measured directly. `experiments/run_loss_inversion.py` optimises
damaged real audio back toward the original under each loss with no network in
the way, which isolates the objective perfectly but is not a training run.

Do not let anyone describe this work as "the model sounds better". It is not
that, and claiming it would be false.

### The next experiment, in priority order

1. **Stage-1 VAE A/B on a GPU.** Same data, same schedule, same seed;
   `spectral_loss="legacy"` vs `spectral_loss="perceptual"`. Score
   reconstructions with `synthgen-eval --input recon --reference originals`.
   This confirms or refutes everything here. Nothing else matters as much.
2. **A pitch-aware loss term** (see failure 2 above). Highest-value open
   problem, and the eval suite already has the metrics to detect whether it
   works.
3. **A discriminator.** `VAELoss`'s docstring has claimed "optional adversarial
   loss" since the first commit and there has never been one. L1 + spectral
   alone is the standard recipe for muffled latent-decoder output. This is a
   known, well-understood gap.
4. **Loopability.** 0/12 production clips can be looped. Nothing in the current
   architecture or objective addresses it, and for a *sample library* it is a
   first-class feature, not a nicety.
5. **Loudness consistency.** `AudioTextDataset` peak-normalises every clip to
   0.95 and then applies +/-3 dB of random gain, so the model has no way to
   learn a consistent output level -- which matches the 25.7 dB RMS spread
   measured in production. An LUFS target in the data pipeline would likely fix
   this more cheaply than anything in the model.

### Reproducing my numbers

```bash
pip install torch numpy scipy soundfile librosa matplotlib pytest
PYTHONPATH=. python experiments/run_loss_sensitivity.py --input <wav-dir> --out results
PYTHONPATH=. python experiments/run_resolution_probe.py --out results
PYTHONPATH=. python experiments/run_snake_stability.py --out results
PYTHONPATH=. python experiments/run_loss_inversion.py --input <wav-dir> --out results --steps 800
python -m pytest tests/ -q
```

Full write-up with tables: `docs/EVALUATION.md`.
