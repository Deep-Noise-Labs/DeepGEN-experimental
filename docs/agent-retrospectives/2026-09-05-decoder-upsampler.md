# The decoder's resamplers (2026-09-05)

Written by the agent that did the work, for the agent that comes next.
**Read `README.md` in this folder first**, then this.

## What I was asked

The standing scheduled prompt: improve DeepGEN towards Spitfire / Serum /
Splice-grade quality, propose evals that define the job, execute, prove it with
real audio, and leave a retrospective so the next agent does not repeat the
work.

## The first thing I found, and it changed the job

**Sixteen open pull requests. Zero merged.** They cluster into three ideas:

| Cluster | PRs |
|---|---|
| Alias-free / anti-aliased activations | #1, #11, #12, #14, #16 |
| Perceptual and/or adversarial VAE objective | #2, #4, #5, #6, #7, #8, #9, #10, #12, #15 |
| Flow-matching timestep sampling | #3 |
| Data preprocessing | #13 |

Three of those - #14, #15, #16 - were written by *this same scheduled routine*
on 2026-09-02, 09-03 and 09-04. The routine has been producing one
well-evidenced PR per day into a queue nobody merges. Adding a seventeenth
version of "improve the VAE objective" would have been the wrong move, and
`docs/OPEN_PR_INDEX.md` is the other half of this session's output: the
clusters, one shortlisted PR per cluster, and a do-not-redo list.

The work I did instead was the job my predecessor explicitly declined and
handed over. From `AGENTS.md` on PR #16, "Where I would go next, in order":

> **1. Measure the transposed-convolution upsampler.** `DecoderBlock` uses
> `ConvTranspose1d(kernel=2·stride, stride=stride)` - the textbook
> checkerboard-artefact setup. I deliberately did *not* change it, because
> BigVGAN keeps transposed convs and I had no measurement to justify touching
> it. Get one. This is the most likely next win.

I checked all sixteen PRs first: **none of them touches the resamplers.** PR
#11's `upsample` hits are the alias-free sandwich *inside the activation*, not
`DecoderBlock`'s `ConvTranspose1d`. The job was genuinely unclaimed.

## What I found

The framing in that handover - "checkerboard artefact" - is not quite the right
diagnosis, and getting it right mattered. Odena's checkerboard comes from
*uneven overlap*, when `kernel_size % stride != 0`. Here `kernel = 2 * stride`,
so the overlap is even and the classic checkerboard is absent.

The real defect is **imaging**. A transposed convolution with stride `s` and
kernel `2s` is a polyphase interpolator with two taps per phase, and two taps
cannot suppress the spectral images that rate conversion creates. The encoder
has the mirror defect: decimation with no anti-alias filter. Both produce
inharmonic content, which is the audible kind.

Results, all with matched weights, all in `docs/UPSAMPLER.md`:

- **75 dB** more image rejection at stride 4 and stride 8 on scored sine probes.
- **79.5 dB** less invented content on the six real renders, mean over 6 of 6
  attributable clips, at **zero in-band fidelity cost** (35.6 dB both arms) and
  **zero added parameters**.
- The encoder currently returns out-of-band content at **+0.2 dB (stride 4) and
  +4.1 dB (stride 8) relative to the wanted signal** - louder than the signal.
- **The architectural ceiling is about -22 dB.** Parks-McClellan says no kernel
  of length `2 * stride` can do better, so this is not something training
  fixes. An empirical linear-interpolation baseline lands at -22.0 dB
  independently, which is a satisfying cross-check.

## What I got wrong, and how it was caught

Five things. Four of them would have shipped a confidently wrong number.

**1. The convolution bias floored the encoder measurement.** When the
anti-alias filter correctly nulls an out-of-band probe, the convolution's input
is ~0 and its output is *just the bias* - a constant that sets the noise floor
of the reading. I measured 8.7 dB of rejection and nearly reported it. Zeroing
the bias in both arms gives 53-57 dB. **If a filter looks like it barely
works, check what the layer outputs when its input is zero.**

**2. Un-aligned SNR flattered the fix, then aligned SNR reversed it, and both
were wrong.** Comparing waveforms without a delay search measures group-delay
difference, not fidelity - and it happens to favour the filtered arm. Adding an
integer-shift search reversed the ranking, so I nearly reported that the fix
costs 4-12 dB of fidelity and wrote a paragraph rationalising it as pass-band
droop. It was neither: the filter length was **even**, so its linear-phase
delay is a half-integer that no integer shift can remove. Forcing odd lengths
made the fixed arm better on both metrics on all six clips.
**An even-length linear-phase FIR will lie to any waveform-domain comparison.**

**3. I chose the filter's transition width by reasoning, and the reasoning was
backwards.** I argued a wide transition must dull the pass-band, so the filter
should be long and sharp, and set `transition=0.25, taps=32*stride`. The
ablation says a *wider* transition is better at fixed length: it raises the
Kaiser beta, which deepens the stop-band far faster than the extra droop costs.
The shipped default is `transition=0.5, taps=16*stride` - half the compute and
better rejection than what I reasoned my way to.

**4. My first Kaiser design was silently a rectangular window.** With
`half_width` set too small for the tap count, the standard attenuation estimate
falls below 21 dB, the beta formula returns 0, and you get an unwindowed
truncated sinc with ~-29 dB sidelobes. It looks like a working filter in code
and in a diff. The stop-band test now asserts an **absolute** criterion
(-60 dB above 1.5x cutoff), deliberately not one derived from the transition
parameter - a test that recomputes its threshold from the parameter it is
checking passes for every value and therefore checks nothing.

**5. My top probe frequency measured the signal as its own image.** The
image-to-signal metric counts energy below `edge / GUARD` as signal; a probe
whose baseband component lands above that gets scored as an image and the ratio
explodes. It showed up as an impossible upward hook at the right-hand end of
the pitch curve. Probes are now bounded by `MAX_PROBE_NU`.

Two lessons I inherited and did **not** have to rediscover, both from the
anti-aliasing session: use 4-term Blackman-Harris rather than `numpy.blackman`
(whose -58 dB sidelobes cap readings in exactly the interesting range), and
score probe frequencies before using them. They saved me hours. Keep the habit.

## What is proven, and what is not

**Proven, reproducibly, with matched weights:** everything in the tables in
`docs/UPSAMPLER.md`. The equivalence tests are what license those claims -
`tests/test_resample.py` asserts the band-limited operators are bit-for-bit
identical to the ones they replace when the filter is a unit impulse.

**Not proven:**

- Anything about a **trained** model. There is still no DeepGEN checkpoint.
  This is an operator property, not a quality result. Do not let it be
  described as "the model sounds better".
- Any listening preference. No panel was run.
- Any comparison against Serum, Spitfire or Splice. Nothing was measured
  against them and nothing should be claimed.
- Compute cost. One depthwise FIR of 65 taps (stride 4) or 129 (stride 8) per
  rate change is real work and I did not time it against a training step.

## Where I would go next, in order

1. **Merge something.** The queue is the bottleneck now, not the ideas. See
   `docs/OPEN_PR_INDEX.md` for one shortlisted PR per cluster. Until a Stage-1
   VAE can actually be trained with these changes, every further PR is
   unfalsifiable.
2. **Stage-1 VAE A/B on a GPU**, `bandlimited=False` vs `True`, same seed, same
   data, same schedule. This is the experiment that turns every operator
   measurement in this repo into a quality result. It needs GPU time, not
   another CPU session. This has now been the top unmet need for three
   sessions running.
3. **A pitch-aware loss term.** Still, in my reading, the highest-value *open*
   idea in the repository, and PR #15's detune sweep already showed why
   multi-resolution STFT cannot supply it. Related: the QC bake-off in
   `dnl-inference-backend` found the production model renders the wrong note on
   6 of 6 real renders, and the whole "QC brain" exists downstream to repair
   it. Whether an upstream pitch-aware objective would remove the need for that
   repair is an **inference, not a measured fact** - but it is the most
   valuable thing left to test.
4. **The encoder's residual stack**, which decimates 1024x. I filtered the
   rate change; I did not look at whether the stack before it is well
   conditioned.
5. **Time the filters.** If the cost is material, `taps=8*stride` still gives
   -82 dB and halves it again.

## Environment (it has been identical for four sessions - plan for it)

- **No GPU.** 4 CPUs, 15 GB RAM, ~29 GB disk.
- **AWS is unavailable.** The MCP connector's token is expired and the
  environment's `AWS_ACCESS_KEY_ID` returns `InvalidAccessKeyId`. This has been
  true every session since 2026-09-02. **Do not spend time on it** - get the
  connector re-authorised out of band, or use committed audio.
- `api.deepnoise.ai` is **403 at the egress proxy**. No fresh generations. This
  is an organisation egress policy, not a fault to route around.
- `download.pytorch.org` is blocked; plain `pip install torch` from PyPI works.
- **`git push` is 403 for every token in the environment.** Reads (clone,
  fetch) work. The only write path that works is the GitHub MCP server, which
  takes **text only** - so figures and audio could not be committed this
  session, and `experiments/` regenerates them instead. Budget for this: it
  cost me an hour at the end.
- Real audio therefore comes from Deep Noise GitHub repositories. The corpus I
  used - six raw production renders with documented S3 provenance and prompts -
  is `dnl-inference-backend/docs/qc-eval/2026-09-02-bakeoff/audio/*_raw_C4.ogg`.
  It is the best committed corpus I found; start there.
- `deepnoise-web-assets/player/track{1..5}.mp3` are five identical files.
  Not a varied corpus.
- Real brand tokens for any report are in `deepnoise-web`'s stylesheet
  (`--color-black #181818`, `--color-brand #16ffc5`, the neutral ramp). The
  brand skill's visual-identity file leaves them as blanks; the repo has them.
