# Agent retrospectives

One file per session that measured or changed DeepGEN. Each records what
existed on arrival, what was measured, what shipped, **what did not work**, and
what to do next - so the next agent or human starts from evidence instead of
repeating the work.

Read this index, then the newest retrospective, then
[`OPEN_PR_INDEX.md`](../OPEN_PR_INDEX.md) before opening a pull request.

Append a new file per session. Do not rewrite earlier ones: a conclusion that
was later corrected is more useful than a tidy history.

| Date | File | One line |
|---|---|---|
| 2026-09-05 | [decoder-upsampler](2026-09-05-decoder-upsampler.md) | The VAE's resamplers are 2-tap-per-phase and cannot suppress images; a fixed Kaiser filter buys 79.5 dB on real renders at zero in-band cost and zero parameters. Also: 16 open PRs, none merged, and what to do about that. |

## Retrospectives that live on unmerged branches

Earlier sessions wrote theirs before this folder existed, so they are still on
their own pull-request branches. Read them there - they contain findings this
folder would otherwise duplicate:

| PR | Where | What it establishes |
|---|---|---|
| #16 | `AGENTS.md` | Alias-free activations: 19.0 dB mean alias reduction in the residual stack. Measurement hygiene that later sessions depend on - 4-term Blackman-Harris, scored probe frequencies, the attributability floor. Its "go next" list is where the 2026-09-05 work came from. |
| #15 | `AGENT_NOTES.md` | The reconstruction objective. Measured facts about the production model from twelve real generations. A list of five things that were tried and failed - including that longer FFT windows do **not** help and that multi-resolution STFT loss cannot fix pitch. |
| #14 | `AGENTS.md`, `docs/EVALS.md` | First eval suite and alias bench. |

**The single most repeated finding across all of them:** the reconstruction
objective and the operators around it are the ceiling on everything, and none
of it can be turned into a quality claim without one Stage-1 VAE training run
on a GPU. That run has been the top unmet need for three sessions.
