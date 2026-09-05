"""
Measure the decoder's transposed-convolution upsampler and the encoder's
strided downsampler, and quantify what a fixed band-limiting filter buys.

Run:
    PYTHONPATH=. python experiments/upsampler_bench.py --out proofs/upsampler

Every arm of every comparison uses identical learnable weights; the two
modules are asserted bit-for-bit equal when the fixed filter is a unit
impulse (``tests/test_resample.py``). So a difference here is the filter's
doing and nothing else's.

Measurement hygiene, inherited from the anti-aliasing session's mistakes
(see docs/agent-retrospectives/):

- 4-term Blackman-Harris analysis window. ``numpy.blackman`` is the 3-term
  window, whose ~-58 dB sidelobes silently cap readings in exactly the range
  these numbers live in.
- Probe frequencies are scored before use, so no probe sits at a rational
  fraction of the rate that hides the defect it is meant to expose.
- A guard band around the band edge is excluded, so filter transition slope
  is never counted as an image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from synthgen.model.resample import (
    DEFAULT_TAPS_PER_STRIDE,
    DEFAULT_TRANSITION,
    BandlimitedDownsample1d,
    BandlimitedUpsample1d,
    StridedDownsample1d,
    TransposedUpsample1d,
    resampling_filter,
)

# Decoder strides, in the order AudioDecoder applies them.
DECODER_STRIDES = (8, 8, 4, 4)
ENCODER_STRIDES = (4, 4, 8, 8)

ANALYSIS_N = 1 << 14
GUARD = 1.25  # exclude +/-25% around the band edge from both regions


# =============================================================================
# Analysis primitives
# =============================================================================


def blackman_harris(n: int) -> torch.Tensor:
    """4-term Blackman-Harris: ~-92 dB sidelobes."""
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    k = torch.arange(n, dtype=torch.float64)
    w = (
        a[0]
        - a[1] * torch.cos(2 * math.pi * k / (n - 1))
        + a[2] * torch.cos(4 * math.pi * k / (n - 1))
        - a[3] * torch.cos(6 * math.pi * k / (n - 1))
    )
    return w.to(torch.float32)


def image_to_signal_db(y: torch.Tensor, stride: int) -> float:
    """
    Energy above the baseband edge, relative to energy inside it, in dB.

    ``y`` is a single-channel output at the upsampled rate. The true signal
    can only occupy ``|f| < 0.5 / stride``; everything above that is an image
    the operator invented.
    """
    y = y.detach().to(torch.float64)
    n = y.shape[-1]
    window = blackman_harris(n).to(torch.float64)
    spec = torch.fft.rfft(y * window)
    power = (spec.abs() ** 2)
    freqs = torch.fft.rfftfreq(n)

    edge = 0.5 / stride
    signal = power[freqs < edge / GUARD].sum().item()
    image = power[freqs > edge * GUARD].sum().item()
    if signal <= 0:
        return float("nan")
    return 10 * math.log10(max(image, 1e-300) / signal)


def alias_to_signal_db(y: torch.Tensor, probe_out_freq: float) -> float:
    """
    For decimation: energy anywhere except the probe's own bin, relative to
    the probe bin. A clean decimator passing an in-band tone leaves only the
    tone.
    """
    y = y.detach().to(torch.float64)
    n = y.shape[-1]
    window = blackman_harris(n).to(torch.float64)
    spec = torch.fft.rfft(y * window)
    power = (spec.abs() ** 2)
    freqs = torch.fft.rfftfreq(n)

    # A narrow band around the probe counts as signal; the window's main lobe
    # is a few bins wide.
    width = 8.0 / n
    on = (freqs > probe_out_freq - width) & (freqs < probe_out_freq + width)
    signal = power[on].sum().item()
    rest = power[~on].sum().item()
    if signal <= 0:
        return float("nan")
    return 10 * math.log10(max(rest, 1e-300) / signal)


def image_visibility(nu: float, stride: int) -> float:
    """
    Score a probe frequency (cycles/sample at the *input* rate).

    Returns the smallest normalised distance, at the output rate, between the
    baseband component and any image. If that distance is tiny the probe
    cannot reveal imaging, because the image lands on top of the signal.
    Bigger is better; anything below ``1 / ANALYSIS_N`` is unusable.
    """
    base = nu / stride
    worst = 1.0
    for k in range(1, stride):
        for image in (k / stride + base, k / stride - base):
            image = abs(image)
            if image > 0.5:
                image = 1.0 - image
            worst = min(worst, abs(image - base))
    return worst


# Highest usable probe. image_to_signal_db counts energy below edge / GUARD as
# signal, so a probe whose own baseband component lands above that is scored as
# its own image and the ratio explodes. The bound is 0.5 / GUARD; back off a
# further 5% for the analysis window's main lobe.
MAX_PROBE_NU = 0.5 / GUARD * 0.95


def pick_probe_frequencies(stride: int, count: int = 12) -> list[float]:
    """
    Evenly spread probes across the input band, each scored before use.

    Two filters apply: the probe must not be so high that the measurement
    counts the signal as an image (``MAX_PROBE_NU``), and its images must be
    resolvable from the signal (``image_visibility``).
    """
    candidates = np.linspace(0.02, MAX_PROBE_NU, count * 4)
    chosen: list[float] = []
    for nu in candidates:
        if image_visibility(float(nu), stride) < 32.0 / ANALYSIS_N:
            continue
        if any(abs(nu - c) < 0.4 / count for c in chosen):
            continue
        chosen.append(float(nu))
        if len(chosen) == count:
            break
    return chosen


def sine(nu: float, n: int) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32)
    return torch.sin(2 * math.pi * nu * t).view(1, 1, n)


# =============================================================================
# Weight settings for the baseline arm
# =============================================================================


def set_nearest_hold(module, stride: int) -> None:
    """Zero-order hold: the interpolator a 2-tap-per-phase kernel does best."""
    with torch.no_grad():
        w = module.conv.weight if hasattr(module, "conv") else module.weight
        b = module.conv.bias if hasattr(module, "conv") else module.bias
        w.fill_(0.0)
        w[0, 0, :stride] = 1.0
        b.fill_(0.0)


def set_linear_interp(module, stride: int) -> None:
    """Linear interpolation: the best *smooth* 2-tap-per-phase kernel."""
    with torch.no_grad():
        w = module.conv.weight if hasattr(module, "conv") else module.weight
        b = module.conv.bias if hasattr(module, "conv") else module.bias
        taps = 1.0 - torch.abs(torch.arange(2 * stride, dtype=torch.float32) - stride) / stride
        w.fill_(0.0)
        w[0, 0, :] = taps
        b.fill_(0.0)


def zero_bias(module) -> None:
    """
    Zero the convolution bias in both arms before measuring.

    This is not cosmetic. When the anti-alias filter does its job on an
    out-of-band probe, the convolution's input is ~0 and its output is just
    the bias -- a constant that floors the reading and makes a working filter
    look like a 9 dB improvement instead of a 70 dB one. Both arms are
    zeroed identically, so the comparison stays matched.
    """
    with torch.no_grad():
        b = module.conv.bias if hasattr(module, "conv") else module.bias
        b.fill_(0.0)


def copy_weights(src, dst) -> None:
    with torch.no_grad():
        sw = src.conv.weight if hasattr(src, "conv") else src.weight
        sb = src.conv.bias if hasattr(src, "conv") else src.bias
        dw = dst.conv.weight if hasattr(dst, "conv") else dst.weight
        db = dst.conv.bias if hasattr(dst, "conv") else dst.bias
        dw.copy_(sw)
        db.copy_(sb)


# =============================================================================
# P1 -- image rejection across the input band, matched weights
# =============================================================================


def bench_image_vs_pitch(stride: int, seeds: int = 8) -> dict:
    probes = pick_probe_frequencies(stride)
    rows = []

    for nu in probes:
        x = sine(nu, ANALYSIS_N // stride)
        entry = {"nu": nu, "visibility": image_visibility(nu, stride)}

        # Hand-set interpolators: deterministic, no seed averaging needed.
        for name, setter in (("hold", set_nearest_hold), ("linear", set_linear_interp)):
            base = TransposedUpsample1d(1, 1, stride)
            fixed = BandlimitedUpsample1d(1, 1, stride)
            setter(base, stride)
            copy_weights(base, fixed)
            entry[f"baseline_{name}_db"] = image_to_signal_db(base(x)[0, 0], stride)
            entry[f"fixed_{name}_db"] = image_to_signal_db(fixed(x)[0, 0], stride)

        # Random init: what training actually starts from.
        b_rand, f_rand = [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            base = TransposedUpsample1d(1, 1, stride)
            fixed = BandlimitedUpsample1d(1, 1, stride)
            zero_bias(base)
            copy_weights(base, fixed)
            b_rand.append(image_to_signal_db(base(x)[0, 0], stride))
            f_rand.append(image_to_signal_db(fixed(x)[0, 0], stride))
        entry["baseline_random_db"] = float(np.mean(b_rand))
        entry["baseline_random_std"] = float(np.std(b_rand))
        entry["fixed_random_db"] = float(np.mean(f_rand))
        entry["fixed_random_std"] = float(np.std(f_rand))

        rows.append(entry)

    return {"stride": stride, "probes": rows}


# =============================================================================
# P2 -- the architectural ceiling: best achievable rejection vs kernel length
# =============================================================================


def optimal_lowpass_rejection(stride: int, kernel_size: int, transition: float) -> float:
    """
    Best stop-band rejection achievable by *any* FIR of this length for this
    rate change, via Parks-McClellan (optimal in the minimax sense).

    This is what makes the argument architectural rather than a matter of
    training: a transposed convolution of kernel ``2 * stride`` is a polyphase
    interpolator whose anti-imaging filter *is* that kernel, so no amount of
    training can beat this number.
    """
    from scipy.signal import freqz, remez

    cutoff = 0.5 / stride
    pass_edge = cutoff * (1.0 - transition)
    stop_edge = cutoff * (1.0 + transition)
    try:
        taps = remez(
            kernel_size,
            [0.0, pass_edge, stop_edge, 0.5],
            [1.0, 0.0],
            fs=1.0,
        )
    except Exception:
        return float("nan")

    w, h = freqz(taps, worN=8192, fs=1.0)
    passband = np.abs(h[w < pass_edge])
    stopband = np.abs(h[w > stop_edge])
    if passband.max() <= 0:
        return float("nan")
    rejection = float(20 * np.log10(stopband.max() / passband.max()))
    # Parks-McClellan stops converging in float64 somewhere past -150 dB and
    # starts returning worse numbers for longer filters. Anything beyond that
    # is a numerical artefact, not a filter, so it is not reported.
    if rejection < -150.0:
        return float("nan")
    return rejection


def bench_ceiling(stride: int) -> dict:
    rows = []
    for taps_per_phase in (2, 4, 6, 8, 12, 16, 24, 32):
        kernel_size = taps_per_phase * stride
        rows.append(
            {
                "taps_per_phase": taps_per_phase,
                "kernel_size": kernel_size,
                "optimal_rejection_db": optimal_lowpass_rejection(
                    stride, kernel_size, DEFAULT_TRANSITION
                ),
                "shipped_rejection_db": measured_filter_rejection(
                    stride, kernel_size, DEFAULT_TRANSITION
                ),
                "shipped_passband_droop_db": measured_passband_droop(
                    stride, kernel_size, DEFAULT_TRANSITION
                ),
            }
        )
    return {"stride": stride, "rows": rows, "current_taps_per_phase": 2}


def measured_passband_droop(stride: int, kernel_size: int, transition: float) -> float:
    """
    Worst attenuation inside the band the filter is meant to pass, in dB.

    This is the cost side of the trade. A filter with a magnificent stop-band
    that dulls the top of its own pass-band has moved the defect rather than
    removed it, so the two numbers are always reported together.
    """
    taps = resampling_filter(stride, kernel_size, transition)[0, 0].to(torch.float64)
    response = torch.fft.rfft(taps, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192)
    cutoff = 0.5 / stride
    passband = response[freqs < cutoff * (1 - transition)]
    if passband.numel() == 0 or passband.max() <= 0:
        return float("nan")
    return float(20 * math.log10(passband.min().item() / passband.max().item()))


def measured_filter_rejection(stride: int, kernel_size: int, transition: float) -> float:
    """Stop-band rejection of the Kaiser filter this repository actually ships."""
    taps = resampling_filter(stride, kernel_size, transition)[0, 0].to(torch.float64)
    response = torch.fft.rfft(taps, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192)
    cutoff = 0.5 / stride
    passband = response[freqs < cutoff * (1 - transition)].max().item()
    stopband = response[freqs > cutoff * (1 + transition)].max().item()
    if passband <= 0:
        return float("nan")
    return float(20 * math.log10(stopband / passband))


# =============================================================================
# P3 -- compounding through the four decoder stages
# =============================================================================


def bench_stage_compounding(nu: float = 0.21, seeds: int = 8) -> dict:
    """
    Run a probe through the four decoder stages in sequence and measure
    imaging after each. The signal's band shrinks relative to the running
    rate at every stage, so the image region is defined by the cumulative
    ratio.

    Two weight settings, because they answer different questions:

    ``linear`` gives every baseline stage a competent hand-built
    interpolator -- the best a 2-tap-per-phase kernel can do smoothly. This
    is the fair comparison and the one to quote.

    ``random`` is initialisation, which is where training starts. It is
    reported for completeness and is not evidence about a trained decoder.
    """
    rows = []
    for weights in ("linear", "random"):
        for arm in ("baseline", "fixed"):
            per_seed = []
            for seed in range(seeds):
                torch.manual_seed(seed)
                mods = []
                for stride in DECODER_STRIDES:
                    base = TransposedUpsample1d(1, 1, stride)
                    zero_bias(base)
                    if weights == "linear":
                        set_linear_interp(base, stride)
                    if arm == "baseline":
                        mods.append(base)
                    else:
                        fixed = BandlimitedUpsample1d(1, 1, stride)
                        copy_weights(base, fixed)
                        mods.append(fixed)

                x = sine(nu, 512)
                cumulative = 1
                trace = []
                for stride, mod in zip(DECODER_STRIDES, mods):
                    x = mod(x)
                    cumulative *= stride
                    # Normalise level so the reading is a ratio, not a gain.
                    x = x / (x.abs().max() + 1e-12)
                    trace.append(image_to_signal_db(x[0, 0], cumulative))
                per_seed.append(trace)
                if weights == "linear":
                    break  # deterministic; one pass is the whole answer
            arr = np.array(per_seed)
            rows.append(
                {
                    "weights": weights,
                    "arm": arm,
                    "mean_db": arr.mean(axis=0).tolist(),
                    "std_db": arr.std(axis=0).tolist(),
                }
            )
    return {
        "strides": list(DECODER_STRIDES),
        "cumulative": [8, 64, 256, 1024],
        "probe_nu": nu,
        "arms": rows,
    }


# =============================================================================
# P4 -- encoder decimation
# =============================================================================


def bench_encoder_alias(stride: int, seeds: int = 8) -> dict:
    """
    Feed tones *above* the post-decimation Nyquist and see where they end up.
    A tone at input frequency nu > 0.5/stride must not appear in the output;
    without a filter it folds back to an audible, inharmonic position.
    """
    rows = []
    edge = 0.5 / stride
    # The filter's stop-band only begins at edge * (1 + transition); probes
    # below that sit in the transition band and are reported as such rather
    # than quietly averaged in.
    stop_edge = edge * (1.0 + DEFAULT_TRANSITION)
    # A reference in-band tone, put through the same weights, turns the
    # reading into a level-independent fold-to-signal ratio.
    reference = sine(edge * 0.5, ANALYSIS_N)

    for nu in np.linspace(edge * 1.1, 0.45, 12):
        nu = float(nu)
        folded = abs(((nu * stride) + 0.5) % 1.0 - 0.5)
        x = sine(nu, ANALYSIS_N)

        b_vals, f_vals = [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            base = StridedDownsample1d(1, 1, stride)
            fixed = BandlimitedDownsample1d(1, 1, stride)
            zero_bias(base)
            copy_weights(base, fixed)

            ref_db = 10 * math.log10(float((base(reference)[0, 0] ** 2).mean()) + 1e-30)
            b_vals.append(
                10 * math.log10(float((base(x)[0, 0] ** 2).mean()) + 1e-30) - ref_db
            )
            f_vals.append(
                10 * math.log10(float((fixed(x)[0, 0] ** 2).mean()) + 1e-30) - ref_db
            )
        rows.append(
            {
                "nu": nu,
                "folds_to": folded,
                "in_stopband": bool(nu > stop_edge),
                "baseline_fold_db": float(np.mean(b_vals)),
                "fixed_fold_db": float(np.mean(f_vals)),
                "rejection_db": float(np.mean(b_vals) - np.mean(f_vals)),
            }
        )
    return {"stride": stride, "stop_edge": stop_edge, "rows": rows}


# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/upsampler")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "decoder_strides": list(DECODER_STRIDES),
            "encoder_strides": list(ENCODER_STRIDES),
            "default_taps_per_stride": DEFAULT_TAPS_PER_STRIDE,
            "default_transition": DEFAULT_TRANSITION,
            "analysis_n": ANALYSIS_N,
            "seeds": args.seeds,
        },
        "image_vs_pitch": {},
        "ceiling": {},
        "encoder_alias": {},
    }

    for stride in (4, 8):
        print(f"[P1] image rejection vs pitch, stride {stride}")
        results["image_vs_pitch"][str(stride)] = bench_image_vs_pitch(stride, args.seeds)
        print(f"[P2] architectural ceiling, stride {stride}")
        results["ceiling"][str(stride)] = bench_ceiling(stride)
        print(f"[P4] encoder decimation, stride {stride}")
        results["encoder_alias"][str(stride)] = bench_encoder_alias(stride, args.seeds)

    print("[P3] stage compounding")
    results["compounding"] = bench_stage_compounding(seeds=args.seeds)

    path = out / "bench.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
