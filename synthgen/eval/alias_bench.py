"""
Alias benchmark: measure how much inharmonic energy the VAE signal path adds.

Two levels of measurement, both weight-matched A/B:

1. ``bench_activation`` -- the non-linearity in isolation, at each internal
   feature rate the VAE actually runs at. Isolates the defect.
2. ``bench_vae`` -- a probe pushed through the complete VAE. **Diagnostic
   only; not a quality measurement.** See the warning on that function.

Only (1) is evidence. It is a genuine measurement because the activation is
pointwise: its aliasing behaviour is fixed by the architecture and cannot be
learned away by training, so measuring it on an untrained model is valid.

(2) is reported for completeness but proves nothing about quality, and was
kept precisely so that nobody re-derives it and mistakes it for a result.
Two reasons, both verified rather than assumed:

* With random weights the output is uncorrelated with the input (measured
  r = 0.0001 on a 1 kHz sine), so the spectrum is dominated by the random
  weights, not by the probe.
* The encoder's strided convolutions are *decimators* and the decoder's
  transposed convolutions are *interpolators*. Those are time-varying
  operations, so -- unlike ordinary convolution -- they do manufacture new
  frequencies. The resampling stack is therefore an alias source in its own
  right, which wrapping the activation does not address.

Nothing in this module measures reconstruction quality; that is a property of
trained weights and cannot be assessed before a checkpoint exists.

Usage:
    uv run python -m synthgen.eval.alias_bench --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synthgen.eval.metrics import (
    alias_to_signal_ratio,
    band_energy_error_db,
    crest_factor_db,
    total_harmonic_distortion,
)
from synthgen.eval.probes import pluck, saw_note, sine
from synthgen.model.vae import AudioVAE, make_activation, remap_state_dict

SAMPLE_RATE = 44100

# The rates the VAE's feature maps actually run at, given strides (4, 4, 8, 8).
# Aliasing gets monotonically worse as the rate drops, because Nyquist falls
# while the non-linearity keeps generating the same harmonics.
FEATURE_RATES = (44100, 11025, 2756, 344)


def _apply(module: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None, :]
    with torch.no_grad():
        return module(tensor)[0, 0].numpy()


def bench_activation(
    frequencies: tuple[float, ...] = (440.0, 1000.0, 2000.0, 4000.0, 8000.0),
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0),
    duration: float = 1.0,
) -> list[dict]:
    """A/B the activation alone, at full rate, across tones and Snake alphas."""
    rows = []
    for alpha in alphas:
        for f0 in frequencies:
            probe = sine(f0, duration, SAMPLE_RATE)
            row: dict[str, float | str] = {"f0_hz": f0, "alpha": alpha}
            for label, antialias in (("baseline", False), ("antialiased", True)):
                activation = make_activation(1, antialias=antialias, alpha_init=alpha)
                out = _apply(activation, probe)
                row[f"asr_db_{label}"] = alias_to_signal_ratio(out, f0, SAMPLE_RATE)
                row[f"thd_db_{label}"] = total_harmonic_distortion(out, f0, SAMPLE_RATE)
            row["asr_improvement_db"] = (
                float(row["asr_db_baseline"]) - float(row["asr_db_antialiased"])
            )
            rows.append(row)
    return rows


def bench_feature_rates(
    f0: float = 2000.0,
    alpha: float = 1.0,
    duration: float = 1.0,
) -> list[dict]:
    """
    A/B the activation at each internal feature rate.

    Shows why the defect compounds with depth: the deeper the block, the lower
    its rate, the lower its Nyquist, and the more of Snake's harmonic output
    folds back.
    """
    rows = []
    for rate in FEATURE_RATES:
        if f0 >= rate / 2:
            continue
        probe = sine(f0, duration, rate)
        row: dict[str, float] = {"feature_rate_hz": rate, "f0_hz": f0, "alpha": alpha}
        for label, antialias in (("baseline", False), ("antialiased", True)):
            activation = make_activation(1, antialias=antialias, alpha_init=alpha)
            out = _apply(activation, probe)
            row[f"asr_db_{label}"] = alias_to_signal_ratio(out, f0, rate)
        row["asr_improvement_db"] = row["asr_db_baseline"] - row["asr_db_antialiased"]
        rows.append(row)
    return rows


def build_matched_pair(seed: int = 0) -> tuple[AudioVAE, AudioVAE]:
    """
    Build baseline and alias-free VAEs holding *identical* weights.

    Without this the two models would differ by random initialisation as well
    as by the fix, and no difference could be attributed to the change.
    """
    torch.manual_seed(seed)
    antialiased = AudioVAE(antialias=True).eval()

    baseline = AudioVAE(antialias=False).eval()
    baseline.load_state_dict(
        remap_state_dict(antialiased.state_dict(), to_antialias=False),
        strict=True,
    )
    return baseline, antialiased


def bench_vae(
    f0: float = 1000.0,
    duration: float = 0.5,
    seed: int = 0,
) -> list[dict]:
    """
    Push probes through the complete (untrained, weight-matched) VAE.

    WARNING: diagnostic only. On an untrained network the output is
    uncorrelated noise, so the resulting ASR describes the random weights and
    not the architecture. Measured values are positive (alias energy exceeding
    harmonic energy), and the alias-free build scores marginally *worse* --
    which is what noise-dominated measurement looks like, not a regression.

    Re-run this once a checkpoint exists; it only becomes a real number then.
    """
    baseline, antialiased = build_matched_pair(seed)

    probes = {
        "sine": sine(f0, duration, SAMPLE_RATE),
        "saw": saw_note(f0, duration, SAMPLE_RATE),
        "pluck": pluck(f0, duration, SAMPLE_RATE),
    }

    rows = []
    for name, probe in probes.items():
        stereo = torch.from_numpy(np.stack([probe, probe]))[None]
        row: dict[str, float | str] = {"probe": name, "f0_hz": f0}
        for label, model in (("baseline", baseline), ("antialiased", antialiased)):
            with torch.no_grad():
                recon, _, _, _ = model(stereo)
            out = recon[0, 0].numpy()
            row[f"asr_db_{label}"] = alias_to_signal_ratio(out, f0, SAMPLE_RATE)
            row[f"crest_db_{label}"] = crest_factor_db(out)
        row["asr_improvement_db"] = (
            float(row["asr_db_baseline"]) - float(row["asr_db_antialiased"])
        )
        rows.append(row)
    return rows


def bench_real_audio(paths: list[Path], duration: float = 4.0) -> list[dict]:
    """
    A/B the activation stage driven with real recorded audio.

    Reports per-band energy error of the alias-free path against the aliased
    one, which localises where the added energy was landing.
    """
    import soundfile as sf

    rows = []
    for path in paths:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio[: int(duration * rate), 0]
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

        baseline_out = _apply(make_activation(1, antialias=False, alpha_init=2.0), audio)
        antialiased_out = _apply(make_activation(1, antialias=True, alpha_init=2.0), audio)

        row: dict[str, float | str] = {
            "file": path.name,
            "sample_rate": rate,
            "crest_db_baseline": crest_factor_db(baseline_out),
            "crest_db_antialiased": crest_factor_db(antialiased_out),
        }
        for band, value in band_energy_error_db(
            antialiased_out, baseline_out, rate
        ).items():
            # Positive = the aliased path put MORE energy here than the clean one.
            row[f"excess_db_{band}"] = value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--audio", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    results = {
        "activation": bench_activation(),
        "feature_rates": bench_feature_rates(),
        "vae": bench_vae(),
    }
    if args.audio:
        results["real_audio"] = bench_real_audio(list(args.audio))

    path = args.out / "alias_bench.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")

    for row in results["activation"]:
        print(
            f"f0={row['f0_hz']:>7.0f} alpha={row['alpha']:<4} "
            f"ASR {row['asr_db_baseline']:>8.2f} -> {row['asr_db_antialiased']:>8.2f} dB "
            f"({row['asr_improvement_db']:>6.2f} dB better)"
        )


if __name__ == "__main__":
    main()
