"""
Audible before/after demo of decoder activation aliasing.

Runs deterministic test signals through ONE decoder activation stage using
the repo's actual activation code, at alpha = 1.0 (the initialisation value):

  before: ``vae.Snake``            - the activation used until this change
  after:  ``antialias.AntiAliasedSnake`` - the 2x oversampled replacement

This isolates the mechanism the change addresses. It is NOT model output
(that requires a trained checkpoint); it is the exact nonlinearity the
decoder applies at the output sample rate, applied to clean synthesizer-like
signals so the fold-back distortion is audible in isolation.

Usage:
    python scripts/aliasing_demo.py --out demo_out

Outputs per signal: <name>_input.wav, <name>_before.wav, <name>_after.wav,
optional spectrogram PNGs, and an aliasing metric summary printed to stdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synthgen.model.antialias import AntiAliasedSnake
from synthgen.model.vae import Snake

SR = 44100


# -----------------------------------------------------------------------------
# Test signals (all mono, peak-normalised to 0.8)
# -----------------------------------------------------------------------------


def exp_sweep(duration: float = 5.0, f_start: float = 40.0, f_end: float = 18000.0) -> np.ndarray:
    """Exponential sine sweep - aliasing shows up as mirrored inharmonic sweeps."""
    t = np.arange(int(duration * SR)) / SR
    k = np.log(f_end / f_start) / duration
    phase = 2 * np.pi * f_start * (np.exp(k * t) - 1) / k
    sweep = np.sin(phase)
    fade = int(0.02 * SR)
    env = np.ones_like(sweep)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return 0.8 * sweep * env


def bl_saw(freq: float, duration: float, n_harmonics: int | None = None) -> np.ndarray:
    """Band-limited additive sawtooth (all harmonics below Nyquist)."""
    t = np.arange(int(duration * SR)) / SR
    if n_harmonics is None:
        n_harmonics = int((SR / 2 - 1) // freq)
    out = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t) / k
    return out


def saw_note(freq: float = 1108.73, duration: float = 3.0) -> np.ndarray:
    """Bright sustained saw note (C#6 by default), organ-style envelope.

    Sustained rather than decaying: the Snake nonlinearity is level-dependent,
    so a steady level keeps the fold-back distortion constant and audible.
    """
    x = bl_saw(freq, duration)
    t = np.arange(len(x)) / SR
    env = np.minimum(t / 0.01, 1.0) * np.clip((duration - t) / 0.15, 0, 1)
    x = x * env
    return 0.8 * x / np.max(np.abs(x))


def supersaw_chord(duration: float = 4.0) -> np.ndarray:
    """Detuned supersaw stack on an A-major chord - dense high-frequency content."""
    rng = np.random.default_rng(7)
    out = np.zeros(int(duration * SR))
    for root in (440.0, 554.37, 659.25):  # A4, C#5, E5
        for detune_cents in (-12, -5, 0, 5, 12):
            f = root * 2 ** (detune_cents / 1200)
            phase_offset = rng.uniform(0, 1)
            x = bl_saw(f, duration)
            shift = int(phase_offset * SR / f)
            out += np.roll(x, shift) / 15.0
    t = np.arange(len(out)) / SR
    env = np.minimum(t / 0.02, 1.0) * np.minimum((duration - t) / 0.3, 1.0)
    out = out * np.clip(env, 0, 1)
    return 0.8 * out / np.max(np.abs(out))


# -----------------------------------------------------------------------------
# Processing and metrics
# -----------------------------------------------------------------------------


def process(x: np.ndarray, module: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32)).view(1, 1, -1)
        y = module(t)
    return y.view(-1).numpy()


def harmonic_mask(n_bins: int, f0: float, n_fft: int, tol_bins: int = 4) -> np.ndarray:
    """Boolean mask of bins within ``tol_bins`` of any harmonic of f0 (or DC)."""
    mask = np.zeros(n_bins, dtype=bool)
    mask[: tol_bins + 1] = True  # DC / envelope sidebands
    k = 1
    while True:
        bin_f = k * f0 * n_fft / SR
        if bin_f > n_bins - 1:
            break
        lo = max(0, int(round(bin_f)) - tol_bins)
        hi = min(n_bins, int(round(bin_f)) + tol_bins + 1)
        mask[lo:hi] = True
        k += 1
    return mask


def aliased_energy_ratio_db(y: np.ndarray, f0: float, n_fft: int = 16384) -> float:
    """
    Ratio (dB) of energy at non-harmonic bins to total energy, averaged over
    frames. For a periodic input, non-harmonic energy is fold-back aliasing.
    """
    hop = n_fft // 2
    window = np.hanning(n_fft)
    mask = harmonic_mask(n_fft // 2 + 1, f0, n_fft)
    ratios = []
    for start in range(0, len(y) - n_fft, hop):
        frame = y[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        total = spec.sum()
        if total < 1e-9:
            continue
        ratios.append(spec[~mask].sum() / total)
    return 10 * np.log10(np.mean(ratios) + 1e-12)


def save_wav(path: Path, x: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), x.astype(np.float32), SR, subtype="PCM_16")


def save_spectrogram(path: Path, x: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_fft, hop = 2048, 512
    window = np.hanning(n_fft)
    frames = [
        np.abs(np.fft.rfft(x[i : i + n_fft] * window))
        for i in range(0, len(x) - n_fft, hop)
    ]
    spec_db = 20 * np.log10(np.array(frames).T + 1e-7)
    vmax = spec_db.max()

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmax - 90,
        vmax=vmax,
        extent=[0, len(x) / SR, 0, SR / 2000],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo_out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    before = Snake(channels=1)
    after = AntiAliasedSnake(channels=1)

    signals = {
        "sweep": (exp_sweep(), None),
        "saw_note": (saw_note(), 1108.73),
        "supersaw_chord": (supersaw_chord(), None),
    }

    metrics = {}
    for name, (x, f0) in signals.items():
        y_before = process(x, before)
        y_after = process(x, after)

        save_wav(args.out / f"{name}_input.wav", x)
        save_wav(args.out / f"{name}_before.wav", y_before)
        save_wav(args.out / f"{name}_after.wav", y_after)
        # What the anti-aliasing removes (dominated by fold-back distortion,
        # plus any legitimate content above the filter cutoff)
        save_wav(args.out / f"{name}_removed.wav", y_before - y_after)

        save_spectrogram(args.out / f"{name}_before.png", y_before, f"{name} - plain Snake (before)")
        save_spectrogram(args.out / f"{name}_after.png", y_after, f"{name} - anti-aliased Snake (after)")

        if f0 is not None:
            m = {
                "input_aliased_db": aliased_energy_ratio_db(x, f0),
                "before_aliased_db": aliased_energy_ratio_db(y_before, f0),
                "after_aliased_db": aliased_energy_ratio_db(y_after, f0),
            }
            m["improvement_db"] = m["before_aliased_db"] - m["after_aliased_db"]
            metrics[name] = m
            print(
                f"{name}: aliased-energy ratio  before={m['before_aliased_db']:.1f} dB  "
                f"after={m['after_aliased_db']:.1f} dB  "
                f"(improvement {m['improvement_db']:.1f} dB; clean input floor "
                f"{m['input_aliased_db']:.1f} dB)"
            )

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {args.out}/")


if __name__ == "__main__":
    main()
