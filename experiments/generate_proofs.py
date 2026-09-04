"""
Generate every measurement, figure and audio render behind the
anti-aliasing change.

Everything this script emits is produced by running the repository's own
code. Nothing is hand-authored, estimated or copied from a paper. Re-run it
to reproduce the numbers in ``docs/ANTIALIASING.md``.

    PYTHONPATH=. python experiments/generate_proofs.py --out proofs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from synthgen.eval.metrics import harmonic_analysis
from synthgen.eval.signals import SWEEP_TEST_FREQS, bandlimited_saw, log_sweep
from synthgen.model.antialias import AntiAliasedSnake, Snake
from synthgen.model.vae import ResidualBlock

SR = 44100
SEED = 7

# Deep Noise brand palette
INK = "#0B0B0F"
PAPER = "#F5F3EF"
BAD = "#E2574C"
GOOD = "#3BAA84"
ACCENT = "#7A5CFF"
MUTED = "#8A8A93"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": MUTED,
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "#DDD9D2",
        "grid.linewidth": 0.6,
    }
)


# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------


def snake_chain(antialias: bool, depth: int) -> nn.Module:
    """A bare cascade of activations - isolates the nonlinearity itself."""
    torch.manual_seed(SEED)
    make = (lambda: AntiAliasedSnake(1, ratio=2)) if antialias else (lambda: Snake(1))
    return nn.Sequential(*[make() for _ in range(depth)]).eval()


def residual_stack(antialias: bool, groups: int, channels: int = 16) -> nn.Module:
    """
    The repository's own ``ResidualBlock``, at audio rate.

    This is the decoder's final stage: the part of the network that runs at
    44.1 kHz and therefore the part where aliasing reaches the output. Both
    arms are built under the same seed, so the convolution weights are
    bit-identical and the activation is the only difference.
    """
    torch.manual_seed(SEED)
    blocks = [
        ResidualBlock(channels, dilation=d, antialias=antialias)
        for _ in range(groups)
        for d in (1, 3, 9)
    ]
    return nn.Sequential(*blocks).eval()


def run_mono(module: nn.Module, x: np.ndarray, channels: int = 1) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x)).float().view(1, 1, -1)
    if channels > 1:
        t = t.expand(1, channels, -1).contiguous()
    with torch.no_grad():
        y = module(t)
    return y[0].mean(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_mono(path: Path, seconds: float, offset: float = 0.0) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SR:
        n = int(round(len(mono) * SR / sr))
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono
        ).astype(np.float32)
    start = int(offset * SR)
    mono = mono[start : start + int(seconds * SR)]
    peak = np.max(np.abs(mono))
    return (mono / peak * 0.7).astype(np.float32) if peak > 0 else mono


def write_wav(path: Path, x: np.ndarray, normalise: bool = True) -> dict:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x)))
    gain_db = 0.0
    if normalise and peak > 0:
        target = 0.85
        gain_db = 20 * np.log10(target / peak)
        x = x * (target / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")
    return {
        "file": path.name,
        "seconds": round(len(x) / SR, 2),
        "applied_gain_db": round(float(gain_db), 2),
    }


# ---------------------------------------------------------------------------
# Proof 1 - alias vs cascade depth
# ---------------------------------------------------------------------------


def proof_depth(out: Path) -> dict:
    f0 = 2090.1
    stimulus = bandlimited_saw(f0, 0.5, SR, 0.5)
    depths = [1, 2, 4, 8, 16, 24, 32]
    rows = {"depth": depths, "before": [], "after": [], "before_sub": [], "after_sub": []}

    for depth in depths:
        before = harmonic_analysis(run_mono(snake_chain(False, depth), stimulus), f0, SR)
        after = harmonic_analysis(run_mono(snake_chain(True, depth), stimulus), f0, SR)
        rows["before"].append(float(before.alias_to_signal_db))
        rows["after"].append(float(after.alias_to_signal_db))
        rows["before_sub"].append(float(before.sub_fundamental_db))
        rows["after_sub"].append(float(after.sub_fundamental_db))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (b, a, title) in zip(
        axes,
        [
            (rows["before"], rows["after"], "Alias-to-signal ratio"),
            (rows["before_sub"], rows["after_sub"], "Sub-fundamental alias energy"),
        ],
    ):
        ax.plot(depths, b, "o-", color=BAD, lw=2, label="Snake (before)")
        ax.plot(depths, a, "o-", color=GOOD, lw=2, label="Anti-aliased Snake (after)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("activations in series")
        ax.set_ylabel("dB relative to note (lower is better)")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
    fig.suptitle(
        f"Aliasing compounds with depth - band-limited saw, f0 = {f0:.0f} Hz",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "p1_alias_vs_depth.png", dpi=140)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Proof 2 - alias vs pitch, on the repo's ResidualBlock
# ---------------------------------------------------------------------------


def proof_pitch(out: Path) -> dict:
    freqs = list(SWEEP_TEST_FREQS)
    before_model = residual_stack(False, groups=4)
    after_model = residual_stack(True, groups=4)
    rows = {"f0": freqs, "before": [], "after": []}

    for f0 in freqs:
        stimulus = bandlimited_saw(f0, 0.5, SR, 0.5)
        rows["before"].append(
            float(harmonic_analysis(run_mono(before_model, stimulus, 16), f0, SR).alias_to_signal_db)
        )
        rows["after"].append(
            float(harmonic_analysis(run_mono(after_model, stimulus, 16), f0, SR).alias_to_signal_db)
        )

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(freqs, rows["before"], "o-", color=BAD, lw=2, label="Snake (before)")
    ax.plot(freqs, rows["after"], "o-", color=GOOD, lw=2, label="Anti-aliased (after)")
    ax.fill_between(freqs, rows["before"], rows["after"], color=GOOD, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{f:.0f}" for f in freqs], rotation=45)
    ax.set_xlabel("fundamental of the played note (Hz)")
    ax.set_ylabel("alias-to-signal ratio (dB)")
    ax.set_title(
        "The higher the note, the worse the aliasing\n"
        "12 ResidualBlocks from synthgen/model/vae.py, identical weights",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "p2_alias_vs_pitch.png", dpi=140)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Proof 3 - sweep spectrograms
# ---------------------------------------------------------------------------


def proof_sweep(out: Path) -> dict:
    sweep = log_sweep(80, 8000, 2.0, SR, 0.5)
    before = run_mono(snake_chain(False, 8), sweep)
    after = run_mono(snake_chain(True, 8), sweep)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, sig, title, colour in [
        (axes[0], sweep, "Input sweep (clean)", MUTED),
        (axes[1], before, "Snake (before)", BAD),
        (axes[2], after, "Anti-aliased Snake (after)", GOOD),
    ]:
        ax.specgram(sig, NFFT=2048, Fs=SR, noverlap=1536, cmap="magma", vmin=-120, vmax=-10)
        ax.set_title(title, fontweight="bold", color=colour)
        ax.set_xlabel("time (s)")
        ax.set_ylim(0, SR / 2)
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle(
        "Aliasing is visible: rising harmonics reflect off Nyquist and travel back down",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "p3_sweep_spectrograms.png", dpi=140)
    plt.close(fig)

    return {
        "sweep_before": write_wav(out / "audio" / "p3_sweep_before.wav", before),
        "sweep_after": write_wav(out / "audio" / "p3_sweep_after.wav", after),
        "sweep_input": write_wav(out / "audio" / "p3_sweep_input.wav", sweep),
    }


# ---------------------------------------------------------------------------
# Proof 4 - spectrum of a single note
# ---------------------------------------------------------------------------


def proof_spectrum(out: Path) -> dict:
    f0 = 2090.1
    stimulus = bandlimited_saw(f0, 0.5, SR, 0.5)
    before = run_mono(snake_chain(False, 8), stimulus)
    after = run_mono(snake_chain(True, 8), stimulus)

    def spec_db(x):
        from synthgen.eval.metrics import blackman_harris

        mag = np.abs(np.fft.rfft(x * blackman_harris(len(x))))
        mag = mag / (mag.max() + 1e-12)
        return np.fft.rfftfreq(len(x), 1 / SR), 20 * np.log10(mag + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    freqs, db_before = spec_db(before)
    _, db_after = spec_db(after)
    ax.plot(freqs, db_before, color=BAD, lw=0.7, label="Snake (before)")
    ax.plot(freqs, db_after, color=GOOD, lw=0.7, alpha=0.85, label="Anti-aliased (after)")
    for k in range(1, int(SR / 2 / f0) + 1):
        ax.axvline(k * f0, color=ACCENT, lw=0.5, alpha=0.28)
    ax.set_xlim(0, SR / 2)
    ax.set_ylim(-120, 3)
    ax.set_xlabel("frequency (Hz)   -   purple lines mark the note's real harmonics")
    ax.set_ylabel("dB relative to peak")
    ax.set_title(
        f"Everything between the purple lines is aliasing (saw, f0 = {f0:.0f} Hz)",
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "p4_spectrum.png", dpi=140)
    plt.close(fig)

    return {
        "f0": f0,
        "before": {k: float(v) for k, v in harmonic_analysis(before, f0, SR).__dict__.items()},
        "after": {k: float(v) for k, v in harmonic_analysis(after, f0, SR).__dict__.items()},
        "audio_before": write_wav(out / "audio" / "p4_note_before.wav", before),
        "audio_after": write_wav(out / "audio" / "p4_note_after.wav", after),
    }


# ---------------------------------------------------------------------------
# Proof 5 - real audio through the real module
# ---------------------------------------------------------------------------


def proof_real_audio(out: Path, sources: list[tuple[str, Path, float]]) -> list[dict]:
    before_model = residual_stack(False, groups=4)
    after_model = residual_stack(True, groups=4)
    results = []

    for name, path, offset in sources:
        source = load_mono(path, seconds=3.0, offset=offset)
        if len(source) < SR:
            continue
        before = run_mono(before_model, source, 16)
        after = run_mono(after_model, source, 16)

        # Level-match the two arms before differencing, so the residual is
        # the artefact and not a gain mismatch.
        scale = float(np.dot(before, after) / (np.dot(after, after) + 1e-12))
        difference = before - scale * after

        entry = {
            "name": name,
            "source_file": str(path),
            "source": write_wav(out / "audio" / f"p5_{name}_source.wav", source),
            "before": write_wav(out / "audio" / f"p5_{name}_before.wav", before),
            "after": write_wav(out / "audio" / f"p5_{name}_after.wav", after),
            "isolated_alias": write_wav(
                out / "audio" / f"p5_{name}_isolated_alias.wav", difference
            ),
            "isolated_alias_level_db": round(
                float(
                    20
                    * np.log10(
                        (np.std(difference) + 1e-12) / (np.std(before) + 1e-12)
                    )
                ),
                2,
            ),
        }
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Proof 6 - trained A/B
# ---------------------------------------------------------------------------


def proof_trained(out: Path, run_dir: Path, holdout: list[tuple[str, Path, float]]) -> dict:
    from synthgen.eval.metrics import (
        high_frequency_retention_db,
        multires_stft_distance,
        si_sdr_db,
    )
    from synthgen.model.vae import AudioVAE

    payload: dict = {"available": False}
    checkpoints = {
        arm: run_dir / f"vae_{arm}.pt" for arm in ("baseline", "antialias")
    }
    if not all(p.exists() for p in checkpoints.values()):
        return payload

    models = {}
    for arm, path in checkpoints.items():
        torch.manual_seed(1234)
        model = AudioVAE(
            in_channels=1,
            latent_dim=32,
            base_channels=16,
            encoder_channel_multipliers=(1, 2, 4),
            decoder_channel_multipliers=(4, 2, 1),
            strides=(4, 4, 4),
            num_residual_per_block=3,
            antialias=(arm == "antialias"),
            antialias_ratio=2,
        )
        model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        models[arm] = model.eval()

    # Loss curves
    histories = {}
    for arm in checkpoints:
        hist_path = run_dir / f"history_{arm}.json"
        if hist_path.exists():
            histories[arm] = json.loads(hist_path.read_text())

    if histories:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for arm, colour, label in [
            ("baseline", BAD, "Snake (before)"),
            ("antialias", GOOD, "Anti-aliased Snake (after)"),
        ]:
            if arm in histories:
                steps = [h["step"] for h in histories[arm]]
                vals = [h["spectral"] for h in histories[arm]]
                ax.plot(steps, vals, color=colour, lw=1.6, label=label)
        ax.set_xlabel("training step")
        ax.set_ylabel("multi-resolution STFT loss")
        ax.set_title(
            "Identical data, identical seed, identical parameter count",
            fontweight="bold",
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "p6_training_curves.png", dpi=140)
        plt.close(fig)

    # Held-out reconstruction
    per_clip = []
    for name, path, offset in holdout:
        source = load_mono(path, seconds=2.0, offset=offset)
        if len(source) < SR // 2:
            continue
        x = torch.from_numpy(source).view(1, 1, -1)
        entry = {"name": name, "source_file": str(path)}
        entry["source"] = write_wav(out / "audio" / f"p6_{name}_source.wav", source)
        for arm, model in models.items():
            with torch.no_grad():
                mean, _ = model.encode(x)
                recon = model.decode(mean)[0, 0].numpy()
            n = min(len(recon), len(source))
            entry[arm] = {
                "audio": write_wav(out / "audio" / f"p6_{name}_{arm}.wav", recon[:n]),
                "si_sdr_db": round(si_sdr_db(recon[:n], source[:n]), 2),
                "multires_stft": round(multires_stft_distance(recon[:n], source[:n]), 4),
                "hf_retention_db": round(
                    high_frequency_retention_db(recon[:n], source[:n], SR), 2
                ),
            }
        per_clip.append(entry)

    # Alias measurement on the trained decoders
    alias = {}
    for arm, model in models.items():
        rows = []
        for f0 in (453.1, 903.7, 2090.1):
            stimulus = bandlimited_saw(f0, 0.5, SR, 0.5)
            x = torch.from_numpy(stimulus).view(1, 1, -1)
            with torch.no_grad():
                mean, _ = model.encode(x)
                recon = model.decode(mean)[0, 0].numpy()
            rows.append(
                {
                    "f0": f0,
                    "alias_to_signal_db": round(
                        harmonic_analysis(recon, f0, SR).alias_to_signal_db, 2
                    ),
                }
            )
        alias[arm] = rows

    payload.update(
        {
            "available": True,
            "histories": {k: v[-1] for k, v in histories.items()},
            "reconstructions": per_clip,
            "alias": alias,
        }
    )
    return payload


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("proofs"))
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/out"))
    parser.add_argument("--skip-trained", action="store_true")
    args = parser.parse_args()

    out = args.out
    (out / "audio").mkdir(parents=True, exist_ok=True)

    repos = Path("/home/user")
    real_sources = [
        ("guitar", repos / "aisynth-vst/assets/guitar.wav", 0.0),
        ("electronic", repos / "audiocraft/assets/electronic.mp3", 0.5),
        ("synth_pad", repos / "audiocraft/dataset/example/electro_1.mp3", 4.0),
    ]
    real_sources = [(n, p, o) for n, p, o in real_sources if p.exists()]

    holdout = [
        (
            "aisynth_fixture",
            repos / "deep-noise-effects-api/tests/fixtures/frontend_nofx_reference.wav",
            0.0,
        ),
        ("bach", repos / "audiocraft/assets/bach.mp3", 0.5),
    ]
    holdout = [(n, p, o) for n, p, o in holdout if p.exists()]

    report: dict = {}
    print("proof 1: alias vs depth")
    report["depth"] = proof_depth(out)
    print("proof 2: alias vs pitch")
    report["pitch"] = proof_pitch(out)
    print("proof 3: sweep spectrograms")
    report["sweep"] = proof_sweep(out)
    print("proof 4: spectrum")
    report["spectrum"] = proof_spectrum(out)
    print("proof 5: real audio")
    report["real_audio"] = proof_real_audio(out, real_sources)
    if not args.skip_trained:
        print("proof 6: trained A/B")
        report["trained"] = proof_trained(out, args.run_dir, holdout)

    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    main()
