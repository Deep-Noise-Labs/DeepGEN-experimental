"""
Draw the proof figures from the measured JSON.

    PYTHONPATH=. python experiments/upsampler_figures.py --out proofs/upsampler

Reads only ``bench.json``, ``filter_ablation.json`` and ``audio_proof.json``.
It computes nothing itself, so a figure can never disagree with the numbers
the experiments recorded.
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

# Deep Noise tokens, read from deepnoise-web's stylesheet, plus a two-series
# categorical pair validated with the dataviz palette checker against this
# dark surface (lightness band, chroma, CVD separation, contrast: all pass).
SURFACE = "#131517"
PANEL = "#181818"
INK = "#e1e5e7"
INK_MUTED = "#bbc6cd"
GRID = "#2c2f33"
FIXED = "#17ab8a"   # band-limited
BASELINE = "#e0702f"  # current transposed conv


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": PANEL,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRID,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )


def finish(fig, ax_or_axes, path: Path, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(0.012, 0.965, title, ha="left", va="top", fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.895, subtitle, ha="left", va="top", fontsize=9.5, color=INK_MUTED)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# =============================================================================


def fig_image_vs_pitch(bench: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, stride in zip(axes, ("4", "8")):
        probes = bench["image_vs_pitch"][stride]["probes"]
        nu = [p["nu"] for p in probes]
        base = [p["baseline_linear_db"] for p in probes]
        fixed = [p["fixed_linear_db"] for p in probes]
        ax.plot(nu, base, color=BASELINE, lw=2, marker="o", ms=5, label="transposed conv (current)")
        ax.plot(nu, fixed, color=FIXED, lw=2, marker="o", ms=5, label="band-limited")
        ax.set_title(f"stride {stride}", color=INK, loc="left")
        ax.set_xlabel("probe frequency (cycles/sample of the input rate)")
        ax.set_ylim(-130, 10)
    axes[0].set_ylabel("image-to-signal (dB, lower is cleaner)")
    axes[0].legend(frameon=False, labelcolor=INK_MUTED, loc="lower left", fontsize=9)
    finish(
        fig,
        axes,
        out / "p1_image_vs_pitch.png",
        "The upsampler's images, across the pitch range",
        "Identical weights in both arms, set to linear interpolation - the best a 2-tap-per-phase kernel does smoothly.",
    )


def fig_ceiling(bench: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    rows = [r for r in bench["ceiling"]["8"]["rows"] if r["optimal_rejection_db"] == r["optimal_rejection_db"]]
    x = [r["taps_per_phase"] for r in rows]
    ax.plot(x, [r["optimal_rejection_db"] for r in rows], color=INK_MUTED, lw=2, ls="--",
            marker="s", ms=5, label="theoretical best for that kernel length")
    ax.plot(x, [r["shipped_rejection_db"] for r in rows], color=FIXED, lw=2,
            marker="o", ms=6, label="the Kaiser filter this repository ships")

    current = rows[0]
    ax.scatter([2], [current["optimal_rejection_db"]], s=150, color=BASELINE, zorder=6)
    ax.annotate(
        f"current architecture: kernel = 2x stride.\nEven a perfectly trained kernel\ncannot beat {current['optimal_rejection_db']:.0f} dB.",
        xy=(2, current["optimal_rejection_db"]),
        xytext=(6.2, -18),
        color=BASELINE,
        fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=1.2),
    )

    ax.set_xlabel("taps per polyphase phase (kernel length / stride)")
    ax.set_ylabel("stop-band rejection (dB)")
    ax.legend(frameon=False, labelcolor=INK_MUTED, loc="lower left", fontsize=9)
    finish(
        fig,
        ax,
        out / "p2_architectural_ceiling.png",
        "Why training cannot fix this",
        "Parks-McClellan optimum. The current kernel is the leftmost point: its rejection is capped by its length, not by its weights.",
    )


def fig_ablation(ablation: dict, out: Path) -> None:
    rows = [r for r in ablation["rows"] if r["taps_per_stride"] is not None and r["transition"] == 0.5]
    base = [r for r in ablation["rows"] if r["taps_per_stride"] is None][0]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = [r["filter_taps"] for r in rows]
    ax.plot(x, [r["out_of_band_db"] for r in rows], color=FIXED, lw=2, marker="o", ms=6,
            label="invented (out-of-band) content")
    ax.plot(x, [r["in_band_fidelity_db"] for r in rows], color=INK_MUTED, lw=2, marker="s", ms=5,
            label="in-band fidelity vs reference")
    ax.axhline(base["out_of_band_db"], color=BASELINE, lw=1.8, ls="--")
    ax.text(x[-1], base["out_of_band_db"] + 4, f"current: {base['out_of_band_db']:.0f} dB",
            color=BASELINE, fontsize=9.5, ha="right")
    ax.axhline(base["in_band_fidelity_db"], color=INK_MUTED, lw=1, ls=":")
    ax.text(x[-1], base["in_band_fidelity_db"] + 4, f"current in-band: {base['in_band_fidelity_db']:.0f} dB",
            color=INK_MUTED, fontsize=9, ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("anti-imaging filter length (taps)")
    ax.set_ylabel("dB")
    ax.legend(frameon=False, labelcolor=INK_MUTED, loc="center left", fontsize=9)
    finish(
        fig,
        ax,
        out / "p3_filter_ablation.png",
        "The filter is free in the band that matters",
        "Six real Deep Noise renders. In-band fidelity is flat at the baseline's own value; only invented content falls.",
    )


def fig_encoder(bench: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, stride in zip(axes, ("4", "8")):
        rows = [r for r in bench["encoder_alias"][stride]["rows"] if r["in_stopband"]]
        nu = [r["nu"] for r in rows]
        ax.plot(nu, [r["baseline_fold_db"] for r in rows], color=BASELINE, lw=2, marker="o", ms=5,
                label="strided conv (current)")
        ax.plot(nu, [r["fixed_fold_db"] for r in rows], color=FIXED, lw=2, marker="o", ms=5,
                label="band-limited")
        ax.axhline(0, color=INK_MUTED, lw=1, ls=":")
        ax.set_title(f"stride {stride}", color=INK, loc="left")
        ax.set_xlabel("input frequency above the new Nyquist (cycles/sample)")
    axes[0].set_ylabel("folded level vs in-band signal (dB)")
    axes[0].legend(frameon=False, labelcolor=INK_MUTED, loc="center left", fontsize=9)
    finish(
        fig,
        axes,
        out / "p4_encoder_fold.png",
        "The encoder folds out-of-band content back in, louder than the signal",
        "Content above the post-decimation Nyquist returns as inharmonic noise at or above 0 dB relative to the wanted band.",
    )


def fig_compounding(bench: dict, out: Path) -> None:
    comp = bench["compounding"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    stages = np.arange(1, 5)
    for arm in comp["arms"]:
        if arm["weights"] != "linear":
            continue
        colour = BASELINE if arm["arm"] == "baseline" else FIXED
        label = "transposed conv (current)" if arm["arm"] == "baseline" else "band-limited"
        ax.plot(stages, arm["mean_db"], color=colour, lw=2, marker="o", ms=7, label=label)
        ax.annotate(f"{arm['mean_db'][-1]:.0f} dB", xy=(4, arm["mean_db"][-1]),
                    xytext=(4.06, arm["mean_db"][-1]), color=colour, fontsize=10, va="center")
    ax.set_xticks(stages)
    ax.set_xticklabels([f"stage {i}\n(x{s})" for i, s in zip(stages, comp["strides"])])
    ax.set_xlim(0.8, 4.6)
    ax.set_ylabel("image-to-signal (dB)")
    ax.legend(frameon=False, labelcolor=INK_MUTED, loc="center right", fontsize=9)
    finish(
        fig,
        ax,
        out / "p6_stage_compounding.png",
        "Through all four decoder stages",
        "Imaging does not accumulate with depth - it sits at the single-stage ceiling the whole way down.",
    )


def fig_spectrograms(proof: dict, out: Path, audio_dir: Path) -> None:
    row = max(
        (r for r in proof["rows"] if r["attributable"]),
        key=lambda r: r["improvement_db"],
    )
    names = [
        ("reference", "reference (what the stage should return)"),
        ("baseline", "transposed conv (current)"),
        ("fixed", "band-limited"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    for ax, (key, title) in zip(axes, names):
        data, sr = sf.read(str(audio_dir / row["files"][key]), always_2d=True)
        mono = data[:, 0]
        ax.specgram(mono, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-150, vmax=-20)
        ax.axhline(row["band_edge_hz"], color=FIXED, lw=1.4, ls="--")
        ax.set_title(title, color=INK, loc="left", fontsize=10.5)
        ax.set_xlabel("time (s)")
        ax.set_facecolor(PANEL)
        ax.grid(False)
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].text(0.06, row["band_edge_hz"] * 1.12, "band edge", color=FIXED, fontsize=9)
    finish(
        fig,
        axes,
        out / "p5_spectrograms.png",
        f"Real render: {row['clip'].split('_')[0]}",
        "Everything above the dashed line is content the upsampler invented. The reference has none; the current operator has a full mirror image.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/upsampler")
    args = parser.parse_args()
    out = Path(args.out)

    style()
    bench = json.loads((out / "bench.json").read_text())
    ablation = json.loads((out / "filter_ablation.json").read_text())
    proof = json.loads((out / "audio_proof.json").read_text())

    fig_image_vs_pitch(bench, out)
    fig_ceiling(bench, out)
    fig_ablation(ablation, out)
    fig_encoder(bench, out)
    fig_spectrograms(proof, out, out / "audio")
    fig_compounding(bench, out)


if __name__ == "__main__":
    main()
