#!/usr/bin/env python3
"""
Controlled A/B of the two VAE objectives.

Same model, same data, same seed, same optimiser, same number of steps. The only
variable is the loss:

  before = VAELoss(legacy=True)  -> 0.1*L1 + 1.0*linear-MRSTFT + 1e-4*KL
  after  = VAELoss()             -> 0.1*L1 + 0.25*linear-MRSTFT
                                    + 15*multi-scale-log-mel(mid/side) + 1e-4*KL

This is a reconstruction-*ceiling* test: a small autoencoder is fitted to a small
set of real clips and we listen to what each objective's optimum sounds like.
Small enough to run on CPU, which means the number to read is the difference
between the two objectives, not absolute fidelity.

The adversarial terms are deliberately excluded: a GAN needs far more than a
CPU-scale budget to become useful, and enabling it here would measure the budget
rather than the idea.

    python scripts/vae_objective_ab.py clip1.wav clip2.wav --steps 3000

Background and results: docs/VAE_OBJECTIVE.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal as sps

# Runnable from a plain checkout, not only from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthgen.model.vae import AudioVAE  # noqa: E402
from synthgen.training.losses import VAELoss  # noqa: E402


def load(path: Path, sample_rate: int, n: int, offset: float) -> np.ndarray:
    audio, file_sr = sf.read(str(path), always_2d=True, dtype="float64")
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if file_sr != sample_rate:
        g = np.gcd(int(file_sr), sample_rate)
        audio = sps.resample_poly(audio, sample_rate // g, file_sr // g, axis=-1)
    start = int(offset * sample_rate)
    audio = audio[:, start:start + n]
    if audio.shape[-1] < n:
        audio = np.pad(audio, ((0, 0), (0, n - audio.shape[-1])))
    fade = int(0.004 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    audio[:, :fade] *= ramp
    audio[:, -fade:] *= ramp[::-1]
    return (audio / (np.max(np.abs(audio)) + 1e-12) * 0.85).astype(np.float32)


def save(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    out = np.asarray(audio).T.astype(np.float32)
    peak = np.max(np.abs(out))
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(path), out, sample_rate, subtype="PCM_16")


# ------------------------------------------------------------------ metrics --
def log_spectral_distance(pred, ref, nfft=2048):
    def spec(a):
        _, _, z = sps.stft(a, nperseg=nfft, noverlap=nfft * 3 // 4, boundary=None)
        return np.abs(z)
    return float(np.mean([
        np.mean(np.abs(20 * np.log10((spec(pred[c]) + 1e-6) / (spec(ref[c]) + 1e-6))))
        for c in range(pred.shape[0])
    ]))


def band_error_db(pred, ref, lo, hi, sample_rate):
    """Band energy error in dB. Negative = too dull, positive = too bright."""
    def energy(a):
        spec = np.fft.rfft(a, axis=-1)
        freqs = np.fft.rfftfreq(a.shape[-1], 1 / sample_rate)
        mask = (freqs >= lo) & (freqs < hi)
        return np.sum(np.abs(spec[:, mask]) ** 2)
    return float(10 * np.log10((energy(pred) + 1e-20) / (energy(ref) + 1e-20)))


def stereo_width(x):
    side, mid = (x[0] - x[1]) / 2, (x[0] + x[1]) / 2
    return float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))


def crest_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)))


# ----------------------------------------------------------------- training --
def train(tag, loss_fn, data, args):
    torch.manual_seed(args.seed)
    model = AudioVAE(
        in_channels=2,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        strides=(4, 4, 4, 4),
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    started = time.time()
    history = []
    for step in range(args.steps):
        idx = torch.randint(0, data.shape[0], (args.batch,), generator=gen)
        recon, target, mean, log_var = model(data[idx])
        out = loss_fn(recon, target, mean, log_var)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            loss_value = float(out["loss"].detach())
            history.append({"step": step + 1, "loss": loss_value})
            elapsed = time.time() - started
            print(f"[{tag}] {step+1}/{args.steps} loss={loss_value:.4f} "
                  f"({elapsed/(step+1):.2f}s/step)", flush=True)
    model.eval()
    return model, history


@torch.no_grad()
def reconstruct(model, data):
    out = []
    for i in range(data.shape[0]):
        mean, _ = model.encode(data[i:i + 1])
        out.append(model.decode(mean)[0].cpu().numpy())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Real audio files to fit")
    parser.add_argument("--output-dir", default="./ab_out")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--clip-s", type=float, default=0.75)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep the clip divisible by the 256x compression ratio
    n = (int(args.clip_s * args.sample_rate) // 256) * 256
    paths = [Path(p) for p in args.inputs]
    raw = [load(p, args.sample_rate, n, args.offset) for p in paths]
    data = torch.from_numpy(np.stack(raw))
    print(f"data {tuple(data.shape)} ({n/args.sample_rate:.2f}s @ "
          f"{args.sample_rate} Hz stereo)", flush=True)

    for path, ref in zip(paths, raw):
        save(out_dir / f"{path.stem}__reference.wav", ref, args.sample_rate)

    objectives = {
        "before": VAELoss(legacy=True),
        "after": VAELoss(sample_rate=args.sample_rate, adv_weight=0.0, fm_weight=0.0),
    }

    report = {"config": vars(args), "runs": {}}
    for tag, loss_fn in objectives.items():
        print(f"\n===== {tag} =====", flush=True)
        model, history = train(tag, loss_fn, data, args)
        rows = []
        for path, ref, rec in zip(paths, raw, reconstruct(model, data)):
            save(out_dir / f"{path.stem}__{tag}.wav", rec, args.sample_rate)
            rows.append({
                "clip": path.stem,
                "lsd_db": log_spectral_distance(rec, ref),
                "band_8k_16k_db": band_error_db(rec, ref, 8000, 16000, args.sample_rate),
                "band_2k_8k_db": band_error_db(rec, ref, 2000, 8000, args.sample_rate),
                "width": stereo_width(rec),
                "width_ref": stereo_width(ref),
                "crest_db": crest_db(rec),
                "crest_ref_db": crest_db(ref),
            })
        report["runs"][tag] = {"history": history, "per_clip": rows}

    print("\n==================== RESULTS ====================")
    print(f"{'clip':<16}{'LSD dB (lower better)':>24}{'8-16 kHz err dB':>21}{'width':>18}")
    for i, path in enumerate(paths):
        b = report["runs"]["before"]["per_clip"][i]
        a = report["runs"]["after"]["per_clip"][i]
        print(f"{path.stem:<16}{b['lsd_db']:>11.2f} ->{a['lsd_db']:>10.2f}"
              f"{b['band_8k_16k_db']:>11.1f} ->{a['band_8k_16k_db']:>8.1f}"
              f"{b['width']:>9.3f} ->{a['width']:>6.3f}  (ref {b['width_ref']:.3f})")

    report["summary"] = {}
    for key, label in [("lsd_db", "log-spectral distance (dB)"),
                       ("band_8k_16k_db", "8-16 kHz energy error (dB)"),
                       ("band_2k_8k_db", "2-8 kHz energy error (dB)")]:
        before = float(np.mean([c[key] for c in report["runs"]["before"]["per_clip"]]))
        after = float(np.mean([c[key] for c in report["runs"]["after"]["per_clip"]]))
        report["summary"][key] = {"before": before, "after": after}
        print(f"\nmean {label:<32} before={before:8.2f}   after={after:8.2f}")

    with open(out_dir / "ab_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWAVs and ab_report.json -> {out_dir}")


if __name__ == "__main__":
    main()
