"""
A/B experiment: baseline VAE objective vs perceptual + adversarial objective.

Trains two identically-initialised AudioVAEs on the same procedurally
generated synth-sound dataset - one with the old objective (L1 + linear
multi-resolution STFT + KL), one with the new objective (adds multi-scale
log-mel + multi-resolution STFT discriminator + feature matching) - then
reconstructs a held-out set with both and writes WAVs plus metrics so the
difference can be heard and measured.

The dataset is generated on the fly (supersaw pads, FM bells, plucks,
sub bass, filter sweeps, chord stabs...), so the experiment needs no
downloads and runs on CPU at reduced scale:

    uv run python scripts/ab_vae_objective.py --steps 3000 --out-dir ./ab_out

At full scale the same comparison applies to Stage-1 training via
`vae_adversarial` in the config (see docs/TRAINING.md).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthgen.model.discriminator import MultiResolutionDiscriminator  # noqa: E402
from synthgen.model.vae import AudioVAE  # noqa: E402
from synthgen.training.losses import (  # noqa: E402
    MultiResolutionSTFTLoss,
    MultiScaleMelSpectrogramLoss,
    VAELoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


# =============================================================================
# Procedural synth-sound dataset
# =============================================================================


def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """Simple linear ADSR envelope of length n samples."""
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s = max(1, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def _supersaw(rng, n, sr, f0):
    """Detuned stack of saw oscillators - the classic trance/EDM pad core."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for _ in range(7):
        detune = 1.0 + rng.uniform(-0.012, 0.012)
        phase = rng.uniform(0, 1)
        out += ((t * f0 * detune + phase) % 1.0) * 2 - 1
    return out / 7


def _fm_bell(rng, n, sr, f0):
    """Two-operator FM bell (DX7-style ratio + decaying mod index)."""
    t = np.arange(n) / sr
    ratio = rng.choice([1.4, 2.0, 3.5, 5.0])
    index = rng.uniform(2.0, 6.0) * np.exp(-t * rng.uniform(2, 5))
    return np.sin(2 * np.pi * f0 * t + index * np.sin(2 * np.pi * f0 * ratio * t))


def _square(rng, n, sr, f0):
    """Square lead with vibrato."""
    t = np.arange(n) / sr
    vib = 1.0 + 0.005 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
    return np.sign(np.sin(2 * np.pi * f0 * vib * t))


def _lowpass_sweep(x, sr, f_start, f_end):
    """One-pole lowpass with exponentially swept cutoff."""
    n = len(x)
    cutoff = np.geomspace(f_start, f_end, n)
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff / sr)
    y = np.zeros(n)
    state = 0.0
    for i in range(n):
        state += alpha[i] * (x[i] - state)
        y[i] = state
    return y


def make_clip(rng: np.random.Generator, sr: int, seconds: float) -> np.ndarray:
    """Generate one synth-style clip, shape (samples,), peak-normalised."""
    n = int(sr * seconds)
    kind = rng.integers(0, 6)
    f0 = float(rng.choice([55, 82.4, 110, 146.8, 220, 293.7, 440]))

    if kind == 0:  # supersaw pad (slow attack)
        x = _supersaw(rng, n, sr, f0)
        x = _lowpass_sweep(x, sr, rng.uniform(800, 2000), rng.uniform(3000, 8000))
        x *= _adsr(n, sr, 0.25, 0.2, 0.7, 0.3)
    elif kind == 1:  # FM bell
        x = _fm_bell(rng, n, sr, f0 * 2)
        x *= _adsr(n, sr, 0.005, 0.4, 0.2, 0.4)
    elif kind == 2:  # pluck (bright saw, fast decay)
        x = _supersaw(rng, n, sr, f0 * 2)
        x = _lowpass_sweep(x, sr, rng.uniform(6000, 9000), rng.uniform(300, 800))
        x *= _adsr(n, sr, 0.002, 0.15, 0.05, 0.2)
    elif kind == 3:  # sub bass with click transient
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * (f0 / 2) * t)
        click = np.zeros(n)
        click[: int(0.004 * sr)] = rng.standard_normal(int(0.004 * sr)) * 0.5
        x = x * _adsr(n, sr, 0.003, 0.1, 0.8, 0.2) + click
    elif kind == 4:  # square lead
        x = _square(rng, n, sr, f0)
        x = _lowpass_sweep(x, sr, rng.uniform(2000, 5000), rng.uniform(2000, 5000))
        x *= _adsr(n, sr, 0.01, 0.1, 0.6, 0.25)
    else:  # chord stab (3-note supersaw)
        x = np.zeros(n)
        for interval in [1.0, 1.26, 1.5]:  # root, major third, fifth
            x += _supersaw(rng, n, sr, f0 * interval)
        x = _lowpass_sweep(x, sr, rng.uniform(4000, 8000), rng.uniform(600, 1500))
        x *= _adsr(n, sr, 0.005, 0.3, 0.2, 0.3)

    peak = np.abs(x).max() + 1e-9
    return (x / peak * 0.9).astype(np.float32)


TEST_CLIP_NAMES = [
    "supersaw_pad", "fm_bell", "pluck", "sub_bass",
    "square_lead", "chord_stab", "supersaw_pad_2", "fm_bell_2",
]


def make_dataset(seed: int, sr: int, seconds: float, num_train: int):
    """Build the train tensor and a named, held-out test set."""
    rng = np.random.default_rng(seed)
    train = np.stack([make_clip(rng, sr, seconds) for _ in range(num_train)])

    test_rng = np.random.default_rng(seed + 10_000)
    test = {}
    kinds = [0, 1, 2, 3, 4, 5, 0, 1]
    for name, kind in zip(TEST_CLIP_NAMES, kinds):
        # Draw clips until we get the desired kind so the test set is diverse
        while True:
            probe = np.random.default_rng(test_rng.integers(0, 2**32))
            if probe.integers(0, 6) == kind:
                clip = make_clip(probe, sr, seconds)
                break
        test[name] = clip

    return (
        torch.from_numpy(train).unsqueeze(1),          # (N, 1, samples)
        {k: torch.from_numpy(v)[None, None] for k, v in test.items()},
    )


# =============================================================================
# Training
# =============================================================================


def build_vae(seed: int, latent_dim: int, base_channels: int) -> AudioVAE:
    torch.manual_seed(seed)
    return AudioVAE(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=base_channels,
        encoder_channel_multipliers=(1, 2, 4, 4),
        decoder_channel_multipliers=(4, 4, 2, 1),
        strides=(4, 4, 4, 4),
    )


def train_vae(
    name: str,
    train_data: torch.Tensor,
    args,
    mel_weight: float,
    adversarial: bool,
) -> tuple[AudioVAE, list[dict]]:
    """Train one VAE; returns the model and per-log-step metric history."""
    device = torch.device(args.device)
    model = build_vae(args.seed, args.latent_dim, args.base_channels).to(device)

    loss_fn = VAELoss(
        mel_weight=mel_weight,
        sample_rate=args.sample_rate,
    ).to(device)

    discriminator = None
    disc_optimizer = None
    if adversarial:
        torch.manual_seed(args.seed + 1)
        discriminator = MultiResolutionDiscriminator(
            resolutions=((1024, 256), (512, 128), (256, 64)),
            channels=args.disc_channels,
        ).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=args.disc_lr, betas=(0.8, 0.99)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    gen = torch.Generator().manual_seed(args.seed + 2)
    history = []
    start = time.time()

    for step in range(args.steps):
        idx = torch.randint(0, train_data.shape[0], (args.batch_size,), generator=gen)
        audio = train_data[idx].to(device)

        reconstruction, target, mean, log_var = model(audio)
        losses = loss_fn(reconstruction, target, mean, log_var)

        adv_active = adversarial and step >= args.adv_start_step
        if adv_active:
            # Discriminator update (fake detached)
            disc_optimizer.zero_grad()
            real_logits, real_features = discriminator(target)
            fake_logits_d, _ = discriminator(reconstruction.detach())
            d_loss = discriminator_loss(real_logits, fake_logits_d)
            d_loss.backward()
            disc_optimizer.step()

            # Generator adversarial terms (discriminator frozen)
            for p in discriminator.parameters():
                p.requires_grad_(False)
            fake_logits_g, fake_features = discriminator(reconstruction)
            adv_loss = generator_adversarial_loss(fake_logits_g)
            fm_loss = feature_matching_loss(
                [[f.detach() for f in maps] for maps in real_features],
                fake_features,
            )
            for p in discriminator.parameters():
                p.requires_grad_(True)

            losses["loss"] = (
                losses["loss"]
                + args.adv_weight * adv_loss
                + args.feature_matching_weight * fm_loss
            )
            losses["disc_loss"] = d_loss.detach()

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            row = {
                "step": step,
                **{k: float(v.detach()) for k, v in losses.items()},
            }
            history.append(row)
            elapsed = time.time() - start
            print(
                f"[{name}] step {step}/{args.steps} "
                f"loss={row['loss']:.4f} spectral={row['spectral_loss']:.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    return model, history


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(model: AudioVAE, test_clips: dict, sample_rate: int) -> tuple[dict, dict]:
    """Reconstruct held-out clips; return reconstructions and metrics."""
    model.eval()
    stft_metric = MultiResolutionSTFTLoss()
    mel_metric = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)

    recons, metrics = {}, {}
    for name, clip in test_clips.items():
        z = model.encode_to_latent(clip)
        recon = model.decode(z)
        min_len = min(recon.shape[-1], clip.shape[-1])
        recon, clip = recon[..., :min_len], clip[..., :min_len]
        recons[name] = recon

        # High-frequency band (>= 4 kHz) log-magnitude distance
        n_fft = 1024
        window = torch.hann_window(n_fft)
        spec_r = torch.stft(recon[0], n_fft, n_fft // 4, window=window, return_complex=True).abs()
        spec_t = torch.stft(clip[0], n_fft, n_fft // 4, window=window, return_complex=True).abs()
        hf_bin = int(4000 / (sample_rate / 2) * (n_fft // 2 + 1))
        hf_dist = F.l1_loss(
            torch.log(spec_r[:, hf_bin:] + 1e-5), torch.log(spec_t[:, hf_bin:] + 1e-5)
        )

        metrics[name] = {
            "stft_loss": float(stft_metric(recon, clip)),
            "mel_loss": float(mel_metric(recon, clip)),
            "l1": float(F.l1_loss(recon, clip)),
            "hf_log_dist": float(hf_dist),
        }
    model.train()
    return recons, metrics


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int):
    import soundfile as sf

    x = audio.squeeze().cpu().numpy()
    peak = np.abs(x).max()
    if peak > 1.0:
        x = x / peak * 0.98
    sf.write(path, x, sample_rate)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="./ab_vae_out")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--adv-start-step", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--num-train", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--disc-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    # Keep the adversarial term small next to reconstruction (DAC uses a
    # roughly 15:1 recon:adv balance); a strong discriminator destabilises
    # training and hurts held-out fidelity.
    parser.add_argument("--adv-weight", type=float, default=0.2)
    parser.add_argument("--feature-matching-weight", type=float, default=1.0)
    parser.add_argument(
        "--arms",
        type=str,
        default="baseline,improved",
        help="Comma list from: baseline (old objective), mel (adds "
        "multi-scale mel only), improved (mel + adversarial)",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    print("Generating procedural synth dataset...", flush=True)
    train_data, test_clips = make_dataset(
        args.seed, args.sample_rate, args.seconds, args.num_train
    )

    for name, clip in test_clips.items():
        save_wav(out_dir / "audio" / f"{name}_original.wav", clip, args.sample_rate)

    arm_defs = {
        "baseline": {"mel_weight": 0.0, "adversarial": False},
        "mel": {"mel_weight": 1.0, "adversarial": False},
        "improved": {"mel_weight": 1.0, "adversarial": True},
    }
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    results = {"args": vars(args)}
    for label in arms:
        spec = arm_defs[label]
        print(f"\n=== Training {label} objective ===", flush=True)
        model, history = train_vae(
            label, train_data, args, spec["mel_weight"], spec["adversarial"]
        )
        recons, metrics = evaluate(model, test_clips, args.sample_rate)
        for name, recon in recons.items():
            save_wav(
                out_dir / "audio" / f"{name}_{label}.wav", recon, args.sample_rate
            )
        results[label] = {"history": history, "metrics": metrics}

    # Aggregate
    for label in arms:
        m = results[label]["metrics"]
        results[label]["mean_metrics"] = {
            key: sum(v[key] for v in m.values()) / len(m)
            for key in next(iter(m.values()))
        }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Mean held-out metrics ===")
    for label in arms:
        print(label, results[label]["mean_metrics"])
    print(f"\nWAVs and results.json written to {out_dir}")


if __name__ == "__main__":
    main()
