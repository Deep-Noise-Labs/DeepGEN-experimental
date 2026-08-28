"""
Controlled A/B of the Stage-1 VAE objective.

Trains two Audio VAEs that are identical in architecture, data, data order,
seed, optimizer and step count, and differ *only* in the training objective:

    before  L1 + multi-resolution STFT + KL            (the objective on `main`)
    after   the above + multi-scale log-mel
            + hinge adversarial + feature matching     (this branch)

It then reconstructs a held-out set of real instrument samples through both
autoencoders and writes the audio plus objective metrics, so the difference can
be listened to rather than argued about.

This is deliberately a *small* experiment - it is sized to run on a CPU in under
an hour, at roughly 1/1000th of the compute of a real Stage-1 run. The latent
bitrate is matched to production (44100 / 256 x 32 dims mono = 5512 floats/s,
the same as 44100 x 2 / 1024 x 64 dims stereo), but both autoencoders are far
from converged. Read the result as a *relative* comparison of two objectives
under an identical budget, not as a sample of SynthGen's eventual quality.

Usage::

    python experiments/vae_objective_ab.py \
        --audio-dir ./data/instrument_samples \
        --output-dir ./experiments/out \
        --steps 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from synthgen.model.vae import AudioVAE
from synthgen.training.discriminator import AudioDiscriminator
from synthgen.training.losses import (
    MelSpectrogramLoss,
    VAELoss,
    discriminator_hinge_loss,
    mel_filterbank,
)
from synthgen.utils.audio import load_audio

SAMPLE_RATE = 44100


def save_float_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Write a 32-bit float WAV.

    Deliberately not 16-bit PCM. An undertrained decoder can sit 30 dB below the
    target, and at that level 16-bit quantisation noise is loud enough to change
    the *ordering* of any metric with an unbounded log floor - the log-spectral
    distance between the two arms reverses. Quantise once, at the point of
    listening, after level matching; never before measuring.
    """
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")


# =============================================================================
# Data
# =============================================================================


class CropDataset(torch.utils.data.Dataset):
    """Random fixed-length mono crops from a list of audio files."""

    def __init__(self, paths: list[Path], crop_samples: int, seed: int = 0):
        self.paths = paths
        self.crop_samples = crop_samples
        self.clips: list[np.ndarray] = []
        for path in paths:
            audio = load_audio(path, sample_rate=SAMPLE_RATE, channels=1)
            peak = float(np.abs(audio).max())
            if peak > 0:
                audio = audio / peak * 0.9
            self.clips.append(audio.astype(np.float32))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        # One "epoch" is a fixed number of random crops per clip.
        return len(self.clips) * 8

    def __getitem__(self, index: int) -> torch.Tensor:
        clip = self.clips[index % len(self.clips)]
        length = clip.shape[-1]
        if length <= self.crop_samples:
            pad = self.crop_samples - length
            crop = np.pad(clip, ((0, 0), (0, pad)))
        else:
            # Bias towards the first half of the file: that is where the note
            # attack lives, and transients are the interesting part.
            limit = max(1, length - self.crop_samples)
            start = self.rng.randrange(0, min(limit, max(1, length // 2)))
            crop = clip[:, start : start + self.crop_samples]
        return torch.from_numpy(np.ascontiguousarray(crop))


def deterministic_eval_crops(
    paths: list[Path], crop_samples: int
) -> list[tuple[str, torch.Tensor]]:
    """One fixed crop per held-out file, taken from the note onset."""
    crops = []
    for path in paths:
        audio = load_audio(path, sample_rate=SAMPLE_RATE, channels=1)
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak * 0.9
        # Start a little before the loudest 10 ms window so the attack is included.
        env = np.abs(audio[0])
        window = SAMPLE_RATE // 100
        if env.shape[0] > window:
            smoothed = np.convolve(env, np.ones(window) / window, mode="same")
            onset = max(0, int(np.argmax(smoothed > 0.15 * smoothed.max())) - window)
        else:
            onset = 0
        onset = min(onset, max(0, audio.shape[-1] - crop_samples))
        crop = audio[:, onset : onset + crop_samples]
        if crop.shape[-1] < crop_samples:
            crop = np.pad(crop, ((0, 0), (0, crop_samples - crop.shape[-1])))
        crops.append((path.stem, torch.from_numpy(crop.astype(np.float32))))
    return crops


# =============================================================================
# Metrics
# =============================================================================


class ReconstructionMetrics:
    """Objective reconstruction metrics, computed identically for both arms."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.mel = MelSpectrogramLoss(sample_rate=sample_rate)
        self.fb = mel_filterbank(1025, 128, sample_rate)
        self.window = torch.hann_window(2048)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x.reshape(-1, x.shape[-1]),
            2048,
            512,
            2048,
            self.window,
            return_complex=True,
        ).abs()

    def _band_db(self, mag: torch.Tensor, low: float, high: float) -> float:
        freqs = torch.linspace(0, self.sample_rate / 2, mag.shape[-2])
        mask = (freqs >= low) & (freqs < high)
        energy = (mag[..., mask, :] ** 2).sum()
        return float(10 * torch.log10(energy + 1e-12))

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        # The decoder's transposed convolutions need not land exactly on the
        # input length; compare only the overlap.
        length = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :length], target[..., :length]
        mag_p, mag_t = self._stft(pred), self._stft(target)

        # Log-spectral distance in dB, the standard spectral-fidelity number.
        lsd = float(
            torch.sqrt(
                (
                    (20 * torch.log10(mag_p + 1e-8) - 20 * torch.log10(mag_t + 1e-8))
                    ** 2
                ).mean()
            )
        )

        # SNR in the time domain: how much of the waveform (phase included) survived.
        noise = pred - target
        snr = float(
            10
            * torch.log10(
                (target**2).sum() / ((noise**2).sum() + 1e-12) + 1e-12
            )
        )

        # Overall output level relative to the target. An undertrained decoder
        # under a pure regression loss converges towards near-silence - that is
        # the safest "average" waveform - so this number is diagnostic in its own
        # right, and it contaminates every band measurement below if ignored.
        rms_p = float(torch.sqrt((pred**2).mean()) + 1e-12)
        rms_t = float(torch.sqrt((target**2).mean()) + 1e-12)
        output_level_db = 20 * math.log10(rms_p / rms_t)

        # Level-matched copy: isolates spectral *balance* from overall gain.
        mag_m = self._stft(pred * (rms_t / rms_p))

        bands = {
            "lf_0_1k": (0.0, 1000.0),
            "mid_1k_4k": (1000.0, 4000.0),
            "hi_4k_11k": (4000.0, 11000.0),
            "air_11k_22k": (11000.0, 22050.0),
        }
        band_error = {}
        for name, (low, high) in bands.items():
            target_db = self._band_db(mag_t, low, high)
            band_error[f"band_db_error_{name}"] = abs(
                self._band_db(mag_p, low, high) - target_db
            )
            band_error[f"band_db_error_matched_{name}"] = abs(
                self._band_db(mag_m, low, high) - target_db
            )

        # Transient sharpness: correlation of the onset envelopes. Conditional-mean
        # smearing shows up here before it shows up in any magnitude metric.
        def envelope(mag: torch.Tensor) -> torch.Tensor:
            energy = mag.sum(dim=-2)
            return torch.diff(energy, dim=-1).clamp(min=0)

        env_p, env_t = envelope(mag_p), envelope(mag_t)
        env_p = env_p - env_p.mean()
        env_t = env_t - env_t.mean()
        transient_corr = float(
            (env_p * env_t).sum()
            / (env_p.norm() * env_t.norm() + 1e-12)
        )

        return {
            "mel_distance": float(self.mel(pred.unsqueeze(0), target.unsqueeze(0))),
            "log_spectral_distance_db": lsd,
            "snr_db": snr,
            "output_level_db": output_level_db,
            "transient_envelope_corr": transient_corr,
            **band_error,
        }


# =============================================================================
# Training
# =============================================================================


def build_vae(args: argparse.Namespace) -> AudioVAE:
    return AudioVAE(
        in_channels=1,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        strides=tuple(args.strides),
    )


def train_arm(
    name: str,
    args: argparse.Namespace,
    dataset: CropDataset,
    adversarial: bool,
    mel_weight: float,
    log_path: Path,
) -> AudioVAE:
    """Train one arm of the A/B. Seeding is reset so both arms see identical data."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    vae = build_vae(args)
    loss_fn = VAELoss(
        l1_weight=0.1,
        spectral_weight=1.0,
        mel_weight=mel_weight,
        kl_weight=1e-4,
        adv_weight=1.0,
        fm_weight=2.0,
        sample_rate=SAMPLE_RATE,
    )

    discriminator = None
    disc_optimizer = None
    if adversarial:
        discriminator = AudioDiscriminator(
            periods=(2, 3, 5),
            period_channels=(16, 32, 64, 128),
            stft_resolutions=((1024, 256, 1024), (512, 128, 512), (256, 64, 256)),
            stft_channels=8,
        )
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9)
        )

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup)
        * (0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))) ** 0.5,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    vae.train()
    if discriminator is not None:
        discriminator.train()

    history: list[dict] = []
    data_iter = iter(loader)
    start = time.time()

    for step in range(args.steps):
        try:
            audio = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio = next(data_iter)

        adv_active = adversarial and step >= args.disc_start

        optimizer.zero_grad(set_to_none=True)
        if disc_optimizer is not None:
            disc_optimizer.zero_grad(set_to_none=True)

        reconstruction, target, mean, log_var = vae(audio)

        if adv_active:
            for param in discriminator.parameters():
                param.requires_grad_(False)
            fake_logits, fake_features = discriminator(reconstruction)
            with torch.no_grad():
                _, real_features = discriminator(target)
            losses = loss_fn(
                reconstruction,
                target,
                mean,
                log_var,
                fake_logits=fake_logits,
                real_features=real_features,
                fake_features=fake_features,
            )
        else:
            losses = loss_fn(reconstruction, target, mean, log_var)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        d_loss_value = 0.0
        if adv_active:
            for param in discriminator.parameters():
                param.requires_grad_(True)
            real_logits, _ = discriminator(target)
            fake_logits_d, _ = discriminator(reconstruction.detach())
            d_loss = discriminator_hinge_loss(real_logits, fake_logits_d)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            disc_optimizer.step()
            d_loss_value = float(d_loss.detach())

        if step % args.log_every == 0 or step == args.steps - 1:
            record = {
                "arm": name,
                "step": step,
                "elapsed_s": round(time.time() - start, 1),
                "adv_active": adv_active,
                "disc_loss": round(d_loss_value, 4),
                **{
                    key: round(float(value.detach()), 5)
                    for key, value in losses.items()
                },
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    log_path.write_text("\n".join(json.dumps(row) for row in history))
    return vae


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./experiments/out")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--disc-start", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--crop-seconds", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--held-out", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["before", "after"],
        choices=["before", "after"],
        help="Which arms to run (both by default)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p
        for p in Path(args.audio_dir).iterdir()
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
    )
    if len(paths) <= args.held_out:
        raise SystemExit(f"Need more than {args.held_out} audio files, found {len(paths)}")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    eval_paths, train_paths = paths[: args.held_out], paths[args.held_out :]
    print(f"train files: {len(train_paths)}  held-out files: {len(eval_paths)}")

    # Round the crop down to a whole number of latent frames so reconstructions
    # come back at exactly the input length and the A/B files line up sample for
    # sample in a DAW.
    compression = 1
    for stride in args.strides:
        compression *= stride
    crop_samples = (int(args.crop_seconds * SAMPLE_RATE) // compression) * compression
    dataset = CropDataset(train_paths, crop_samples, seed=args.seed)
    eval_crops = deterministic_eval_crops(eval_paths, crop_samples)

    arms = {
        "before": dict(adversarial=False, mel_weight=0.0),
        "after": dict(adversarial=True, mel_weight=15.0),
    }
    arms = {name: arms[name] for name in args.arms}

    models = {}
    for name, settings in arms.items():
        print(f"\n=== training arm: {name} ({settings}) ===", flush=True)
        models[name] = train_arm(
            name=name,
            args=args,
            dataset=dataset,
            log_path=output_dir / f"train_{name}.jsonl",
            **settings,
        )

    metrics_fn = ReconstructionMetrics(SAMPLE_RATE)
    results: dict[str, dict] = {}

    for stem, crop in eval_crops:
        save_float_wav(output_dir / "audio" / f"{stem}__original.wav", crop.numpy())
        results[stem] = {}
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                latent = model.encode_to_latent(crop.unsqueeze(0))
                recon = model.decode(latent)[0]
            recon = recon[..., : crop.shape[-1]]
            save_float_wav(output_dir / "audio" / f"{stem}__{name}.wav", recon.numpy())
            results[stem][name] = metrics_fn(recon, crop)

    # Aggregate: mean per metric per arm.
    summary: dict[str, dict[str, float]] = {}
    metric_names = list(next(iter(results.values()))[next(iter(models))].keys())
    for arm in models:
        summary[arm] = {
            metric: float(np.mean([results[s][arm][metric] for s in results]))
            for metric in metric_names
        }

    payload = {
        "config": vars(args),
        "train_files": [p.name for p in train_paths],
        "eval_files": [p.name for p in eval_paths],
        "per_clip": results,
        "summary": summary,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    print("\n=== summary (mean over held-out clips) ===")
    for metric in metric_names:
        row = "  ".join(f"{arm}={summary[arm][metric]:9.4f}" for arm in models)
        print(f"{metric:34s} {row}")


if __name__ == "__main__":
    main()
