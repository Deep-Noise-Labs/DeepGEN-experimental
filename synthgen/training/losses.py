"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + spectral + mel + adversarial)
- DiT training (flow matching velocity MSE)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MultiResolutionSTFTLoss",
    "MultiResolutionMelLoss",
    "VAELoss",
    "feature_matching_loss",
    "generator_adversarial_loss",
    "discriminator_adversarial_loss",
    "FlowMatchingLoss",
]


# =============================================================================
# VAE Losses
# =============================================================================


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for audio reconstruction quality.

    Computes spectral convergence and log-magnitude loss at multiple
    FFT sizes to capture both fine and coarse frequency details.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256),
        hop_sizes: tuple = (512, 256, 128, 64),
        win_sizes: tuple = (2048, 1024, 512, 256),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> torch.Tensor:
        """Compute STFT magnitude."""
        # x shape: (batch, samples) or (batch, channels, samples)
        if x.dim() == 3:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, samples)
        else:
            batch = x.shape[0]

        window = torch.hann_window(win_size, device=x.device)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True,
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        total_loss = 0.0

        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)
            target_mag = self._stft(target, fft_size, hop_size, win_size)

            # Spectral convergence loss
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )

            # Log-magnitude loss
            log_loss = F.l1_loss(
                torch.log(pred_mag + 1e-8),
                torch.log(target_mag + 1e-8),
            )

            total_loss += sc_loss + log_loss

        return total_loss / len(self.fft_sizes)


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """HTK mel scale."""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """
    Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Implemented in torch rather than pulled from librosa so the loss is a pure
    tensor op that lives on whatever device the training step is on.
    """
    fmax = sample_rate / 2.0 if fmax is None else fmax
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(fmin)).item(),
        hz_to_mel(torch.tensor(fmax)).item(),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)

    # Slopes of each triangle against every FFT bin centre.
    diff = hz_points[1:] - hz_points[:-1]
    ramps = hz_points.unsqueeze(1) - fft_freqs.unsqueeze(0)  # (n_mels + 2, n_bins)

    lower = -ramps[:-2] / diff[:-1].unsqueeze(1)
    upper = ramps[2:] / diff[1:].unsqueeze(1)
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)

    # Slaney-style area normalisation: equal energy per band, not equal peak.
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    return weights * enorm.unsqueeze(1)


class MultiResolutionMelLoss(nn.Module):
    """
    L1 on log-mel spectrograms at several time/frequency resolutions.

    Why this and not just :class:`MultiResolutionSTFTLoss`: a linear-frequency
    STFT L1 spends its capacity in proportion to bin count, and half of a
    linear spectrum's bins sit in the top octave (11--22 kHz) where the ear has
    almost no frequency resolution and very little sensitivity. Meanwhile the
    2--3 bins covering 20--200 Hz -- where the fundamental of a bass patch or a
    piano's bottom register lives -- contribute almost nothing to the gradient.
    A mel-spaced criterion redistributes that weight to match human hearing,
    which is what "sounds right" actually means.

    The short windows (64--256 samples) resolve transients -- the attack of a
    pluck, the click of a percussive hit -- which a 2048-sample window smears
    across 46 ms. The long windows resolve steady-state partials. Both matter
    for sample-library material; neither alone is sufficient.

    Defaults follow the Descript Audio Codec recipe, trimmed to five
    resolutions to keep the step cost reasonable at 44.1 kHz.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        window_sizes: tuple[int, ...] = (2048, 1024, 512, 128, 64),
        n_mels: tuple[int, ...] = (320, 160, 80, 20, 10),
        # 20 Hz floor: below the audible band, and it keeps DC offset and
        # sub-audio rumble from taking gradient away from things you can hear.
        fmin: float = 20.0,
        fmax: float | None = None,
        log_weight: float = 1.0,
        mag_weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if len(window_sizes) != len(n_mels):
            raise ValueError("window_sizes and n_mels must be the same length")

        self.window_sizes = window_sizes
        self.hop_sizes = tuple(w // 4 for w in window_sizes)
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.eps = eps

        for i, (win, mels) in enumerate(zip(window_sizes, n_mels)):
            self.register_buffer(
                f"fb_{i}",
                mel_filterbank(sample_rate, win, mels, fmin=fmin, fmax=fmax),
                persistent=False,
            )
            self.register_buffer(
                f"window_{i}", torch.hann_window(win), persistent=False
            )

    def _log_mel(self, x: torch.Tensor, index: int) -> torch.Tensor:
        win = self.window_sizes[index]
        spec = torch.stft(
            x,
            n_fft=win,
            hop_length=self.hop_sizes[index],
            win_length=win,
            window=getattr(self, f"window_{index}").to(x.device, x.dtype),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        ).abs()
        fb = getattr(self, f"fb_{index}").to(x.device, x.dtype)
        return fb @ spec

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            batch, channels, samples = pred.shape
            pred = pred.reshape(batch * channels, samples)
            target = target.reshape(batch * channels, samples)

        # STFT in fp32 regardless of the autocast context.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            total = pred.new_zeros(())
            for i in range(len(self.window_sizes)):
                pred_mel = self._log_mel(pred, i)
                target_mel = self._log_mel(target, i)
                total = total + self.mag_weight * F.l1_loss(pred_mel, target_mel)
                total = total + self.log_weight * F.l1_loss(
                    torch.log(pred_mel.clamp(min=self.eps)),
                    torch.log(target_mel.clamp(min=self.eps)),
                )

        return total / len(self.window_sizes)


class VAELoss(nn.Module):
    """
    Reconstruction + KL objective for the audio autoencoder.

    Components:
    - L1 on the waveform
    - Multi-resolution STFT (linear frequency)
    - Multi-resolution mel (perceptual frequency weighting)
    - KL divergence

    The adversarial and feature-matching terms live in
    :func:`generator_adversarial_loss` / :func:`feature_matching_loss` and are
    combined by the trainer, which owns the discriminator and its optimiser.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
        mel_weight: float = 1.0,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight

        self.spectral_loss = MultiResolutionSTFTLoss()
        self.mel_loss = (
            MultiResolutionMelLoss(sample_rate=sample_rate) if mel_weight > 0 else None
        )

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstruction: Reconstructed audio.
            target: Original audio.
            mean: Latent mean from encoder.
            log_var: Latent log-variance from encoder.

        Returns:
            Dictionary with total loss and individual components.
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(reconstruction, target)

        # Multi-resolution STFT loss
        spectral_loss = self.spectral_loss(reconstruction, target)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mean.pow(2) - log_var.exp()
        )

        # Total loss
        total_loss = (
            self.l1_weight * l1_loss
            + self.spectral_weight * spectral_loss
            + self.kl_weight * kl_loss
        )

        losses = {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "kl_loss": kl_loss,
        }

        if self.mel_loss is not None:
            mel_loss = self.mel_loss(reconstruction, target)
            losses["mel_loss"] = mel_loss
            losses["loss"] = total_loss + self.mel_weight * mel_loss

        return losses


# =============================================================================
# Adversarial losses
# =============================================================================


def feature_matching_loss(
    features_real: list[list[torch.Tensor]],
    features_fake: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 between the discriminator's intermediate activations on real vs fake.

    This is what keeps adversarial autoencoder training stable: it gives the
    generator a dense, well-conditioned signal ("match what the critic notices")
    instead of only the single scalar of the adversarial term, which on its own
    is happy to be satisfied by artefacts that fool the critic without
    resembling the target.
    """
    total = None
    count = 0
    for real_stack, fake_stack in zip(features_real, features_fake):
        for real, fake in zip(real_stack, fake_stack):
            term = F.l1_loss(fake, real.detach())
            total = term if total is None else total + term
            count += 1
    if total is None:
        raise ValueError("feature_matching_loss received no features")
    return total / count


def generator_adversarial_loss(logits_fake: list[torch.Tensor]) -> torch.Tensor:
    """
    Hinge generator loss: ``mean(-D(fake))`` over every sub-discriminator.

    Hinge rather than least-squares: it saturates once the generator has
    convinced a given critic, so gradient budget moves to the critics that are
    still winning instead of being spent pushing an already-won logit higher.
    """
    total = None
    for logit in logits_fake:
        term = -logit.mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("generator_adversarial_loss received no logits")
    return total / len(logits_fake)


def discriminator_adversarial_loss(
    logits_real: list[torch.Tensor],
    logits_fake: list[torch.Tensor],
) -> torch.Tensor:
    """Hinge discriminator loss: ``mean(relu(1 - D(real)) + relu(1 + D(fake)))``."""
    total = None
    for real, fake in zip(logits_real, logits_fake):
        term = F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("discriminator_adversarial_loss received no logits")
    return total / len(logits_real)


# =============================================================================
# DiT Losses
# =============================================================================


class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for DiT training.

    Simple MSE between predicted and target velocity fields.
    Optionally supports loss weighting based on timestep.
    """

    def __init__(self, weighting: str = "uniform"):
        """
        Args:
            weighting: Timestep weighting strategy.
                - "uniform": Equal weight for all timesteps.
                - "snr": Signal-to-noise ratio weighting.
                - "min_snr": Min-SNR weighting (capped at gamma=5).
        """
        super().__init__()
        self.weighting = weighting

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            v_pred: Predicted velocity (batch, latent_dim, seq_len).
            v_target: Target velocity (batch, latent_dim, seq_len).
            t: Timesteps (batch,) for weighted loss.

        Returns:
            Scalar loss value.
        """
        # Per-sample MSE
        mse = F.mse_loss(v_pred, v_target, reduction="none")
        mse = mse.mean(dim=[1, 2])  # Average over latent_dim and seq_len

        if self.weighting == "uniform" or t is None:
            return mse.mean()

        elif self.weighting == "snr":
            # SNR weighting: higher weight for noisier timesteps
            snr = t / (1 - t + 1e-8)
            weights = 1.0 / (snr + 1.0)
            weights = weights / weights.mean()
            return (mse * weights).mean()

        elif self.weighting == "min_snr":
            # Min-SNR-gamma weighting (gamma=5)
            gamma = 5.0
            snr = t / (1 - t + 1e-8)
            weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
            weights = weights / weights.mean()
            return (mse * weights).mean()

        else:
            return mse.mean()
