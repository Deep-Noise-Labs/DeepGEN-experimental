"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + spectral + adversarial)
- DiT training (flow matching velocity MSE)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """
    Build a triangular mel filterbank matrix (HTK mel scale), pure torch.

    Returns:
        Tensor of shape (n_mels, n_fft // 2 + 1).
    """
    fmax = fmax or sample_rate / 2

    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_points = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    bins = torch.clamp(bins, 0, n_fft // 2)

    fbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fbank[i, left:center] = (
                torch.arange(left, center) - left
            ).float() / max(1, (center - left).item())
        if right > center:
            fbank[i, center:right] = (
                right - torch.arange(center, right)
            ).float() / max(1, (right - center).item())
    return fbank


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Multi-scale log-mel spectrogram loss (DAC / Stable Audio recipe).

    L1 distance between log-mel spectrograms at several FFT resolutions.
    Compared to a linear-frequency magnitude loss, the mel warping spends
    its capacity where hearing does, so errors in the perceptually dense
    low/mid bands are weighted appropriately while the loss still covers
    the full bandwidth at every scale.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_sizes: tuple = (2048, 1024, 512, 256, 128),
        n_mels: tuple = (160, 80, 40, 20, 10),
        clamp_eps: float = 1e-5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.clamp_eps = clamp_eps

        for i, (fft_size, mels) in enumerate(zip(fft_sizes, n_mels)):
            self.register_buffer(
                f"fbank_{i}",
                _mel_filterbank(sample_rate, fft_size, mels),
                persistent=False,
            )
            self.register_buffer(
                f"window_{i}", torch.hann_window(fft_size), persistent=False
            )

    def _log_mel(self, x: torch.Tensor, index: int) -> torch.Tensor:
        fft_size = self.fft_sizes[index]
        if x.dim() == 3:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, samples)

        window = getattr(self, f"window_{index}")
        fbank = getattr(self, f"fbank_{index}")

        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            win_length=fft_size,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs()  # (batch, freq, frames)
        mel = torch.matmul(fbank, magnitude)
        return torch.log(mel.clamp(min=self.clamp_eps))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        total_loss = 0.0
        for i in range(len(self.fft_sizes)):
            total_loss += F.l1_loss(self._log_mel(pred, i), self._log_mel(target, i))
        return total_loss / len(self.fft_sizes)


# =============================================================================
# Adversarial Losses (VAE stage)
# =============================================================================


def discriminator_loss(
    real_logits: list[torch.Tensor],
    fake_logits: list[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for the discriminator.

    Args:
        real_logits: Logit maps for real audio, one per resolution.
        fake_logits: Logit maps for reconstructed audio (detached).
    """
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss += F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


def generator_adversarial_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Hinge generator loss: push discriminator logits on fakes up."""
    loss = 0.0
    for fake in fake_logits:
        loss += (-fake).mean()
    return loss / len(fake_logits)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 distance between discriminator feature maps of real and fake audio.

    Stabilises adversarial training and acts as a learned perceptual loss.
    Real features are detached - the discriminator is not trained by this term.
    """
    loss = 0.0
    count = 0
    for real_maps, fake_maps in zip(real_features, fake_features):
        for real, fake in zip(real_maps, fake_maps):
            loss += F.l1_loss(fake, real.detach())
            count += 1
    return loss / max(1, count)


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Reconstruction loss (L1 + multi-resolution STFT + multi-scale log-mel)
    - KL divergence loss

    Adversarial and feature-matching terms are computed in the trainer
    (they need the discriminator) and added to the total there.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
        mel_weight: float = 0.0,
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
            MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
            if mel_weight > 0
            else None
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
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "kl_loss": kl_loss,
        }

        # Multi-scale log-mel loss
        if self.mel_loss is not None:
            mel_loss = self.mel_loss(reconstruction, target)
            total_loss = total_loss + self.mel_weight * mel_loss
            losses["mel_loss"] = mel_loss

        losses["loss"] = total_loss
        return losses


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
