"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + spectral + adversarial)
- Adversarial VAE training (hinge GAN + feature matching, with the
  multi-scale STFT discriminator in synthgen.model.discriminator)
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


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Reconstruction loss (L1 + multi-resolution STFT)
    - KL divergence loss
    - Optional adversarial loss
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight

        self.spectral_loss = MultiResolutionSTFTLoss()

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

        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "kl_loss": kl_loss,
        }


# =============================================================================
# Adversarial VAE Losses
# =============================================================================


def discriminator_loss(
    real_logits: list[torch.Tensor],
    fake_logits: list[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for the discriminator, averaged over scales.

    Pushes real logits above +1 and fake logits below -1.

    Args:
        real_logits: Per-scale logit maps for real audio.
        fake_logits: Per-scale logit maps for reconstructed audio
            (detached from the generator graph).

    Returns:
        Scalar discriminator loss.
    """
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


def generator_adversarial_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """
    Hinge generator loss, averaged over scales.

    Pushes the discriminator's logits on reconstructions upward.

    Args:
        fake_logits: Per-scale logit maps for reconstructed audio
            (with gradients flowing back to the generator).

    Returns:
        Scalar generator adversarial loss.
    """
    loss = 0.0
    for fake in fake_logits:
        loss = loss - fake.mean()
    return loss / len(fake_logits)


def feature_matching_loss(
    fake_features: list[list[torch.Tensor]],
    real_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss between discriminator activations.

    L1 distance between the discriminator's intermediate feature maps for
    real and reconstructed audio, normalized per layer by the mean
    magnitude of the real features (as in EnCodec). This stabilizes GAN
    training and acts as a learned perceptual reconstruction loss.

    Args:
        fake_features: Per-scale lists of feature maps for reconstructions.
        real_features: Per-scale lists of feature maps for real audio
            (detached; no gradients flow into the discriminator).

    Returns:
        Scalar feature matching loss.
    """
    loss = 0.0
    num_layers = 0
    for scale_fake, scale_real in zip(fake_features, real_features):
        for fake, real in zip(scale_fake, scale_real):
            real = real.detach()
            loss = loss + F.l1_loss(fake, real) / (real.abs().mean() + 1e-8)
            num_layers += 1
    return loss / max(num_layers, 1)


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
