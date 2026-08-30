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

    Computes spectral convergence and log-magnitude terms at multiple FFT
    sizes, plus two terms that a magnitude-only objective cannot express:

    **Phase (complex) term.** Spectral convergence and log-magnitude both
    depend on ``|STFT(x)|`` alone, so they are exactly invariant to phase.
    Any two signals sharing a magnitude spectrogram -- including a signal and
    its own polarity inversion -- score identically, however different they
    sound. Transient smear and the "underwater" quality typical of latent
    audio autoencoders live entirely in this blind spot. The complex term
    measures ``|STFT(pred) - STFT(target)|``, normalised by the mean target
    magnitude so it stays scale-invariant and O(1).

    **Stereo (mid/side) term.** The magnitude terms flatten ``(B, C, T)`` to
    ``(B*C, T)`` and score each channel independently, so the relationship
    between channels is unconstrained. A decoder can preserve both channel
    magnitudes perfectly while destroying the stereo image -- and a
    polarity-flipped channel disappears on mono fold-down. Scoring the
    mid/side decomposition constrains the image directly.

    Both extra terms default to on. Set ``phase_weight=0.0`` and
    ``stereo_weight=0.0`` to recover the previous magnitude-only behaviour.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256),
        hop_sizes: tuple = (512, 256, 128, 64),
        win_sizes: tuple = (2048, 1024, 512, 256),
        sc_weight: float = 1.0,
        log_mag_weight: float = 1.0,
        phase_weight: float = 1.0,
        stereo_weight: float = 1.0,
        log_eps: float = 1e-5,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.sc_weight = sc_weight
        self.log_mag_weight = log_mag_weight
        self.phase_weight = phase_weight
        self.stereo_weight = stereo_weight
        self.log_eps = log_eps

        # Cache one Hann window per resolution instead of reallocating
        # on every forward pass.
        for win_size in sorted(set(win_sizes)):
            self.register_buffer(
                f"window_{win_size}",
                torch.hann_window(win_size),
                persistent=False,
            )

    def _stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> torch.Tensor:
        """Compute the complex STFT, flattening any channel dimension."""
        # x shape: (batch, samples) or (batch, channels, samples)
        if x.dim() == 3:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, samples)

        window = getattr(self, f"window_{win_size}").to(x.device, x.dtype)
        return torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True,
        )

    @staticmethod
    def _mid_side(x: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 2, samples) L/R audio to stacked mid/side."""
        left, right = x[:, 0], x[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        return torch.stack([mid, side], dim=1)

    def _spectral_terms(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Accumulate the per-resolution terms for one channel layout."""
        sc_total = pred.new_zeros(())
        log_total = pred.new_zeros(())
        phase_total = pred.new_zeros(())

        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            pred_stft = self._stft(pred, fft_size, hop_size, win_size)
            target_stft = self._stft(target, fft_size, hop_size, win_size)

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Spectral convergence loss
            sc_total = sc_total + torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )

            # Log-magnitude loss. Clamping rather than adding the epsilon
            # keeps near-silent bins from dominating the gradient: with a
            # 1e-8 additive floor an empty bin contributes log(1e-8) ~= -18.4.
            log_total = log_total + F.l1_loss(
                torch.log(pred_mag.clamp_min(self.log_eps)),
                torch.log(target_mag.clamp_min(self.log_eps)),
            )

            if self.phase_weight > 0:
                # Complex error modulus, normalised by mean target magnitude.
                # The +1e-12 keeps the gradient of sqrt finite at zero error.
                diff = pred_stft - target_stft
                modulus = torch.sqrt(diff.real ** 2 + diff.imag ** 2 + 1e-12)
                phase_total = phase_total + modulus.mean() / (
                    target_mag.mean() + 1e-8
                )

        num_resolutions = len(self.fft_sizes)
        return {
            "sc": sc_total / num_resolutions,
            "log_mag": log_total / num_resolutions,
            "phase": phase_total / num_resolutions,
        }

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return the individual loss terms, for logging and diagnostics.

        Keys: ``sc``, ``log_mag``, ``phase``, ``stereo``, ``total``.
        """
        terms = self._spectral_terms(pred, target)

        stereo = pred.new_zeros(())
        if self.stereo_weight > 0 and pred.dim() == 3 and pred.shape[1] == 2:
            ms_terms = self._spectral_terms(
                self._mid_side(pred), self._mid_side(target)
            )
            stereo = ms_terms["sc"] + ms_terms["log_mag"]
            if self.phase_weight > 0:
                stereo = stereo + self.phase_weight * ms_terms["phase"]

        total = (
            self.sc_weight * terms["sc"]
            + self.log_mag_weight * terms["log_mag"]
            + self.phase_weight * terms["phase"]
            + self.stereo_weight * stereo
        )

        return {**terms, "stereo": stereo, "total": total}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        return self.components(pred, target)["total"]


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
        phase_weight: float = 1.0,
        stereo_weight: float = 1.0,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight

        self.spectral_loss = MultiResolutionSTFTLoss(
            phase_weight=phase_weight,
            stereo_weight=stereo_weight,
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

        # Multi-resolution STFT loss (magnitude + phase + stereo image)
        spectral = self.spectral_loss.components(reconstruction, target)
        spectral_loss = spectral["total"]

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
            # Diagnostics: which part of the spectral objective is unhappy.
            "spectral_sc": spectral["sc"],
            "spectral_log_mag": spectral["log_mag"],
            "spectral_phase": spectral["phase"],
            "spectral_stereo": spectral["stereo"],
        }


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
