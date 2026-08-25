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

    The default resolution bank spans 64 to 8192 samples. At 44.1 kHz the
    8192-sample window resolves ~5.4 Hz - enough to pin down the fundamental
    of a sub bass or 808 - while the 64/128-sample windows localise energy at
    1.5-3 ms, so smeared attack transients register as magnitude error even
    though the loss itself is phase-free. The previous bank (256-2048) had
    neither end: 21.5 Hz bins in the sub-bass octave and >=6 ms time blur.

    For stereo signals, ``sum_and_difference=True`` additionally applies the
    log-magnitude term to the mid ((L+R)/2) and side ((L-R)/2) signals
    (Steinmetz et al., auraloss; used by Stable Audio's autoencoder). A
    per-channel loss cannot see stereo-image errors - collapsing a wide pad
    to mono leaves each channel's magnitude spectrum almost unchanged - but
    the side signal exposes them directly. Spectral convergence is skipped
    for mid/side because it normalises by target energy, which is exactly
    zero for the side channel of mono-in-stereo material.

    The log-magnitude term uses ``log_eps`` as an audibility floor. The
    previous value (1e-8, roughly -160 dBFS) meant most bins of typical
    synth material sat at the float32 FFT noise floor, where bin-to-bin
    roundoff noise - not audio - dominated the average and the gradients.
    The default (1e-5, -100 dBFS for peak-normalised audio) makes the term
    spend its capacity on bins that can actually be heard.

    STFTs are computed in float32 regardless of input dtype, so the loss is
    safe (and numerically stable) under bf16/fp16 autocast.
    """

    def __init__(
        self,
        fft_sizes: tuple = (8192, 2048, 1024, 512, 256, 128, 64),
        hop_sizes: tuple = (2048, 512, 256, 128, 64, 32, 16),
        win_sizes: tuple = (8192, 2048, 1024, 512, 256, 128, 64),
        sum_and_difference: bool = True,
        sd_weight: float = 1.0,
        log_eps: float = 1e-5,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.sum_and_difference = sum_and_difference
        self.sd_weight = sd_weight
        self.log_eps = log_eps

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

        x = x.float()
        window = torch.hann_window(win_size, device=x.device)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True,
        )
        return stft.abs()

    def _multi_resolution_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sc_weight: float = 1.0,
    ) -> torch.Tensor:
        """Average SC + log-magnitude loss over all usable resolutions."""
        total_loss = pred.new_zeros((), dtype=torch.float32)
        num_used = 0

        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            # torch.stft centre-pads by fft_size // 2 (reflect), which
            # requires the signal to be longer than the padding.
            if pred.shape[-1] <= fft_size // 2:
                continue

            pred_mag = self._stft(pred, fft_size, hop_size, win_size)
            target_mag = self._stft(target, fft_size, hop_size, win_size)

            if sc_weight > 0:
                # Spectral convergence loss
                sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                    torch.norm(target_mag, p="fro") + 1e-8
                )
                total_loss = total_loss + sc_weight * sc_loss

            # Log-magnitude loss
            log_loss = F.l1_loss(
                torch.log(pred_mag + self.log_eps),
                torch.log(target_mag + self.log_eps),
            )

            total_loss = total_loss + log_loss
            num_used += 1

        return total_loss / max(num_used, 1)

    @staticmethod
    def _to_mid_side(x: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 2, samples) stereo to stacked mid/side channels."""
        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.5 * (x[:, 0] - x[:, 1])
        return torch.stack([mid, side], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        total_loss = self._multi_resolution_loss(pred, target, sc_weight=1.0)

        if self.sum_and_difference and pred.dim() == 3 and pred.shape[1] == 2:
            pred_ms = self._to_mid_side(pred)
            target_ms = self._to_mid_side(target)
            total_loss = total_loss + self.sd_weight * self._multi_resolution_loss(
                pred_ms, target_ms, sc_weight=0.0
            )

        return total_loss


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
# DiT Losses
# =============================================================================


class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for DiT training.

    Simple MSE between predicted and target velocity fields.
    Optionally supports loss weighting based on timestep.
    """

    def __init__(self, weighting: str = "uniform", gamma: float = 5.0):
        """
        Args:
            weighting: Timestep weighting strategy.
                - "uniform": Equal weight for all timesteps.
                - "snr": Signal-to-noise ratio weighting.
                - "min_snr": Min-SNR-gamma weighting (Hang et al. 2023),
                  in the velocity-prediction form min(SNR, gamma) / (SNR + 1).
            gamma: SNR cap for "min_snr".
        """
        super().__init__()
        self.weighting = weighting
        self.gamma = gamma

    def _snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        SNR of the flow-matching interpolant at timestep t.

        For x_t = t * x_0 + (1 - t) * noise the signal coefficient is t and
        the noise coefficient is (1 - t), so SNR(t) = t^2 / (1 - t)^2.
        t is clamped away from {0, 1} where the SNR is 0 / infinite and the
        weight expressions would otherwise divide by zero.
        """
        t = t.clamp(1e-4, 1.0 - 1e-4)
        return (t / (1.0 - t)) ** 2

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
            snr = self._snr(t)
            weights = 1.0 / (snr + 1.0)
            weights = weights / weights.mean()
            return (mse * weights).mean()

        elif self.weighting == "min_snr":
            # Min-SNR-gamma weighting, velocity-prediction form:
            # min(SNR, gamma) / (SNR + 1) (Hang et al. 2023, sec. 4;
            # the epsilon-prediction form divides by SNR instead).
            snr = self._snr(t)
            weights = torch.minimum(snr, torch.full_like(snr, self.gamma)) / (
                snr + 1.0
            )
            weights = weights / weights.mean()
            return (mse * weights).mean()

        else:
            return mse.mean()
