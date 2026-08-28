"""
Verbatim copy of the Stage-1 VAE objective as it exists on ``main``
(commit 4907d7f, ``synthgen/training/losses.py``).

Kept so the ablation can score both objectives inside a single process without
checking out two working trees. Do not "fix" anything in here — its whole
purpose is to be the *before* side of the comparison. The real objective lives
in ``synthgen/training/losses.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LegacyMultiResolutionSTFTLoss", "LegacyVAELoss"]


class LegacyMultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss, as on ``main``."""

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

    def _stft(self, x, fft_size, hop_size, win_size):
        if x.dim() == 3:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, samples)
        window = torch.hann_window(win_size, device=x.device)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window, return_complex=True
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)
            target_mag = self._stft(target, fft_size, hop_size, win_size)

            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )
            log_loss = F.l1_loss(
                torch.log(pred_mag + 1e-8),
                torch.log(target_mag + 1e-8),
            )
            total_loss += sc_loss + log_loss

        return total_loss / len(self.fft_sizes)


class LegacyVAELoss(nn.Module):
    """Combined VAE loss, as on ``main``."""

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
        self.spectral_loss = LegacyMultiResolutionSTFTLoss()

    def forward(self, reconstruction, target, mean, log_var):
        l1_loss = F.l1_loss(reconstruction, target)
        spectral_loss = self.spectral_loss(reconstruction, target)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

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
