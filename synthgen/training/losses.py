"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + spectral + adversarial)
- DiT training (flow matching velocity MSE)

On the VAE objective
--------------------
The original objective here was ``0.1 * L1(waveform) + 1.0 * MRSTFT(magnitude)``.
Measured against real audio, that objective ranks a 41 ms all-pass transient
smear as only ~27% as costly as a mild 9 kHz low-pass, even though the smear
leaves the magnitude spectrum within 0.6 dB of the reference and is far more
audible. Anything an objective scores as cheap is inside its optimum, so the
decoder is free to converge there - which is exactly the smeared, dull, narrow
character that separates model output from a Spitfire or Splice sample.

Three additions fix the three blind spots, see ``docs/VAE_OBJECTIVE.md``:

1. ``MultiScaleMelSpectrogramLoss`` - mel weighting so the loss budget follows
   the ear instead of following bass energy (a Frobenius norm over linear
   magnitudes is dominated by the loudest low bins).
2. ``mid_side=True`` - spectral error is measured on mid/side rather than L/R,
   so collapsing the stereo image is actually penalised.
3. Adversarial + feature-matching terms against a waveform/complex-STFT
   discriminator - the only part of the objective that sees phase at all.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared spectral helpers
# =============================================================================


def to_mid_side(x: torch.Tensor) -> torch.Tensor:
    """
    Convert (batch, 2, samples) L/R audio to mid/side.

    A loss computed independently on L and R is only weakly sensitive to stereo
    width: shrinking the side signal moves L and R towards each other, and both
    stay close to their targets. Measuring on mid/side makes the side channel a
    first-class target instead.

    Non-stereo inputs are returned unchanged.
    """
    if x.dim() != 3 or x.shape[1] != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    return torch.stack([mid, side], dim=1)


def hz_to_mel(freq: torch.Tensor | float) -> torch.Tensor | float:
    """HTK mel scale."""
    if isinstance(freq, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Inverse HTK mel scale."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Triangular mel filterbank of shape (n_mels, n_fft // 2 + 1).

    Implemented in plain PyTorch rather than via torchaudio/librosa so the loss
    has no extra runtime dependency and stays differentiable on any device.
    """
    f_max = f_max if f_max is not None else sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate / 2.0, n_freqs, dtype=dtype)

    m_pts = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2, dtype=dtype)
    f_pts = mel_to_hz(m_pts)

    # slopes[i, j] = f_pts[j] - all_freqs[i]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    d = f_pts[1:] - f_pts[:-1]
    d = torch.clamp(d, min=1e-8)

    down = -slopes[:, :-2] / d[:-1]
    up = slopes[:, 2:] / d[1:]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    return fb.T.contiguous()


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


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Multi-scale log-mel reconstruction loss (DAC-style).

    Two differences from ``MultiResolutionSTFTLoss`` matter:

    - **Mel weighting.** A spectral-convergence term over linear magnitudes is a
      Frobenius norm, so it is dominated by whichever bins hold the most energy -
      in practice the bottom two octaves. Mel bands spread the loss budget the
      way hearing does, which is what stops the decoder trading away the top end.
    - **Several band resolutions, down to 5 bands / 32-sample windows.** Short
      windows resolve transients; the coarse band counts supervise gross
      spectral balance. Together they constrain attack shape far more tightly
      than four long-window resolutions can.

    Set ``mid_side=True`` to measure on mid/side instead of L/R, which makes the
    loss sensitive to stereo width.
    """

    def __init__(
        self,
        window_lengths: tuple = (2048, 1024, 512, 128, 32),
        n_mels: tuple = (160, 80, 40, 20, 5),
        sample_rate: int = 44100,
        mid_side: bool = True,
        log_eps: float = 1e-5,
        log_weight: float = 1.0,
        mag_weight: float = 1.0,
    ):
        super().__init__()
        if len(window_lengths) != len(n_mels):
            raise ValueError("window_lengths and n_mels must have the same length")

        self.window_lengths = tuple(window_lengths)
        self.n_mels = tuple(n_mels)
        self.sample_rate = sample_rate
        self.mid_side = mid_side
        self.log_eps = log_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight

        for i, (win, mels) in enumerate(zip(self.window_lengths, self.n_mels)):
            n_freqs = win // 2 + 1
            if mels > n_freqs:
                raise ValueError(
                    f"n_mels={mels} exceeds the {n_freqs} FFT bins available at "
                    f"window_length={win}"
                )
            self.register_buffer(
                f"fb_{i}", mel_filterbank(win, mels, sample_rate), persistent=False
            )
            self.register_buffer(
                f"window_{i}", torch.hann_window(win), persistent=False
            )

    def _mel(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """(batch * channels, samples) -> (batch * channels, n_mels, frames)."""
        win = self.window_lengths[index]
        window = getattr(self, f"window_{index}").to(device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x.float(),
            n_fft=win,
            hop_length=win // 4,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        fb = getattr(self, f"fb_{index}").to(device=x.device, dtype=torch.float32)
        return torch.einsum("mf,bft->bmt", fb, mag)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mid_side:
            pred = to_mid_side(pred)
            target = to_mid_side(target)

        if pred.dim() == 3:
            batch, channels, samples = pred.shape
            pred = pred.reshape(batch * channels, samples)
            target = target.reshape(batch * channels, samples)

        total = pred.new_zeros(())
        for i in range(len(self.window_lengths)):
            p = self._mel(pred, i)
            t = self._mel(target, i)
            total = total + self.mag_weight * F.l1_loss(p, t)
            total = total + self.log_weight * F.l1_loss(
                torch.log10(p + self.log_eps), torch.log10(t + self.log_eps)
            )
        return total / len(self.window_lengths)


# =============================================================================
# Adversarial losses
# =============================================================================


class DiscriminatorAdversarialLoss(nn.Module):
    """
    Discriminator objective. ``hinge`` (DAC/EnCodec) or ``lsgan`` (HiFi-GAN).
    """

    def __init__(self, mode: str = "hinge"):
        super().__init__()
        if mode not in ("hinge", "lsgan"):
            raise ValueError(f"Unknown adversarial mode: {mode}")
        self.mode = mode

    def forward(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = real_logits[0].new_zeros(())
        for real, fake in zip(real_logits, fake_logits):
            if self.mode == "hinge":
                loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
            else:
                loss = loss + ((real - 1.0) ** 2).mean() + (fake ** 2).mean()
        return loss / max(len(real_logits), 1)


class GeneratorAdversarialLoss(nn.Module):
    """Generator side of the adversarial objective."""

    def __init__(self, mode: str = "hinge"):
        super().__init__()
        if mode not in ("hinge", "lsgan"):
            raise ValueError(f"Unknown adversarial mode: {mode}")
        self.mode = mode

    def forward(self, fake_logits: list[torch.Tensor]) -> torch.Tensor:
        loss = fake_logits[0].new_zeros(())
        for fake in fake_logits:
            if self.mode == "hinge":
                loss = loss + (-fake).mean()
            else:
                loss = loss + ((fake - 1.0) ** 2).mean()
        return loss / max(len(fake_logits), 1)


class FeatureMatchingLoss(nn.Module):
    """
    L1 between the discriminator's intermediate activations for real and fake.

    This is what keeps adversarial training stable: it gives the generator a
    dense, well-behaved signal pointing at the real distribution instead of only
    the scalar "fooled / not fooled" gradient.
    """

    def forward(
        self,
        real_features: list[list[torch.Tensor]],
        fake_features: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        loss = None
        count = 0
        for real_maps, fake_maps in zip(real_features, fake_features):
            for real, fake in zip(real_maps, fake_maps):
                term = F.l1_loss(fake, real.detach())
                loss = term if loss is None else loss + term
                count += 1
        if loss is None:
            raise ValueError("FeatureMatchingLoss received no feature maps")
        return loss / count


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Reconstruction loss (L1 + multi-resolution STFT + multi-scale log-mel)
    - KL divergence loss
    - Optional adversarial + feature-matching loss (see ``adv_weight``)

    Defaults changed in the perceptual-objective rework: the multi-scale mel
    term carries most of the reconstruction weight and the legacy linear-magnitude
    MRSTFT term is kept at a reduced weight as a broadband anchor. Passing
    ``legacy=True`` restores the previous objective exactly, which is what the
    A/B in ``docs/VAE_OBJECTIVE.md`` compares against.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 0.25,
        l1_weight: float = 0.1,
        mel_weight: float = 15.0,
        adv_weight: float = 1.0,
        fm_weight: float = 2.0,
        sample_rate: int = 44100,
        mid_side: bool = True,
        adv_mode: str = "hinge",
        legacy: bool = False,
    ):
        super().__init__()
        self.legacy = legacy
        if legacy:
            # Exactly the pre-rework objective, kept so the A/B is reproducible.
            spectral_weight, l1_weight = 1.0, 0.1
            mel_weight, adv_weight, fm_weight = 0.0, 0.0, 0.0
            mid_side = False

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight

        self.spectral_loss = MultiResolutionSTFTLoss()
        self.mel_loss = (
            MultiScaleMelSpectrogramLoss(sample_rate=sample_rate, mid_side=mid_side)
            if mel_weight > 0
            else None
        )
        self.adv_loss = GeneratorAdversarialLoss(mode=adv_mode)
        self.fm_loss = FeatureMatchingLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        fake_logits: Optional[list[torch.Tensor]] = None,
        real_features: Optional[list[list[torch.Tensor]]] = None,
        fake_features: Optional[list[list[torch.Tensor]]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstruction: Reconstructed audio.
            target: Original audio.
            mean: Latent mean from encoder.
            log_var: Latent log-variance from encoder.
            fake_logits: Discriminator logits for the reconstruction. When None,
                the adversarial and feature-matching terms are skipped, so the
                same object works before and after the adversarial warmup.
            real_features: Discriminator feature maps for the target.
            fake_features: Discriminator feature maps for the reconstruction.

        Returns:
            Dictionary with total loss and individual components.
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(reconstruction, target)

        # Multi-resolution STFT loss (linear magnitude, broadband anchor)
        spectral_loss = self.spectral_loss(reconstruction, target)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mean.pow(2) - log_var.exp()
        )

        total_loss = (
            self.l1_weight * l1_loss
            + self.spectral_weight * spectral_loss
            + self.kl_weight * kl_loss
        )

        components = {
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "kl_loss": kl_loss,
        }

        # Multi-scale log-mel (mid/side) reconstruction
        if self.mel_loss is not None and self.mel_weight > 0:
            mel_loss = self.mel_loss(reconstruction, target)
            total_loss = total_loss + self.mel_weight * mel_loss
            components["mel_loss"] = mel_loss

        # Adversarial terms - only once the discriminator is warm
        if self.adv_weight > 0 and fake_logits is not None:
            adv_loss = self.adv_loss(fake_logits)
            total_loss = total_loss + self.adv_weight * adv_loss
            components["adv_loss"] = adv_loss

        if self.fm_weight > 0 and real_features is not None and fake_features is not None:
            fm_loss = self.fm_loss(real_features, fake_features)
            total_loss = total_loss + self.fm_weight * fm_loss
            components["fm_loss"] = fm_loss

        return {"loss": total_loss, **components}


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
