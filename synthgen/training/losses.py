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

    .. deprecated::
        Retained for regression comparison only. At 44.1 kHz its longest
        window (2048) resolves only 21.5 Hz, so it cannot separate adjacent
        bass partials, and its unweighted linear-frequency bins leave the
        air band and the stereo image effectively unconstrained. Use
        :class:`PerceptualSampleLoss` for training. See docs/EVALUATION.md.
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


class PerceptualSampleLoss(nn.Module):
    """
    Spectral reconstruction loss tuned for 44.1 kHz instrument samples.

    ``MultiResolutionSTFTLoss`` is the objective most latent audio
    autoencoders inherit, and it is the ceiling on everything downstream: the
    decoder can only learn to preserve what the loss can see. Measured against
    real audio (see docs/EVALUATION.md), the inherited version is close to
    blind in exactly the places that separate a usable sample from a
    commercial one. This loss closes three of them.

    **1. Log floor.** The old ``log(mag + 1e-8)`` gives a bin 160 dB of range
    to be wrong in, so near-silent bins produce enormous errors and the loss
    spends its gradient budget on content nobody can hear. The floor here is
    *relative*: each scale is clamped at ``dynamic_range_db`` below that
    scale's own peak target magnitude. An absolute floor does not work,
    because STFT magnitudes scale with both window length and clip level, so
    the same constant means a different number of dB on every clip -- this was
    measured, not assumed (see docs/EVALUATION.md).

    **2. Band weighting.** Linear-frequency bins are counted uniformly, which
    hands most of the loss to the crowded top octaves while the sub and air
    bands -- the two a producer notices first -- contribute almost nothing.
    Per-band weights make that explicit and tunable.

    **3. Stereo.** The old loss flattens ``(B, C, T)`` to ``(B*C, T)`` and
    compares channels independently, so any decoder output with the right
    per-channel magnitudes scores identically no matter what it does to the
    stereo image, including collapsing it to mono. Adding a mid/side term
    puts width and mono-compatibility into the objective.

    A transient term on spectral flux is also included: it penalises smeared
    attacks directly rather than hoping the magnitude terms catch them.

    **What is deliberately absent: longer windows.** The obvious fourth change
    is to extend the window set upward, on the reasoning that a 2048-sample
    window at 44.1 kHz resolves only 21.5 Hz per bin and cannot separate
    adjacent bass partials. That was tried and measured, and it does not hold
    up. Adding 4096 and 8192 windows changed every sensitivity share on real
    audio by less than 0.01, and a detune sweep showed that *no* window size
    gives this family of losses a usable pitch gradient: both a 2048-max and an
    8192-max loss saturate at a 5-cent (0.12 Hz) detune, registering "different"
    without encoding "how different". Bass tuning needs an explicitly pitch-
    aware term, not a bigger FFT. The long windows were therefore dropped
    rather than shipped, since an 8192-point STFT is the single most expensive
    term in the loss. Numbers in docs/EVALUATION.md.

    Args:
        fft_sizes: STFT window sizes, longest first.
        hop_ratio: Hop length as a fraction of each window.
        sample_rate: Used to map band edges onto FFT bins.
        band_weights: Per-band multipliers on the magnitude term. Keys must be
            a subset of :data:`synthgen.eval.metrics.BANDS`.
        mid_side_weight: Weight of the mid/side spectral term.
        transient_weight: Weight of the spectral-flux term.
        dynamic_range_db: Log floor, in dB below each scale's peak target
            magnitude. 60 dB is far below audibility in the presence of the
            peak; lowering it further starts to reward chasing the noise floor.
        log_eps: Absolute magnitude backstop, for an all-silent target.
    """

    #: Band edges in Hz, matching ``synthgen.eval.metrics.BANDS``.
    BAND_EDGES: dict[str, tuple[float, float]] = {
        "sub": (20.0, 60.0),
        "bass": (60.0, 250.0),
        "low_mid": (250.0, 1000.0),
        "mid": (1000.0, 4000.0),
        "presence": (4000.0, 10000.0),
        "air": (10000.0, 22050.0),
    }

    #: Defaults lift the two bands the inherited loss under-weights most.
    DEFAULT_BAND_WEIGHTS: dict[str, float] = {
        "sub": 2.0,
        "bass": 1.5,
        "low_mid": 1.0,
        "mid": 1.0,
        "presence": 1.25,
        "air": 2.0,
    }

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256, 128),
        hop_ratio: float = 0.25,
        sample_rate: int = 44100,
        band_weights: Optional[dict[str, float]] = None,
        mid_side_weight: float = 0.5,
        transient_weight: float = 0.5,
        dynamic_range_db: float = 60.0,
        log_eps: float = 1e-8,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(max(1, int(n * hop_ratio)) for n in self.fft_sizes)
        self.sample_rate = sample_rate
        self.band_weights = dict(band_weights or self.DEFAULT_BAND_WEIGHTS)
        self.mid_side_weight = mid_side_weight
        self.transient_weight = transient_weight
        self.dynamic_range_db = dynamic_range_db
        self.log_eps = log_eps

        for n_fft in self.fft_sizes:
            self.register_buffer(
                f"window_{n_fft}", torch.hann_window(n_fft), persistent=False
            )
            self.register_buffer(
                f"bandw_{n_fft}", self._band_weight_vector(n_fft), persistent=False
            )

    def _band_weight_vector(self, n_fft: int) -> torch.Tensor:
        """Per-FFT-bin weight vector built from the band table."""
        freqs = torch.linspace(0.0, self.sample_rate / 2.0, n_fft // 2 + 1)
        weights = torch.ones_like(freqs)
        for name, (low, high) in self.BAND_EDGES.items():
            weight = self.band_weights.get(name)
            if weight is None:
                continue
            weights[(freqs >= low) & (freqs < high)] = weight
        # Normalise so total loss magnitude stays comparable across configs.
        return weights / weights.mean()

    def _magnitude(self, x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        """STFT magnitude of ``(B, T)``, returned as ``(B, freq, frames)``."""
        window = getattr(self, f"window_{n_fft}").to(x.device, x.dtype)
        # A window longer than the signal makes torch.stft raise; pad first.
        if x.shape[-1] < n_fft:
            x = F.pad(x, (0, n_fft - x.shape[-1]))
        stft = torch.stft(
            x, n_fft, hop, n_fft, window, return_complex=True, center=True
        )
        return stft.abs()

    def _scale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        n_fft: int,
        hop: int,
    ) -> torch.Tensor:
        """Convergence + weighted log-magnitude + flux, for one resolution."""
        pred_mag = self._magnitude(pred, n_fft, hop)
        target_mag = self._magnitude(target, n_fft, hop)
        weights = getattr(self, f"bandw_{n_fft}").to(pred_mag.device, pred_mag.dtype)
        weights = weights.view(1, -1, 1)

        # Spectral convergence, computed per item so one loud clip in the batch
        # cannot dominate the whole term.
        num = torch.linalg.vector_norm(target_mag - pred_mag, dim=(1, 2))
        den = torch.linalg.vector_norm(target_mag, dim=(1, 2)) + 1e-8
        sc_loss = (num / den).mean()

        # Log-magnitude with a per-item relative floor and per-band weighting.
        # The floor tracks each item's own spectral peak, so the same number of
        # dB is protected regardless of window length or how loud the clip is.
        peak = target_mag.amax(dim=(1, 2), keepdim=True)
        floor = (peak * (10.0 ** (-self.dynamic_range_db / 20.0))).clamp(
            min=self.log_eps
        )
        pred_log = torch.log(torch.maximum(pred_mag, floor))
        target_log = torch.log(torch.maximum(target_mag, floor))
        log_loss = (weights * (pred_log - target_log).abs()).mean()

        # Spectral flux: frame-to-frame change in log magnitude. Matching this
        # is what forces attacks to stay sharp instead of being averaged out.
        if pred_log.shape[-1] > 1:
            pred_flux = pred_log[..., 1:] - pred_log[..., :-1]
            target_flux = target_log[..., 1:] - target_log[..., :-1]
            flux_loss = (pred_flux - target_flux).abs().mean()
        else:
            flux_loss = pred_log.new_zeros(())

        return sc_loss + log_loss + self.transient_weight * flux_loss

    def _flatten_channels(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` or ``(B, T)`` to ``(B*C, T)``."""
        if x.dim() == 3:
            return x.reshape(x.shape[0] * x.shape[1], x.shape[-1])
        return x

    @staticmethod
    def _mid_side(x: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Mid and side signals, or None for mono / non-stereo input."""
        if x.dim() != 3 or x.shape[1] != 2:
            return None
        return (x[:, 0] + x[:, 1]) / 2.0, (x[:, 0] - x[:, 1]) / 2.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred: Predicted audio, ``(batch, channels, samples)``.
            target: Target audio, same shape.

        Returns:
            Scalar loss.
        """
        pred_flat = self._flatten_channels(pred)
        target_flat = self._flatten_channels(target)

        total = pred.new_zeros(())
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            total = total + self._scale_loss(pred_flat, target_flat, n_fft, hop)
        total = total / len(self.fft_sizes)

        # Mid/side term: without this the stereo image is unconstrained.
        pred_ms = self._mid_side(pred)
        target_ms = self._mid_side(target)
        if pred_ms is not None and target_ms is not None and self.mid_side_weight > 0:
            ms_total = pred.new_zeros(())
            for p, t in zip(pred_ms, target_ms):
                for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
                    ms_total = ms_total + self._scale_loss(p, t, n_fft, hop)
            ms_total = ms_total / (2 * len(self.fft_sizes))
            total = total + self.mid_side_weight * ms_total

        return total


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Reconstruction loss (L1 + perceptual multi-resolution spectral loss)
    - KL divergence loss
    - Optional adversarial loss

    Args:
        recon_weight: Reserved for an external reconstruction term.
        kl_weight: Weight on the KL divergence.
        spectral_weight: Weight on the spectral reconstruction term.
        l1_weight: Weight on the waveform L1 term.
        sample_rate: Passed to the spectral loss for band-edge placement.
        spectral_loss: ``"perceptual"`` (default) or ``"legacy"``. The legacy
            option restores the pre-2026-09 ``MultiResolutionSTFTLoss`` and
            exists so regressions can be reproduced, not for new training.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
        sample_rate: int = 44100,
        spectral_loss: str = "perceptual",
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight

        if spectral_loss == "perceptual":
            self.spectral_loss = PerceptualSampleLoss(sample_rate=sample_rate)
        elif spectral_loss == "legacy":
            self.spectral_loss = MultiResolutionSTFTLoss()
        else:
            raise ValueError(
                f"Unknown spectral_loss {spectral_loss!r}; "
                "expected 'perceptual' or 'legacy'"
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
