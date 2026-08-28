"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + spectral + mel + stereo + adversarial)
- DiT training (flow matching velocity MSE)

Design note — why the VAE objective looks like this
---------------------------------------------------
SynthGen is a two-stage system: the DiT generates latents, the VAE decoder turns
them into audio. The decoder is therefore a hard ceiling on final quality — no
amount of DiT training can produce audio the decoder cannot render. The VAE
objective is the single highest-leverage thing in the repository for perceived
sound quality, so it is built to match what production audio autoencoders
(EnCodec, DAC, Stable Audio) actually use rather than the minimal
L1 + magnitude-STFT recipe:

1. **Analysis windows sized for 44.1 kHz.** A 2048-point FFT gives 21.5 Hz bins,
   so a 41 Hz bass fundamental lands in bin 2 and sub-bass is effectively
   unconstrained. The ladder starts at 8192 (5.4 Hz bins).
2. **Perceptual frequency weighting.** Half the bins of a linear STFT sit above
   11 kHz. A linear-frequency loss spends most of its capacity on the top octave
   and almost none on 100 Hz – 1 kHz where musical fundamentals live. A mel-scaled
   term rebalances this.
3. **Per-item normalisation.** Spectral convergence normalised over a whole batch
   lets one loud item dominate; quiet items (soft pads, release tails) then
   contribute almost nothing.
4. **A magnitude floor.** ``log(mag + 1e-8)`` maps digital silence to -18.4, so
   numerical noise in inaudible bins produces enormous log errors and the loss
   ends up optimising the noise floor. The same floor bounds the spectral
   convergence denominator, which otherwise divides by ~0 on a silent item.
5. **Stereo image.** Folding channels into the batch scores L and R independently
   and never constrains mid/side, so width collapses.
6. **An adversarial term.** Magnitude losses are phase blind. Phase blindness is
   what makes neural-codec audio sound smeared and "watery"; a complex-STFT
   critic is the standard fix.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthgen.model.discriminator import (
    feature_matching_loss,
    generator_hinge_loss,
)

# =============================================================================
# Spectral helpers
# =============================================================================


def _hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor | float:
    """HTK mel scale."""
    if isinstance(hz, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """
    Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Implemented in pure PyTorch so the loss stays differentiable and free of
    device/dtype juggling, and so the training objective does not depend on
    librosa or torchaudio transform internals.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_points = torch.linspace(
        float(_hz_to_mel(fmin)), float(_hz_to_mel(fmax)), n_mels + 2
    )
    hz_points = _mel_to_hz(mel_points)

    filterbank = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, centre, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (fft_freqs - left) / torch.clamp(centre - left, min=1e-8)
        falling = (right - fft_freqs) / torch.clamp(right - centre, min=1e-8)
        filterbank[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)

    # Deliberately *not* Slaney area-normalised. This filterbank feeds a
    # log-domain L1, where a per-band constant is an additive offset that
    # cancels in the difference — its only real effect would be to push wide
    # high-frequency bands down towards the log floor and make the loss blind
    # up there.
    return filterbank


class _WindowCache:
    """Lazily built, per-(length, device, dtype) Hann windows."""

    def __init__(self) -> None:
        self._cache: dict[tuple, torch.Tensor] = {}

    def get(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (length, str(device), dtype)
        window = self._cache.get(key)
        if window is None:
            window = torch.hann_window(length, device=device, dtype=dtype)
            self._cache[key] = window
        return window


def _stft_magnitude(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_size: int,
    window_cache: _WindowCache,
) -> torch.Tensor:
    """
    STFT magnitude of ``(batch, channels, samples)`` audio.

    Returns ``(batch, channels, freq, frames)`` — channels are kept as their own
    axis (rather than folded into the batch) so downstream losses can normalise
    per item and reason about the stereo pair.

    Magnitudes are normalised by the window sum, so a full-scale sinusoid gives
    the same bin magnitude at every FFT size. Without that, an 8192-point
    analysis produces magnitudes ~32x larger than a 256-point one and a single
    magnitude floor cannot mean "inaudible" at both ends of the ladder.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    batch, channels, samples = x.shape

    flat = x.reshape(batch * channels, samples)
    # STFT is not implemented for reduced precision on all backends, and the
    # loss should not be computed in bf16 regardless.
    flat = flat.float()

    window = window_cache.get(win_size, flat.device, flat.dtype)
    spec = torch.stft(
        flat,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        return_complex=True,
    )
    magnitude = spec.abs() / window.sum().clamp_min(1e-8)
    return magnitude.reshape(batch, channels, *magnitude.shape[-2:])


# =============================================================================
# VAE Losses
# =============================================================================


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for audio reconstruction quality.

    Computes spectral convergence and log-magnitude loss at multiple FFT sizes
    to capture both fine and coarse frequency details.

    The default resolution ladder runs from 8192 down to 256 samples. At
    44.1 kHz the 8192-point window resolves 5.4 Hz, which is what makes
    sub-bass and low-mid content actually visible to the objective; the
    256-point window resolves 5.8 ms, which is what keeps transients sharp.

    Args:
        fft_sizes: FFT sizes for each resolution.
        hop_sizes: Hop per resolution. Defaults to ``fft_size // 4``.
        win_sizes: Window length per resolution. Defaults to ``fft_size``.
        mag_floor: Magnitudes are clamped to this before the log. Prevents the
            loss from being dominated by inaudible near-silent bins.
        per_item: Normalise spectral convergence per batch item rather than over
            the whole batch, so quiet sounds are weighted like loud ones.
    """

    def __init__(
        self,
        fft_sizes: tuple = (8192, 4096, 2048, 1024, 512, 256),
        hop_sizes: tuple | None = None,
        win_sizes: tuple | None = None,
        mag_floor: float = 1e-5,
        per_item: bool = True,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes) if hop_sizes is not None else tuple(
            n // 4 for n in self.fft_sizes
        )
        self.win_sizes = tuple(win_sizes) if win_sizes is not None else self.fft_sizes
        self.mag_floor = mag_floor
        self.per_item = per_item
        self._windows = _WindowCache()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        total_loss = pred.new_zeros((), dtype=torch.float32)
        used = 0

        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            # Skip resolutions the input is too short to support rather than
            # letting torch.stft raise on short clips.
            if pred.shape[-1] < win_size:
                continue

            pred_mag = _stft_magnitude(pred, fft_size, hop_size, win_size, self._windows)
            target_mag = _stft_magnitude(
                target, fft_size, hop_size, win_size, self._windows
            )

            # Spectral convergence, normalised per batch item.
            #
            # The denominator is floored at the norm a signal sitting entirely
            # on the magnitude floor would have. Without that, a silent item —
            # a release tail, a gap between hits, a quiet pad — divides by ~0
            # and produces a loss spike orders of magnitude above the rest of
            # the batch.
            batch = pred_mag.shape[0]
            if self.per_item:
                diff = (target_mag - pred_mag).reshape(batch, -1)
                ref = target_mag.reshape(batch, -1)
                floor = self.mag_floor * math.sqrt(ref.shape[1])
                sc_loss = (
                    torch.norm(diff, p=2, dim=1)
                    / torch.norm(ref, p=2, dim=1).clamp_min(floor)
                ).mean()
            else:
                floor = self.mag_floor * math.sqrt(target_mag.numel())
                sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(
                    target_mag, p="fro"
                ).clamp_min(floor)

            # Log-magnitude loss with a floor, so inaudible bins cannot dominate.
            log_loss = F.l1_loss(
                torch.log(pred_mag.clamp_min(self.mag_floor)),
                torch.log(target_mag.clamp_min(self.mag_floor)),
            )

            total_loss = total_loss + sc_loss + log_loss
            used += 1

        if used == 0:
            return total_loss
        return total_loss / used


class MelSpectrogramLoss(nn.Module):
    """
    Multi-scale log-mel L1 loss.

    A linear-frequency STFT loss allocates half its bins to the top octave. Mel
    spacing puts the loss capacity where hearing (and musical content) actually
    is: fundamentals, formants and the low-mid body of a patch.

    Multiple window sizes are used with a mel band count that scales with the
    window, following the DAC recipe: long windows with many bands resolve
    pitch, short windows with few bands resolve envelope.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_sizes: tuple = (2048, 512),
        n_mels: tuple = (128, 64),
        fmin: float = 0.0,
        fmax: float | None = None,
        log_floor: float = 1e-5,
    ):
        super().__init__()
        if len(fft_sizes) != len(n_mels):
            raise ValueError("fft_sizes and n_mels must have the same length")

        self.sample_rate = sample_rate
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(n // 4 for n in self.fft_sizes)
        self.log_floor = log_floor
        self._windows = _WindowCache()

        for index, (n_fft, mels) in enumerate(zip(self.fft_sizes, n_mels)):
            self.register_buffer(
                f"mel_fb_{index}",
                build_mel_filterbank(n_fft, mels, sample_rate, fmin=fmin, fmax=fmax),
                persistent=False,
            )
        self.num_scales = len(self.fft_sizes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_zeros((), dtype=torch.float32)
        used = 0

        for index, (n_fft, hop) in enumerate(zip(self.fft_sizes, self.hop_sizes)):
            if pred.shape[-1] < n_fft:
                continue

            filterbank: torch.Tensor = getattr(self, f"mel_fb_{index}")
            pred_mag = _stft_magnitude(pred, n_fft, hop, n_fft, self._windows)
            target_mag = _stft_magnitude(target, n_fft, hop, n_fft, self._windows)

            filterbank = filterbank.to(pred_mag.device, pred_mag.dtype)
            # (n_mels, freq) @ (..., freq, frames) -> (..., n_mels, frames)
            pred_mel = torch.einsum("mf,bcft->bcmt", filterbank, pred_mag)
            target_mel = torch.einsum("mf,bcft->bcmt", filterbank, target_mag)

            total = total + F.l1_loss(
                torch.log(pred_mel.clamp_min(self.log_floor)),
                torch.log(target_mel.clamp_min(self.log_floor)),
            )
            used += 1

        if used == 0:
            return total
        return total / used


class StereoCoherenceLoss(nn.Module):
    """
    Mid/side spectral loss.

    Scoring left and right independently leaves the *relationship* between them
    unconstrained, and the cheapest way for a decoder to reduce a per-channel
    loss is to make both channels more alike — which collapses the stereo image.
    Encoding the pair as mid = (L+R)/2 and side = (L-R)/2 and scoring both makes
    width part of the objective. Mono input is a no-op.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 512),
        log_floor: float = 1e-5,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(n // 4 for n in self.fft_sizes)
        self.log_floor = log_floor
        self._windows = _WindowCache()

    @staticmethod
    def _mid_side(x: torch.Tensor) -> torch.Tensor | None:
        if x.dim() != 3 or x.shape[1] != 2:
            return None
        left, right = x[:, 0], x[:, 1]
        return torch.stack([(left + right) * 0.5, (left - right) * 0.5], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_ms = self._mid_side(pred)
        target_ms = self._mid_side(target)
        if pred_ms is None or target_ms is None:
            return pred.new_zeros((), dtype=torch.float32)

        total = pred.new_zeros((), dtype=torch.float32)
        used = 0
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            if pred.shape[-1] < n_fft:
                continue
            pred_mag = _stft_magnitude(pred_ms, n_fft, hop, n_fft, self._windows)
            target_mag = _stft_magnitude(target_ms, n_fft, hop, n_fft, self._windows)
            total = total + F.l1_loss(
                torch.log(pred_mag.clamp_min(self.log_floor)),
                torch.log(target_mag.clamp_min(self.log_floor)),
            )
            used += 1

        if used == 0:
            return total
        return total / used


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Reconstruction loss (L1 + multi-resolution STFT + multi-scale log-mel)
    - Stereo mid/side coherence loss
    - KL divergence loss
    - Adversarial + feature-matching loss against a complex-STFT critic

    The adversarial terms are active only when ``discriminator`` is passed to
    :meth:`forward`; without it this behaves as a (much better weighted) purely
    reconstructive objective, so the loss can still be used for smoke runs and
    CPU tests with no critic attached.

    Args:
        recon_weight: Retained for configuration compatibility; scales the
            combined reconstruction group.
        kl_weight: Weight on the KL term.
        spectral_weight: Weight on the multi-resolution STFT term.
        l1_weight: Weight on the waveform L1 term.
        mel_weight: Weight on the multi-scale log-mel term.
        stereo_weight: Weight on the mid/side coherence term.
        adversarial_weight: Weight on the generator hinge term.
        feature_matching_weight: Weight on discriminator feature matching.
        sample_rate: Sample rate, used to build the mel filterbanks.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
        mel_weight: float = 1.0,
        stereo_weight: float = 0.25,
        adversarial_weight: float = 0.1,
        feature_matching_weight: float = 2.0,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight
        self.stereo_weight = stereo_weight
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight

        self.spectral_loss = MultiResolutionSTFTLoss()
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.stereo_loss = StereoCoherenceLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        discriminator: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstruction: Reconstructed audio.
            target: Original audio.
            mean: Latent mean from encoder.
            log_var: Latent log-variance from encoder.
            discriminator: Optional critic. When supplied, adversarial and
                feature-matching terms are added to the total.

        Returns:
            Dictionary with total loss and individual components.
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(reconstruction, target)

        # Multi-resolution STFT loss
        spectral_loss = self.spectral_loss(reconstruction, target)

        # Perceptually weighted multi-scale mel loss
        mel_loss = self.mel_loss(reconstruction, target)

        # Stereo image (mid/side) coherence
        stereo_loss = self.stereo_loss(reconstruction, target)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mean.pow(2) - log_var.exp()
        )

        reconstruction_group = (
            self.l1_weight * l1_loss
            + self.spectral_weight * spectral_loss
            + self.mel_weight * mel_loss
            + self.stereo_weight * stereo_loss
        )

        total_loss = self.recon_weight * reconstruction_group + self.kl_weight * kl_loss

        losses = {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "mel_loss": mel_loss,
            "stereo_loss": stereo_loss,
            "kl_loss": kl_loss,
        }

        if discriminator is not None:
            fake_logits, fake_features = discriminator(reconstruction)
            with torch.no_grad():
                _, real_features = discriminator(target)

            adv_loss = generator_hinge_loss(fake_logits)
            fm_loss = feature_matching_loss(real_features, fake_features)

            losses["adversarial_loss"] = adv_loss
            losses["feature_matching_loss"] = fm_loss
            losses["loss"] = (
                total_loss
                + self.adversarial_weight * adv_loss
                + self.feature_matching_weight * fm_loss
            )

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
        t: torch.Tensor | None = None,
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
