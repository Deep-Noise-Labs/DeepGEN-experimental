"""
Loss functions for SynthGen training.

Includes losses for:
- VAE training (reconstruction + KL + perceptual mel + adversarial + feature matching)
- DiT training (flow matching velocity MSE)

On the Stage-1 objective
------------------------
The Audio VAE is the quality ceiling of the whole system - the DiT only ever
produces latents, so anything the decoder cannot render is unrecoverable. Two
properties of a purely reconstructive ``L1 + multi-resolution STFT`` objective
make it a poor ceiling for production-grade instrument and synthesiser samples:

1. **It is a conditional-mean estimator.** Many waveforms share the same
   magnitude spectrum, and an L-p distance between them is minimised by their
   *average*, not by any one of them. Averaging over phase is what smears
   transients, hollows out the stereo image, and leaves the characteristic
   "under water" quality of a purely regression-trained audio autoencoder. No
   amount of extra spectral resolution fixes this: it is a property of the
   estimator, not of the analysis. This is why every production neural codec -
   SoundStream, EnCodec, DAC, and the Stable Audio autoencoder - is trained
   adversarially. Before this module gained ``discriminator_hinge_loss`` and
   ``FeatureMatchingLoss``, SynthGen had no such term at all.

2. **Its frequency axis is linear, so its attention is badly allocated.** With
   ``n_fft=2048`` at 44.1 kHz, 514 of the 1025 bins sit above 11 kHz and only 70
   sit below 1.5 kHz - half of the loss's measurement units are spent on the top
   octave, where instrument content is typically at or near the noise floor,
   and 7% on the region where pitch and timbre are actually resolved. A mel
   axis reverses that allocation (34% of bands below 1.5 kHz, 17% above 11 kHz)
   because it spaces bands the way the cochlea does.

``MelSpectrogramLoss`` addresses (2) and, just as importantly, is the stable
perceptual anchor that keeps the adversarial terms from wandering:
``mel + adversarial + feature-matching`` is the DAC recipe. The linear-frequency
``MultiResolutionSTFTLoss`` is retained alongside it - it is the term that keeps
absolute spectral magnitudes honest.

References:
    HiFi-GAN, Kong et al. 2020 - https://arxiv.org/abs/2010.05646
    Descript Audio Codec, Kumar et al. 2023 - https://arxiv.org/abs/2306.06546
    Stable Audio Open, Evans et al. 2024 - https://arxiv.org/abs/2407.14358
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

        # Cache the analysis windows so they are not rebuilt (and re-uploaded to
        # the device) on every forward pass.
        for win_size in set(win_sizes):
            self.register_buffer(
                f"window_{win_size}", torch.hann_window(win_size), persistent=False
            )

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

        window = getattr(self, f"window_{win_size}").to(device=x.device)
        # torch.stft has no complex autograd path under autocast; force fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            stft = torch.stft(
                x.float(), fft_size, hop_size, win_size, window,
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


def _hz_to_mel(freq: torch.Tensor | float) -> torch.Tensor:
    """Slaney-style Hz to mel conversion (linear below 1 kHz, log above)."""
    freq = torch.as_tensor(freq, dtype=torch.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_region = freq >= min_log_hz
    mels = torch.where(
        log_region,
        min_log_mel + torch.log(freq.clamp(min=1e-8) / min_log_hz) / logstep,
        mels,
    )
    return mels


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_hz_to_mel`."""
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_region = mels >= min_log_mel
    freqs = torch.where(
        log_region,
        min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
        freqs,
    )
    return freqs


def mel_filterbank(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """
    Build a Slaney-normalised triangular mel filterbank.

    Implemented here rather than pulled from torchaudio/librosa so that the loss
    has no extra runtime dependency and is bit-identical across environments.

    Args:
        n_freqs: Number of FFT bins (``n_fft // 2 + 1``).
        n_mels: Number of mel bands.
        sample_rate: Audio sample rate in Hz.
        f_min: Lowest band edge in Hz.
        f_max: Highest band edge in Hz. Defaults to Nyquist.

    Returns:
        Filterbank of shape (n_mels, n_freqs).
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    all_freqs = torch.linspace(0, sample_rate / 2, n_freqs, dtype=torch.float64)

    m_min = _hz_to_mel(f_min)
    m_max = _hz_to_mel(f_max)
    m_pts = torch.linspace(float(m_min), float(m_max), n_mels + 2, dtype=torch.float64)
    f_pts = _mel_to_hz(m_pts)

    # Triangular filters: rising slope to the centre, falling slope after it.
    f_diff = f_pts[1:] - f_pts[:-1]                       # (n_mels + 1,)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)

    # Slaney normalisation: equal area per filter, so the bank does not
    # implicitly re-weight the spectrum by bandwidth.
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    fb *= enorm.unsqueeze(0)

    return fb.T.to(torch.float32)


class MelSpectrogramLoss(nn.Module):
    """
    Multi-scale log-mel spectrogram distance - the perceptual reconstruction term.

    Two things distinguish it from :class:`MultiResolutionSTFTLoss`:

    - **Log-frequency band pooling.** Mel bands are spaced the way the cochlea
      resolves frequency, so the loss allocates its measurement units in
      proportion to how finely we actually hear, instead of uniformly across a
      linear frequency axis (where, at ``n_fft=2048``, half the bins fall in the
      top octave).
    - **A bounded log floor.** Magnitudes are clamped at ``clamp_eps`` before the
      logarithm. Without that bound, bins at digital silence contribute an
      essentially unbounded error - the log of a near-zero target is dominated by
      numerical noise, and gradient budget goes to content below the threshold of
      hearing.

    Both the linear and the log magnitude distance are summed, following DAC: the
    linear term keeps loud partials accurate in absolute terms, the log term
    keeps quiet-but-audible detail from being written off as rounding error.

    Multiple window lengths are used for the same reason as in
    :class:`MultiResolutionSTFTLoss`: short windows carry transient information,
    long windows carry the exact harmonic structure. Band counts are scaled with
    the window so that every scale stays well conditioned.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        window_lengths: tuple[int, ...] = (2048, 1024, 512, 256, 128),
        n_mels: tuple[int, ...] = (160, 128, 80, 40, 20),
        f_min: float = 0.0,
        f_max: float | None = None,
        log_weight: float = 1.0,
        mag_weight: float = 1.0,
        clamp_eps: float = 1e-5,
    ):
        super().__init__()
        if len(window_lengths) != len(n_mels):
            raise ValueError("window_lengths and n_mels must have the same length")

        self.sample_rate = sample_rate
        self.window_lengths = window_lengths
        self.n_mels = n_mels
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps

        for win_length, n_mel in zip(window_lengths, n_mels):
            n_freqs = win_length // 2 + 1
            if n_mel > n_freqs:
                raise ValueError(
                    f"n_mels={n_mel} exceeds the {n_freqs} FFT bins available "
                    f"at window_length={win_length}"
                )
            self.register_buffer(
                f"fb_{win_length}",
                mel_filterbank(n_freqs, n_mel, sample_rate, f_min, f_max),
                persistent=False,
            )
            self.register_buffer(
                f"win_{win_length}", torch.hann_window(win_length), persistent=False
            )

    def _mel(self, x: torch.Tensor, win_length: int) -> torch.Tensor:
        """Mel magnitude spectrogram of a (batch * channels, samples) signal."""
        window = getattr(self, f"win_{win_length}").to(device=x.device)
        fb = getattr(self, f"fb_{win_length}").to(device=x.device)

        with torch.autocast(device_type=x.device.type, enabled=False):
            stft = torch.stft(
                x.float(),
                n_fft=win_length,
                hop_length=win_length // 4,
                win_length=win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
        magnitude = stft.abs()                    # (batch, freq, frames)
        return torch.matmul(fb, magnitude)        # (batch, n_mels, frames)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted audio (batch, channels, samples).
            target: Target audio (batch, channels, samples).

        Returns:
            Scalar loss value.
        """
        if pred.dim() == 3:
            batch, channels, samples = pred.shape
            pred = pred.reshape(batch * channels, samples)
            target = target.reshape(batch * channels, -1)

        total = pred.new_zeros((), dtype=torch.float32)
        for win_length in self.window_lengths:
            # Windows longer than the signal produce a degenerate STFT; skip them
            # so short clips and unit tests stay valid.
            if win_length > pred.shape[-1]:
                continue

            mel_pred = self._mel(pred, win_length)
            mel_target = self._mel(target, win_length)

            total = total + self.mag_weight * F.l1_loss(mel_pred, mel_target)
            total = total + self.log_weight * F.l1_loss(
                mel_pred.clamp(min=self.clamp_eps).log10(),
                mel_target.clamp(min=self.clamp_eps).log10(),
            )

        return total / max(1, len(self.window_lengths))


# =============================================================================
# Adversarial losses
# =============================================================================


def discriminator_hinge_loss(
    real_logits: list[torch.Tensor],
    fake_logits: list[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for the discriminator, averaged over sub-discriminators.

    ``mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))``. The hinge saturates
    once a sample is classified with margin, which stops an over-confident
    critic from producing the runaway gradients that destabilise audio GANs.
    """
    loss = real_logits[0].new_zeros(())
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


def generator_hinge_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Non-saturating hinge generator loss: ``-mean(D(fake))``."""
    loss = fake_logits[0].new_zeros(())
    for fake in fake_logits:
        loss = loss + (-fake).mean()
    return loss / len(fake_logits)


class FeatureMatchingLoss(nn.Module):
    """
    L1 distance between the discriminator's internal activations for real and
    reconstructed audio.

    This is what keeps adversarial autoencoder training stable. The raw
    adversarial term only says "this does not look real" without saying how to
    fix it; feature matching turns the critic into a learned perceptual metric
    and supplies a dense, well-behaved gradient. In practice it does most of the
    work - the adversarial term mainly stops the decoder from hedging.
    """

    def forward(
        self,
        real_features: list[list[torch.Tensor]],
        fake_features: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            real_features: Per-sub-discriminator feature maps for real audio.
            fake_features: Per-sub-discriminator feature maps for reconstructions.

        Returns:
            Scalar loss value.
        """
        loss = fake_features[0][0].new_zeros(())
        count = 0
        for real_maps, fake_maps in zip(real_features, fake_features):
            for real, fake in zip(real_maps, fake_maps):
                loss = loss + F.l1_loss(fake, real.detach())
                count += 1
        return loss / max(1, count)


class VAELoss(nn.Module):
    """
    Combined VAE loss for audio autoencoder training.

    Components:
    - Waveform L1 reconstruction loss
    - Multi-resolution STFT loss (linear magnitude)
    - Multi-scale log-mel spectrogram loss (perceptual)
    - KL divergence loss
    - Adversarial + feature-matching losses, when discriminator outputs are supplied

    The adversarial terms are opt-in per call: pass ``fake_logits`` /
    ``real_features`` / ``fake_features`` once the discriminator warm-up is over
    and they are added to the total; omit them and the loss degrades gracefully
    to the purely reconstructive objective.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        spectral_weight: float = 1.0,
        l1_weight: float = 0.1,
        mel_weight: float = 15.0,
        adv_weight: float = 1.0,
        fm_weight: float = 2.0,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight

        self.spectral_loss = MultiResolutionSTFTLoss()
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.feature_matching_loss = FeatureMatchingLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        fake_logits: list[torch.Tensor] | None = None,
        real_features: list[list[torch.Tensor]] | None = None,
        fake_features: list[list[torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstruction: Reconstructed audio.
            target: Original audio.
            mean: Latent mean from encoder.
            log_var: Latent log-variance from encoder.
            fake_logits: Discriminator logits for the reconstruction. When given,
                the adversarial term is included.
            real_features: Discriminator feature maps for the target audio.
            fake_features: Discriminator feature maps for the reconstruction.

        Returns:
            Dictionary with total loss and individual components.
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(reconstruction, target)

        # Multi-resolution STFT loss
        spectral_loss = self.spectral_loss(reconstruction, target)

        # Perceptual multi-scale log-mel loss
        mel_loss = self.mel_loss(reconstruction, target)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mean.pow(2) - log_var.exp()
        )

        # Total loss
        total_loss = (
            self.l1_weight * l1_loss
            + self.spectral_weight * spectral_loss
            + self.mel_weight * mel_loss
            + self.kl_weight * kl_loss
        )

        losses = {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "spectral_loss": spectral_loss,
            "mel_loss": mel_loss,
            "kl_loss": kl_loss,
        }

        if fake_logits is not None:
            adv_loss = generator_hinge_loss(fake_logits)
            losses["adv_loss"] = adv_loss
            total_loss = total_loss + self.adv_weight * adv_loss

        if real_features is not None and fake_features is not None:
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            losses["fm_loss"] = fm_loss
            total_loss = total_loss + self.fm_weight * fm_loss

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
