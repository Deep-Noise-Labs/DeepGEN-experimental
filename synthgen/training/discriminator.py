"""
Discriminators for adversarial Audio VAE (Stage-1) training.

The Stage-1 autoencoder is the hard ceiling on SynthGen's output quality: the
DiT can only ever produce latents, and whatever the decoder cannot render is
gone for good. A reconstruction objective built purely from L1 + magnitude STFT
distances is a conditional-mean estimator, so it converges on the *average* of
every plausible waveform that fits the coarse spectrum. For audio that average
is audible as smeared transients, phase-incoherent stereo, and a dull top end.

Adversarial training removes that averaging: the discriminator only has to tell
"real" from "reconstructed", so the decoder is pushed to commit to one sharp,
plausible waveform rather than hedging. This module provides the two
discriminator families that the neural-codec literature converged on:

- ``MultiPeriodDiscriminator`` (HiFi-GAN) reshapes the waveform by a set of
  co-prime periods and looks at it with 2-D convolutions. Periodic structure -
  pitch, the buzz of a saw wave, the beating of detuned oscillators - becomes
  spatially local and therefore easy to police.
- ``MultiResolutionSTFTDiscriminator`` (EnCodec / DAC) looks at the complex STFT
  at several resolutions, split into frequency sub-bands so that each band gets
  its own critic. Without the sub-band split the discriminator ignores the top
  octaves, which carry very little energy but almost all of the perceived "air".

Both return per-scale logits together with their intermediate feature maps, so
the trainer can add a feature-matching loss on top of the adversarial term.

References:
    HiFi-GAN, Kong et al. 2020 - https://arxiv.org/abs/2010.05646
    EnCodec, Défossez et al. 2022 - https://arxiv.org/abs/2210.13438
    Descript Audio Codec, Kumar et al. 2023 - https://arxiv.org/abs/2306.06546
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature maps are returned as a list per sub-discriminator.
FeatureMaps = list[torch.Tensor]

LRELU_SLOPE = 0.1


def _weight_norm(module: nn.Module) -> nn.Module:
    """Apply weight norm using the non-deprecated parametrization API."""
    return nn.utils.parametrizations.weight_norm(module)


def _fold_channels(x: torch.Tensor) -> torch.Tensor:
    """
    Collapse audio channels into the batch dimension.

    Discriminators judge single-channel waveforms. Folding stereo into the batch
    keeps every channel independently critiqued, which is what prevents the
    decoder from collapsing the stereo image to a doubled mono signal.

    Args:
        x: Audio of shape (batch, channels, samples).

    Returns:
        Audio of shape (batch * channels, 1, samples).
    """
    batch, channels, samples = x.shape
    return x.reshape(batch * channels, 1, samples)


# =============================================================================
# Multi-Period Discriminator (HiFi-GAN)
# =============================================================================


class PeriodDiscriminator(nn.Module):
    """
    Single-period discriminator.

    Reshapes a 1-D waveform of length ``T`` into a 2-D map of shape
    ``(T // period, period)`` and applies 2-D convolutions with ``(k, 1)``
    kernels. Samples that are ``period`` apart become vertical neighbours, so
    the network can compare successive cycles of any component whose period
    divides ``period``.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
    ):
        super().__init__()
        self.period = period

        pad = (kernel_size - 1) // 2
        convs = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            # The last block keeps stride 1 so the receptive field stops growing
            # once it already spans several cycles.
            block_stride = stride if i < len(channels) - 1 else 1
            convs.append(
                _weight_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(block_stride, 1),
                        padding=(pad, 0),
                    )
                )
            )
            in_ch = out_ch
        self.convs = nn.ModuleList(convs)
        self.conv_post = _weight_norm(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, FeatureMaps]:
        """
        Args:
            x: Waveform of shape (batch, 1, samples).

        Returns:
            Tuple of (logits, feature_maps).
        """
        batch, _, samples = x.shape

        # Right-pad so the sample count is divisible by the period.
        remainder = samples % self.period
        if remainder != 0:
            x = F.pad(x, (0, self.period - remainder), mode="reflect")
            samples = x.shape[-1]

        x = x.view(batch, 1, samples // self.period, self.period)

        features: FeatureMaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)

        logits = self.conv_post(x)
        features.append(logits)
        return logits.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    """
    Bank of :class:`PeriodDiscriminator` at co-prime periods.

    Co-prime periods guarantee that no single periodic artefact can hide from
    every member of the bank.
    """

    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=p, channels=channels) for p in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[FeatureMaps]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tuple of (list of logits, list of feature-map lists) - one entry per
            sub-discriminator.
        """
        x = _fold_channels(x)
        all_logits: list[torch.Tensor] = []
        all_features: list[FeatureMaps] = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features


# =============================================================================
# Multi-Resolution complex-STFT Discriminator (EnCodec / DAC)
# =============================================================================


class STFTSubBandDiscriminator(nn.Module):
    """
    Complex-STFT discriminator at one resolution, split into frequency bands.

    Operating on the *complex* STFT (real and imaginary parts as two channels)
    rather than the magnitude is what makes this critic sensitive to phase, and
    therefore to transient smearing and stereo collapse.

    The frequency axis is cut into sub-bands, each with its own convolutional
    stack. This matters more than it looks: the top octave of a 44.1 kHz signal
    typically sits 40-60 dB below the fundamental, so a single full-band critic
    can drive its loss to near zero while completely ignoring the "air" band. A
    dedicated critic per band cannot make that trade.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        channels: int = 32,
        bands: tuple[tuple[float, float], ...] = (
            (0.0, 0.1),
            (0.1, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0),
        ),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.bands = bands

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        def band_stack() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    _weight_norm(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
                    _weight_norm(
                        nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))
                    ),
                    _weight_norm(
                        nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))
                    ),
                    _weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
                ]
            )

        self.band_convs = nn.ModuleList([band_stack() for _ in bands])
        self.conv_post = _weight_norm(
            nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))
        )

    def _spectrogram(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute the complex STFT and slice it into sub-bands.

        Args:
            x: Waveform of shape (batch, 1, samples).

        Returns:
            List of tensors of shape (batch, 2, frames, band_bins).
        """
        x = x.squeeze(1)

        # torch.stft has no complex autograd path under autocast; run in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            stft = torch.stft(
                x.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(x.device),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )

        # (batch, freq, frames) complex -> (batch, 2, frames, freq) real
        spec = torch.view_as_real(stft)          # (batch, freq, frames, 2)
        spec = spec.permute(0, 3, 2, 1)          # (batch, 2, frames, freq)

        n_bins = spec.shape[-1]
        slices = []
        for low, high in self.bands:
            lo = int(low * n_bins)
            hi = max(lo + 1, int(high * n_bins))
            slices.append(spec[..., lo:hi])
        return slices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, FeatureMaps]:
        """
        Args:
            x: Waveform of shape (batch, 1, samples).

        Returns:
            Tuple of (logits, feature_maps).
        """
        band_specs = self._spectrogram(x)

        features: FeatureMaps = []
        outputs = []
        for spec, convs in zip(band_specs, self.band_convs):
            h = spec
            for conv in convs:
                h = F.leaky_relu(conv(h), LRELU_SLOPE)
                features.append(h)
            outputs.append(h)

        # Re-join the bands along the frequency axis for a final joint verdict.
        joined = torch.cat(outputs, dim=-1)
        logits = self.conv_post(joined)
        features.append(logits)
        return logits.flatten(1), features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """
    Bank of :class:`STFTSubBandDiscriminator` at several STFT resolutions.

    Short windows resolve transients (the pick attack, the hammer, the click at
    the front of a drum), long windows resolve steady-state partials (the exact
    harmonic ratios that make a string section sound like a string section).
    """

    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = (
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
            (128, 32, 128),
        ),
        channels: int = 32,
    ):
        """
        Args:
            resolutions: Tuple of ``(n_fft, hop_length, win_length)`` triples.
            channels: Hidden width of each sub-band convolutional stack.
        """
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTSubBandDiscriminator(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    channels=channels,
                )
                for n_fft, hop_length, win_length in resolutions
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[FeatureMaps]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tuple of (list of logits, list of feature-map lists).
        """
        x = _fold_channels(x)
        all_logits: list[torch.Tensor] = []
        all_features: list[FeatureMaps] = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features


# =============================================================================
# Combined discriminator
# =============================================================================


class AudioDiscriminator(nn.Module):
    """
    The full Stage-1 critic: multi-period + multi-resolution complex STFT.

    A single forward pass returns the concatenated logits and feature maps of
    every sub-discriminator, which is all the trainer needs for the hinge
    adversarial loss and the feature-matching loss.
    """

    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        stft_resolutions: tuple[tuple[int, int, int], ...] = (
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
            (128, 32, 128),
        ),
        stft_channels: int = 32,
        period_channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
        use_period: bool = True,
        use_stft: bool = True,
    ):
        super().__init__()
        if not (use_period or use_stft):
            raise ValueError("AudioDiscriminator needs at least one sub-discriminator")

        self.mpd = (
            MultiPeriodDiscriminator(periods=periods, channels=period_channels)
            if use_period
            else None
        )
        self.mrd = (
            MultiResolutionSTFTDiscriminator(
                resolutions=stft_resolutions, channels=stft_channels
            )
            if use_stft
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[FeatureMaps]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tuple of (list of logits, list of feature-map lists) across every
            sub-discriminator of both banks.
        """
        logits: list[torch.Tensor] = []
        features: list[FeatureMaps] = []

        if self.mpd is not None:
            p_logits, p_features = self.mpd(x)
            logits.extend(p_logits)
            features.extend(p_features)

        if self.mrd is not None:
            s_logits, s_features = self.mrd(x)
            logits.extend(s_logits)
            features.extend(s_features)

        return logits, features
