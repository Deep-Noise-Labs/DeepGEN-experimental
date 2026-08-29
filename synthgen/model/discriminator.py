"""
Discriminators for adversarial training of the SynthGen Audio VAE.

Why this exists
---------------
Every loss term the VAE was previously trained with - L1 on the waveform and a
multi-resolution STFT loss on *magnitudes* - is either phase-blind or nearly so.
An STFT magnitude loss cannot distinguish a clean pluck from the same pluck put
through an all-pass filter that smears its partials over tens of milliseconds:
the magnitude spectrum is identical by construction, so the gradient is
(near) zero. Phase is what makes a transient sound like a pick hitting a string
rather than a swell, and it is the single biggest thing separating "AI-ish"
audio from a Spitfire or Splice sample.

A discriminator that looks at the *waveform* is the only term in the objective
that sees phase directly. Two complementary families are used, following
HiFi-GAN / EnCodec / DAC:

- ``MultiPeriodDiscriminator``   - reshapes the 1D signal into 2D by period and
  convolves, so each sub-discriminator specialises in a different periodic
  structure. This is what recovers pitch stability and harmonic detail.
- ``MultiResolutionSTFTDiscriminator`` - operates on the *complex* STFT
  (real and imaginary parts as channels, not magnitude), at several
  resolutions. Because it is fed real/imag rather than magnitude, it is
  phase-sensitive; because it is multi-resolution it covers both transient
  (short window) and tonal (long window) structure.

Stereo handling
---------------
Both discriminators fold the channel axis into the batch axis, so they operate
on single channels. When ``mid_side=True`` (the default) the L/R pair is
converted to mid/side first. The side channel then receives real adversarial
pressure of its own, instead of being something the model can quietly collapse
because it barely moves an L/R-averaged loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

weight_norm = nn.utils.parametrizations.weight_norm


# =============================================================================
# Helpers
# =============================================================================


def to_mid_side(x: torch.Tensor) -> torch.Tensor:
    """
    Convert (batch, 2, samples) L/R audio to mid/side.

    Mono or multi-channel inputs other than stereo are returned untouched.
    """
    if x.shape[1] != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    return torch.stack([mid, side], dim=1)


def _fold_channels(x: torch.Tensor, mid_side: bool) -> torch.Tensor:
    """(batch, channels, samples) -> (batch * channels, 1, samples)."""
    if mid_side:
        x = to_mid_side(x)
    batch, channels, samples = x.shape
    return x.reshape(batch * channels, 1, samples)


# =============================================================================
# Multi-Period Discriminator (HiFi-GAN)
# =============================================================================


class PeriodDiscriminator(nn.Module):
    """
    Single period sub-discriminator.

    The 1D waveform is reshaped to (samples // period, period) and processed
    with 2D convolutions that stride only along the "time" axis. Samples that
    are ``period`` apart therefore land in the same column, which makes the
    module sensitive to periodic structure at that period - i.e. to pitch.
    """

    def __init__(
        self,
        period: int,
        channels: tuple = (32, 128, 512, 1024),
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period

        convs = []
        in_ch = 1
        for out_ch in channels:
            convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_ch, out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                )
            )
            in_ch = out_ch
        # final feature layer without striding
        convs.append(
            weight_norm(
                nn.Conv2d(in_ch, in_ch, kernel_size=(kernel_size, 1), padding=(2, 0))
            )
        )
        self.convs = nn.ModuleList(convs)
        self.conv_post = weight_norm(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Audio of shape (batch, 1, samples).

        Returns:
            (logits, feature_maps) - logits are flattened, feature_maps are the
            intermediate activations used for the feature-matching loss.
        """
        batch, _, samples = x.shape

        # Pad so the length is divisible by the period, then fold to 2D
        remainder = samples % self.period
        if remainder != 0:
            x = F.pad(x, (0, self.period - remainder), mode="reflect")
            samples = x.shape[-1]
        x = x.view(batch, 1, samples // self.period, self.period)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)

        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    """Bank of ``PeriodDiscriminator``s over co-prime periods."""

    def __init__(
        self,
        periods: tuple = (2, 3, 5, 7, 11),
        channels: tuple = (32, 128, 512, 1024),
        mid_side: bool = True,
    ):
        super().__init__()
        self.mid_side = mid_side
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=p, channels=channels) for p in periods]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        x = _fold_channels(x, self.mid_side)
        logits, features = [], []
        for disc in self.discriminators:
            logit, feats = disc(x)
            logits.append(logit)
            features.append(feats)
        return logits, features


# =============================================================================
# Multi-Resolution STFT Discriminator (EnCodec / DAC)
# =============================================================================


class STFTSubDiscriminator(nn.Module):
    """
    Single-resolution complex-STFT sub-discriminator.

    Consumes the *complex* spectrogram as two real channels (real, imaginary)
    rather than the magnitude. This is deliberate: magnitude-only inputs would
    reproduce the exact phase blindness this whole module exists to fix.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        channels: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 1, samples) -> (batch, 2, freq, frames) real/imag."""
        x = x.squeeze(1)
        # STFT is not implemented for half precision; run it in fp32.
        spec = torch.stft(
            x.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device).float(),
            center=True,
            return_complex=True,
        )
        return torch.stack([spec.real, spec.imag], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        y = self._stft(x).to(next(self.parameters()).dtype)
        features = []
        for conv in self.convs:
            y = F.leaky_relu(conv(y), 0.1)
            features.append(y)
        y = self.conv_post(y)
        features.append(y)
        return y.flatten(1, -1), features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Bank of complex-STFT sub-discriminators at several resolutions."""

    def __init__(
        self,
        resolutions: tuple = ((2048, 512), (1024, 256), (512, 128), (256, 64), (128, 32)),
        channels: int = 32,
        mid_side: bool = True,
    ):
        super().__init__()
        self.mid_side = mid_side
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(
                n_fft=n_fft, hop_length=hop, win_length=n_fft, channels=channels
            )
            for n_fft, hop in resolutions
        ])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        x = _fold_channels(x, self.mid_side)
        logits, features = [], []
        for disc in self.discriminators:
            logit, feats = disc(x)
            logits.append(logit)
            features.append(feats)
        return logits, features


# =============================================================================
# Combined
# =============================================================================


class CombinedDiscriminator(nn.Module):
    """
    Multi-period + multi-resolution-STFT discriminator, as used for the VAE
    stage. Returns the concatenated logits and feature maps of both banks.
    """

    def __init__(
        self,
        periods: tuple = (2, 3, 5, 7, 11),
        mpd_channels: tuple = (32, 128, 512, 1024),
        stft_resolutions: tuple = ((2048, 512), (1024, 256), (512, 128), (256, 64), (128, 32)),
        stft_channels: int = 32,
        mid_side: bool = True,
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            periods=periods, channels=mpd_channels, mid_side=mid_side
        )
        self.mrd = MultiResolutionSTFTDiscriminator(
            resolutions=stft_resolutions, channels=stft_channels, mid_side=mid_side
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            (logits, features) aggregated over both discriminator banks.
        """
        mpd_logits, mpd_features = self.mpd(x)
        mrd_logits, mrd_features = self.mrd(x)
        return mpd_logits + mrd_logits, mpd_features + mrd_features


def build_discriminator(config: Optional[object] = None) -> CombinedDiscriminator:
    """Construct the discriminator from a TrainingConfig-like object."""
    if config is None:
        return CombinedDiscriminator()
    return CombinedDiscriminator(
        periods=tuple(getattr(config, "disc_periods", (2, 3, 5, 7, 11))),
        mpd_channels=tuple(getattr(config, "disc_mpd_channels", (32, 128, 512, 1024))),
        stft_channels=int(getattr(config, "disc_stft_channels", 32)),
        mid_side=bool(getattr(config, "disc_mid_side", True)),
    )
