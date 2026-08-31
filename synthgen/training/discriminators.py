"""
Discriminators for adversarial training of the SynthGen audio VAE.

Why this module exists
----------------------
The Stage-1 objective was L1 on the waveform plus a magnitude multi-resolution
STFT term. Both are *magnitude* criteria averaged over a spectrogram, and both
are minimised by an output that is spectrally close on average. Nothing in them
distinguishes a crisp transient from a smeared one, or coherent phase from
incoherent phase, so the optimum they point at is the conditional mean of the
data -- the blurry, slightly phasey, slightly "underwater" reconstruction that
every purely-regression audio autoencoder converges to.

Every autoencoder that reaches production audio quality (HiFi-GAN, EnCodec,
DAC, Stable Audio, BigVGAN) closes that gap the same way: a discriminator that
looks at structure the magnitude loss is blind to, plus a feature-matching term
that stabilises it.

Two complementary families are implemented:

``MultiPeriodDiscriminator``
    Reshapes the waveform into 2D by a set of coprime periods and convolves.
    Catches periodic structure -- pitch, phase coherence within a cycle,
    per-cycle waveshape. This is what makes a saw sound like a saw rather than a
    band-limited approximation of one.

``MultiResolutionSTFTDiscriminator``
    Convolves over the *complex* STFT (real and imaginary parts as channels) at
    several resolutions. Because it sees real/imag rather than magnitude, it can
    penalise phase incoherence directly -- the thing that makes reconstructions
    sound smeared and that the magnitude loss cannot see at all.

Both take ``(batch, channels, samples)`` and fold channels into the batch, so
stereo is handled without a separate code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MultiPeriodDiscriminator",
    "MultiResolutionSTFTDiscriminator",
    "CombinedDiscriminator",
]


def _weight_norm(module: nn.Module) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(module)


def _fold_channels(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, T)`` -> ``(B * C, 1, T)`` so stereo needs no special casing."""
    batch, channels, samples = x.shape
    return x.reshape(batch * channels, 1, samples)


# =============================================================================
# Multi-period discriminator
# =============================================================================


class PeriodDiscriminator(nn.Module):
    """
    One period of the multi-period discriminator (HiFi-GAN, Kong et al. 2020).

    The waveform is reshaped to ``(T/period, period)`` and convolved with 2D
    kernels that are tall and one sample wide, so each kernel sees the same
    phase position across successive cycles.
    """

    def __init__(
        self,
        period: int,
        channels: tuple[int, ...] = (32, 128, 512, 1024),
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(
                _weight_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0),
                    )
                )
            )
            in_ch = out_ch

        # Final feature layer runs at stride 1 so the receptive field stays dense.
        layers.append(
            _weight_norm(
                nn.Conv2d(in_ch, 1024, kernel_size=(kernel_size, 1), padding=(2, 0))
            )
        )
        self.convs = nn.ModuleList(layers)
        self.post = _weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Pad to a whole number of periods, then fold time into 2D.
        remainder = x.shape[-1] % self.period
        if remainder != 0:
            x = F.pad(x, (0, self.period - remainder), mode="reflect")
        batch, channels, samples = x.shape
        x = x.view(batch, channels, samples // self.period, self.period)

        features: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    """Bank of :class:`PeriodDiscriminator` over coprime periods."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period) for period in periods]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        x = _fold_channels(x)
        logits, features = [], []
        for disc in self.discriminators:
            logit, feats = disc(x)
            logits.append(logit)
            features.append(feats)
        return logits, features


# =============================================================================
# Multi-resolution complex-STFT discriminator
# =============================================================================


class STFTDiscriminator(nn.Module):
    """
    One resolution of the complex-STFT discriminator (EnCodec / DAC family).

    Operates on ``(real, imag)`` stacked as input channels, so phase is visible
    to the network rather than being thrown away by a magnitude operator.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int | None = None,
        channels: int = 32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )

        def conv(in_ch: int, out_ch: int, stride: tuple[int, int]) -> nn.Module:
            return _weight_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 9), stride=stride, padding=(1, 4))
            )

        self.convs = nn.ModuleList(
            [
                conv(2, channels, (1, 1)),
                conv(channels, channels, (2, 2)),
                conv(channels, channels, (2, 2)),
                conv(channels, channels, (2, 2)),
                conv(channels, channels, (2, 2)),
            ]
        )
        self.post = _weight_norm(
            nn.Conv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, T) -> (N, 2, freq, frames)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device, x.dtype),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        return torch.stack([spec.real, spec.imag], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # STFT is numerically unhappy in low precision; keep it in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self._stft(x.float())

        features: list[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)
        return x.flatten(1), features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Bank of :class:`STFTDiscriminator` across FFT sizes."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (2048, 1024, 512),
        hop_sizes: tuple[int, ...] = (512, 256, 128),
        channels: int = 32,
    ):
        super().__init__()
        if len(fft_sizes) != len(hop_sizes):
            raise ValueError("fft_sizes and hop_sizes must be the same length")
        self.discriminators = nn.ModuleList(
            [
                STFTDiscriminator(n_fft=n, hop_length=h, channels=channels)
                for n, h in zip(fft_sizes, hop_sizes)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        x = _fold_channels(x)
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
    Multi-period + multi-resolution-STFT discriminators behind one interface.

    Returns a flat list of per-sub-discriminator logits and a flat list of the
    matching intermediate feature stacks (used for feature matching).
    """

    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        fft_sizes: tuple[int, ...] = (2048, 1024, 512),
        hop_sizes: tuple[int, ...] = (512, 256, 128),
        stft_channels: int = 32,
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.mrd = MultiResolutionSTFTDiscriminator(
            fft_sizes=fft_sizes, hop_sizes=hop_sizes, channels=stft_channels
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        mpd_logits, mpd_features = self.mpd(x)
        mrd_logits, mrd_features = self.mrd(x)
        return mpd_logits + mrd_logits, mpd_features + mrd_features
