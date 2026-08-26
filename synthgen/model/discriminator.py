"""
Multi-scale STFT discriminator for adversarial VAE training.

Spectral reconstruction losses (L1 + multi-resolution STFT) alone are known
to produce audio with smeared transients, dull high frequencies and
"swishy" noise textures, because magnitude-only objectives are blind to
phase coherence and reward the model for averaging over plausible
reconstructions. Every state-of-the-art neural audio codec (EnCodec,
Descript Audio Codec, the Stable Audio autoencoder, HiFi-GAN-family
vocoders) closes this gap with an adversarial objective plus feature
matching.

This module implements a multi-scale complex-STFT discriminator following
the MS-STFT design introduced in EnCodec (Defossez et al., 2022): several
sub-discriminators, each operating on the real/imaginary STFT of the
waveform at a different resolution, with 2D convolutions over the
(frequency, time) plane. Each sub-discriminator returns its intermediate
feature maps so the generator can be trained with a feature-matching loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTSubDiscriminator(nn.Module):
    """
    Single-resolution complex-STFT discriminator.

    The input waveform is transformed with a complex STFT; real and
    imaginary parts are stacked as channels (2 per audio channel), and a
    stack of 2D convolutions with growing time-dilation scores the
    resulting (frequency, time) image. Outputs a logit map plus the
    intermediate feature maps for feature matching.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: Optional[int] = None,
        win_size: Optional[int] = None,
        in_channels: int = 2,
        filters: int = 32,
        max_filters: int = 512,
        dilations: tuple = (1, 2, 4),
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size or fft_size // 4
        self.win_size = win_size or fft_size
        self.in_channels = in_channels

        self.register_buffer(
            "window", torch.hann_window(self.win_size), persistent=False
        )

        wn = nn.utils.parametrizations.weight_norm

        convs = []
        # Real + imaginary parts stacked as channels.
        current = in_channels * 2
        convs.append(wn(nn.Conv2d(current, filters, kernel_size=(3, 9), padding=(1, 4))))
        current = filters

        # Downsample time, grow dilation along the time axis.
        for dilation in dilations:
            out = min(current * 2, max_filters)
            convs.append(
                wn(
                    nn.Conv2d(
                        current,
                        out,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        dilation=(1, dilation),
                        padding=(1, 4 * dilation),
                    )
                )
            )
            current = out

        convs.append(wn(nn.Conv2d(current, current, kernel_size=(3, 3), padding=(1, 1))))
        self.convs = nn.ModuleList(convs)
        self.conv_post = wn(nn.Conv2d(current, 1, kernel_size=(3, 3), padding=(1, 1)))

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Complex STFT as a (batch, 2 * channels, freq, frames) tensor."""
        batch, channels, samples = x.shape
        # STFT is computed in float32 for numerical stability under autocast.
        x = x.reshape(batch * channels, samples).float()
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.float(),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # (batch * channels, freq, frames, 2)
        spec = spec.permute(0, 3, 1, 2)  # (batch * channels, 2, freq, frames)
        freq, frames = spec.shape[-2], spec.shape[-1]
        return spec.reshape(batch, channels * 2, freq, frames)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Waveform of shape (batch, channels, samples).

        Returns:
            Tuple of (logit map, list of intermediate feature maps).
        """
        h = self._spectrogram(x)

        features = []
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, 0.2)
            features.append(h)

        logits = self.conv_post(h)
        return logits, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-scale STFT discriminator: one sub-discriminator per resolution.

    Multiple STFT resolutions cover the time/frequency trade-off — long
    windows resolve harmonic structure (sustained pads, detuned stacks),
    short windows resolve transients (plucks, drum hits) — so no artifact
    class can hide from every scale at once.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256, 128),
        in_channels: int = 2,
        filters: int = 32,
        max_filters: int = 512,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTSubDiscriminator(
                    fft_size=fft_size,
                    in_channels=in_channels,
                    filters=filters,
                    max_filters=max_filters,
                )
                for fft_size in fft_sizes
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: Waveform of shape (batch, channels, samples).

        Returns:
            Tuple of (list of logit maps, list of per-scale feature lists).
        """
        logits = []
        features = []
        for disc in self.discriminators:
            scale_logits, scale_features = disc(x)
            logits.append(scale_logits)
            features.append(scale_features)
        return logits, features
