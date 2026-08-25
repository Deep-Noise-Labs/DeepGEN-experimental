"""
Multi-scale STFT discriminator for adversarial VAE training.

Magnitude-only spectral losses (L1 + multi-resolution STFT) are phase-blind:
a decoder can satisfy them while producing smeared transients and a dull,
"underwater" top end. Following EnCodec, Descript Audio Codec and Stable
Audio, this module adds a set of discriminators operating on the *complex*
STFT at several resolutions. The adversarial signal pushes the decoder
towards phase-coherent, crisp reconstructions, and the discriminator's
intermediate feature maps provide a learned perceptual distance
(feature-matching loss).

Each sub-discriminator views the complex STFT of every audio channel as a
2D image (real and imaginary parts stacked as input channels) and applies a
stack of strided 2D convolutions with LeakyReLU, returning both the final
logit map and the intermediate feature maps.
"""

from typing import Sequence

import torch
import torch.nn as nn


class STFTSubDiscriminator(nn.Module):
    """
    Single-resolution complex-STFT discriminator.

    Operates on the complex spectrogram (real + imaginary stacked as
    channels, one pair per audio channel) with 2D convolutions that are
    strided along the frequency axis and dilated along the time axis.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        audio_channels: int = 2,
        base_filters: int = 32,
        max_filters: int = 512,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.audio_channels = audio_channels

        self.register_buffer(
            "window", torch.hann_window(win_size), persistent=False
        )

        in_channels = 2 * audio_channels  # real + imag per audio channel

        filters = [
            base_filters,
            min(base_filters * 2, max_filters),
            min(base_filters * 4, max_filters),
            min(base_filters * 8, max_filters),
            min(base_filters * 16, max_filters),
        ]

        convs = []
        prev = in_channels
        # First layer: no stride, capture local time-frequency structure
        convs.append(nn.Conv2d(prev, filters[0], kernel_size=(3, 9), padding=(1, 4)))
        prev = filters[0]
        # Strided layers: downsample along time, keep frequency resolution,
        # increasing dilation along the frequency axis widens the receptive
        # field over harmonics
        for i, dilation in enumerate((1, 2, 4)):
            convs.append(
                nn.Conv2d(
                    prev,
                    filters[i + 1],
                    kernel_size=(3, 9),
                    stride=(1, 2),
                    dilation=(dilation, 1),
                    padding=(dilation, 4),
                )
            )
            prev = filters[i + 1]
        convs.append(nn.Conv2d(prev, filters[4], kernel_size=(3, 3), padding=(1, 1)))
        prev = filters[4]

        self.convs = nn.ModuleList(
            [nn.utils.parametrizations.weight_norm(c) for c in convs]
        )
        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(prev, 1, kernel_size=(3, 3), padding=(1, 1))
        )
        self.activation = nn.LeakyReLU(0.1)

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex STFT as a real-valued image.

        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tensor of shape (batch, 2 * channels, freq_bins, frames).
        """
        batch, channels, samples = x.shape
        x = x.reshape(batch * channels, samples)
        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (batch * channels, freq, frames)
        spec = torch.view_as_real(stft)  # (batch * channels, freq, frames, 2)
        spec = spec.permute(0, 3, 1, 2)  # (batch * channels, 2, freq, frames)
        freq, frames = spec.shape[-2], spec.shape[-1]
        return spec.reshape(batch, channels * 2, freq, frames)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tuple of (logits, features):
            - logits: Final discriminator map (batch, 1, freq', frames').
            - features: Intermediate feature maps for feature matching.
        """
        # STFT of a discriminator input is not part of the generator graph
        # numerically sensitive in half precision; keep it in fp32
        spec = self._spectrogram(x.float())

        features = []
        h = spec
        for conv in self.convs:
            h = self.activation(conv(h))
            features.append(h)
        logits = self.conv_post(h)
        return logits, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Ensemble of complex-STFT discriminators at multiple resolutions.

    Multiple FFT sizes give the ensemble simultaneous views with fine
    frequency resolution (large FFT — sustained harmonics, timbre) and fine
    time resolution (small FFT — transients, attacks), so neither can be
    cheated in isolation.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128),
        audio_channels: int = 2,
        base_filters: int = 32,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.discriminators = nn.ModuleList(
            [
                STFTSubDiscriminator(
                    fft_size=n,
                    hop_size=n // 4,
                    win_size=n,
                    audio_channels=audio_channels,
                    base_filters=base_filters,
                )
                for n in self.fft_sizes
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples).

        Returns:
            Tuple of (all_logits, all_features), one entry per scale.
        """
        all_logits = []
        all_features = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features
