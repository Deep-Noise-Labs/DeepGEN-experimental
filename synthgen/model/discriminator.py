"""
Discriminators for adversarial VAE training.

The autoencoder is the fidelity ceiling of the whole latent-diffusion stack:
the DiT can never sound better than what the VAE decoder can reconstruct.
Purely regression-based objectives (L1 + magnitude STFT) are known to produce
over-smoothed reconstructions - dull high end, smeared transients - because
they reward the decoder for predicting the *average* of all plausible
waveforms. A discriminator removes that incentive by penalising outputs that
are distinguishable from real audio.

This module implements a multi-resolution complex-STFT discriminator in the
style of DAC (Descript Audio Codec) and Stable Audio: each sub-discriminator
sees the real and imaginary STFT channels (so phase and transients are
visible to it) at a different time/frequency resolution.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wn_conv2d(*args, **kwargs) -> nn.Conv2d:
    """Conv2d with weight normalization."""
    return nn.utils.parametrizations.weight_norm(nn.Conv2d(*args, **kwargs))


class STFTSubDiscriminator(nn.Module):
    """
    Single-resolution complex-STFT discriminator.

    Operates on the stacked [real, imag] channels of the STFT, using 2D
    convolutions with strides along the frequency axis. Returns the final
    logit map plus intermediate feature maps for feature-matching loss.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: Optional[int] = None,
        channels: int = 32,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size or fft_size

        self.register_buffer(
            "window", torch.hann_window(self.win_size), persistent=False
        )

        self.convs = nn.ModuleList([
            _wn_conv2d(2, channels, kernel_size=(3, 9), padding=(1, 4)),
            _wn_conv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            _wn_conv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            _wn_conv2d(channels, channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            _wn_conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        ])
        self.conv_out = _wn_conv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1))

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute complex STFT as stacked real/imag channels.

        Args:
            x: Audio of shape (batch, samples).

        Returns:
            Tensor of shape (batch, 2, frames, freq_bins).
        """
        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
        )
        # (batch, freq, frames) -> (batch, 2, frames, freq)
        spec = torch.view_as_real(stft)  # (batch, freq, frames, 2)
        return spec.permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Audio of shape (batch, samples). Multi-channel audio should be
               folded into the batch dimension by the caller.

        Returns:
            Tuple of (logits, feature_maps).
        """
        spec = self._spectrogram(x)

        features = []
        h = spec
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)

        logits = self.conv_out(h)
        return logits, features


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-resolution STFT discriminator (DAC / Stable Audio style).

    Runs several STFTSubDiscriminators at complementary time/frequency
    resolutions so that both fast transients and fine harmonic structure
    are adversarially supervised.
    """

    def __init__(
        self,
        resolutions: tuple = ((2048, 512), (1024, 256), (512, 128)),
        channels: int = 32,
    ):
        """
        Args:
            resolutions: Tuple of (fft_size, hop_size) pairs.
            channels: Convolutional channel width of each sub-discriminator.
        """
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(
                fft_size=fft_size,
                hop_size=hop_size,
                channels=channels,
            )
            for fft_size, hop_size in resolutions
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: Audio of shape (batch, channels, samples) or (batch, samples).

        Returns:
            Tuple of (logits_per_resolution, features_per_resolution).
        """
        if x.dim() == 3:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, samples)

        all_logits = []
        all_features = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)

        return all_logits, all_features
