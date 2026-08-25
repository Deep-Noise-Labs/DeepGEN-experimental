"""
Multi-resolution STFT discriminator for adversarial VAE training.

Magnitude-only spectral losses (L1 + multi-resolution STFT) leave the
decoder free to produce phase-incoherent audio: transients smear, unison
detune turns into chorus-like "swirl", and high frequencies sound diffuse.
Every production-grade audio autoencoder (EnCodec, Descript Audio Codec,
Stable Audio) closes this gap with a waveform-domain adversarial loss.

This module implements the modern standard: a set of sub-discriminators,
each operating on the *complex* STFT (real + imaginary planes) of the
input waveform at a different resolution. Complex input makes the
discriminator directly sensitive to phase structure, which the existing
magnitude losses cannot see.

Reference designs: EnCodec's MS-STFT discriminator and DAC's
multi-resolution discriminator.
"""

import torch
import torch.nn as nn


class STFTSubDiscriminator(nn.Module):
    """
    Single-resolution discriminator over the complex STFT.

    The waveform is transformed with a fixed STFT; real and imaginary
    planes of every audio channel are stacked as 2D input channels and
    processed by a stack of strided 2D convolutions. Returns the final
    logit map plus intermediate feature maps for feature matching.
    """

    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_size: int,
        in_channels: int = 2,
        base_channels: int = 32,
        max_channels: int = 512,
        num_layers: int = 5,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer(
            "window", torch.hann_window(win_size), persistent=False
        )

        wn = nn.utils.parametrizations.weight_norm

        layers = []
        # Real + imaginary plane per audio channel
        channels_in = in_channels * 2
        channels_out = base_channels
        for i in range(num_layers):
            # Stride along frequency; keep time resolution until later layers
            stride = (2, 1) if i < 2 else (2, 2)
            layers.append(
                nn.Sequential(
                    wn(
                        nn.Conv2d(
                            channels_in,
                            channels_out,
                            kernel_size=(5, 5),
                            stride=stride,
                            padding=(2, 2),
                        )
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
            channels_in = channels_out
            channels_out = min(channels_out * 2, max_channels)

        self.layers = nn.ModuleList(layers)
        self.output_conv = wn(
            nn.Conv2d(channels_in, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Complex STFT stacked as (batch, 2 * channels, freq, time)."""
        batch, channels, samples = x.shape
        # STFT in float32 for numerical stability under autocast
        x = x.reshape(batch * channels, samples).float()
        spec = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # (batch * channels, freq, time, 2)
        spec = spec.permute(0, 3, 1, 2)  # (batch * channels, 2, freq, time)
        return spec.reshape(batch, channels * 2, spec.shape[2], spec.shape[3])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Waveform of shape (batch, channels, samples).

        Returns:
            Tuple of (logits, feature_maps). Logits have shape
            (batch, 1, freq', time'); feature_maps is the list of
            activations after every conv block.
        """
        h = self._spectrogram(x)
        features = []
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        logits = self.output_conv(h)
        return logits, features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """
    Ensemble of STFT sub-discriminators at multiple resolutions.

    Short windows catch transient/attack fidelity; long windows catch
    tonal steadiness and unison/detune structure. All resolutions vote,
    so the decoder cannot hide artefacts at any single time-frequency
    trade-off.
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256, 128),
        in_channels: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTSubDiscriminator(
                    fft_size=n,
                    hop_size=n // 4,
                    win_size=n,
                    in_channels=in_channels,
                    base_channels=base_channels,
                )
                for n in fft_sizes
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            x: Waveform of shape (batch, channels, samples).

        Returns:
            Tuple of (logits_list, features_list) with one entry per
            resolution.
        """
        all_logits = []
        all_features = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features
