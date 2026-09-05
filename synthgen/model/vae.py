"""
Audio Variational Autoencoder (VAE) for SynthGen.

Compresses raw waveforms into a compact latent space suitable for
diffusion-based generation. Architecture inspired by Stable Audio's
autoencoder with Snake activations and strided convolutions.

The encoder maps stereo 44.1kHz audio to continuous latents with a
compression ratio of 2048x (e.g., 10s stereo audio = 882,000 samples
per channel → 431 latent frames of dimension 64).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthgen.model.resample import BandlimitedDownsample1d, BandlimitedUpsample1d


# =============================================================================
# Activation Functions
# =============================================================================


class Snake(nn.Module):
    """
    Snake activation function: x + (1/alpha) * sin^2(alpha * x).

    Provides periodic inductive bias that is beneficial for audio synthesis,
    as shown in BigVGAN and Stable Audio.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((1, channels, 1), alpha_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


# =============================================================================
# Building Blocks
# =============================================================================


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions and Snake activation."""

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.block = nn.Sequential(
            Snake(channels),
            nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            Snake(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

        # Weight normalization for stability
        self.block[1] = nn.utils.parametrizations.weight_norm(self.block[1])
        self.block[3] = nn.utils.parametrizations.weight_norm(self.block[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: residual layers followed by downsampling.

    ``bandlimited`` inserts a fixed anti-alias low-pass before the strided
    convolution decimates. Without it, content above the post-decimation
    Nyquist folds back as inharmonic noise; see ``synthgen/model/resample.py``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        num_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        bandlimited: bool = True,
    ):
        super().__init__()

        # Residual layers with increasing dilation
        self.residual_layers = nn.Sequential(*[
            ResidualBlock(in_channels, dilation=d)
            for d in dilations[:num_residual]
        ])

        # Downsampling via strided convolution
        if bandlimited:
            resampler = BandlimitedDownsample1d(in_channels, out_channels, stride)
            conv = resampler.conv
        else:
            resampler = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            )
            conv = resampler

        self.downsample = nn.Sequential(Snake(in_channels), resampler)
        nn.utils.parametrizations.weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_layers(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsampling followed by residual layers.

    ``bandlimited`` inserts a fixed anti-imaging low-pass between the
    zero-stuffing and the learnable kernel of the transposed convolution.
    Without it the stage emits spectral images -- inharmonic mirror copies of
    the signal -- that no downstream layer can remove; see
    ``synthgen/model/resample.py``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        num_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        bandlimited: bool = True,
    ):
        super().__init__()

        # Upsampling via transposed convolution
        if bandlimited:
            resampler = BandlimitedUpsample1d(in_channels, out_channels, stride)
        else:
            resampler = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            )

        self.upsample = nn.Sequential(Snake(in_channels), resampler)
        nn.utils.parametrizations.weight_norm(resampler)

        # Residual layers
        self.residual_layers = nn.Sequential(*[
            ResidualBlock(out_channels, dilation=d)
            for d in dilations[:num_residual]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.residual_layers(x)
        return x


# =============================================================================
# Encoder
# =============================================================================


class AudioEncoder(nn.Module):
    """
    Audio encoder that compresses waveforms into latent representations.

    Architecture:
    - Input convolution
    - 4 encoder blocks with progressive downsampling (4, 4, 8, 8 = 1024x total)
    - Bottleneck convolution to latent dimension
    - Outputs mean and log-variance for VAE sampling
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        strides: tuple = (4, 4, 8, 8),
        num_residual_per_block: int = 3,
        bandlimited: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.bandlimited = bandlimited
        self.latent_dim = latent_dim

        # Compute compression ratio
        self.compression_ratio = 1
        for s in strides:
            self.compression_ratio *= s

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        self.input_conv = nn.utils.parametrizations.weight_norm(self.input_conv)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels

        for i, (mult, stride) in enumerate(zip(channel_multipliers, strides)):
            out_channels = base_channels * mult
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=stride,
                    num_residual=num_residual_per_block,
                    bandlimited=bandlimited,
                )
            )
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(current_channels, dilation=1),
            ResidualBlock(current_channels, dilation=3),
            Snake(current_channels),
        )

        # Output projection to latent space (mean and log-var)
        self.to_latent = nn.Conv1d(current_channels, latent_dim * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to latent distribution parameters.

        Args:
            x: Audio tensor of shape (batch, channels, samples).

        Returns:
            Tuple of (mean, log_var), each of shape (batch, latent_dim, latent_length).
        """
        x = self.input_conv(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.bottleneck(x)
        x = self.to_latent(x)

        mean, log_var = x.chunk(2, dim=1)
        return mean, log_var


# =============================================================================
# Decoder
# =============================================================================


class AudioDecoder(nn.Module):
    """
    Audio decoder that reconstructs waveforms from latent representations.

    Architecture mirrors the encoder with transposed convolutions for upsampling.
    Does NOT use tanh at the output to avoid harmonic distortion.
    """

    def __init__(
        self,
        out_channels: int = 2,
        latent_dim: int = 64,
        base_channels: int = 64,
        channel_multipliers: tuple = (8, 4, 2, 1),
        strides: tuple = (8, 8, 4, 4),
        num_residual_per_block: int = 3,
        bandlimited: bool = True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.bandlimited = bandlimited
        self.latent_dim = latent_dim

        # Input projection from latent
        first_channels = base_channels * channel_multipliers[0]
        self.from_latent = nn.Conv1d(latent_dim, first_channels, kernel_size=1)

        # Pre-bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(first_channels, dilation=1),
            ResidualBlock(first_channels, dilation=3),
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        current_channels = first_channels

        for i, (mult, stride) in enumerate(zip(channel_multipliers, strides)):
            if i == 0:
                in_ch = current_channels
            else:
                in_ch = current_channels
            out_ch = base_channels * (channel_multipliers[i + 1] if i + 1 < len(channel_multipliers) else 1)

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    stride=stride,
                    num_residual=num_residual_per_block,
                    bandlimited=bandlimited,
                )
            )
            current_channels = out_ch

        # Output projection (no tanh to avoid harmonic distortion)
        self.output_conv = nn.Sequential(
            Snake(current_channels),
            nn.Conv1d(current_channels, out_channels, kernel_size=7, padding=3),
        )
        self.output_conv[1] = nn.utils.parametrizations.weight_norm(self.output_conv[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio waveform.

        Args:
            z: Latent tensor of shape (batch, latent_dim, latent_length).

        Returns:
            Reconstructed audio of shape (batch, channels, samples).
        """
        x = self.from_latent(z)
        x = self.bottleneck(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.output_conv(x)
        return x


# =============================================================================
# Full VAE
# =============================================================================


class AudioVAE(nn.Module):
    """
    Full Audio Variational Autoencoder.

    Combines encoder and decoder with reparameterization trick for
    training with the VAE objective (reconstruction + KL divergence).
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        base_channels: int = 64,
        encoder_channel_multipliers: tuple = (1, 2, 4, 8),
        decoder_channel_multipliers: tuple = (8, 4, 2, 1),
        strides: tuple = (4, 4, 8, 8),
        num_residual_per_block: int = 3,
        bandlimited: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.bandlimited = bandlimited
        self.compression_ratio = 1
        for s in strides:
            self.compression_ratio *= s

        self.encoder = AudioEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_multipliers=encoder_channel_multipliers,
            strides=strides,
            num_residual_per_block=num_residual_per_block,
            bandlimited=bandlimited,
        )

        self.decoder = AudioDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_multipliers=decoder_channel_multipliers,
            strides=tuple(reversed(strides)),
            num_residual_per_block=num_residual_per_block,
            bandlimited=bandlimited,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio."""
        return self.decoder(z)

    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE training."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through VAE.

        Args:
            x: Audio tensor of shape (batch, channels, samples).

        Returns:
            Tuple of (reconstruction, input, mean, log_var).
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decode(z)

        # Ensure reconstruction matches input length
        if reconstruction.shape[-1] != x.shape[-1]:
            min_len = min(reconstruction.shape[-1], x.shape[-1])
            reconstruction = reconstruction[..., :min_len]
            x = x[..., :min_len]

        return reconstruction, x, mean, log_var

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent (using mean, no sampling)."""
        mean, _ = self.encode(x)
        return mean

    def get_latent_length(self, audio_length: int) -> int:
        """Compute latent sequence length from audio sample length."""
        return audio_length // self.compression_ratio
