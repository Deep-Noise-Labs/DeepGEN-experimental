"""
Audio Variational Autoencoder (VAE) for SynthGen.

Compresses raw waveforms into a compact latent space suitable for
diffusion-based generation. Architecture inspired by Stable Audio's
autoencoder with Snake activations and strided convolutions.

The encoder maps stereo 44.1kHz audio to continuous latents with a
compression ratio of 1024x (e.g., 10s stereo audio = 441,000 samples
per channel → 430 latent frames of dimension 64).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthgen.model.antialias import AntiAliasedActivation


# =============================================================================
# Activation Functions
# =============================================================================


class Snake(nn.Module):
    """
    Snake activation function: x + (1/alpha) * sin^2(alpha * x).

    Provides periodic inductive bias that is beneficial for audio synthesis,
    as shown in BigVGAN and Stable Audio.

    NOTE: Snake is a periodic non-linearity and therefore generates harmonics
    above the Nyquist rate of whatever feature map it is applied to. Used bare
    it aliases badly -- measured at -17 dB alias-to-signal on an 8 kHz tone at
    alpha=2. Prefer ``make_activation(..., antialias=True)``, which wraps this
    in oversampling and band-limiting. This class is kept unwrapped so that
    the aliasing baseline stays reproducible and for checkpoint compatibility.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((1, channels, 1), alpha_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


def make_activation(
    channels: int,
    antialias: bool = True,
    ratio: int = 2,
    alpha_init: float = 1.0,
) -> nn.Module:
    """
    Build the activation used throughout the VAE.

    Args:
        channels: Feature channels the activation is applied to.
        antialias: Wrap Snake in oversample -> activate -> band-limit.
        ratio: Oversampling factor used when ``antialias`` is set.
        alpha_init: Initial Snake alpha.

    Returns:
        Either a bare ``Snake`` or an alias-free wrapper around one. Both have
        an identical shape contract, so they are interchangeable.
    """
    snake = Snake(channels, alpha_init=alpha_init)
    if not antialias:
        return snake
    return AntiAliasedActivation(snake, ratio=ratio)


def remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    to_antialias: bool,
) -> dict[str, torch.Tensor]:
    """
    Translate a VAE state dict between the aliased and alias-free layouts.

    Wrapping Snake in ``AntiAliasedActivation`` nests it one level deeper, so
    ``...block.0.alpha`` becomes ``...block.0.activation.alpha``. No weights
    change -- the wrapper holds only fixed FIR buffers -- so a checkpoint
    trained in one layout can be loaded into the other by renaming keys.

    This also makes an exactly weight-matched A/B possible: build one model,
    remap, and the only difference between the two is the resampling.

    Args:
        state_dict: Source state dict.
        to_antialias: ``True`` converts bare -> wrapped, ``False`` the reverse.

    Returns:
        A new state dict with renamed keys.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if to_antialias:
            # Snake's only parameter is `alpha`; move it under the wrapper.
            new_key = (
                key[: -len("alpha")] + "activation.alpha"
                if key.endswith("alpha")
                else key
            )
        else:
            new_key = key.replace(".activation.alpha", ".alpha")
        remapped[new_key] = value
    return remapped


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
        antialias: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.block = nn.Sequential(
            make_activation(channels, antialias=antialias),
            nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            make_activation(channels, antialias=antialias),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

        # Weight normalization for stability
        self.block[1] = nn.utils.parametrizations.weight_norm(self.block[1])
        self.block[3] = nn.utils.parametrizations.weight_norm(self.block[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: residual layers followed by downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        num_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        antialias: bool = True,
    ):
        super().__init__()

        # Residual layers with increasing dilation
        self.residual_layers = nn.Sequential(*[
            ResidualBlock(in_channels, dilation=d, antialias=antialias)
            for d in dilations[:num_residual]
        ])

        # Downsampling via strided convolution
        self.downsample = nn.Sequential(
            make_activation(in_channels, antialias=antialias),
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            ),
        )
        self.downsample[1] = nn.utils.parametrizations.weight_norm(self.downsample[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_layers(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsampling followed by residual layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        num_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        antialias: bool = True,
    ):
        super().__init__()

        # Upsampling via transposed convolution
        self.upsample = nn.Sequential(
            make_activation(in_channels, antialias=antialias),
            nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            ),
        )
        self.upsample[1] = nn.utils.parametrizations.weight_norm(self.upsample[1])

        # Residual layers
        self.residual_layers = nn.Sequential(*[
            ResidualBlock(out_channels, dilation=d, antialias=antialias)
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
        antialias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
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
                    antialias=antialias,
                )
            )
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(current_channels, dilation=1, antialias=antialias),
            ResidualBlock(current_channels, dilation=3, antialias=antialias),
            make_activation(current_channels, antialias=antialias),
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
        antialias: bool = True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.latent_dim = latent_dim

        # Input projection from latent
        first_channels = base_channels * channel_multipliers[0]
        self.from_latent = nn.Conv1d(latent_dim, first_channels, kernel_size=1)

        # Pre-bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(first_channels, dilation=1, antialias=antialias),
            ResidualBlock(first_channels, dilation=3, antialias=antialias),
        )

        # Decoder blocks. Each block widens by `stride` and steps to the next
        # multiplier in the schedule; the final block holds `base_channels`.
        self.decoder_blocks = nn.ModuleList()
        current_channels = first_channels

        for i, stride in enumerate(strides):
            next_mult = (
                channel_multipliers[i + 1] if i + 1 < len(channel_multipliers) else 1
            )
            out_ch = base_channels * next_mult

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    stride=stride,
                    num_residual=num_residual_per_block,
                    antialias=antialias,
                )
            )
            current_channels = out_ch

        # Output projection (no tanh to avoid harmonic distortion)
        self.output_conv = nn.Sequential(
            make_activation(current_channels, antialias=antialias),
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
        antialias: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.antialias = antialias
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
            antialias=antialias,
        )

        self.decoder = AudioDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_multipliers=decoder_channel_multipliers,
            strides=tuple(reversed(strides)),
            num_residual_per_block=num_residual_per_block,
            antialias=antialias,
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
