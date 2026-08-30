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


# =============================================================================
# Activation Functions
# =============================================================================


class Snake(nn.Module):
    """
    Snake activation function: x + (1/alpha) * sin^2(alpha * x).

    Provides periodic inductive bias that is beneficial for audio synthesis,
    as shown in BigVGAN and Stable Audio.

    Note the numerical hazard: ``alpha`` is unconstrained, so a training step
    that drives it towards zero makes ``1/alpha`` explode. ``SnakeBeta`` below
    removes that failure mode by parameterising in log space.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((1, channels, 1), alpha_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


class SnakeBeta(nn.Module):
    """
    Snake activation with an independent magnitude parameter:

        x + (1/beta) * sin^2(alpha * x)

    ``alpha`` controls the frequency of the periodic component and ``beta``
    its magnitude. Both are stored in log space, so they are strictly positive
    for any parameter value and the reciprocal can never blow up.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.full((1, channels, 1), math.log(alpha_init))
        )
        self.log_beta = nn.Parameter(
            torch.full((1, channels, 1), math.log(beta_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


# =============================================================================
# Alias-free resampling (Kaiser-windowed sinc)
# =============================================================================
#
# A pointwise nonlinearity generates harmonics above the Nyquist frequency of
# the signal it is applied to. At the native rate those harmonics fold back
# down as inharmonic partials -- the "cheap digital" grit that separates a
# neural synth from Serum or a Spitfire library. The standard remedy, from
# Karras et al. (Alias-Free GAN) and BigVGAN, is to run the nonlinearity at
# 2x rate between a matched pair of low-pass resamplers, so the harmonics it
# creates are removed before decimation instead of aliasing.
#
# Filter design follows the reference alias-free-torch implementation (MIT).


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    """
    Build a 1D low-pass FIR kernel as a Kaiser-windowed sinc.

    Args:
        cutoff: Normalised cutoff frequency (cycles/sample), in (0, 0.5).
        half_width: Normalised width of the transition band.
        kernel_size: Number of taps.

    Returns:
        Kernel of shape (1, 1, kernel_size), normalised to unit DC gain.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser beta from the required stop-band attenuation
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros(1, 1, kernel_size)

    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    """Band-limited 1D upsampling by an integer ratio."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or (6 * ratio // 2) * 2
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1).to(x.dtype),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Band-limited 1D decimation by an integer ratio."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or (6 * ratio // 2) * 2
        self.even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(self.even)
        self.pad_right = self.kernel_size // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            x,
            self.filter.expand(channels, -1, -1).to(x.dtype),
            stride=self.ratio,
            groups=channels,
        )


class AliasFreeSnake(nn.Module):
    """
    Snake activation evaluated at ``ratio`` times the native sample rate.

    Upsample -> SnakeBeta -> downsample. The harmonics the nonlinearity
    generates above the original Nyquist frequency are removed by the
    decimation filter rather than folding back into the audible band.

    Costs roughly ``ratio`` times the activation compute of a plain Snake;
    the convolutions around it are unchanged, so end-to-end VAE step time
    grows by substantially less than ``ratio``.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        ratio: int = 2,
        kernel_size: int = 12,
    ):
        super().__init__()
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        self.activation = SnakeBeta(channels, alpha_init=alpha_init)
        self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.activation(self.upsample(x)))


def make_activation(channels: int, antialias: bool = True) -> nn.Module:
    """Return the configured activation for a given channel count."""
    if antialias:
        return AliasFreeSnake(channels)
    return Snake(channels)


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
            make_activation(channels, antialias),
            nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            make_activation(channels, antialias),
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
            make_activation(in_channels, antialias),
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
            make_activation(in_channels, antialias),
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
            make_activation(current_channels, antialias),
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
                    antialias=antialias,
                )
            )
            current_channels = out_ch

        # Output projection (no tanh to avoid harmonic distortion)
        self.output_conv = nn.Sequential(
            make_activation(current_channels, antialias),
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
