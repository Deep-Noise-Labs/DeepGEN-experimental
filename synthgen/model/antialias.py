"""
Anti-aliased activation for the SynthGen VAE decoder.

The Snake nonlinearity (x + sin^2(alpha*x)/alpha) is periodic and therefore
generates an infinite series of harmonics of its input. When it is applied
inside the decoder at (or near) the output sample rate, every harmonic that
lands above Nyquist folds back into the audible band as an inharmonic
"aliasing" artifact - the metallic fizz that separates consumer-grade digital
synthesis from professional oscillators.

BigVGAN (Lee et al., 2023, https://arxiv.org/abs/2206.04658) showed that
wrapping each nonlinearity in a 2x oversampling stage removes most of this
aliasing: upsample with a windowed-sinc low-pass, apply the nonlinearity at
the doubled rate (so its harmonics have headroom before folding), then
low-pass and downsample back. The filters are fixed (non-learned) Kaiser
windowed sinc kernels, following the alias-free design of "Alias-Free GAN"
(Karras et al., 2021, https://arxiv.org/abs/2106.12423).

``AntiAliasedSnake`` is a drop-in replacement for ``vae.Snake``: it exposes
the same learnable ``alpha`` parameter (same name, same shape) and registers
its filters as non-persistent buffers, so state dicts remain byte-compatible
with checkpoints trained before this change.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    """
    Design a low-pass FIR filter as a Kaiser-windowed sinc.

    Args:
        cutoff: Normalised cutoff frequency (0.5 = Nyquist of the signal rate).
        half_width: Transition band half-width (normalised).
        kernel_size: Number of filter taps.

    Returns:
        Filter tensor of shape (1, 1, kernel_size), normalised to unity DC gain.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser window beta from the required stop-band attenuation
    # (standard Kaiser design formula, Oppenheim & Schafer).
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

    filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filt = filt / filt.sum()
    return filt.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    """2x (or ``ratio``x) upsampling via zero-insertion + windowed-sinc low-pass."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
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
            self.filter.expand(channels, -1, -1).to(dtype=x.dtype),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """``ratio``x downsampling via windowed-sinc anti-aliasing low-pass + decimation."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
        self.stride = ratio
        self.pad_left = (self.kernel_size - self.stride) // 2
        self.pad_right = (self.kernel_size - self.stride + 1) // 2
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
            self.filter.expand(channels, -1, -1).to(dtype=x.dtype),
            stride=self.stride,
            groups=channels,
        )


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)."""
    return x + (1.0 / (alpha + 1e-8)) * torch.sin(alpha * x) ** 2


class AntiAliasedSnake(nn.Module):
    """
    Snake activation evaluated at 2x oversampled rate (BigVGAN-style).

    Drop-in replacement for ``vae.Snake`` with an identical state dict
    (single learnable parameter ``alpha`` of shape (1, channels, 1); the
    resampling filters are fixed, non-persistent buffers). Output length
    equals input length.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        ratio: int = 2,
        kernel_size: int = 12,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = snake(x, self.alpha)
        x = self.downsample(x)
        return x
