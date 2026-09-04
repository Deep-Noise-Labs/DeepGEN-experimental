"""
Alias-free signal-processing primitives for the SynthGen synthesis stack.

Why this module exists
----------------------
The SynthGen decoder is a stack of Snake activations. Snake is

    snake(x) = x + (1/a) * sin^2(a * x)

which is a *memoryless nonlinearity*. Every memoryless nonlinearity
generates harmonics of its input, and ``sin^2`` generates them without
bound. When such a function is evaluated on a discrete-time signal at
44.1 kHz, every harmonic it produces above Nyquist (22.05 kHz) does not
disappear - it **folds back** into the audible band at

    f_alias = | k * f0 - n * fs |

These folded partials are almost never at multiples of f0, so they are
*inharmonic*: they do not fuse with the note, and the ear hears them as
a separate metallic/gritty layer. This is the exact defect that
separates a cheap oscillator from Serum, and it is the reason BigVGAN
(Lee et al., ICLR 2023) and Alias-Free GAN (Karras et al., NeurIPS 2021)
wrap their nonlinearities in an oversample/filter/decimate sandwich.

The fix is to evaluate the nonlinearity at a higher rate, low-pass the
result below the original Nyquist, then decimate:

    x -> upsample(R) -> snake -> lowpass -> downsample(R)

This does not remove aliasing entirely (harmonics above R*Nyquist still
fold) but it pushes the alias floor down by a large, measurable margin.
``synthgen.eval`` measures exactly how much.

References
----------
- Karras et al., "Alias-Free Generative Adversarial Networks" (2021)
- Lee et al., "BigVGAN: A Universal Neural Vocoder with Large-Scale
  Training" (ICLR 2023)
- Pons et al., "Upsampling artifacts in neural audio synthesis" (2021)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Snake",
    "kaiser_sinc_filter1d",
    "LowPassFilter1d",
    "UpSample1d",
    "DownSample1d",
    "AntiAliasedActivation",
    "AntiAliasedSnake",
    "make_activation",
]


# =============================================================================
# The raw nonlinearity
# =============================================================================


class Snake(nn.Module):
    """
    Snake activation: ``x + (1/alpha) * sin^2(alpha * x)``.

    Periodic inductive bias that suits audio synthesis, but - evaluated at
    the base rate - a broadband alias generator. Prefer
    :class:`AntiAliasedSnake` in any module that runs at audio rate.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


# =============================================================================
# Windowed-sinc low-pass filter
# =============================================================================


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    """
    Build a windowed-sinc low-pass FIR kernel using a Kaiser window.

    Args:
        cutoff: Normalised cutoff frequency in cycles/sample, i.e. the
            fraction of the sample rate. ``0.5`` is Nyquist.
        half_width: Normalised width of the transition band. A narrower
            transition needs a longer kernel.
        kernel_size: Number of filter taps. Even values are supported;
            the kernel is built on a half-sample-shifted grid so the
            filter stays linear-phase.

    Returns:
        Tensor of shape ``(1, 1, kernel_size)``, unit DC gain.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser window beta from the desired stopband attenuation, following
    # Kaiser's standard design formula.
    delta_f = 4 * half_width
    attenuation = 2.285 * (kernel_size - 1) * math.pi * delta_f + 7.95
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

    if cutoff <= 0.0:
        filt = torch.zeros_like(time)
    else:
        filt = 2 * cutoff * torch.special.sinc(2 * cutoff * time)

    filt = filt * window
    # Normalise to unit DC gain so the filter is level-preserving.
    filt = filt / filt.sum()
    return filt.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    """Depthwise linear-phase low-pass filter for ``(B, C, T)`` audio."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if cutoff > 0.5:
            raise ValueError("cutoff must not exceed 0.5 (Nyquist)")

        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding

        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if self.padding:
            # Replicate padding avoids injecting a step discontinuity at the
            # boundaries, which would itself be broadband.
            x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        weight = self.filter.expand(channels, -1, -1).to(dtype=x.dtype)
        return F.conv1d(x, weight, stride=self.stride, groups=channels)


# =============================================================================
# Resampling
# =============================================================================


class UpSample1d(nn.Module):
    """Anti-imaging upsampler: zero-stuff by ``ratio`` then interpolate."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
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
        weight = self.filter.expand(channels, -1, -1).to(dtype=x.dtype)
        # ratio gain compensates for the energy lost to zero-stuffing.
        x = self.ratio * F.conv_transpose1d(
            x, weight, stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Anti-aliasing decimator: low-pass then keep every ``ratio``-th sample."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


# =============================================================================
# Anti-aliased activation
# =============================================================================


class AntiAliasedActivation(nn.Module):
    """
    Wrap any pointwise activation in an oversample/filter/decimate sandwich.

    The wrapped activation is evaluated at ``ratio`` times the base rate, so
    the harmonics it generates have ``ratio`` times more headroom before
    they fold. The low-pass in :class:`DownSample1d` removes everything
    above the base-rate Nyquist before decimation.

    Output length is guaranteed to match input length.
    """

    def __init__(self, activation: nn.Module, ratio: int = 2):
        super().__init__()
        self.ratio = ratio
        self.activation = activation
        self.upsample = UpSample1d(ratio)
        self.downsample = DownSample1d(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        x = self.upsample(x)
        x = self.activation(x)
        x = self.downsample(x)
        if x.shape[-1] != length:
            if x.shape[-1] > length:
                x = x[..., :length]
            else:
                x = F.pad(x, (0, length - x.shape[-1]))
        return x


class AntiAliasedSnake(nn.Module):
    """
    Snake evaluated at ``ratio`` times the base rate, band-limited on the
    way back down. Drop-in replacement for :class:`Snake`.

    The ``alpha`` parameter lives on the inner Snake, so a checkpoint
    trained with :class:`Snake` can be loaded by remapping
    ``<prefix>.alpha`` to ``<prefix>.act.activation.alpha``.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0, ratio: int = 2):
        super().__init__()
        self.act = AntiAliasedActivation(
            Snake(channels, alpha_init=alpha_init), ratio=ratio
        )

    @property
    def alpha(self) -> nn.Parameter:
        return self.act.activation.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


def make_activation(
    channels: int,
    antialias: bool = True,
    ratio: int = 2,
    alpha_init: float = 1.0,
) -> nn.Module:
    """
    Activation factory used throughout the VAE.

    Args:
        channels: Number of channels the activation runs on.
        antialias: If ``True`` return :class:`AntiAliasedSnake`, else the
            raw :class:`Snake`. ``False`` reproduces the pre-change model
            exactly and exists so the two can be A/B compared.
        ratio: Oversampling ratio for the anti-aliased path.
        alpha_init: Initial Snake alpha.
    """
    if antialias:
        return AntiAliasedSnake(channels, alpha_init=alpha_init, ratio=ratio)
    return Snake(channels, alpha_init=alpha_init)
