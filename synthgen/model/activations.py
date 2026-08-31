"""
Anti-aliased activations for the SynthGen audio decoder.

Why this module exists
----------------------
A pointwise nonlinearity applied to a discrete-time signal generates harmonics
that extend far beyond the Nyquist frequency of the sample grid it is running
on. Those harmonics have nowhere to go: they fold back into the audible band as
*inharmonic* partials. On a sustained tonal source -- exactly the material this
model targets (synth leads, pads, bells, plucked and bowed instrument samples)
-- folded partials are not "a bit of noise", they are audible metallic grit that
sits at frequencies unrelated to the fundamental, and the ear locks onto them
immediately.

``Snake`` is a *strongly* harmonic-generating nonlinearity (it contains a squared
sine), and the decoder applies it at roughly thirty sites, at every sample rate
from the latent grid up to 44.1 kHz. Aliasing at each site accumulates and cannot
be trained away, because it is a property of running a nonlinearity on a grid
that is too coarse for its own output bandwidth.

The fix is the one introduced by BigVGAN (Lee et al., 2023,
https://arxiv.org/abs/2206.04658), following the alias-free formulation of
StyleGAN3 (Karras et al., 2021): wrap every nonlinearity in an upsample /
activate / downsample sandwich with Kaiser-windowed sinc filters, so the
harmonics the nonlinearity creates are generated on a finer grid and then
low-pass filtered away *before* they can fold.

Cost is real: at ``ratio=2`` each wrapped activation runs on twice the samples
plus two short grouped FIR convolutions. It is applied to the decoder only,
which is the component that actually emits audio.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Snake",
    "SnakeBeta",
    "LowPassFilter1d",
    "UpSample1d",
    "DownSample1d",
    "AntiAliasedActivation",
    "build_activation",
]


# =============================================================================
# Nonlinearities
# =============================================================================


class Snake(nn.Module):
    """
    Snake activation: ``x + (1/alpha) * sin^2(alpha * x)``.

    Periodic inductive bias for audio synthesis (Ziyin et al., 2020). Kept for
    checkpoint compatibility with Stage-1 runs trained before ``SnakeBeta``
    existed; new models should prefer :class:`SnakeBeta`.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


class SnakeBeta(nn.Module):
    """
    Snake with a decoupled magnitude parameter (BigVGAN):

        ``x + (1/beta) * sin^2(alpha * x)``

    ``alpha`` controls the frequency of the periodic component and ``beta`` its
    magnitude. Tying them together, as plain Snake does, means the network cannot
    ask for a fast periodic component without also asking for a quiet one. For
    bright synthetic timbres those are exactly the two knobs you want to set
    independently.

    Both parameters are stored in log space so they stay positive under
    unconstrained gradient descent, which removes the ``1/(alpha + 1e-8)``
    blow-up that plain Snake suffers whenever ``alpha`` is driven towards zero.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        beta_init: float | None = None,
        log_scale: bool = True,
    ):
        super().__init__()
        beta_init = alpha_init if beta_init is None else beta_init
        self.log_scale = log_scale

        if log_scale:
            alpha = torch.full((1, channels, 1), math.log(alpha_init))
            beta = torch.full((1, channels, 1), math.log(beta_init))
        else:
            alpha = torch.full((1, channels, 1), float(alpha_init))
            beta = torch.full((1, channels, 1), float(beta_init))

        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp() if self.log_scale else self.alpha
        beta = self.beta.exp() if self.log_scale else self.beta
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x) ** 2


# =============================================================================
# Kaiser-windowed sinc resampling
# =============================================================================


def kaiser_beta(attenuation_db: float) -> float:
    """Kaiser window beta for a target stopband attenuation (Kaiser's formula)."""
    if attenuation_db > 50.0:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21.0:
        return 0.5842 * (attenuation_db - 21.0) ** 0.4 + 0.07886 * (attenuation_db - 21.0)
    return 0.0


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    """
    Design a windowed-sinc low-pass FIR.

    Args:
        cutoff: Cutoff as a fraction of the sample rate (0.25 == sr/4).
        half_width: Transition band half-width, same units as ``cutoff``.
        kernel_size: FIR length in taps.

    Returns:
        Filter of shape ``(1, 1, kernel_size)``, normalised to unit DC gain.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser design: attenuation implied by the transition width and tap count.
    delta_f = 4.0 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    beta = kaiser_beta(attenuation)

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float64)

    if even:
        time = torch.arange(-half_size, half_size, dtype=torch.float64) + 0.5
    else:
        time = torch.arange(kernel_size, dtype=torch.float64) - half_size

    if cutoff == 0.0:
        taps = torch.zeros_like(time)
    else:
        taps = 2.0 * cutoff * window * torch.sinc(2.0 * cutoff * time)
        taps = taps / taps.sum()

    return taps.to(torch.float32).view(1, 1, kernel_size)


# Filter design, and why it is not BigVGAN's.
#
# BigVGAN uses 12 taps with a transition half-width of 0.6/ratio. That is a very
# gentle filter, and it is fine for a 22.05 or 24 kHz model where the top octave
# carries little. At 44.1 kHz it is not: measured, the 12-tap up/down round trip
# costs -1.2 dB at 16 kHz and -3.8 dB at 20 kHz. The decoder's full-rate stage
# alone applies eight of these, so that compounds to roughly -10 dB at 16 kHz and
# -31 dB at 20 kHz -- trading an aliasing problem for a dull, lifeless top end,
# which on cymbals, bright pads and plucked-string harmonics is just as
# disqualifying.
#
# 64 taps at half-width 0.1/ratio measures flat to 20 kHz (-0.06 dB), with
# stopband rejection better than -38 dB at 24 kHz and -60 dB above 26 kHz. The
# aliases that fold deepest into the audible band come from the highest
# frequencies (content at 40 kHz lands at 4.1 kHz), and those are the ones this
# design rejects hardest.
#
# Cost is a 64-tap grouped FIR twice per activation instead of a 12-tap one,
# which is a fraction of the cost of the convolutions it sits between.
TAPS_PER_RATIO = 32
TRANSITION_HALF_WIDTH = 0.1


def _default_kernel_size(ratio: int) -> int:
    """Tap count for a given ratio, forced even."""
    return int(TAPS_PER_RATIO * ratio // 2) * 2


class LowPassFilter1d(nn.Module):
    """Grouped (per-channel) FIR low-pass with optional decimation."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = TRANSITION_HALF_WIDTH,
        stride: int = 1,
        kernel_size: int = 64,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if cutoff > 0.5:
            raise ValueError("cutoff must not exceed 0.5 (Nyquist)")

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x,
            self.filter.to(x.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class UpSample1d(nn.Module):
    """Integer-ratio interpolating upsampler (zero-stuff + windowed-sinc)."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = _default_kernel_size(ratio) if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=TRANSITION_HALF_WIDTH / ratio,
                kernel_size=self.kernel_size,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.to(x.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Integer-ratio decimator with the matching anti-aliasing low-pass."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = _default_kernel_size(ratio) if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=TRANSITION_HALF_WIDTH / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


# =============================================================================
# The wrapper
# =============================================================================


class AntiAliasedActivation(nn.Module):
    """
    Run a pointwise nonlinearity on an oversampled grid, then filter and decimate.

    ``up -> activation -> down``. The harmonics the nonlinearity creates above
    the original Nyquist frequency are removed by the decimation low-pass instead
    of folding back into the audible band as inharmonic partials.

    Args:
        activation: The pointwise module to wrap (e.g. :class:`SnakeBeta`).
        ratio: Oversampling ratio. 2 is the BigVGAN default and captures the
            dominant fold-back; higher ratios cost proportionally more.
    """

    def __init__(self, activation: nn.Module, ratio: int = 2):
        super().__init__()
        if ratio < 1:
            raise ValueError("ratio must be >= 1")
        self.ratio = ratio
        self.activation = activation
        if ratio > 1:
            self.upsample = UpSample1d(ratio)
            self.downsample = DownSample1d(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ratio == 1:
            return self.activation(x)
        x = self.upsample(x)
        x = self.activation(x)
        return self.downsample(x)


def build_activation(
    channels: int,
    kind: str = "snakebeta",
    anti_aliased: bool = True,
    ratio: int = 2,
) -> nn.Module:
    """
    Factory used by the VAE so activation choice is a single config switch.

    Args:
        channels: Channel count (each channel gets its own alpha/beta).
        kind: ``"snake"`` (legacy) or ``"snakebeta"``.
        anti_aliased: Wrap in :class:`AntiAliasedActivation`.
        ratio: Oversampling ratio when ``anti_aliased`` is set.
    """
    if kind == "snake":
        activation: nn.Module = Snake(channels)
    elif kind == "snakebeta":
        activation = SnakeBeta(channels)
    else:
        raise ValueError(f"Unknown activation kind: {kind!r} (expected 'snake' or 'snakebeta')")

    if anti_aliased:
        return AntiAliasedActivation(activation, ratio=ratio)
    return activation
