"""
Alias-free resampling and activations for SynthGen.

Periodic activations such as Snake (``x + (1/a) sin^2(a x)``) are strongly
non-linear: applied to a band-limited feature map they synthesise harmonics
far above the Nyquist rate of that feature map. Those harmonics cannot be
represented at the working rate, so they fold back into the audible band as
inharmonic partials -- aliasing. In a synthesiser context this is the single
most damaging artefact class, because it breaks the exact property that makes
a sound read as "professional": a clean, strictly harmonic partial series.

The fix follows the alias-free design introduced by BigVGAN (Lee et al., 2023),
itself an application of StyleGAN3's alias-free reasoning to audio: run the
non-linearity at an oversampled rate, then band-limit before returning to the
working rate.

    x -> upsample(ratio) -> activation -> lowpass -> downsample(ratio) -> y

Resampling uses Kaiser-windowed sinc FIR filters, which give an explicitly
controllable stopband attenuation rather than the uncontrolled response of a
learned or naive interpolation kernel.

References:
    BigVGAN: A Universal Neural Vocoder with Large-Scale Training,
    https://arxiv.org/abs/2206.04658
    Alias-Free Generative Adversarial Networks (StyleGAN3),
    https://arxiv.org/abs/2106.12423
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Filter design
# =============================================================================


def kaiser_beta(attenuation_db: float) -> float:
    """
    Kaiser window beta for a target stopband attenuation.

    Standard design formula (Oppenheim & Schafer, Discrete-Time Signal
    Processing, sec. 7.5).
    """
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
    Build a Kaiser-windowed sinc low-pass FIR kernel.

    Args:
        cutoff: Normalised cutoff frequency in cycles/sample (0 < cutoff < 0.5).
        half_width: Normalised width of the transition band.
        kernel_size: Number of FIR taps. Larger is sharper and more expensive.

    Returns:
        Kernel of shape ``(1, 1, kernel_size)``, normalised to unit DC gain.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser design: transition width -> required beta.
    delta_f = 4.0 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    beta = kaiser_beta(attenuation)

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # Sample the ideal sinc on the (possibly half-sample offset) time grid.
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros(1, 1, kernel_size)

    filter_ = 2.0 * cutoff * window * torch.sinc(2.0 * cutoff * time)
    # Unit DC gain keeps the block signal-preserving at low frequencies.
    filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


# =============================================================================
# Resamplers
# =============================================================================


# Taps per resampling filter. BigVGAN's reference uses ``6 * ratio`` (12 at
# ratio 2); measured on this architecture that leaves ~15 dB round-trip error
# in the 15 kHz "air" band, which is exactly the region that carries the
# perceived expensiveness of a sample library. Cost here is memory-bound
# rather than tap-bound, so a longer kernel is close to free -- 32 taps buys
# roughly 13 dB of alias suppression and 15 dB of air-band accuracy for a
# single-digit percentage of runtime. See docs/EVALS.md for the sweep.
DEFAULT_FILTER_TAPS = 32


class UpSample1d(nn.Module):
    """Band-limited integer upsampling via transposed convolution with a sinc kernel."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            DEFAULT_FILTER_TAPS if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter_ = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter_, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        # ratio gain compensates the energy lost to zero-stuffing.
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1).to(x.dtype),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Band-limited integer decimation: low-pass then stride."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            DEFAULT_FILTER_TAPS if kernel_size is None else kernel_size
        )
        self.pad_left = self.kernel_size // 2 - 1
        self.pad_right = self.kernel_size // 2
        filter_ = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter_, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            x,
            self.filter.expand(channels, -1, -1).to(x.dtype),
            stride=self.ratio,
            groups=channels,
        )


# =============================================================================
# Alias-free activation wrapper
# =============================================================================


class AntiAliasedActivation(nn.Module):
    """
    Run a pointwise non-linearity at an oversampled rate and band-limit the result.

    Wrapping any activation in this module leaves its shape contract unchanged,
    so it is a drop-in replacement for the bare activation.

    Args:
        activation: The pointwise non-linearity to make alias-free.
        ratio: Oversampling factor. 2 removes the second-order fold-back that
            dominates Snake's harmonic output; higher ratios buy more headroom
            at linear cost.
    """

    def __init__(self, activation: nn.Module, ratio: int = 2):
        super().__init__()
        self.ratio = ratio
        self.activation = activation
        self.upsample = UpSample1d(ratio)
        self.downsample = DownSample1d(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.activation(x)
        return self.downsample(x)
