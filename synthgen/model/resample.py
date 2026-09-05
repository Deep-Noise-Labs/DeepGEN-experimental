"""
Band-limited resampling for the audio VAE.

Why this file exists
--------------------
The VAE moves audio across a 1024x sample-rate change: the encoder decimates
by 4, 4, 8, 8 with strided convolutions, and the decoder puts it back with
transposed convolutions of the same strides. Both operators are written as a
single convolution of kernel length ``2 * stride``.

That kernel length is the problem. A transposed convolution with stride ``s``
and kernel ``2s`` is exactly a polyphase interpolator with ``s`` phases of
**two taps each**. Two taps per phase cannot suppress the spectral images that
rate conversion creates, so every decoder stage emits mirrored, inharmonic
copies of the signal around multiples of its input rate. The encoder has the
mirror-image defect: it decimates without an anti-alias filter, so content
above the new Nyquist folds back down.

For sustained, harmonically dense material -- which is what a synthesiser
model generates -- inharmonic imaging and folding are the most audible defect
class there is. It is the difference between a patch that sounds like a
sampler instrument and one that sounds "cheap digital".

The fix, following Alias-Free GAN (Karras et al.) and BigVGAN, is to put a
fixed, non-learnable Kaiser-windowed sinc filter on the rate-changing step:
low-pass **after** zero-stuffing on the way up, and **before** decimating on
the way down.

Matched-weight property
-----------------------
Both replacements below are written so that when the fixed filter is replaced
by a unit impulse they are *bit-for-bit identical* to the operators they
replace, with the same learnable weight tensor and the same parameter count.
``tests/test_resample.py`` asserts exactly that. The consequence is that any
measured difference between the two is attributable to the filter alone, not
to a change of weights, initialisation, capacity or level.

The filters are normalised to unit DC gain (``sum(h) == 1``), which is also
what a unit impulse has, so the pass-band level of the two arms matches and
the learnable kernel that follows sees the same signal scale it saw before.
No added parameters; roughly one extra FIR per rate change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "kaiser_sinc_filter1d",
    "resampling_filter",
    "unit_impulse_filter1d",
    "TransposedUpsample1d",
    "BandlimitedUpsample1d",
    "StridedDownsample1d",
    "BandlimitedDownsample1d",
    "DEFAULT_TAPS_PER_STRIDE",
    "DEFAULT_TRANSITION",
]

# Filter length as a multiple of the resampling ratio, and transition
# half-width as a fraction of the cutoff.
#
# Chosen from the measured ablation in docs/UPSAMPLER.md, not by intuition.
# At stride 4 on real renders this pair gives about -112 dB image-to-signal
# with a 65-tap filter, which is far below the 16-bit noise floor, and a
# reconstruction SNR indistinguishable from filters four times longer.
#
# Two things that ablation corrected, both of which had produced a wrong
# conclusion first time round:
#
# 1. A wider transition is *better* at a fixed length, not worse. Widening it
#    raises the Kaiser beta, which deepens the stop-band far more than the
#    extra pass-band droop (about 1.5 dB at 90% of the band) costs. The
#    droop is also a fixed linear response the learnable kernel after it can
#    compensate; an image is signal-dependent and cannot be undone.
# 2. The apparent fidelity cost of filtering was never droop at all. It was
#    the half-sample delay of an even-length linear-phase FIR. See
#    ``resampling_filter``, which forces odd lengths.
DEFAULT_TAPS_PER_STRIDE = 16
DEFAULT_TRANSITION = 0.5


# =============================================================================
# Filter design
# =============================================================================


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    """
    Design a low-pass Kaiser-windowed sinc filter, normalised to unit DC gain.

    Args:
        cutoff: Normalised cutoff frequency in cycles/sample, in (0, 0.5).
            For rate conversion by ``s`` this is ``0.5 / s``.
        half_width: Half the transition-band width, in cycles/sample.
        kernel_size: Number of taps. Longer means sharper transition and
            deeper stop-band.

    Returns:
        Tensor of shape ``(1, 1, kernel_size)`` whose taps sum to 1.

    A unit impulse is the degenerate member of this family, and passing it
    through either resampler below reproduces the unfiltered operator exactly.
    """
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if not 0.0 < cutoff < 0.5:
        raise ValueError(f"cutoff must be in (0, 0.5), got {cutoff}")

    even = kernel_size % 2 == 0
    half = kernel_size // 2

    # Kaiser beta from the standard stop-band-attenuation estimate.
    delta_f = 2 * half_width
    attenuation = 2.285 * (kernel_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, periodic=False, beta=beta, dtype=torch.float64)

    if even:
        time = torch.arange(-half, half, dtype=torch.float64) + 0.5
    else:
        time = torch.arange(kernel_size, dtype=torch.float64) - half

    taps = 2 * cutoff * window * torch.special.sinc(2 * cutoff * time)
    taps = taps / taps.sum()  # unit DC gain
    return taps.to(torch.float32).view(1, 1, kernel_size)


def unit_impulse_filter1d(kernel_size: int) -> torch.Tensor:
    """A unit impulse of the given length: the identity member of the family."""
    taps = torch.zeros(kernel_size)
    taps[kernel_size // 2] = 1.0
    return taps.view(1, 1, kernel_size)


def resampling_filter(
    stride: int,
    kernel_size: int,
    transition: float = DEFAULT_TRANSITION,
) -> torch.Tensor:
    """
    The fixed filter for a rate change of ``stride``.

    Cutoff sits at the new Nyquist, ``0.5 / stride``. ``kernel_size <= 1``
    yields a unit impulse, which reduces the band-limited operators to the
    unfiltered ones they replace.
    """
    if kernel_size <= 1:
        return unit_impulse_filter1d(1)
    # Force an odd length. A linear-phase FIR of even length delays the signal
    # by a half-integer number of samples, which no integer-shift alignment
    # can undo; it shows up as a fidelity loss at high frequencies and is
    # easily mistaken for the filter damaging the pass-band. Odd length gives
    # an exactly integer group delay of (N - 1) / 2 and symmetric padding.
    if kernel_size % 2 == 0:
        kernel_size += 1
    cutoff = 0.5 / stride
    return kaiser_sinc_filter1d(
        cutoff=cutoff,
        half_width=transition * cutoff,
        kernel_size=kernel_size,
    )


# =============================================================================
# Upsampling (decoder side)
# =============================================================================


class TransposedUpsample1d(nn.Module):
    """
    The repository's current decoder upsampler, isolated so it can be measured.

    ``ConvTranspose1d(in, out, kernel_size=2 * stride, stride=stride,
    padding=stride // 2)`` -- identical to ``DecoderBlock.upsample``'s
    convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = stride * 2
        self.padding = stride // 2
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BandlimitedUpsample1d(nn.Module):
    """
    Drop-in replacement for :class:`TransposedUpsample1d` that band-limits the
    rate change.

    The transposed convolution is decomposed into its two real steps --
    zero-stuff by ``stride``, then convolve with the learnable kernel -- and a
    fixed anti-imaging low-pass is inserted between them:

        zero-stuff -> fixed Kaiser sinc low-pass -> learnable kernel

    The learnable kernel has the same shape and the same semantics as the
    ``ConvTranspose1d`` weight it replaces, so a checkpoint trained with one
    can be loaded into the other. With ``filter_kernel_size=1`` the fixed
    filter is a unit impulse and this module is numerically identical to
    :class:`TransposedUpsample1d`.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Upsampling ratio.
        filter_kernel_size: Taps in the fixed anti-imaging filter. Defaults to
            ``DEFAULT_TAPS_PER_STRIDE * stride``; see
            ``experiments/upsampler_bench.py`` for the length-vs-rejection
            curve that chose it.
        transition: Transition half-width as a fraction of the cutoff. Smaller
            is sharper but needs more taps for the same stop-band depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        filter_kernel_size: int | None = None,
        transition: float = DEFAULT_TRANSITION,
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = stride * 2
        self.padding = stride // 2
        self.in_channels = in_channels

        # Same weight tensor layout as ConvTranspose1d: (in, out, kernel).
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self._reset_parameters()

        if filter_kernel_size is None:
            filter_kernel_size = DEFAULT_TAPS_PER_STRIDE * stride
        taps = resampling_filter(stride, filter_kernel_size, transition)
        # resampling_filter may round the length up to keep it odd, so the
        # padding must follow the tensor rather than the request.
        self.filter_kernel_size = taps.shape[-1]
        self.register_buffer("filter_taps", taps, persistent=False)

    def _reset_parameters(self) -> None:
        """Match ``nn.ConvTranspose1d``'s initialisation exactly."""
        fan_in = self.weight.shape[1] * self.weight.shape[2]  # out_channels * kernel
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def _zero_stuff(self, x: torch.Tensor) -> torch.Tensor:
        """Insert ``stride - 1`` zeros after every input sample."""
        batch, channels, length = x.shape
        if self.stride == 1:
            return x
        out = x.new_zeros(batch, channels, length, self.stride)
        out[..., 0] = x
        return out.view(batch, channels, length * self.stride)

    def _antiimage(self, z: torch.Tensor) -> torch.Tensor:
        """Depthwise fixed low-pass. Symmetric padding keeps the length."""
        k = self.filter_kernel_size
        if k <= 1:
            return z
        taps = self.filter_taps.to(z.dtype).expand(self.in_channels, 1, k)
        left = (k - 1) // 2
        right = k - 1 - left
        z = F.pad(z, (left, right))
        return F.conv1d(z, taps, groups=self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._zero_stuff(x)
        z = self._antiimage(z)

        # conv_transpose1d(x, W, stride=s, padding=p) is zero-stuffing followed
        # by a convolution with the flipped, transposed kernel. Trailing
        # zero-stuffed taps are dropped so the output length matches.
        weight = self.weight.transpose(0, 1).flip(-1)
        pad = self.kernel_size - self.padding - 1
        y = F.conv1d(z, weight, bias=self.bias, padding=pad)
        expected = (x.shape[-1] - 1) * self.stride - 2 * self.padding + self.kernel_size
        return y[..., :expected]


# =============================================================================
# Downsampling (encoder side)
# =============================================================================


class StridedDownsample1d(nn.Module):
    """
    The repository's current encoder downsampler, isolated so it can be
    measured.

    ``Conv1d(in, out, kernel_size=2 * stride, stride=stride,
    padding=stride // 2)`` -- identical to ``EncoderBlock.downsample``'s
    convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = stride * 2
        self.padding = stride // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BandlimitedDownsample1d(nn.Module):
    """
    Drop-in replacement for :class:`StridedDownsample1d` that band-limits the
    rate change.

    A fixed Kaiser sinc low-pass is applied at the input rate, before the
    strided convolution decimates, so content above the new Nyquist is
    attenuated instead of folded back. With ``filter_kernel_size=1`` the
    filter is a unit impulse and this module is numerically identical to
    :class:`StridedDownsample1d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        filter_kernel_size: int | None = None,
        transition: float = DEFAULT_TRANSITION,
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = stride * 2
        self.padding = stride // 2
        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        if filter_kernel_size is None:
            filter_kernel_size = DEFAULT_TAPS_PER_STRIDE * stride
        taps = resampling_filter(stride, filter_kernel_size, transition)
        # resampling_filter may round the length up to keep it odd, so the
        # padding must follow the tensor rather than the request.
        self.filter_kernel_size = taps.shape[-1]
        self.register_buffer("filter_taps", taps, persistent=False)

    def _antialias(self, x: torch.Tensor) -> torch.Tensor:
        k = self.filter_kernel_size
        if k <= 1:
            return x
        taps = self.filter_taps.to(x.dtype).expand(self.in_channels, 1, k)
        left = (k - 1) // 2
        right = k - 1 - left
        x = F.pad(x, (left, right))
        return F.conv1d(x, taps, groups=self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._antialias(x))
