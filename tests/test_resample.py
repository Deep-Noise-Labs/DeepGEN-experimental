"""
Tests for band-limited resampling.

The load-bearing tests are the equivalence ones. Every measured claim about
the anti-imaging and anti-aliasing filters depends on the two arms of the
comparison differing *only* by the filter, so that equivalence is asserted
here rather than assumed.
"""

import math

import pytest
import torch

from synthgen.model.resample import (
    DEFAULT_TAPS_PER_STRIDE,
    DEFAULT_TRANSITION,
    BandlimitedDownsample1d,
    BandlimitedUpsample1d,
    StridedDownsample1d,
    TransposedUpsample1d,
    kaiser_sinc_filter1d,
    resampling_filter,
    unit_impulse_filter1d,
)

STRIDES = [2, 4, 8]


# =============================================================================
# Filter design
# =============================================================================


def test_kaiser_filter_has_unit_dc_gain():
    """Unit DC gain is what makes the two arms level-matched."""
    for stride in STRIDES:
        taps = resampling_filter(stride, DEFAULT_TAPS_PER_STRIDE * stride)
        assert taps.sum().item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("stride", STRIDES)
def test_resampling_filter_is_odd_length(stride):
    """
    Odd length means an exactly integer group delay. With an even-length
    linear-phase FIR the output sits half a sample off the input, which no
    integer alignment can correct and which reads as high-frequency fidelity
    loss -- easy to misattribute to the filter damaging the pass-band.
    """
    for requested in (DEFAULT_TAPS_PER_STRIDE * stride, 2 * stride, 2 * stride + 1):
        taps = resampling_filter(stride, requested)
        assert taps.shape[-1] % 2 == 1, f"even-length filter for request {requested}"
        assert taps.shape[-1] >= requested


def test_unit_impulse_has_unit_dc_gain():
    assert unit_impulse_filter1d(1).sum().item() == pytest.approx(1.0)
    assert resampling_filter(8, 1).numel() == 1


@pytest.mark.parametrize("stride", STRIDES)
def test_default_filter_rejects_the_stop_band(stride):
    """
    The defaults must actually stop above the new Nyquist. Without this the
    whole change is decorative -- an under-designed filter (the Kaiser beta
    collapsing to a rectangular window) still *looks* like a fix in code.
    """
    taps = resampling_filter(stride, DEFAULT_TAPS_PER_STRIDE * stride)[0, 0]
    response = torch.fft.rfft(taps, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192)
    cutoff = 0.5 / stride

    # Absolute design intent, deliberately not derived from DEFAULT_TRANSITION:
    # a test that recomputes its own threshold from the parameter it is
    # checking passes for every parameter value and therefore checks nothing.
    passband = response[freqs < cutoff * 0.8].max()
    stopband = response[freqs > cutoff * 1.5].max()
    rejection_db = 20 * math.log10(stopband.item() / passband.item())
    assert rejection_db < -60.0, f"stop-band only {rejection_db:.1f} dB down"


@pytest.mark.parametrize("stride", STRIDES)
def test_default_filter_keeps_the_passband_flat(stride):
    """A filter that dulls the band it is meant to pass is not a fix."""
    taps = resampling_filter(stride, DEFAULT_TAPS_PER_STRIDE * stride)[0, 0]
    response = torch.fft.rfft(taps, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192)
    cutoff = 0.5 / stride

    # Flat across the useful band, again as an absolute criterion.
    passband = response[freqs < cutoff * 0.8]
    ripple_db = 20 * math.log10((passband.max() / passband.min()).item())
    assert ripple_db < 2.0, f"pass-band ripple {ripple_db:.2f} dB"


def test_kaiser_filter_rejects_bad_arguments():
    with pytest.raises(ValueError):
        kaiser_sinc_filter1d(0.6, 0.01, 32)
    with pytest.raises(ValueError):
        kaiser_sinc_filter1d(0.1, 0.01, 0)


# =============================================================================
# Matched-weight equivalence: the foundation of every A/B in this repository
# =============================================================================


@pytest.mark.parametrize("stride", STRIDES)
def test_bandlimited_upsample_with_impulse_equals_transposed_conv(stride):
    """
    With the fixed filter reduced to a unit impulse, the band-limited
    upsampler must reproduce ``ConvTranspose1d`` bit-for-bit -- same output
    shape, same values. This is what licenses attributing any measured
    difference to the filter alone.
    """
    torch.manual_seed(0)
    baseline = TransposedUpsample1d(3, 5, stride)
    fixed = BandlimitedUpsample1d(3, 5, stride, filter_kernel_size=1)

    with torch.no_grad():
        fixed.weight.copy_(baseline.conv.weight)
        fixed.bias.copy_(baseline.conv.bias)

    x = torch.randn(2, 3, 37)
    got = fixed(x)
    want = baseline(x)

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max().item()


@pytest.mark.parametrize("stride", STRIDES)
def test_bandlimited_downsample_with_impulse_equals_strided_conv(stride):
    torch.manual_seed(0)
    baseline = StridedDownsample1d(3, 5, stride)
    fixed = BandlimitedDownsample1d(3, 5, stride, filter_kernel_size=1)

    with torch.no_grad():
        fixed.conv.weight.copy_(baseline.conv.weight)
        fixed.conv.bias.copy_(baseline.conv.bias)

    x = torch.randn(2, 3, 37 * stride)
    got = fixed(x)
    want = baseline(x)

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max().item()


@pytest.mark.parametrize("stride", STRIDES)
def test_filtered_upsampler_preserves_shape_and_parameter_count(stride):
    """The replacement must be a true drop-in: same shape out, same params."""
    baseline = TransposedUpsample1d(3, 5, stride)
    fixed = BandlimitedUpsample1d(3, 5, stride)

    x = torch.randn(2, 3, 37)
    assert fixed(x).shape == baseline(x).shape

    n_baseline = sum(p.numel() for p in baseline.parameters())
    n_fixed = sum(p.numel() for p in fixed.parameters())
    assert n_fixed == n_baseline, "the filter must add no learnable parameters"


@pytest.mark.parametrize("stride", STRIDES)
def test_upsampler_weight_tensor_is_checkpoint_compatible(stride):
    """The learnable weight keeps ConvTranspose1d's (in, out, kernel) layout."""
    baseline = TransposedUpsample1d(3, 5, stride)
    fixed = BandlimitedUpsample1d(3, 5, stride)
    assert fixed.weight.shape == baseline.conv.weight.shape
    assert fixed.bias.shape == baseline.conv.bias.shape


# =============================================================================
# The defect itself
# =============================================================================


def test_transposed_conv_images_a_sine_and_the_filter_removes_it():
    """
    End-to-end statement of the problem in one test: upsample a band-limited
    sine by 8 through both arms with identical weights, and measure energy in
    the image region. The filtered arm must be far cleaner.

    Weights are set to a pure interpolation kernel (all channel pairs equal)
    so the measurement is about the operator, not about random weights.
    """
    stride = 8
    torch.manual_seed(0)

    baseline = TransposedUpsample1d(1, 1, stride)
    fixed = BandlimitedUpsample1d(1, 1, stride)
    with torch.no_grad():
        # Nearest-neighbour hold: the honest baseline interpolator that a
        # 2-tap-per-phase kernel can represent exactly.
        baseline.conv.weight.fill_(0.0)
        baseline.conv.weight[0, 0, :stride] = 1.0
        baseline.conv.bias.fill_(0.0)
        fixed.weight.copy_(baseline.conv.weight)
        fixed.bias.copy_(baseline.conv.bias)

    n = 2048
    t = torch.arange(n, dtype=torch.float32)
    x = torch.sin(2 * math.pi * 0.113 * t).view(1, 1, n)

    def image_to_signal_db(y: torch.Tensor) -> float:
        spec = torch.fft.rfft(y[0, 0] * torch.hann_window(y.shape[-1]))
        mag = spec.abs() ** 2
        freqs = torch.fft.rfftfreq(y.shape[-1])
        # Baseband occupies < 0.5/stride; everything above it is an image.
        signal = mag[freqs < 0.5 / stride].sum()
        image = mag[freqs > 0.5 / stride].sum()
        return 10 * math.log10((image / signal).item())

    isr_baseline = image_to_signal_db(baseline(x))
    isr_fixed = image_to_signal_db(fixed(x))

    assert isr_fixed < isr_baseline - 20.0, (
        f"filter bought only {isr_baseline - isr_fixed:.1f} dB "
        f"(baseline {isr_baseline:.1f}, fixed {isr_fixed:.1f})"
    )
