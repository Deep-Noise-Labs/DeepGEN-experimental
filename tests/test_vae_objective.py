"""
Tests for the perceptual / adversarial VAE objective.

The behavioural tests below are regression tests for the *reason* this
objective exists: the previous one could not hear phase, could not hear stereo
width, and spent its loss budget on bass. Each test states the property in
terms of what a listener would say, then asserts the loss agrees.
"""

import math

import pytest
import torch

from synthgen.model.discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionSTFTDiscriminator,
    to_mid_side,
)
from synthgen.model.vae import AudioVAE, Snake
from synthgen.training.losses import (
    DiscriminatorAdversarialLoss,
    FeatureMatchingLoss,
    GeneratorAdversarialLoss,
    MultiResolutionSTFTLoss,
    MultiScaleMelSpectrogramLoss,
    VAELoss,
    mel_filterbank,
)

SR = 44100


# =============================================================================
# Fixtures / signal helpers
# =============================================================================


def _pluck(n: int = 16384, f0: float = 220.0, sr: int = SR) -> torch.Tensor:
    """A decaying harmonic pluck: sharp attack, several partials, stereo."""
    t = torch.arange(n, dtype=torch.float32) / sr
    env = torch.exp(-t * 12.0)
    sig = sum((1.0 / k) * torch.sin(2 * math.pi * f0 * k * t) for k in range(1, 12))
    left = sig * env
    right = sig * env * 0.8 + torch.roll(sig * env, 37) * 0.2
    return torch.stack([left, right]).unsqueeze(0) * 0.3


def _all_pass_smear(x: torch.Tensor, strength: float = 600.0) -> torch.Tensor:
    """
    Apply a frequency-dependent group delay with a pure all-pass filter.

    |H(f)| == 1 for every f, so the magnitude spectrum is mathematically
    unchanged and only phase moves. Audibly this turns a pluck into a smear.
    """
    n = x.shape[-1]
    nfft = 1 << int(math.ceil(math.log2(n * 2)))
    freqs = torch.fft.rfftfreq(nfft, 1 / SR)
    w = freqs / (SR / 2)
    phase = -strength * (w ** 2) * math.pi
    h = torch.polar(torch.ones_like(phase), phase)
    spec = torch.fft.rfft(x, n=nfft, dim=-1)
    return torch.fft.irfft(spec * h, n=nfft, dim=-1)[..., :n]


def _collapse_stereo(x: torch.Tensor, side_gain: float = 0.15) -> torch.Tensor:
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * side_gain
    return torch.stack([mid + side, mid - side], dim=1)


def _rms_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Level-match so loudness is never the thing being scored."""
    return x * (ref.pow(2).mean().sqrt() / (x.pow(2).mean().sqrt() + 1e-12))


# =============================================================================
# Mel filterbank
# =============================================================================


def test_mel_filterbank_shape_and_support():
    fb = mel_filterbank(n_fft=2048, n_mels=64, sample_rate=SR)
    assert fb.shape == (64, 1025)
    assert torch.all(fb >= 0)
    # every band must have some support, or it contributes no gradient
    assert torch.all(fb.sum(dim=1) > 0)


def test_mel_filterbank_is_ordered_low_to_high():
    fb = mel_filterbank(n_fft=1024, n_mels=32, sample_rate=SR)
    centres = (fb * torch.arange(fb.shape[1], dtype=fb.dtype)).sum(1) / fb.sum(1)
    assert torch.all(centres[1:] > centres[:-1])


# =============================================================================
# Mel loss behaviour
# =============================================================================


def test_mel_loss_zero_on_identical_input():
    loss = MultiScaleMelSpectrogramLoss(sample_rate=SR)
    x = _pluck()
    assert float(loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_mel_loss_penalises_stereo_collapse_more_than_lr_loss():
    """
    A collapsed stereo image is a real, audible defect. Measured on L/R it is
    nearly free, because both channels move towards each other and each stays
    close to its target. Measured on mid/side it is expensive.
    """
    ref = _pluck()
    collapsed = _rms_match(_collapse_stereo(ref), ref)

    ms = MultiScaleMelSpectrogramLoss(sample_rate=SR, mid_side=True)
    lr = MultiScaleMelSpectrogramLoss(sample_rate=SR, mid_side=False)

    ms_penalty = float(ms(collapsed, ref))
    lr_penalty = float(lr(collapsed, ref))
    # ~3x on this deliberately narrow test signal (side/mid ~= 0.14); the gap
    # widens with source width - ~5x measured on real wide programme material.
    assert ms_penalty > 2.5 * lr_penalty, (
        f"mid/side penalty {ms_penalty:.5f} should dominate L/R {lr_penalty:.5f}"
    )


def test_mel_loss_is_more_sensitive_to_phase_smear_than_legacy_mrstft():
    """
    The regression this whole objective exists for.

    Compare two defects against the same reference:
      A) a 9 kHz-ish loss of top end        - audible, but mild
      B) an all-pass transient smear        - far more audible, and leaves the
                                              magnitude spectrum untouched

    Both losses are scale-free only within themselves, so compare each
    objective's own ratio B/A. The legacy magnitude objective under-rates the
    smear; the mel objective rates it much closer to its true cost.
    """
    ref = _pluck()

    smeared = _rms_match(_all_pass_smear(ref), ref)

    # simple, smooth top-end loss via a moving-average low-pass
    kernel = torch.ones(1, 1, 9) / 9
    dull = torch.nn.functional.conv1d(
        ref.reshape(-1, 1, ref.shape[-1]), kernel, padding=4
    ).reshape(ref.shape)
    dull = _rms_match(dull, ref)

    legacy = MultiResolutionSTFTLoss()
    mel = MultiScaleMelSpectrogramLoss(sample_rate=SR, mid_side=True)

    legacy_ratio = float(legacy(smeared, ref)) / float(legacy(dull, ref))
    mel_ratio = float(mel(smeared, ref)) / float(mel(dull, ref))

    assert mel_ratio > legacy_ratio, (
        f"mel objective should rank the transient smear relatively higher than "
        f"the legacy magnitude objective (mel {mel_ratio:.3f} vs "
        f"legacy {legacy_ratio:.3f})"
    )


def test_mel_loss_rejects_more_bands_than_bins():
    with pytest.raises(ValueError):
        MultiScaleMelSpectrogramLoss(window_lengths=(32,), n_mels=(64,), sample_rate=SR)


# =============================================================================
# to_mid_side
# =============================================================================


def test_to_mid_side_round_trips():
    x = _pluck()
    ms = to_mid_side(x)
    left = ms[:, 0] + ms[:, 1]
    right = ms[:, 0] - ms[:, 1]
    assert torch.allclose(torch.stack([left, right], 1), x, atol=1e-6)


def test_to_mid_side_passes_through_non_stereo():
    x = torch.randn(1, 1, 512)
    assert torch.equal(to_mid_side(x), x)


# =============================================================================
# Snake activation
# =============================================================================


def test_snake_stays_finite_for_any_parameter_value():
    """
    The previous Snake divided by ``alpha + 1e-8`` with alpha unconstrained: a
    run whose alpha drifted through zero hit a ~1e8 gain and NaN'd. Log-space
    alpha/beta cannot reach zero.
    """
    snake = Snake(channels=4)
    with torch.no_grad():
        snake.log_alpha.fill_(-30.0)   # alpha ~ 9e-14
        snake.log_beta.fill_(-30.0)
    out = snake(torch.randn(2, 4, 256))
    assert torch.isfinite(out).all()

    with torch.no_grad():
        snake.log_alpha.fill_(5.0)     # alpha ~ 148
        snake.log_beta.fill_(5.0)
    assert torch.isfinite(snake(torch.randn(2, 4, 256))).all()


def test_snake_gradients_are_finite():
    snake = Snake(channels=4)
    x = torch.randn(2, 4, 256, requires_grad=True)
    snake(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(snake.log_alpha.grad).all()
    assert torch.isfinite(snake.log_beta.grad).all()


def test_snake_upgrades_legacy_alpha_checkpoints():
    """Old checkpoints stored a raw ``alpha``; loading one must still work."""
    snake = Snake(channels=3)
    legacy = {"alpha": torch.tensor([2.0, 1.0, 0.5]).reshape(1, 3, 1)}
    missing, unexpected = snake.load_state_dict(legacy, strict=False)
    assert "alpha" not in unexpected
    assert torch.allclose(
        snake.log_alpha.exp().flatten(), torch.tensor([2.0, 1.0, 0.5]), atol=1e-5
    )


def test_snake_upgrade_clamps_non_positive_alpha():
    """A legacy alpha at or below zero was already broken; it must not become NaN."""
    snake = Snake(channels=2)
    snake.load_state_dict({"alpha": torch.tensor([0.0, -3.0]).reshape(1, 2, 1)}, strict=False)
    assert torch.isfinite(snake(torch.randn(1, 2, 128))).all()


# =============================================================================
# Discriminators
# =============================================================================


@pytest.mark.parametrize("channels", [1, 2])
def test_combined_discriminator_shapes(channels):
    disc = CombinedDiscriminator(
        mpd_channels=(8, 16), stft_channels=8,
        stft_resolutions=((512, 128), (128, 32)),
    )
    logits, features = disc(torch.randn(2, channels, 8192) * 0.1)
    assert len(logits) == len(disc.mpd.discriminators) + len(disc.mrd.discriminators)
    assert len(features) == len(logits)
    for logit in logits:
        assert logit.dim() == 2
        assert torch.isfinite(logit).all()
    for maps in features:
        assert len(maps) > 0


def test_discriminator_handles_lengths_not_divisible_by_period():
    mpd = MultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11), channels=(8, 16))
    logits, _ = mpd(torch.randn(1, 2, 4099) * 0.1)  # prime-ish length
    assert all(torch.isfinite(logit).all() for logit in logits)


def test_stft_discriminator_is_phase_sensitive():
    """
    The point of feeding real/imag rather than magnitude: an all-pass smear must
    change the discriminator's view of the signal, even though the magnitude
    spectrum is unchanged.
    """
    torch.manual_seed(0)
    mrd = MultiResolutionSTFTDiscriminator(
        resolutions=((512, 128),), channels=8
    ).eval()
    ref = _pluck()
    smeared = _all_pass_smear(ref)

    with torch.no_grad():
        a, _ = mrd(ref)
        b, _ = mrd(smeared)
    rel = (a[0] - b[0]).abs().mean() / (a[0].abs().mean() + 1e-8)
    assert rel > 1e-3, f"discriminator response barely moved ({rel:.2e})"


def test_discriminator_gradients_reach_the_input():
    disc = CombinedDiscriminator(
        mpd_channels=(8,), stft_channels=8, stft_resolutions=((256, 64),)
    )
    x = (torch.randn(1, 2, 4096) * 0.1).requires_grad_(True)
    logits, _ = disc(x)
    sum(logit.sum() for logit in logits).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# =============================================================================
# Adversarial losses
# =============================================================================


@pytest.mark.parametrize("mode", ["hinge", "lsgan"])
def test_discriminator_loss_prefers_correct_classification(mode):
    loss = DiscriminatorAdversarialLoss(mode=mode)
    confident = loss([torch.full((2, 4), 3.0)], [torch.full((2, 4), -3.0)])
    confused = loss([torch.full((2, 4), -3.0)], [torch.full((2, 4), 3.0)])
    assert float(confident) < float(confused)


@pytest.mark.parametrize("mode", ["hinge", "lsgan"])
def test_generator_loss_prefers_fooling_the_discriminator(mode):
    loss = GeneratorAdversarialLoss(mode=mode)
    fooled = loss([torch.full((2, 4), 1.0)])
    caught = loss([torch.full((2, 4), -1.0)])
    assert float(fooled) < float(caught)


def test_feature_matching_zero_when_features_match():
    feats = [[torch.randn(2, 4, 8), torch.randn(2, 4, 8)]]
    assert float(FeatureMatchingLoss()(feats, feats)) == pytest.approx(0.0, abs=1e-7)


def test_feature_matching_does_not_backprop_into_real_features():
    real = [[torch.randn(2, 4, 8, requires_grad=True)]]
    fake = [[torch.randn(2, 4, 8, requires_grad=True)]]
    FeatureMatchingLoss()(real, fake).backward()
    assert real[0][0].grad is None
    assert fake[0][0].grad is not None


def test_unknown_adversarial_mode_is_rejected():
    with pytest.raises(ValueError):
        GeneratorAdversarialLoss(mode="wasserstein")
    with pytest.raises(ValueError):
        DiscriminatorAdversarialLoss(mode="wasserstein")


# =============================================================================
# VAELoss wiring
# =============================================================================


def _vae_batch():
    torch.manual_seed(0)
    vae = AudioVAE(latent_dim=8, base_channels=8, strides=(4, 4, 4, 4))
    audio = _pluck(n=8192)
    return vae, vae(audio)


def test_vae_loss_without_discriminator_skips_adversarial_terms():
    _, (recon, target, mean, log_var) = _vae_batch()
    out = VAELoss(sample_rate=SR)(recon, target, mean, log_var)
    assert "mel_loss" in out
    assert "adv_loss" not in out and "fm_loss" not in out
    assert torch.isfinite(out["loss"])


def test_vae_loss_includes_adversarial_terms_when_given_logits():
    _, (recon, target, mean, log_var) = _vae_batch()
    disc = CombinedDiscriminator(
        mpd_channels=(8,), stft_channels=8, stft_resolutions=((256, 64),)
    )
    fake_logits, fake_features = disc(recon)
    _, real_features = disc(target)
    out = VAELoss(sample_rate=SR)(
        recon, target, mean, log_var,
        fake_logits=fake_logits,
        real_features=real_features,
        fake_features=fake_features,
    )
    assert "adv_loss" in out and "fm_loss" in out
    assert torch.isfinite(out["loss"])


def test_legacy_flag_reproduces_the_previous_objective():
    """`legacy=True` must be byte-for-byte the old objective, for A/B runs."""
    _, (recon, target, mean, log_var) = _vae_batch()
    out = VAELoss(legacy=True)(recon, target, mean, log_var)
    assert set(out) == {"loss", "l1_loss", "spectral_loss", "kl_loss"}

    expected = (
        0.1 * torch.nn.functional.l1_loss(recon, target)
        + 1.0 * MultiResolutionSTFTLoss()(recon, target)
        + 1e-4 * (-0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp()))
    )
    assert float(out["loss"]) == pytest.approx(float(expected), rel=1e-5)


def test_vae_loss_backward_reaches_the_model():
    vae, (recon, target, mean, log_var) = _vae_batch()
    VAELoss(sample_rate=SR)(recon, target, mean, log_var)["loss"].backward()
    grads = [p.grad for p in vae.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
