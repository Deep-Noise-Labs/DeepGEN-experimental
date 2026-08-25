"""Tests for the anti-aliased Snake activation and its VAE integration."""

import torch

from synthgen.model.antialias import AntiAliasedSnake, DownSample1d, UpSample1d
from synthgen.model.vae import AudioDecoder, AudioVAE, Snake


def test_upsample_downsample_lengths():
    x = torch.randn(2, 4, 1000)
    up = UpSample1d(ratio=2)
    down = DownSample1d(ratio=2)
    y = up(x)
    assert y.shape == (2, 4, 2000)
    z = down(y)
    assert z.shape == x.shape


def test_antialiased_snake_preserves_shape():
    act = AntiAliasedSnake(channels=8)
    for length in (256, 1000, 4410, 44100):
        x = torch.randn(1, 8, length)
        y = act(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_antialiased_snake_state_dict_matches_snake():
    """Checkpoints must transfer between the two activation variants."""
    plain = Snake(channels=16)
    aa = AntiAliasedSnake(channels=16)

    assert set(plain.state_dict().keys()) == set(aa.state_dict().keys()) == {"alpha"}

    # Load in both directions
    aa.load_state_dict(plain.state_dict())
    plain.load_state_dict(aa.state_dict())


def test_decoder_state_dict_compatible_across_antialias_flag():
    kwargs = dict(
        out_channels=2,
        latent_dim=8,
        base_channels=8,
        channel_multipliers=(4, 2, 1, 1),
        strides=(4, 4, 2, 2),
        num_residual_per_block=1,
    )
    dec_old = AudioDecoder(antialias=False, **kwargs)
    dec_new = AudioDecoder(antialias=True, **kwargs)
    dec_new.load_state_dict(dec_old.state_dict())
    dec_old.load_state_dict(dec_new.state_dict())


def test_vae_forward_with_antialias():
    vae = AudioVAE(
        in_channels=2,
        latent_dim=8,
        base_channels=8,
        strides=(2, 2, 4, 4),
        num_residual_per_block=1,
        decoder_antialias=True,
    )
    x = torch.randn(1, 2, 4096)
    recon, target, mean, log_var = vae(x)
    assert recon.shape == target.shape
    assert torch.isfinite(recon).all()

    # Gradients flow through the oversampled activation
    loss = recon.abs().mean()
    loss.backward()
    grads = [p.grad for p in vae.decoder.parameters() if p.grad is not None]
    assert len(grads) > 0


def _aliased_energy_db(y: torch.Tensor, f0_bin: int, n: int) -> float:
    """Energy (dB) at the folded images of the harmonics of ``f0_bin``."""
    window = torch.hann_window(n)
    spectrum = torch.fft.rfft(y * window).abs()
    total = 0.0
    for k in (2, 3, 4, 6):
        harmonic = k * f0_bin
        folded = harmonic % (2 * (n // 2))
        if folded > n // 2:
            folded = 2 * (n // 2) - folded
        # Only count bins that are actual fold-backs (above-Nyquist harmonics)
        if harmonic > n // 2:
            total += spectrum[folded - 2 : folded + 3].pow(2).sum().item()
    return 10 * torch.log10(torch.tensor(total + 1e-12)).item()


def test_antialiased_snake_reduces_aliasing():
    """
    A sine whose Snake harmonics land above Nyquist must produce less
    fold-back energy through AntiAliasedSnake than through plain Snake.
    """
    n = 8192
    f0_bin = 2867  # ~0.35 * fs, so 2*f0 and above alias
    t = torch.arange(n, dtype=torch.float64)
    x = 0.8 * torch.sin(2 * torch.pi * f0_bin * t / n)
    x = x.float().view(1, 1, n)

    with torch.no_grad():
        y_plain = Snake(1)(x)
        y_aa = AntiAliasedSnake(1)(x)

    e_plain = _aliased_energy_db(y_plain.view(-1), f0_bin, n)
    e_aa = _aliased_energy_db(y_aa.view(-1), f0_bin, n)

    # Expect a clearly audible reduction (>10 dB) in aliased energy
    assert e_plain - e_aa > 10.0, f"plain={e_plain:.1f} dB, aa={e_aa:.1f} dB"
