"""
Discriminators for adversarial training of the SynthGen Audio VAE.

Purely reconstructive objectives (L1 + magnitude STFT) are *phase blind*: the
loss cannot distinguish a crisp transient from a smeared one, nor a coherent
high band from a noisy one, as long as the magnitude spectrogram matches. That
is the dominant cause of the "watery" / "metallic" character that separates a
research-grade neural codec from a production sample library.

Every production-grade audio autoencoder (EnCodec, DAC, Stable Audio) therefore
trains its decoder against a discriminator that sees *complex* STFT frames, so
phase and inter-frame coherence become part of the objective.

This module provides:

- ``STFTDiscriminator``   — one complex-STFT resolution, 2-D convolutional.
- ``MultiScaleSTFTDiscriminator`` — a bank of the above across resolutions.
- ``discriminator_hinge_loss`` / ``generator_hinge_loss`` — hinge GAN losses.
- ``feature_matching_loss`` — L1 between intermediate discriminator features.

The discriminator is a *training-only* module: it is never exported to Triton
and never used at inference.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "STFTDiscriminator",
    "MultiScaleSTFTDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "feature_matching_loss",
]


def _weight_norm(module: nn.Module) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(module)


class STFTDiscriminator(nn.Module):
    """
    Discriminator operating on the complex STFT at a single resolution.

    Real and imaginary parts are stacked as two input channels so the network
    sees phase directly. Convolutions are 2-D over the (frequency, time) plane
    and stride only in frequency, which keeps full temporal resolution — the
    thing that matters for transients.

    Args:
        n_fft: FFT size for this resolution.
        hop_length: STFT hop.
        win_length: STFT window length.
        channels: Base channel count.
        num_layers: Number of strided conv layers.
        max_channels: Channel ceiling, to bound parameter count.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        channels: int = 32,
        num_layers: int = 4,
        max_channels: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        layers: list[nn.Module] = [
            _weight_norm(nn.Conv2d(2, channels, kernel_size=(7, 7), padding=(3, 3)))
        ]

        current = channels
        for _ in range(num_layers):
            out_channels = min(current * 2, max_channels)
            layers.append(
                _weight_norm(
                    nn.Conv2d(
                        current,
                        out_channels,
                        kernel_size=(5, 3),
                        # Stride in frequency only: keep time resolution intact
                        # so the critic can still see attack transients.
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                )
            )
            current = out_channels

        self.layers = nn.ModuleList(layers)
        self.output_conv = _weight_norm(
            nn.Conv2d(current, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Return the complex STFT as a real 2-channel image."""
        # x: (batch * channels, samples).
        # torch.stft has no reduced-precision kernel, and the trainer calls the
        # critic from inside an autocast block, so the input arrives as bf16.
        # Analyse in fp32 and let autocast re-cast for the convolutions.
        x = x.float()
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype, device=x.device),
            return_complex=True,
        )
        # (N, freq, time) complex -> (N, 2, freq, time) real
        return torch.stack([spec.real, spec.imag], dim=1)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            audio: (batch, channels, samples).

        Returns:
            ``(logits, features)`` where ``logits`` is the un-reduced critic map
            and ``features`` are the intermediate activations used for feature
            matching.
        """
        batch, channels, samples = audio.shape
        x = self._stft(audio.reshape(batch * channels, samples))

        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        logits = self.output_conv(x)
        return logits, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Bank of :class:`STFTDiscriminator` across several STFT resolutions.

    Short windows police transients; long windows police steady-state partials
    and the low end. Defaults follow the EnCodec / DAC resolution ladder,
    extended downwards with a 2048-sample window because SynthGen targets
    44.1 kHz material where sub-bass detail matters.
    """

    def __init__(
        self,
        n_ffts: tuple = (2048, 1024, 512, 256, 128),
        channels: int = 32,
        num_layers: int = 4,
        max_channels: int = 256,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTDiscriminator(
                    n_fft=n,
                    hop_length=n // 4,
                    win_length=n,
                    channels=channels,
                    num_layers=num_layers,
                    max_channels=max_channels,
                )
                for n in n_ffts
            ]
        )

    def forward(
        self, audio: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Args:
            audio: (batch, channels, samples).

        Returns:
            ``(logits_per_scale, features_per_scale)``.
        """
        all_logits: list[torch.Tensor] = []
        all_features: list[list[torch.Tensor]] = []
        for disc in self.discriminators:
            logits, features = disc(audio)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features


# =============================================================================
# Adversarial losses
# =============================================================================


def discriminator_hinge_loss(
    real_logits: list[torch.Tensor],
    fake_logits: list[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for the discriminator: push real above +1 and fake below -1.

    Hinge (rather than least-squares) is used because it saturates, which keeps
    the critic from overpowering the decoder early in training.
    """
    loss = real_logits[0].new_zeros(())
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / max(len(real_logits), 1)


def generator_hinge_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the generator (the VAE decoder)."""
    loss = fake_logits[0].new_zeros(())
    for fake in fake_logits:
        loss = loss + F.relu(1.0 - fake).mean()
    return loss / max(len(fake_logits), 1)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 distance between discriminator activations on real and reconstructed
    audio, normalised by the magnitude of the real activations.

    Feature matching is what stabilises adversarial autoencoder training: it
    gives the decoder a dense gradient even when the critic's scalar output has
    saturated.
    """
    if not real_features:
        return torch.zeros(())

    loss = real_features[0][0].new_zeros(())
    count = 0
    for real_scale, fake_scale in zip(real_features, fake_features):
        for real, fake in zip(real_scale, fake_scale):
            loss = loss + F.l1_loss(fake, real.detach()) / (
                real.detach().abs().mean() + 1e-5
            )
            count += 1
    return loss / max(count, 1)
