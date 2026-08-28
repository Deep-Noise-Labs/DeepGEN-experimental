"""
Unit tests for the adversarial components of the VAE objective.
"""

import torch

from synthgen.model.discriminator import (
    MultiScaleSTFTDiscriminator,
    STFTDiscriminator,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)


def _disc(**kwargs):
    return MultiScaleSTFTDiscriminator(
        n_ffts=(256, 128), channels=8, num_layers=2, max_channels=32, **kwargs
    )


class TestSTFTDiscriminator:
    def test_forward_shapes_and_features(self):
        disc = STFTDiscriminator(
            n_fft=256, hop_length=64, win_length=256, channels=8, num_layers=2
        )
        audio = torch.randn(2, 2, 8192)
        logits, features = disc(audio)

        # Channels are folded into the batch: 2 items x 2 channels.
        assert logits.shape[0] == 4
        # One feature map per layer, including the input convolution.
        assert len(features) == 3
        assert all(f.shape[0] == 4 for f in features)

    def test_gradients_flow_to_audio(self):
        disc = STFTDiscriminator(
            n_fft=256, hop_length=64, win_length=256, channels=8, num_layers=2
        )
        audio = torch.randn(1, 2, 8192, requires_grad=True)
        logits, _ = disc(audio)
        logits.mean().backward()

        assert audio.grad is not None
        assert torch.isfinite(audio.grad).all()
        assert audio.grad.abs().sum() > 0

    def test_mono_input(self):
        disc = STFTDiscriminator(
            n_fft=256, hop_length=64, win_length=256, channels=8, num_layers=2
        )
        logits, _ = disc(torch.randn(3, 1, 8192))
        assert logits.shape[0] == 3


class TestMultiScaleSTFTDiscriminator:
    def test_one_output_per_scale(self):
        disc = _disc()
        logits, features = disc(torch.randn(2, 2, 8192))
        assert len(logits) == 2
        assert len(features) == 2


class TestAdversarialLosses:
    def test_discriminator_hinge_rewards_correct_separation(self):
        confident = [torch.full((2, 1, 4, 4), 3.0)]
        wrong = [torch.full((2, 1, 4, 4), -3.0)]

        # Real high / fake low is the correct separation.
        good = discriminator_hinge_loss(confident, wrong)
        bad = discriminator_hinge_loss(wrong, confident)
        assert good.item() < bad.item()
        assert good.item() == 0.0

    def test_generator_hinge_is_non_negative(self):
        loss = generator_hinge_loss([torch.randn(2, 1, 4, 4)])
        assert loss.item() >= 0

    def test_generator_hinge_falls_when_critic_is_fooled(self):
        fooled = [torch.full((2, 1, 4, 4), 2.0)]
        caught = [torch.full((2, 1, 4, 4), -2.0)]
        assert generator_hinge_loss(fooled).item() < generator_hinge_loss(caught).item()

    def test_feature_matching_zero_for_identical_features(self):
        features = [[torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 8)]]
        loss = feature_matching_loss(features, features)
        assert loss.item() < 1e-6

    def test_feature_matching_positive_for_different_features(self):
        real = [[torch.randn(2, 4, 8, 8)]]
        fake = [[torch.randn(2, 4, 8, 8)]]
        assert feature_matching_loss(real, fake).item() > 0

    def test_full_adversarial_round_trip(self):
        """A generator step and a critic step both produce finite gradients."""
        disc = _disc()
        real = torch.randn(2, 2, 8192)
        fake = torch.randn(2, 2, 8192, requires_grad=True)

        fake_logits, fake_features = disc(fake)
        with torch.no_grad():
            _, real_features = disc(real)

        gen_loss = generator_hinge_loss(fake_logits) + feature_matching_loss(
            real_features, fake_features
        )
        gen_loss.backward()
        assert torch.isfinite(fake.grad).all()

        disc.zero_grad(set_to_none=True)
        real_logits, _ = disc(real)
        detached_logits, _ = disc(fake.detach())
        disc_loss = discriminator_hinge_loss(real_logits, detached_logits)
        disc_loss.backward()

        grads = [p.grad for p in disc.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


class TestAdversarialVAEStep:
    def test_one_full_training_step(self):
        """
        Integration check mirroring what the trainer does each adversarial
        step: VAE forward, generator backward through the critic, then a critic
        update on the detached reconstruction.
        """
        from synthgen.model.vae import AudioVAE
        from synthgen.training.losses import VAELoss

        torch.manual_seed(0)
        vae = AudioVAE(
            in_channels=2,
            latent_dim=8,
            base_channels=4,
            strides=(4, 4, 4, 4),
            num_residual_per_block=1,
        )
        disc = _disc()
        loss_fn = VAELoss()

        gen_opt = torch.optim.AdamW(vae.parameters(), lr=1e-4)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-4)

        audio = torch.randn(1, 2, 8192) * 0.1
        reconstruction, target, mean, log_var = vae(audio)
        losses = loss_fn(reconstruction, target, mean, log_var, discriminator=disc)

        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() for p in vae.parameters()
        )
        gen_opt.step()

        # The generator backward leaves gradients on the critic; the trainer
        # clears them before the critic's own step. Verify that ordering works.
        disc_opt.zero_grad(set_to_none=True)
        real_logits, _ = disc(target)
        fake_logits, _ = disc(reconstruction.detach())
        disc_loss = discriminator_hinge_loss(real_logits, fake_logits)
        disc_loss.backward()
        disc_opt.step()

        assert torch.isfinite(disc_loss)

    def test_trainer_config_exposes_adversarial_controls(self):
        from synthgen.training.trainer import TrainingConfig

        config = TrainingConfig()
        assert config.vae_adversarial is True
        assert config.adv_start_step > 0
        assert config.disc_learning_rate > 0
        assert len(config.disc_n_ffts) > 1
