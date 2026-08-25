"""
Unit tests for SynthGen model architecture.

Tests verify tensor shapes, forward pass execution, and component integration
without requiring GPU or large model weights.
"""

import pytest
import torch

from synthgen.model.vae import AudioVAE, AudioEncoder, AudioDecoder, Snake
from synthgen.model.dit import (
    DiffusionTransformer,
    DiTBlock,
    RotaryPositionalEmbedding,
    TimestepEmbedding,
    TimingEmbedding,
)
from synthgen.model.text_encoder import TextEncoderDummy
from synthgen.model.synthgen import SynthGen, ConditionalFlowMatchingScheduler


# =============================================================================
# VAE Tests
# =============================================================================


class TestSnakeActivation:
    def test_output_shape(self):
        snake = Snake(channels=64)
        x = torch.randn(2, 64, 100)
        out = snake(x)
        assert out.shape == x.shape

    def test_non_zero_gradient(self):
        snake = Snake(channels=32)
        x = torch.randn(1, 32, 50, requires_grad=True)
        out = snake(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestAudioEncoder:
    def test_output_shape(self):
        encoder = AudioEncoder(in_channels=2, latent_dim=64)
        x = torch.randn(2, 2, 44100)  # 1 second of stereo audio at 44.1kHz
        mean, log_var = encoder(x)

        expected_length = 44100 // encoder.compression_ratio
        assert mean.shape == (2, 64, expected_length)
        assert log_var.shape == (2, 64, expected_length)

    def test_compression_ratio(self):
        encoder = AudioEncoder(strides=(4, 4, 8, 8))
        assert encoder.compression_ratio == 4 * 4 * 8 * 8  # 1024


class TestAudioDecoder:
    def test_output_shape(self):
        decoder = AudioDecoder(out_channels=2, latent_dim=64)
        z = torch.randn(2, 64, 43)  # Latent sequence
        out = decoder(z)
        assert out.shape[0] == 2
        assert out.shape[1] == 2
        # Output length should be approximately 43 * 1024
        assert out.shape[2] > 40000


class TestAudioVAE:
    @pytest.fixture
    def vae(self):
        return AudioVAE(
            in_channels=2,
            latent_dim=64,
            base_channels=32,  # Smaller for testing
            encoder_channel_multipliers=(1, 2, 4, 8),
            decoder_channel_multipliers=(8, 4, 2, 1),
            strides=(4, 4, 4, 4),  # 256x compression for faster test
        )

    def test_forward_pass(self, vae):
        x = torch.randn(2, 2, 8192)
        reconstruction, target, mean, log_var = vae(x)
        assert reconstruction.shape[0] == 2
        assert reconstruction.shape[1] == 2
        assert mean.shape[1] == 64
        assert log_var.shape[1] == 64

    def test_encode_to_latent(self, vae):
        x = torch.randn(2, 2, 8192)
        latent = vae.encode_to_latent(x)
        expected_length = 8192 // vae.compression_ratio
        assert latent.shape == (2, 64, expected_length)

    def test_get_latent_length(self, vae):
        audio_length = 44100 * 10  # 10 seconds
        latent_length = vae.get_latent_length(audio_length)
        assert latent_length == audio_length // vae.compression_ratio


# =============================================================================
# DiT Tests
# =============================================================================


class TestTimestepEmbedding:
    def test_output_shape(self):
        embed = TimestepEmbedding(dim=768)
        t = torch.rand(4)
        out = embed(t)
        assert out.shape == (4, 768)


class TestTimingEmbedding:
    def test_output_shape(self):
        embed = TimingEmbedding(dim=768)
        duration = torch.tensor([5.0, 10.0, 15.0])
        out = embed(duration)
        assert out.shape == (3, 768)


class TestRotaryPositionalEmbedding:
    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(dim=32, max_seq_len=512)
        x = torch.randn(2, 8, 100, 64)  # (batch, heads, seq, head_dim)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape[-1] == 32  # dim
        assert cos.shape[-2] >= 100  # seq_len


class TestDiTBlock:
    def test_forward_pass(self):
        block = DiTBlock(dim=256, num_heads=4, cond_dim=256)
        x = torch.randn(2, 50, 256)
        cond = torch.randn(2, 256)
        context = torch.randn(2, 20, 256)
        out = block(x, cond=cond, context=context)
        assert out.shape == (2, 50, 256)


class TestDiffusionTransformer:
    @pytest.fixture
    def dit(self):
        return DiffusionTransformer(
            latent_dim=32,
            model_dim=128,
            num_heads=4,
            num_layers=2,
            cond_dim=128,
            max_seq_len=256,
        )

    def test_forward_pass(self, dit):
        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, 32, seq_len)
        t = torch.rand(batch_size)
        text_embeds = torch.randn(batch_size, 20, 768)
        duration = torch.tensor([10.0, 5.0])

        out = dit(x, t, text_embeds, duration)
        assert out.shape == (batch_size, 32, seq_len)

    def test_output_matches_input_shape(self, dit):
        for seq_len in [10, 50, 100]:
            x = torch.randn(1, 32, seq_len)
            t = torch.rand(1)
            text_embeds = torch.randn(1, 20, 768)
            duration = torch.tensor([10.0])
            out = dit(x, t, text_embeds, duration)
            assert out.shape == x.shape


# =============================================================================
# Flow Matching Tests
# =============================================================================


class TestConditionalFlowMatchingScheduler:
    @pytest.fixture
    def scheduler(self):
        return ConditionalFlowMatchingScheduler()

    def test_sample_timestep(self, scheduler):
        t = scheduler.sample_timestep(batch_size=8, device=torch.device("cpu"))
        assert t.shape == (8,)
        assert (t >= 0).all() and (t <= 1).all()

    def test_add_noise(self, scheduler):
        x_0 = torch.randn(2, 64, 50)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0.0, 1.0])

        x_t = scheduler.add_noise(x_0, noise, t)
        assert x_t.shape == x_0.shape

    def test_get_velocity(self, scheduler):
        x_0 = torch.randn(2, 64, 50)
        noise = torch.randn_like(x_0)
        v = scheduler.get_velocity(x_0, noise)
        # v = x_0 - noise
        assert torch.allclose(v, x_0 - noise)


class TestTimestepSampling:
    def test_default_is_logit_normal(self):
        scheduler = ConditionalFlowMatchingScheduler()
        assert scheduler.timestep_sampling == "logit_normal"

    def test_logit_normal_range_and_shape(self):
        scheduler = ConditionalFlowMatchingScheduler(timestep_sampling="logit_normal")
        t = scheduler.sample_timestep(batch_size=4096, device=torch.device("cpu"))
        assert t.shape == (4096,)
        assert (t > 0).all() and (t < 1).all()

    def test_logit_normal_concentrates_mid_timesteps(self):
        # Pure logit-normal: P(0.25 < t < 0.75) ~ 0.73 vs 0.5 for uniform.
        torch.manual_seed(0)
        scheduler = ConditionalFlowMatchingScheduler(
            timestep_sampling="logit_normal", uniform_mix_prob=0.0
        )
        t = scheduler.sample_timestep(batch_size=20000, device=torch.device("cpu"))
        mid_fraction = ((t > 0.25) & (t < 0.75)).float().mean().item()
        assert mid_fraction > 0.65

    def test_default_mixture_keeps_tail_coverage(self):
        # The 25% uniform floor must keep the extremes covered: for the
        # default mixture, P(t > 0.9) ~ 0.25*0.10 + 0.75*0.014 ~ 0.035,
        # vs ~0.014 for pure logit-normal and 0.10 for uniform.
        torch.manual_seed(0)
        scheduler = ConditionalFlowMatchingScheduler()
        assert scheduler.uniform_mix_prob == 0.25
        t = scheduler.sample_timestep(batch_size=50000, device=torch.device("cpu"))
        tail_fraction = (t > 0.9).float().mean().item()
        assert 0.028 < tail_fraction < 0.045
        mid_fraction = ((t > 0.25) & (t < 0.75)).float().mean().item()
        assert mid_fraction > 0.6

    def test_invalid_mix_prob_rejected(self):
        with pytest.raises(ValueError):
            ConditionalFlowMatchingScheduler(uniform_mix_prob=1.5)

    def test_uniform_mode_preserved(self):
        torch.manual_seed(0)
        scheduler = ConditionalFlowMatchingScheduler(timestep_sampling="uniform")
        t = scheduler.sample_timestep(batch_size=20000, device=torch.device("cpu"))
        assert (t >= 0).all() and (t <= 1).all()
        mid_fraction = ((t > 0.25) & (t < 0.75)).float().mean().item()
        assert abs(mid_fraction - 0.5) < 0.02

    def test_logit_mean_shifts_distribution(self):
        torch.manual_seed(0)
        low = ConditionalFlowMatchingScheduler(logit_mean=-1.0)
        high = ConditionalFlowMatchingScheduler(logit_mean=1.0)
        t_low = low.sample_timestep(batch_size=20000, device=torch.device("cpu"))
        t_high = high.sample_timestep(batch_size=20000, device=torch.device("cpu"))
        assert t_low.mean().item() < 0.4 < 0.6 < t_high.mean().item()

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ConditionalFlowMatchingScheduler(timestep_sampling="cosine")

    def test_synthgen_passes_through_sampling_config(self):
        model = SynthGen(
            vae_latent_dim=32,
            vae_base_channels=16,
            vae_strides=(4, 4, 4, 4),
            dit_model_dim=128,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_cond_dim=128,
            timestep_sampling="uniform",
            use_dummy_text_encoder=True,
        )
        assert model.scheduler.timestep_sampling == "uniform"


# =============================================================================
# Full Model Tests
# =============================================================================


class TestSynthGen:
    @pytest.fixture
    def model(self):
        return SynthGen(
            vae_latent_dim=32,
            vae_base_channels=16,
            vae_strides=(4, 4, 4, 4),  # 256x compression
            dit_model_dim=128,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_cond_dim=128,
            use_dummy_text_encoder=True,
        )

    def test_compute_loss(self, model):
        audio = torch.randn(2, 2, 8192)
        captions = ["warm pad sound", "bright lead synth"]
        durations = torch.tensor([10.0, 5.0])

        losses = model.compute_loss(audio, captions, durations)
        assert "loss" in losses
        assert losses["loss"].ndim == 0  # Scalar
        assert losses["loss"].item() > 0

    def test_generate(self, model):
        audio = model.generate(
            prompts=["test sound"],
            duration=3.0,
            num_steps=5,
            cfg_scale=1.0,
            seed=42,
        )
        assert audio.shape[0] == 1  # batch
        assert audio.shape[1] == 2  # stereo

    def test_parameter_count(self, model):
        counts = model.get_parameter_count()
        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["trainable"] <= counts["total"]


class TestTextEncoderDummy:
    def test_output_shape(self):
        encoder = TextEncoderDummy(output_dim=768, max_length=256)
        texts = ["hello world", "test prompt"]
        out = encoder.encode(texts)
        assert out.shape == (2, 256, 768)
