"""
SynthGen: Full model assembly.

Combines the Audio VAE, Text Encoder, and Diffusion Transformer into
a complete text-to-audio generation system using Conditional Flow Matching.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from synthgen.model.dit import DiffusionTransformer
from synthgen.model.text_encoder import T5TextEncoder, TextEncoderDummy
from synthgen.model.vae import AudioVAE


# =============================================================================
# Flow Matching Scheduler
# =============================================================================


class ConditionalFlowMatchingScheduler:
    """
    Conditional Flow Matching (CFM) scheduler.

    Implements the optimal transport path between noise and data:
        x_t = (1 - t) * noise + t * data

    The model learns to predict the velocity field:
        v = data - noise

    This results in straight interpolation paths, enabling efficient
    sampling with fewer steps compared to DDPM.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        uniform_mix_prob: float = 0.25,
    ):
        """
        Args:
            sigma_min: Minimum noise level for numerical stability.
            timestep_sampling: Training-time timestep distribution.
                - "logit_normal": t = sigmoid(n), n ~ N(logit_mean, logit_std),
                  mixed with a uniform floor (see uniform_mix_prob).
                  Concentrates training on intermediate timesteps, where the
                  velocity target is hardest to predict (SD3, arXiv:2403.03206).
                - "uniform": t ~ U(0, 1) (legacy behaviour).
            logit_mean: Mean of the underlying normal (logit-normal only).
            logit_std: Std of the underlying normal (logit-normal only).
            uniform_mix_prob: With logit-normal sampling, probability of drawing
                a uniform timestep instead. The floor guarantees coverage of the
                path extremes; pure logit-normal (prob 0.0) undersamples t near
                0 and 1 and measurably degrades the final denoising steps that
                clean decays and silences. 0.25 validated in-repo; see PR notes.
        """
        if timestep_sampling not in ("logit_normal", "uniform"):
            raise ValueError(
                f"Unknown timestep_sampling: {timestep_sampling!r} "
                "(expected 'logit_normal' or 'uniform')"
            )
        if not 0.0 <= uniform_mix_prob <= 1.0:
            raise ValueError(
                f"uniform_mix_prob must be in [0, 1], got {uniform_mix_prob}"
            )
        self.sigma_min = sigma_min
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.uniform_mix_prob = uniform_mix_prob

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample training timesteps from [0, 1].

        With logit-normal sampling, mid-range timesteps are drawn far more
        often than the near-noise/near-data extremes, which improves sample
        quality at equal training compute; the uniform floor keeps the
        extremes covered so decays and silences stay clean.
        """
        if self.timestep_sampling == "logit_normal":
            n = torch.randn(batch_size, device=device)
            t = torch.sigmoid(n * self.logit_std + self.logit_mean)
            if self.uniform_mix_prob > 0:
                t_uniform = torch.rand(batch_size, device=device)
                pick = torch.rand(batch_size, device=device) < self.uniform_mix_prob
                t = torch.where(pick, t_uniform, t)
            return t
        return torch.rand(batch_size, device=device)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate between noise and data at timestep t.

        Args:
            x_0: Clean data (latents).
            noise: Random noise.
            t: Timestep in [0, 1], shape (batch,).

        Returns:
            Noisy sample x_t.
        """
        t = t[:, None, None]  # Expand for broadcasting: (batch, 1, 1)
        sigma = self.sigma_min + (1 - self.sigma_min) * (1 - t)
        x_t = t * x_0 + sigma * noise
        return x_t

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity field.

        For optimal transport CFM: v = x_0 - noise
        """
        return x_0 - noise

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        noise: torch.Tensor,
        num_steps: int = 25,
        cfg_scale: float = 3.0,
        text_embeds: Optional[torch.Tensor] = None,
        text_embeds_uncond: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Euler ODE solver with classifier-free guidance.

        Args:
            model_fn: Function that takes (x_t, t, text_embeds, duration) and returns velocity.
            noise: Initial noise tensor.
            num_steps: Number of integration steps.
            cfg_scale: Classifier-free guidance scale.
            text_embeds: Conditional text embeddings.
            text_embeds_uncond: Unconditional text embeddings (for CFG).
            duration: Target duration conditioning.

        Returns:
            Generated latents.
        """
        dt = 1.0 / num_steps
        x_t = noise

        for i in range(num_steps):
            t = torch.full(
                (noise.shape[0],),
                i * dt,
                device=noise.device,
                dtype=noise.dtype,
            )

            # Classifier-free guidance
            if cfg_scale > 1.0 and text_embeds_uncond is not None:
                # Conditional prediction
                v_cond = model_fn(x_t, t, text_embeds, duration)
                # Unconditional prediction
                v_uncond = model_fn(x_t, t, text_embeds_uncond, duration)
                # Guided velocity
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model_fn(x_t, t, text_embeds, duration)

            # Euler step
            x_t = x_t + v * dt

        return x_t


# =============================================================================
# Full SynthGen Model
# =============================================================================


class SynthGen(nn.Module):
    """
    Complete SynthGen text-to-audio generation model.

    Combines:
    - Audio VAE for waveform compression/reconstruction
    - T5 text encoder for prompt understanding
    - DiT with flow matching for latent generation

    Training workflow:
    1. Encode audio to latents via VAE encoder
    2. Sample noise and timestep
    3. Create noisy latent via flow matching interpolation
    4. Predict velocity with DiT conditioned on text + timing
    5. Compute MSE loss between predicted and target velocity

    Inference workflow:
    1. Encode text prompt with T5
    2. Sample noise of appropriate length
    3. Iteratively denoise with Euler ODE solver
    4. Decode latents to waveform via VAE decoder
    """

    def __init__(
        self,
        # VAE parameters
        vae_latent_dim: int = 64,
        vae_base_channels: int = 64,
        vae_strides: tuple = (4, 4, 8, 8),
        # DiT parameters
        dit_model_dim: int = 1024,
        dit_num_heads: int = 16,
        dit_num_layers: int = 20,
        dit_mlp_ratio: float = 4.0,
        dit_cond_dim: int = 768,
        # Text encoder
        text_encoder_name: str = "t5-base",
        text_max_length: int = 256,
        # Audio parameters
        sample_rate: int = 44100,
        audio_channels: int = 2,
        # Training parameters
        cfg_dropout_prob: float = 0.1,
        timestep_sampling: str = "logit_normal",
        timestep_logit_mean: float = 0.0,
        timestep_logit_std: float = 1.0,
        timestep_uniform_mix_prob: float = 0.25,
        # Use dummy text encoder for testing
        use_dummy_text_encoder: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.cfg_dropout_prob = cfg_dropout_prob
        self.vae_latent_dim = vae_latent_dim

        # Audio VAE
        self.vae = AudioVAE(
            in_channels=audio_channels,
            latent_dim=vae_latent_dim,
            base_channels=vae_base_channels,
            strides=vae_strides,
        )

        # Text Encoder
        if use_dummy_text_encoder:
            self.text_encoder = TextEncoderDummy(
                output_dim=768,
                max_length=text_max_length,
            )
        else:
            self.text_encoder = T5TextEncoder(
                model_name=text_encoder_name,
                max_length=text_max_length,
                freeze=True,
            )

        # Diffusion Transformer
        self.dit = DiffusionTransformer(
            latent_dim=vae_latent_dim,
            model_dim=dit_model_dim,
            num_heads=dit_num_heads,
            num_layers=dit_num_layers,
            mlp_ratio=dit_mlp_ratio,
            cond_dim=dit_cond_dim,
        )

        # Flow Matching Scheduler
        self.scheduler = ConditionalFlowMatchingScheduler(
            timestep_sampling=timestep_sampling,
            logit_mean=timestep_logit_mean,
            logit_std=timestep_logit_std,
            uniform_mix_prob=timestep_uniform_mix_prob,
        )

    def compute_loss(
        self,
        audio: torch.Tensor,
        captions: list[str],
        durations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss for the DiT.

        Args:
            audio: Audio tensor (batch, channels, samples).
            captions: List of text captions.
            durations: Duration in seconds (batch,).

        Returns:
            Dictionary with 'loss' and optional auxiliary losses.
        """
        device = audio.device
        batch_size = audio.shape[0]

        # Encode audio to latents (no gradient through VAE)
        with torch.no_grad():
            latents = self.vae.encode_to_latent(audio)

        # Encode text
        text_embeds = self.text_encoder.encode(captions, device=device)

        # Classifier-free guidance dropout: randomly drop text conditioning
        if self.training and self.cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            if mask.any():
                # Replace dropped embeddings with zeros (unconditional)
                text_embeds[mask] = 0.0

        # Sample noise and timestep
        noise = torch.randn_like(latents)
        t = self.scheduler.sample_timestep(batch_size, device)

        # Create noisy latent
        x_t = self.scheduler.add_noise(latents, noise, t)

        # Predict velocity
        v_pred = self.dit(x_t, t, text_embeds, durations)

        # Target velocity
        v_target = self.scheduler.get_velocity(latents, noise)

        # MSE loss on velocity prediction
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        duration: float = 10.0,
        num_steps: int = 25,
        cfg_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate audio from text prompts.

        Args:
            prompts: List of text descriptions.
            duration: Target duration in seconds.
            num_steps: Number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Generated audio tensor (batch, channels, samples).
        """
        device = next(self.dit.parameters()).device
        batch_size = len(prompts)

        if seed is not None:
            torch.manual_seed(seed)

        # Compute latent sequence length from duration
        audio_samples = int(duration * self.sample_rate)
        latent_length = self.vae.get_latent_length(audio_samples)

        # Encode text (conditional)
        text_embeds = self.text_encoder.encode(prompts, device=device)

        # Unconditional embeddings (zeros)
        text_embeds_uncond = torch.zeros_like(text_embeds)

        # Duration conditioning
        dur_tensor = torch.full(
            (batch_size,), duration, device=device, dtype=torch.float32
        )

        # Sample initial noise
        noise = torch.randn(
            batch_size, self.vae_latent_dim, latent_length,
            device=device,
        )

        # Define model function for scheduler
        def model_fn(x_t, t, text_emb, dur):
            return self.dit(x_t, t, text_emb, dur)

        # Generate latents via flow matching
        latents = self.scheduler.sample(
            model_fn=model_fn,
            noise=noise,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            text_embeds=text_embeds,
            text_embeds_uncond=text_embeds_uncond,
            duration=dur_tensor,
        )

        # Decode latents to audio
        audio = self.vae.decode(latents)

        # Trim to exact target length
        target_samples = int(duration * self.sample_rate)
        audio = audio[..., :target_samples]

        return audio

    def get_parameter_count(self) -> dict[str, int]:
        """Get parameter counts for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "vae_total": count_params(self.vae),
            "vae_trainable": count_trainable(self.vae),
            "dit_total": count_params(self.dit),
            "dit_trainable": count_trainable(self.dit),
            "text_encoder_total": count_params(self.text_encoder),
            "text_encoder_trainable": count_trainable(self.text_encoder),
            "total": count_params(self),
            "trainable": count_trainable(self),
        }
