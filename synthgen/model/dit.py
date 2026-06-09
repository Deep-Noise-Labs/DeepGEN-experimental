"""
Diffusion Transformer (DiT) for SynthGen.

A transformer-based generative model that operates in the latent space
of the Audio VAE. Uses Conditional Flow Matching for training and
generates audio latents conditioned on text, timing, and diffusion timestep.

Architecture features:
- Rotary Positional Embeddings (RoPE) for sequence position encoding
- Cross-attention for text conditioning
- Adaptive Layer Norm (adaLN) for timestep conditioning
- Gated MLP blocks
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Positional Encoding
# =============================================================================


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Applied to half of the head dimension for key and query tensors,
    following the approach used in Stable Audio.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, :x.shape[-1]],
            self.sin_cached[:, :, :seq_len, :x.shape[-1]],
        )


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # Split into two halves
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[..., :x1.shape[-1]]
    sin = sin[..., :x1.shape[-1]]

    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)
    return rotated


# =============================================================================
# Timestep Embedding
# =============================================================================


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by MLP projection.

    Maps scalar timestep to a high-dimensional embedding vector.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (batch,) with values in [0, 1].

        Returns:
            Embedding of shape (batch, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)


class TimingEmbedding(nn.Module):
    """
    Embedding for audio duration/timing conditioning.

    Encodes the target duration in seconds as a conditioning signal.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, duration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            duration: Duration in seconds, shape (batch,) or (batch, 1).

        Returns:
            Embedding of shape (batch, dim).
        """
        if duration.ndim == 1:
            duration = duration.unsqueeze(-1)
        return self.mlp(duration)


# =============================================================================
# Attention and MLP
# =============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and cross-attention support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim if not is_cross_attention else dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim if not is_cross_attention else dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            context: Optional context for cross-attention (batch, ctx_len, dim).
            rope_cos, rope_sin: Rotary embeddings (only for self-attention).
        """
        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        kv_input = context if (self.is_cross_attention and context is not None) else x
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to half of head dim (self-attention only)
        if rope_cos is not None and rope_sin is not None and not self.is_cross_attention:
            rope_dim = self.head_dim // 2
            q_rope, q_pass = q[..., :rope_dim * 2], q[..., rope_dim * 2:]
            k_rope, k_pass = k[..., :rope_dim * 2], k[..., rope_dim * 2:]

            q_rope = apply_rotary_emb(q_rope, rope_cos, rope_sin)
            k_rope = apply_rotary_emb(k_rope, rope_cos, rope_sin)

            q = torch.cat([q_rope, q_pass], dim=-1)
            k = torch.cat([k_rope, k_pass], dim=-1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        return self.to_out(attn_output)


class GatedMLP(nn.Module):
    """Gated MLP block (SwiGLU variant)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


# =============================================================================
# Adaptive Layer Norm (adaLN)
# =============================================================================


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embedding.

    Modulates the normalized output with learned scale and shift
    parameters derived from the conditioning signal.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

        # Initialize to identity transform
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            cond: Conditioning tensor of shape (batch, cond_dim).
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# Transformer Block
# =============================================================================


class DiTBlock(nn.Module):
    """
    Single Diffusion Transformer block.

    Consists of:
    1. Self-attention with adaLN and RoPE
    2. Cross-attention for text conditioning
    3. Gated MLP with adaLN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        cond_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = AdaptiveLayerNorm(dim, cond_dim)
        self.self_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            is_cross_attention=True,
        )

        # MLP
        self.norm3 = AdaptiveLayerNorm(dim, cond_dim)
        self.mlp = GatedMLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim).
            cond: Timestep conditioning (batch, cond_dim).
            context: Text conditioning (batch, ctx_len, dim).
            rope_cos, rope_sin: Rotary embeddings.
        """
        # Self-attention with adaLN
        residual = x
        x = self.norm1(x, cond)
        x = self.self_attn(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, context=context)
        x = residual + x

        # MLP with adaLN
        residual = x
        x = self.norm3(x, cond)
        x = self.mlp(x)
        x = residual + x

        return x


# =============================================================================
# Full DiT Model
# =============================================================================


class DiffusionTransformer(nn.Module):
    """
    Full Diffusion Transformer for audio latent generation.

    Generates latent representations conditioned on:
    - Text prompt (via cross-attention with T5 embeddings)
    - Duration/timing (via prepend conditioning)
    - Diffusion timestep (via adaLN modulation)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        model_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 20,
        mlp_ratio: float = 4.0,
        cond_dim: int = 768,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        """
        Args:
            latent_dim: Dimension of input/output latents from VAE.
            model_dim: Internal transformer dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            mlp_ratio: MLP hidden dimension multiplier.
            cond_dim: Conditioning dimension (timestep + timing).
            max_seq_len: Maximum sequence length for RoPE.
            dropout: Dropout rate.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_layers = num_layers

        # Input projection: latent_dim -> model_dim
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # Output projection: model_dim -> latent_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, latent_dim),
        )

        # Timestep embedding
        self.timestep_embed = TimestepEmbedding(cond_dim)

        # Timing/duration embedding
        self.timing_embed = TimingEmbedding(cond_dim)

        # Combined conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Text conditioning projection (T5 output dim -> model_dim)
        self.text_proj = nn.Linear(768, model_dim)  # T5-base output is 768

        # Rotary positional embeddings
        self.rope = RotaryPositionalEmbedding(
            dim=model_dim // num_heads // 2,  # Applied to half of head dim
            max_seq_len=max_seq_len,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-initialize output projection for residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the DiT.

        Args:
            x: Noisy latent tensor of shape (batch, latent_dim, seq_len).
            t: Diffusion timestep of shape (batch,), values in [0, 1].
            text_embeds: Text encoder output of shape (batch, text_len, 768).
            duration: Target duration in seconds of shape (batch,).

        Returns:
            Predicted velocity field of shape (batch, latent_dim, seq_len).
        """
        batch_size, _, seq_len = x.shape

        # Transpose to (batch, seq_len, latent_dim) for transformer
        x = x.transpose(1, 2)

        # Project to model dimension
        x = self.input_proj(x)

        # Compute conditioning
        t_embed = self.timestep_embed(t)
        dur_embed = self.timing_embed(duration)
        cond = self.cond_proj(torch.cat([t_embed, dur_embed], dim=-1))

        # Project text embeddings
        context = self.text_proj(text_embeds)

        # Get rotary embeddings
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond=cond, context=context, rope_cos=rope_cos, rope_sin=rope_sin)

        # Project back to latent dimension
        x = self.output_proj(x)

        # Transpose back to (batch, latent_dim, seq_len)
        x = x.transpose(1, 2)

        return x
