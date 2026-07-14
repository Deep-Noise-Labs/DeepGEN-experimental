"""
Text Encoder for SynthGen.

Wraps a frozen T5-base encoder to produce text conditioning embeddings
for the Diffusion Transformer. T5-base provides 768-dimensional embeddings
and has been validated across multiple audio generation systems.
"""

from typing import Optional

import torch
import torch.nn as nn


class T5TextEncoder(nn.Module):
    """
    Frozen T5-base text encoder for conditioning the DiT.

    Produces contextualized text embeddings from natural language prompts
    describing the desired audio characteristics.

    The encoder is kept frozen during training to preserve its language
    understanding capabilities and reduce memory requirements.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        max_length: int = 256,
        freeze: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for T5.
            max_length: Maximum token sequence length.
            freeze: Whether to freeze encoder weights.
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = 768  # T5-base hidden size

        # Lazy initialization to avoid importing transformers at module level
        self._encoder = None
        self._tokenizer = None
        self._freeze = freeze

    def _init_model(self):
        """Initialize T5 model and tokenizer (lazy loading)."""
        if self._encoder is not None:
            return

        from transformers import T5EncoderModel, T5Tokenizer

        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_length,
        )
        self._encoder = T5EncoderModel.from_pretrained(self.model_name)

        if self._freeze:
            self._encoder.eval()
            for param in self._encoder.parameters():
                param.requires_grad = False

    @property
    def tokenizer(self):
        self._init_model()
        return self._tokenizer

    @property
    def encoder(self):
        self._init_model()
        return self._encoder

    def forward(
        self,
        text: list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode text prompts to embeddings.

        Args:
            text: List of text prompts (batch_size strings).
            device: Target device for output tensors.

        Returns:
            Text embeddings of shape (batch_size, seq_len, 768).
        """
        self._init_model()

        if device is None:
            # Prefer the parent module device when T5 was lazy-loaded onto CPU
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = next(self.encoder.parameters()).device

        # Lazy-loaded T5 starts on CPU; move once to the request/parent device
        enc_device = next(self.encoder.parameters()).device
        if enc_device != device:
            self._encoder = self._encoder.to(device)

        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Encode
        with torch.no_grad() if self._freeze else torch.enable_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Return last hidden state
        return outputs.last_hidden_state

    def encode(self, text: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Alias for forward() for cleaner API."""
        return self.forward(text, device=device)


class TextEncoderDummy(nn.Module):
    """
    Dummy text encoder for testing without T5 dependency.

    Produces random embeddings of the correct shape for architecture testing.
    """

    def __init__(self, output_dim: int = 768, max_length: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length

    def forward(
        self,
        text: list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        batch_size = len(text)
        if device is None:
            device = torch.device("cpu")
        return torch.randn(batch_size, self.max_length, self.output_dim, device=device)

    def encode(self, text: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        return self.forward(text, device=device)
