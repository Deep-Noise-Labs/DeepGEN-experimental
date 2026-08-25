"""
Inference pipeline for SynthGen audio generation.

Provides a simple interface for generating audio from text prompts
using a trained SynthGen model.
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from synthgen.model.synthgen import SynthGen
from synthgen.utils.audio import normalize_audio, save_audio

logger = logging.getLogger(__name__)


class SynthGenPipeline:
    """
    High-level inference pipeline for SynthGen.

    Handles model loading, text encoding, latent generation,
    and audio decoding in a single call.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint.
            device: Device to run inference on. None for auto-detect.
            dtype: Data type for inference (float32 or bfloat16).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.model.to(self.device, dtype=dtype)

        logger.info(f"Model loaded on {self.device} with dtype {dtype}")

    def _load_model(self, checkpoint_path: str) -> SynthGen:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config from checkpoint
        config = checkpoint.get("config", {})

        model = SynthGen(
            vae_latent_dim=config.get("vae_latent_dim", 64),
            vae_decoder_antialias=config.get("vae_decoder_antialias", True),
            dit_model_dim=config.get("dit_model_dim", 1024),
            dit_num_heads=config.get("dit_num_heads", 16),
            dit_num_layers=config.get("dit_num_layers", 20),
        )

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        num_steps: int = 25,
        cfg_scale: float = 3.5,
        seed: int | None = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio from a text prompt.

        Args:
            prompt: Text description of the desired sound.
            duration: Target duration in seconds (3-15).
            num_steps: Number of sampling steps (more = higher quality).
            cfg_scale: Classifier-free guidance scale (higher = more prompt adherence).
            seed: Random seed for reproducibility.
            normalize: Whether to normalize output audio.

        Returns:
            Tuple of (audio_tensor, sample_rate).
            Audio tensor shape: (channels, samples).
        """
        # Validate duration
        duration = max(3.0, min(15.0, duration))

        start_time = time.time()

        # Generate
        audio = self.model.generate(
            prompts=[prompt],
            duration=duration,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )

        # Remove batch dimension
        audio = audio[0].cpu().float().numpy()

        # Normalize
        if normalize:
            audio = normalize_audio(audio, target_db=-3.0)

        elapsed = time.time() - start_time
        rtf = elapsed / duration
        logger.info(
            f"Generated {duration:.1f}s audio in {elapsed:.2f}s "
            f"(RTF: {rtf:.3f}, {1/rtf:.1f}x real-time)"
        )

        return torch.from_numpy(audio), self.model.sample_rate

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        duration: float = 10.0,
        num_steps: int = 25,
        cfg_scale: float = 3.5,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio for multiple prompts in a batch.

        Args:
            prompts: List of text descriptions.
            duration: Target duration in seconds.
            num_steps: Number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed.

        Returns:
            Tuple of (audio_batch_tensor, sample_rate).
            Audio tensor shape: (batch, channels, samples).
        """
        duration = max(3.0, min(15.0, duration))

        audio = self.model.generate(
            prompts=prompts,
            duration=duration,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )

        return audio.cpu(), self.model.sample_rate


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate audio with SynthGen")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text description of the desired sound"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration in seconds (3-15)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--num-steps", type=int, default=25,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=3.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Log generation params and local output path to ClearML (no media upload)",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="synthgen",
        help="ClearML project name",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tracker = None
    if args.clearml:
        from types import SimpleNamespace

        from synthgen.tracking import build_tracker
        from synthgen.tracking.inference import log_generation_run

        tracker = build_tracker(
            SimpleNamespace(
                use_clearml=True,
                use_wandb=False,
                clearml_project=args.clearml_project,
                clearml_task_name="synthgen-generate",
                clearml_tags=["inference"],
                clearml_upload_checkpoints=False,
                stage="generate",
            ),
            rank=0,
        )

    try:
        # Initialize pipeline
        pipeline = SynthGenPipeline(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

        # Generate
        audio, sample_rate = pipeline.generate(
            prompt=args.prompt,
            duration=args.duration,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )

        # Save
        output_path = Path(args.output)
        save_audio(output_path, audio.numpy(), sample_rate)
        logger.info(f"Saved audio to: {output_path}")

        if tracker is not None:
            log_generation_run(
                tracker=tracker,
                prompt=args.prompt,
                duration=args.duration,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                checkpoint_path=args.checkpoint,
                output_path=str(output_path.resolve()),
            )
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
