"""
Learning rate schedulers for SynthGen training.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_steps
    2. Cosine decay from base_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class WarmupConstantScheduler(_LRScheduler):
    """
    Constant learning rate with linear warmup.

    Useful for fine-tuning or when combined with gradient accumulation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            scale = 1.0

        return [base_lr * scale for base_lr in self.base_lrs]
