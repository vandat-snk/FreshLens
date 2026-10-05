"""Reproducibility controls for FreshLens CNN training."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed):
    """Apply the exact deterministic settings used by the legacy trainer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
