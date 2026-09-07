"""Shared utilities for integrating AFP with a VLA policy.

Provides AFP model loading and mask-processing helpers used by the auxiliary
attention loss in :mod:`afp_integrations.attention_aux_loss`.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: make the sibling ``afp`` package importable when this file is
# used from a checkout without installing the repository.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# AFP checkpoint configuration
# ---------------------------------------------------------------------------

AFP_CHECKPOINT_DIR = _REPO_ROOT / "afp" / "checkpoints"
AFP_CHECKPOINT_PATH = AFP_CHECKPOINT_DIR / "checkpoint.pth"


# ---------------------------------------------------------------------------
# AFP model loading & mask processing
# ---------------------------------------------------------------------------


def load_afp_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """Load the AFP model from a local checkpoint.

    Place the AFP checkpoint at ``afp/checkpoints/checkpoint.pth`` (or pass an
    explicit ``checkpoint_path``) before calling this function.
    """
    if checkpoint_path is None:
        checkpoint_path = str(AFP_CHECKPOINT_PATH)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"AFP checkpoint not found at {checkpoint_path}. "
            f"Place the checkpoint file at {AFP_CHECKPOINT_PATH} or pass "
            f"`checkpoint_path` explicitly."
        )

    from afp import AFPModel

    model = AFPModel(
        checkpoint_path=checkpoint_path,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        clip_device="cpu",
    )
    logging.info("AFP model loaded.")
    return model


def get_afp_mask(afp_model, image: np.ndarray, task_text: Optional[str] = None) -> np.ndarray:
    """Return the AFP alpha mask (H, W) float32 in [0, 1] for a single image."""
    return afp_model.predict_single(image, task_text=task_text)


def pool_mask_to_grid(mask: np.ndarray, grid_h: int = 16, grid_w: int = 16) -> np.ndarray:
    """Average-pool a full-resolution mask to the policy's image-token grid.

    The defaults (16 x 16) match SigLIP at 224 px as used by PaLI-Gemma based
    policies such as pi0 / pi0.5.
    """
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    pooled = torch.nn.functional.adaptive_avg_pool2d(t, (grid_h, grid_w))
    return pooled.squeeze().numpy()


def mask_to_token_weights(mask: np.ndarray, grid_h: int = 16, grid_w: int = 16) -> torch.Tensor:
    """Convert an AFP mask to a flat (grid_h*grid_w,) weight tensor in [0, 1]."""
    pooled = pool_mask_to_grid(mask, grid_h, grid_w)
    return torch.from_numpy(pooled.flatten()).float()
