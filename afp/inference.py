"""
AFP (Artificial Foveated Perception) inference API.

High-level interface for loading the AFP model and predicting task-conditioned
masks for RGB frames.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .models.afp_net import build_afp


# ImageNet normalization used during AFPNet training
_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _default_args():
    """Return the default model arguments matching the trained checkpoint."""
    args = argparse.Namespace(
        model='afp',
        version='v1',
        backbone='mv3',
        dilation=False,
        position_embedding='sine',
        num_feature_levels=3,
        num_queries=1,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=1024,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        dec_n_points=4,
        enc_n_points=4,
        masks=False,
        mask_out_stride=4,
        query_temporal='weight_sum',
        fpn_temporal=True,
        aux_loss=True,
        num_frames=5,
        lr_backbone=1e-5,
        use_text_conditioning=True,
        text_clip_model='ViT-B/32',
        text_clip_device='cpu',
        text_cache_size=4096,
        text_default_prompt='A generic robotic manipulation task.',
        text_condition_dropout=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return args


class AFPModel:
    """
    Artificial Foveated Perception task-conditioned mask predictor.

    Wraps AFPNet with its CLIP text conditioning for inference. Produces a
    continuous mask in [0, 1] per RGB frame, optionally conditioned on a task
    description string.

    Usage::

        model = AFPModel('afp/checkpoints/checkpoint.pth')
        # frames: numpy array of shape (N, H, W, 3) uint8
        masks = model.predict(frames, task_text='pick up the mug')
        # masks: list of numpy arrays, each (H, W) float32 in [0, 1]
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 clip_device: str = 'cpu',
                 num_frames: int = 5):
        """
        Args:
            checkpoint_path: Path to the trained AFP checkpoint (.pth).
            device: Device for model inference ('cuda', 'cpu', or None for auto).
            clip_device: Device for the CLIP text encoder ('cpu' recommended).
            num_frames: Number of frames per inference window (must match training).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.num_frames = num_frames

        args = _default_args()
        args.device = device
        args.num_frames = num_frames
        args.text_clip_device = clip_device

        # Build model without downloading pretrained backbone weights
        # (they will be overwritten by the checkpoint anyway)
        self.model = build_afp(args, pretrained_backbone=False)
        self.model.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict(self,
                frames: Union[np.ndarray, List[np.ndarray]],
                task_text: Optional[str] = None) -> List[np.ndarray]:
        """
        Predict masks for a batch of RGB frames.

        Args:
            frames: Either a numpy array of shape (N, H, W, 3) uint8,
                    or a list of (H, W, 3) uint8 numpy arrays.
            task_text: Optional task description for CLIP text conditioning.

        Returns:
            List of mask numpy arrays, each (H, W) float32 in [0, 1].
        """
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        elif isinstance(frames, list):
            frame_list = frames
        else:
            raise ValueError(f'frames must be a (N,H,W,3) array or list of (H,W,3) arrays, got {type(frames)}')

        all_mattes = []

        # Process in windows of num_frames
        for start in range(0, len(frame_list), self.num_frames):
            window = frame_list[start:start + self.num_frames]
            window_tensors = []
            for frame in window:
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                else:
                    img = frame
                window_tensors.append(_TRANSFORM(img).unsqueeze(0).to(self.device))

            clip_tensor = torch.cat(window_tensors, dim=0)  # (nf, 3, H, W)
            self.model.num_frames = clip_tensor.shape[0]

            outputs = self.model.inference(
                clip_tensor,
                clip_tensor.shape[-1],
                clip_tensor.shape[-2],
                task_text=task_text
            )

            for mask in outputs:
                mask = F.interpolate(
                    mask,
                    (clip_tensor.shape[-2], clip_tensor.shape[-1]),
                    mode='bilinear',
                    align_corners=False
                )
                alpha = mask[0][0].sigmoid().cpu().numpy().astype(np.float32)
                all_mattes.append(alpha)

        return all_mattes

    @torch.no_grad()
    def predict_single(self,
                       frame: np.ndarray,
                       task_text: Optional[str] = None) -> np.ndarray:
        """
        Predict the mask for a single RGB frame.

        Args:
            frame: A (H, W, 3) uint8 numpy array.
            task_text: Optional task description for CLIP text conditioning.

        Returns:
            Mask as a (H, W) float32 numpy array in [0, 1].
        """
        mattes = self.predict([frame], task_text=task_text)
        return mattes[0]
