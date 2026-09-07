"""AFP auxiliary attention loss and projected-gradient (PCGrad) combination.

This module contains the AFP-specific *training-time* components used to
fine-tune a vision-language-action (VLA) policy with an auxiliary loss that
aligns the policy's attention over image tokens with the AFP mask::

    total_loss = action_loss + aux_weight * attention_afp_loss

The auxiliary loss is computed per attention layer as the MSE between the
L1-normalised attention over image-token key positions (averaged over heads
and prefix query positions) and the L1-normalised AFP mask pooled to the
image-token grid, then averaged over layers.  Because the auxiliary gradient
can conflict with the action gradient, the two are accumulated separately and
combined with a projected-gradient (PCGrad) rule: when the two gradients have
negative cosine similarity, the component of the auxiliary gradient that
opposes the action gradient is projected out before the weighted sum.

The module depends only on PyTorch, plus ``transformers`` if you use
:class:`AttentionCaptureForLoss` to hook a Hugging Face Gemma attention
implementation.  Dataset conversion, optimiser construction, checkpointing and
logging belong to the surrounding training script and are not part of this
module; see "Usage" below for the training-step skeleton.

Components
----------
attention_afp_loss
    Per-layer loss between attention weights and the AFP mask.
AttentionCaptureForLoss
    Monkey-patches ``eager_attention_forward`` of a transformers Gemma model so
    the per-layer loss is accumulated on-the-fly during the forward pass.  This
    keeps gradients intact and avoids storing attention tensors.
compute_batch_afp_masks
    Runs the AFP model on a batch of images and pools the alpha masks to the
    image-token grid.
compute_prefix_len
    Number of prefix (image + language) query positions that the loss is
    applied to; expert (action/state) queries are excluded.
AuxGradientCombiner
    Two-pass gradient accumulation and combination of the action gradient and
    the auxiliary gradient: plain weighting, PCGrad, PCGrad + norm rescaling,
    or gradient-norm balancing.

Usage
-----
Sketch of the training loop (single GPU or DDP)::

    capture = AttentionCaptureForLoss()
    capture.install()                      # patch Gemma eager attention

    combiner = AuxGradientCombiner(trainable_params, mode="pcgrad_norm",
                                   aux_weight=0.1, grad_accum_steps=1)

    for observation, actions in loader:
        with torch.no_grad():
            afp_masks = compute_batch_afp_masks(afp_model, images, prompts).to(device)

        capture.clear()
        capture.set_afp_masks(afp_masks)
        capture.set_prefix_len(compute_prefix_len(n_images, valid_lang_len))
        capture.enable()
        action_loss = model(observation, actions).mean()
        capture.disable()
        aux_loss = capture.get_loss()

        combiner.backward(action_loss, aux_loss, ddp_model=model)
        if combiner.ready():
            combiner.all_reduce()          # no-op outside torch.distributed
            stats = combiner.combine()     # writes p.grad for every param
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    capture.uninstall()

Notes
-----
* Gradient checkpointing must be disabled on the policy: recomputation makes
  the attention hook fire twice per layer with inconsistent tensor counts.
* The policy must run an *eager* attention implementation so that attention
  weights are materialised and returned by ``eager_attention_forward``.
* The default image-token grid is 16 x 16 = 256 tokens (SigLIP at 224 px).
  Pass ``n_img_tokens`` / ``grid_size`` to match other vision encoders.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

from .afp_utils import pool_mask_to_grid

__all__ = [
    "GRID_SIZE",
    "N_IMG_TOKENS",
    "attention_afp_loss",
    "AttentionCaptureForLoss",
    "compute_batch_afp_masks",
    "compute_prefix_len",
    "AuxGradientCombiner",
]

# Image-token grid of the policy's vision encoder (SigLIP, 224 px -> 16 x 16).
GRID_SIZE = 16
N_IMG_TOKENS = GRID_SIZE * GRID_SIZE  # 256


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def attention_afp_loss(
    attn_weights: torch.Tensor,
    afp_masks: torch.Tensor,
    prefix_len: Optional[int] = None,
    n_img_tokens: int = N_IMG_TOKENS,
    min_mask_energy: float = 0.01,
) -> Optional[torch.Tensor]:
    """Attention-AFP loss for a single attention layer.

    Args:
        attn_weights: ``(B, H, T_q, T_kv)`` attention probabilities of one layer.
            Image tokens are assumed to occupy the first ``n_img_tokens`` key
            positions.
        afp_masks: ``(B, n_img_tokens)`` pooled AFP mask values in ``[0, 1]``.
        prefix_len: Number of leading query positions (image + language prefix)
            to average over.  ``None`` uses all query positions.
        n_img_tokens: Number of image-token key positions.
        min_mask_energy: Samples whose mask sums to less than this are skipped;
            L1-normalising a near-empty mask would amplify noise into a spurious
            uniform target.

    Returns:
        Scalar loss with gradients attached, or ``None`` if the key sequence is
        too short to contain the image tokens (the layer is skipped).
    """
    T_kv = attn_weights.shape[-1]
    if T_kv <= n_img_tokens:
        # KV length too short to contain image tokens - skip this layer.
        return None

    # Only use prefix (image + language) query positions, excluding expert
    # action/state queries that should not be constrained.
    q_end = prefix_len if prefix_len is not None else attn_weights.shape[2]

    # Slice to prefix queries x image keys, then average over heads and query
    # positions -> (B, n_img_tokens).  Upcast to fp32 *after* the reduction:
    # the auxiliary signal is tiny (~1e-4) and in bf16 the subtraction
    # (avg_attn_norm - target) suffers catastrophic cancellation.  Casting
    # after the mean keeps the fp32 footprint at O(B * n_img) instead of
    # O(B * H * T_q * n_img).
    img_attn = attn_weights[:, :, :q_end, :n_img_tokens]
    avg_attn = img_attn.mean(dim=(1, 2)).float()

    # Normalise to a probability distribution (L1).  Min-max normalisation
    # explodes the gradient when the attention range is tiny.  The denominator
    # is detached so no gradient flows through 1/sum - this breaks the
    # positive-feedback loop where decreasing image-token attention -> larger
    # 1/sum gradient -> further decrease -> cascade at peak LR.
    avg_attn_norm = avg_attn / avg_attn.sum(dim=-1, keepdim=True).detach().clamp(min=1e-8)

    target = afp_masks.float()
    target_energy = target.sum(dim=-1, keepdim=True)
    valid = (target_energy > min_mask_energy).float()  # (B, 1)
    target = target / target_energy.clamp(min=1e-8)

    per_sample = ((avg_attn_norm - target) ** 2).mean(dim=-1)  # (B,)
    n_valid = valid.squeeze(-1).sum().clamp(min=1.0)
    return (per_sample * valid.squeeze(-1)).sum() / n_valid


class AttentionCaptureForLoss:
    """Accumulates the auxiliary attention-AFP loss during the forward pass.

    Instead of storing full attention tensors (which breaks gradient flow when
    detached and is incompatible with gradient checkpointing), the hook
    computes the per-layer loss on-the-fly and accumulates it into a running
    sum.  This keeps gradients intact and avoids large memory overhead.

    Before each forward pass call :meth:`set_afp_masks` to provide the target
    mask and :meth:`set_prefix_len` to bound the query positions, then wrap
    the forward call in :meth:`enable` / :meth:`disable`.  After the forward
    pass, :meth:`get_loss` returns the mean per-layer loss.
    """

    def __init__(self, n_img_tokens: int = N_IMG_TOKENS):
        self.n_img_tokens = n_img_tokens
        self._enabled = False
        self._patched: Optional[tuple] = None  # (module, fn_name, original_fn)
        self._afp_masks: Optional[torch.Tensor] = None  # (B, n_img_tokens)
        self._prefix_len: Optional[int] = None
        self._loss_sum: Optional[torch.Tensor] = None
        self._layer_count: int = 0

    # -- configuration ------------------------------------------------------

    def set_afp_masks(self, masks: torch.Tensor) -> None:
        """Set the target AFP mask ``(B, n_img_tokens)`` for the next forward pass."""
        self._afp_masks = masks

    def set_prefix_len(self, n: int) -> None:
        """Set the number of prefix (image + language) query positions."""
        self._prefix_len = n

    # -- hook management ----------------------------------------------------

    def install(self, target_module=None, fn_name: str = "eager_attention_forward") -> None:
        """Monkey-patch an eager attention function to accumulate the loss.

        By default patches ``transformers.models.gemma.modeling_gemma.
        eager_attention_forward`` (used by PaLI-Gemma based policies such as
        pi0 / pi0.5).  Pass ``target_module`` / ``fn_name`` to hook a different
        implementation with the same ``(module, query, key, value,
        attention_mask, scaling, dropout=0.0, **kwargs) -> (output, weights)``
        signature.
        """
        if self._patched is not None:
            raise RuntimeError("AttentionCaptureForLoss is already installed; call uninstall() first.")
        if target_module is None:
            from transformers.models.gemma import modeling_gemma as target_module

        original_fn = getattr(target_module, fn_name)
        capture = self

        def _capturing_eager_attn(module, query, key, value, attention_mask, scaling,
                                  dropout=0.0, **kwargs):
            attn_output, attn_weights = original_fn(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
            )
            if capture._enabled and capture._afp_masks is not None:
                capture.accumulate(attn_weights)
            return attn_output, attn_weights

        setattr(target_module, fn_name, _capturing_eager_attn)
        self._patched = (target_module, fn_name, original_fn)

    def uninstall(self) -> None:
        """Restore the original attention function."""
        if self._patched is not None:
            target_module, fn_name, original_fn = self._patched
            setattr(target_module, fn_name, original_fn)
            self._patched = None

    # -- accumulation -------------------------------------------------------

    def accumulate(self, attn_weights: torch.Tensor) -> None:
        """Add the loss of one layer's ``(B, H, T_q, T_kv)`` attention weights.

        Call this directly if you capture attention weights yourself instead
        of using :meth:`install`.
        """
        layer_loss = attention_afp_loss(
            attn_weights, self._afp_masks, self._prefix_len, self.n_img_tokens
        )
        if layer_loss is None:
            return
        self._loss_sum = layer_loss if self._loss_sum is None else self._loss_sum + layer_loss
        self._layer_count += 1

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._loss_sum = None
        self._layer_count = 0

    def get_loss(self) -> torch.Tensor:
        """Mean per-layer auxiliary loss with gradients intact.

        Returns a zero scalar (without gradient) if no layer was captured.
        """
        if self._loss_sum is None or self._layer_count == 0:
            return torch.tensor(0.0)
        return self._loss_sum / self._layer_count


# ---------------------------------------------------------------------------
# Mask preparation helpers
# ---------------------------------------------------------------------------


def compute_batch_afp_masks(
    afp_model,
    images_tensor: torch.Tensor,
    prompts: Sequence[str],
    grid_size: int = GRID_SIZE,
) -> torch.Tensor:
    """Compute pooled AFP masks for a batch of images.

    Args:
        afp_model: Loaded :class:`afp.AFPModel`.
        images_tensor: ``(B, C, H, W)`` uint8 or float (in ``[0, 1]``) tensor.
        prompts: Per-sample task text (length ``B``).  Missing entries run the
            AFP model unconditioned.
        grid_size: Side of the square image-token grid to pool to.

    Returns:
        ``(B, grid_size * grid_size)`` float32 tensor of pooled mask values.
    """
    batch_size = images_tensor.shape[0]
    masks = []
    for i in range(batch_size):
        # Convert (C, H, W) -> (H, W, C) uint8
        if images_tensor.dtype in (torch.float32, torch.bfloat16, torch.float16):
            img_np = (images_tensor[i].permute(1, 2, 0).cpu().float().clamp(0, 1) * 255).byte().numpy()
        else:
            img_np = images_tensor[i].permute(1, 2, 0).cpu().numpy()

        alpha = afp_model.predict_single(img_np, task_text=prompts[i] if i < len(prompts) else None)
        pooled = pool_mask_to_grid(alpha, grid_size, grid_size).flatten()
        masks.append(pooled)

    return torch.tensor(np.stack(masks), dtype=torch.float32)


def compute_prefix_len(n_images: int, valid_lang_len: int, n_img_tokens: int = N_IMG_TOKENS) -> int:
    """Number of prefix query positions the auxiliary loss is applied to.

    Use the per-batch *valid* language length (max over the batch of the
    tokenised-prompt mask sum), not the padded maximum token length: padded
    query positions hold zero-vector embeddings whose attention rows are pure
    noise and would corrupt the averaged attention distribution.
    """
    return n_img_tokens * n_images + valid_lang_len


# ---------------------------------------------------------------------------
# Projected-gradient combination of action and auxiliary gradients
# ---------------------------------------------------------------------------


class AuxGradientCombiner:
    """Accumulate action and auxiliary gradients separately and combine them.

    Modes:

    ``"weighted"``
        Single backward pass of ``action_loss + aux_weight * aux_loss``.
        No projection; provided as the baseline.
    ``"pcgrad"``
        Two-pass backward.  If the accumulated action and auxiliary gradients
        conflict (negative dot product), the component of the auxiliary
        gradient along the action gradient is projected out.  The result is
        ``g_action + aux_weight * g_aux_projected``: PCGrad fixes the
        *direction*, ``aux_weight`` controls the *magnitude*.  Without the
        weight the auxiliary gradient dominates the combined direction and
        gradient clipping attenuates the action signal, causing loss spikes
        once the learning rate is large.
    ``"pcgrad_norm"``
        As ``"pcgrad"``, then the projected auxiliary gradient is rescaled so
        its norm matches the action-gradient norm (ratio clamped to
        ``[scale_min, scale_max]``) before applying ``aux_weight``.
    ``"grad_norm"``
        No projection; the auxiliary gradient is rescaled to the action
        gradient norm (clamped) and added.  ``aux_weight`` is ignored.

    The projection and norm statistics are restricted to the subset of
    parameters that actually receive auxiliary gradients (determined on the
    first optimiser step).  Parameters with zero auxiliary gradient (e.g. the
    last layer's value/output projections) would otherwise be contaminated
    with spurious, amplified action gradients.

    With gradient accumulation the raw action and auxiliary gradients are
    summed across micro-batches and the projection is applied once on the
    full-batch estimate, which is far less noisy than per-micro-batch
    projection.  Under ``DistributedDataParallel`` the two backward passes
    are wrapped in ``no_sync()`` and the buffers are all-reduced once per
    optimiser step via :meth:`all_reduce`, so the cosine similarity,
    projection and scaling all see true full-batch gradients.
    """

    MODES = ("weighted", "pcgrad", "pcgrad_norm", "grad_norm")

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        mode: str = "pcgrad",
        aux_weight: float = 0.1,
        grad_accum_steps: int = 1,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.params: List[torch.nn.Parameter] = list(params)
        self.mode = mode
        self.aux_weight = float(aux_weight)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        self.two_pass = mode != "weighted"
        self._action_buf: Optional[List[torch.Tensor]] = (
            [torch.zeros_like(p.data) for p in self.params] if self.two_pass else None
        )
        self._aux_buf: Optional[List[torch.Tensor]] = (
            [torch.zeros_like(p.data) for p in self.params] if self.two_pass else None
        )
        # Which parameters receive auxiliary gradients; built on the first
        # optimiser step from the accumulated (and all-reduced) buffers.
        self._aux_has_grad: Optional[List[bool]] = None

        self._micro_steps = 0
        self.action_loss_accum = 0.0
        self.aux_loss_accum = 0.0

    # -- per micro-batch ----------------------------------------------------

    def _zero_grads(self) -> None:
        for p in self.params:
            p.grad = None

    def backward(self, action_loss: torch.Tensor, aux_loss: torch.Tensor, ddp_model=None) -> None:
        """Run backward for one micro-batch.

        Args:
            action_loss: Scalar primary (action) loss.
            aux_loss: Scalar auxiliary loss from :meth:`AttentionCaptureForLoss.get_loss`.
            ddp_model: The ``DistributedDataParallel`` wrapper, if any, so that
                the two-pass backward can be run under ``no_sync()``.

        In the two-pass modes ``aux_loss`` must carry a gradient.  The grad-less
        zero that :meth:`AttentionCaptureForLoss.get_loss` returns when no layer
        was captured raises, so a mis-configured hook fails on the first step
        instead of silently training without the auxiliary signal.
        """
        inv_accum = 1.0 / self.grad_accum_steps

        if self.two_pass:
            # Calling backward() twice per forward without no_sync() leaves the
            # DDP reducer in an ill-defined state (the all-reduce may fire per
            # pass, reducing action and aux gradients separately, or silently
            # skip one).  Defer reduction to all_reduce().
            no_sync = (
                ddp_model.no_sync()
                if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel)
                else nullcontext()
            )
            with no_sync:
                self._zero_grads()
                action_loss.backward(retain_graph=True)
                for buf, p in zip(self._action_buf, self.params):
                    if p.grad is not None:
                        buf.add_(p.grad, alpha=inv_accum)

                self._zero_grads()
                if not aux_loss.requires_grad:
                    raise RuntimeError(
                        "aux_loss carries no gradient: AttentionCaptureForLoss captured no "
                        "attention layer. Check that install() and enable() were called, that "
                        "set_afp_masks() was set before the forward pass, and that the policy "
                        "runs eager attention (attn_implementation='eager')."
                    )
                aux_loss.backward()
                for buf, p in zip(self._aux_buf, self.params):
                    if p.grad is not None:
                        buf.add_(p.grad, alpha=inv_accum)

                self._zero_grads()
        else:
            total_loss = (action_loss + self.aux_weight * aux_loss) / self.grad_accum_steps
            total_loss.backward()

        self.action_loss_accum += action_loss.item() / self.grad_accum_steps
        self.aux_loss_accum += aux_loss.item() / self.grad_accum_steps
        self._micro_steps += 1

    def ready(self) -> bool:
        """True when a full accumulation window has been processed."""
        return self._micro_steps > 0 and self._micro_steps % self.grad_accum_steps == 0

    # -- per optimiser step -------------------------------------------------

    def all_reduce(self) -> None:
        """Average the accumulated buffers across ``torch.distributed`` ranks.

        No-op in ``"weighted"`` mode (DDP reduces those gradients itself) or
        when the process group is not initialised.  Call *before*
        :meth:`combine`.
        """
        if not self.two_pass:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        for buf in self._action_buf:
            dist.all_reduce(buf, op=dist.ReduceOp.AVG)
        for buf in self._aux_buf:
            dist.all_reduce(buf, op=dist.ReduceOp.AVG)

    def combine(self) -> Dict[str, float]:
        """Combine the accumulated gradients into ``p.grad`` and reset buffers.

        Returns a dictionary of statistics for logging: accumulated
        ``action_loss`` / ``aux_loss``, ``g_action_norm``, ``g_aux_norm``,
        ``cos_sim`` (action/aux cosine similarity over the aux subspace),
        ``projected`` (whether PCGrad projection fired), ``grad_norm_scale``
        and ``effective_aux_scale`` (the factor actually applied to the
        auxiliary gradient).
        """
        stats: Dict[str, float] = {
            "action_loss": self.action_loss_accum,
            "aux_loss": self.aux_loss_accum,
            "g_action_norm": 0.0,
            "g_aux_norm": 0.0,
            "cos_sim": 0.0,
            "projected": 0.0,
            "grad_norm_scale": 1.0,
            "effective_aux_scale": self.aux_weight,
        }
        self.action_loss_accum = 0.0
        self.aux_loss_accum = 0.0

        if not self.two_pass:
            # Gradients already live in p.grad from the single backward pass.
            return stats

        n = len(self.params)
        if self._aux_has_grad is None:
            self._aux_has_grad = [self._aux_buf[i].abs().max().item() > 0 for i in range(n)]
            logging.info(
                "Aux-gradient mask: %d/%d params receive aux gradients",
                sum(self._aux_has_grad), n,
            )
        mask = self._aux_has_grad

        if self.mode in ("pcgrad", "pcgrad_norm"):
            # Dot product and norms only over parameters that receive aux
            # gradients, giving a correctly scaled projection coefficient.
            dot_val = 0.0
            act_sq = 0.0
            act_sq_full = 0.0
            aux_sq = 0.0
            for i in range(n):
                a_g = self._action_buf[i]
                x_g = self._aux_buf[i]
                act_sq_full += a_g.float().pow(2).sum().item()
                if mask[i]:
                    dot_val += torch.sum(a_g * x_g).item()
                    act_sq += a_g.float().pow(2).sum().item()
                    aux_sq += x_g.float().pow(2).sum().item()
            act_norm = act_sq ** 0.5
            act_norm_full = act_sq_full ** 0.5
            aux_norm = aux_sq ** 0.5
            cos_sim = dot_val / (act_norm * aux_norm + 1e-8)

            # PCGrad projection (direction correction), only over the
            # subspace where aux gradients exist.
            projected = dot_val < 0
            if projected:
                coeff = dot_val / (act_sq + 1e-8)
                for i in range(n):
                    if mask[i]:
                        self._aux_buf[i] = self._aux_buf[i] - coeff * self._action_buf[i]

            # Norm rescaling (pcgrad_norm only): scale the projected aux
            # gradient to match the action-gradient norm in the aux subspace.
            if self.mode == "pcgrad_norm":
                proj_aux_sq = sum(
                    self._aux_buf[i].float().pow(2).sum().item() for i in range(n) if mask[i]
                )
                proj_aux_norm = proj_aux_sq ** 0.5
                gn_scale = min(max(act_norm / max(proj_aux_norm, 1e-8), self.scale_min), self.scale_max)
            else:
                gn_scale = 1.0

            effective_aux_scale = self.aux_weight * gn_scale
            for i, p in enumerate(self.params):
                p.grad = self._action_buf[i] + effective_aux_scale * self._aux_buf[i]

            stats.update(
                g_action_norm=act_norm_full,
                g_aux_norm=aux_norm,
                cos_sim=cos_sim,
                projected=float(projected),
                grad_norm_scale=gn_scale,
                effective_aux_scale=effective_aux_scale,
            )

        elif self.mode == "grad_norm":
            g_action = sum(a.float().pow(2).sum().item() for a in self._action_buf) ** 0.5
            g_aux = sum(x.float().pow(2).sum().item() for x in self._aux_buf) ** 0.5
            gn_scale = min(max(g_action / max(g_aux, 1e-8), self.scale_min), self.scale_max)
            for i, p in enumerate(self.params):
                p.grad = self._action_buf[i] + gn_scale * self._aux_buf[i]

            stats.update(
                g_action_norm=g_action,
                g_aux_norm=g_aux,
                grad_norm_scale=gn_scale,
                effective_aux_scale=gn_scale,
            )

        for buf in self._action_buf:
            buf.zero_()
        for buf in self._aux_buf:
            buf.zero_()
        return stats
