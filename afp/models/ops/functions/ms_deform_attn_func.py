from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
    _HAS_CUDA_MSDA = True
except ImportError:
    _HAS_CUDA_MSDA = False


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    # Convert tensor spatial shapes to list of (H, W) tuples for split
    if isinstance(value_spatial_shapes, torch.Tensor):
        split_sizes = [int(H_ * W_) for H_, W_ in value_spatial_shapes.tolist()]
        shapes_list = value_spatial_shapes.tolist()
    else:
        split_sizes = [int(H_ * W_) for H_, W_ in value_spatial_shapes]
        shapes_list = list(value_spatial_shapes)

    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(shapes_list):
        H_, W_ = int(H_), int(W_)
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        if _HAS_CUDA_MSDA:
            ctx.im2col_step = im2col_step
            output = MSDA.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, ctx.im2col_step)
            ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index,
                                  sampling_locations, attention_weights)
            return output
        else:
            # Pure Python fallback (inference-only, no backward support)
            return ms_deform_attn_core_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not _HAS_CUDA_MSDA:
            raise NotImplementedError(
                'Backward pass requires the compiled MultiScaleDeformableAttention CUDA extension. '
                'This inference-only build uses the pure Python fallback which does not support training.')
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
