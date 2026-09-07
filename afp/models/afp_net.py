# Inference-only AFPNet model (training losses / SetCriterion removed).

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         accuracy, get_world_size, interpolate,
                         is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .deformable_afp import build_deformable_afp
from .text_condition import FrozenCLIPTextEncoder, TextConditioningAdapter


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AFPNet(nn.Module):
    def __init__(self,
                 backbone,
                 transformer,
                 num_frames,
                 num_queries,
                 num_feature_levels,
                 aux_loss=True,
                 fpn_temporal=False,
                 use_text_conditioning=False,
                 text_clip_model='ViT-B/32',
                 text_clip_device='cpu',
                 text_cache_size=4096,
                 text_default_prompt='A generic robotic manipulation task.',
                 text_condition_dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer.decoder.bbox_embed = None

        if self.backbone.modelname == 'mv3':
            self.mask_head = MaskHeadSmallConv(hidden_dim, hidden_dim, True, fpn_temporal)
        else:
            self.mask_head = MaskHeadSmallConv(hidden_dim, hidden_dim, False, fpn_temporal)

        self.use_text_conditioning = use_text_conditioning
        if self.use_text_conditioning:
            self.text_encoder = FrozenCLIPTextEncoder(
                model_name=text_clip_model,
                clip_device=text_clip_device,
                cache_size=text_cache_size,
                default_prompt=text_default_prompt,
            )
            self.text_adapter = TextConditioningAdapter(
                hidden_dim=hidden_dim,
                text_embed_dim=self.text_encoder.embed_dim,
                dropout=text_condition_dropout,
            )
        else:
            self.text_encoder = None
            self.text_adapter = None

    def _prepare_text_inputs(self, batch_size, device, targets=None, task_text=None):
        if not self.use_text_conditioning:
            return None, None

        texts = []
        has_text = []

        if targets is not None:
            for i in range(batch_size):
                target_i = targets[i] if i < len(targets) else {}
                txt = target_i.get('task_description', '')
                if isinstance(txt, (list, tuple)):
                    txt = txt[0] if len(txt) > 0 else ''
                txt = '' if txt is None else str(txt)
                has_flag = target_i.get('has_task_text', 0)
                if torch.is_tensor(has_flag):
                    has_flag = float(has_flag.item())
                has_text.append(1.0 if float(has_flag) > 0.5 else (1.0 if len(txt.strip()) > 0 else 0.0))
                texts.append(txt)
        elif task_text is not None:
            if isinstance(task_text, str):
                texts = [task_text] * batch_size
            elif isinstance(task_text, (list, tuple)):
                texts = list(task_text)
                if len(texts) == 1 and batch_size > 1:
                    texts = texts * batch_size
                if len(texts) != batch_size:
                    raise ValueError(f'task_text length ({len(texts)}) must match batch size ({batch_size})')
            else:
                texts = [str(task_text)] * batch_size
            has_text = [1.0 if len(str(t).strip()) > 0 else 0.0 for t in texts]
        else:
            texts = [''] * batch_size
            has_text = [0.0] * batch_size

        text_embed = self.text_encoder.encode_texts(texts, out_device=device)
        has_text = torch.tensor(has_text, dtype=torch.float32, device=device)
        return text_embed, has_text

    def _apply_text_conditioning(self, hs, text_embed, has_text):
        if not self.use_text_conditioning:
            return hs
        if text_embed is None or has_text is None:
            return hs

        conditioned = []
        for lvl in range(hs.shape[0]):
            conditioned.append(self.text_adapter(hs[lvl], text_embed, has_text))
        return torch.stack(conditioned, dim=0)

    @torch.no_grad()
    def prime_text_cache(self, task_text):
        if not self.use_text_conditioning:
            return
        if isinstance(task_text, str):
            task_text = [task_text]
        elif not isinstance(task_text, (list, tuple)):
            task_text = [str(task_text)]
        _ = self.text_encoder.encode_texts(list(task_text), out_device=self.query_embed.weight.device)

    def inference(self, samples: NestedTensor, orig_w, orig_h, task_text=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        if self.backbone.modelname == 'mv3':
            largefeature = features[0].decompose()[0]
            features = features[1:]
            pos = pos[1:]
        else:
            largefeature = None

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)

            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n // self.num_frames, self.num_frames, c, h, w)

            mask = mask.reshape(n // self.num_frames, self.num_frames, h, w)

            np_, cp, hp, wp = pos[l].shape
            pos_l = pos[l].reshape(np_ // self.num_frames, self.num_frames, cp, hp, wp)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        query_embeds = self.query_embed.weight

        hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(srcs, masks, poses, query_embeds)
        if hs.dim() != 5:
            raise RuntimeError(f'Unexpected decoder state shape: {list(hs.shape)}')

        batch_size = hs.shape[1]
        text_embed, has_text = self._prepare_text_inputs(batch_size=batch_size,
                                                         device=hs.device,
                                                         targets=None,
                                                         task_text=task_text)
        hs = self._apply_text_conditioning(hs, text_embed, has_text)

        reference = inter_references[-1]
        lvl_masks = self.forward_mask_head_train(hs[-1], memory, spatial_shapes, reference, largefeature)

        return [lvl_mask[-1] for lvl_mask in lvl_masks]

    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, largefeature):
        bs, n_f, _, c = feats.shape

        encod_feat_l = []
        spatial_indx = 0

        for feat_l in range(self.num_feature_levels):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, :, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.num_frames, h, w, c).permute(0, 4, 1, 2, 3)
            encod_feat_l.append(mem_l)
            spatial_indx += h * w

        pred_masks = []
        tmp_feature = None

        if largefeature is not None:
            _, C, H, W = largefeature.shape
            largefeature = largefeature.reshape(bs, self.num_frames, C, H, W)

        for iframe in range(self.num_frames):
            encod_feat_f = []
            for lvl in range(self.num_feature_levels):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :])
            if largefeature is not None:
                decod_feat_f, tmp_feature = self.mask_head(encod_feat_f, tmp_feature, largef=largefeature[:, iframe, :, :, :])
            else:
                decod_feat_f, tmp_feature = self.mask_head(encod_feat_f, largef=None)
            query_frame_embed = outputs[:, iframe, :, :]
            mask_pyramid = []
            for decod_feat in decod_feat_f:
                mask_f = torch.einsum("bqc,bchw->bqhw", query_frame_embed, decod_feat)
                mask_pyramid.append(mask_f)
            pred_masks.append(mask_pyramid)
        return pred_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_mask):
        return [{'pred_masks': a} for a in outputs_mask[:-1]]


class ConvTmp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tmp_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.cur_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.ac = nn.Tanh()

    def forward(self, tmp_f, cur_f):
        tmp = self.ac(self.tmp_conv(tmp_f) + self.cur_conv(cur_f))
        return tmp, tmp


class MaskHeadSmallConv(nn.Module):
    def __init__(self, dim, context_dim, large_feature=False, fpn_temporal=False):
        super().__init__()
        self.lay2 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.lay4 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.dim = dim
        self.context_dim = context_dim
        self.fpn_temporal = fpn_temporal

        if self.fpn_temporal:
            self.temporal4 = ConvTmp(context_dim // 2)
            self.temporal3 = ConvTmp(context_dim // 2)
            self.temporal2 = ConvTmp(context_dim // 2)

        self.large_feature = large_feature

        if self.large_feature:
            self.proj = nn.Sequential(
                nn.Conv2d(16, context_dim, kernel_size=1),
                nn.GroupNorm(32, context_dim),
            )
            self.lay_up = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, tmp_feature, largef):
        out = []
        if self.fpn_temporal:
            current_tmp_feature = []

        fused_x = x[-1]

        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[0]
                tmp_f_update, fused_x_r = self.temporal4(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)

        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="bilinear", align_corners=False)

        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[1]
                tmp_f_update, fused_x_r = self.temporal3(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)

        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="bilinear", align_corners=False)

        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[2]
                tmp_f_update, fused_x_r = self.temporal2(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)

        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        if self.large_feature:
            fused_x = self.proj(largef) + F.interpolate(fused_x, size=largef.shape[-2:], mode="bilinear", align_corners=False)
            fused_x = self.lay_up(fused_x)
            fused_x = F.relu(fused_x)
        out.append(fused_x)

        if self.fpn_temporal:
            return out, current_tmp_feature
        else:
            return out, None


def build_afp(args, pretrained_backbone=True):
    device = torch.device(args.device)

    backbone = build_backbone(args, pretrained=pretrained_backbone)
    transformer = build_deformable_afp(args)

    model = AFPNet(
        backbone,
        transformer,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        fpn_temporal=args.fpn_temporal,
        use_text_conditioning=getattr(args, 'use_text_conditioning', False),
        text_clip_model=getattr(args, 'text_clip_model', 'ViT-B/32'),
        text_clip_device=getattr(args, 'text_clip_device', 'cpu'),
        text_cache_size=getattr(args, 'text_cache_size', 4096),
        text_default_prompt=getattr(args, 'text_default_prompt', 'A generic robotic manipulation task.'),
        text_condition_dropout=getattr(args, 'text_condition_dropout', 0.1),
    )

    return model
