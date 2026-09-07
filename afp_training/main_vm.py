# ------------------------------------------------------------------------
# Training script for AFP
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_vm
from models import build_model

try:
    import wandb
except ImportError:
    wandb = None

def get_args_parser():
    parser = argparse.ArgumentParser('AFP', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=[40], type=int, nargs='+')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true') 

    # Model parameters
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='mv3', type=str,
                        help="Name of the convolutional backbone to use: [mobilenetv3, resnet50]")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'temporal'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--query_temporal', type=str, default=None,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--fpn_temporal', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--name', default='vm')
    parser.add_argument('--version', default='v1')
    parser.add_argument('--mask_out_stride', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--l1_loss_coef', default=1, type=float)
    parser.add_argument('--lap_loss_coef', default=1, type=float)
    parser.add_argument('--temporal_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vm')
    parser.add_argument('--model', default='vm')
    parser.add_argument('--vm_path', default='data/post_processed', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # evaluation options
    parser.add_argument('--visualize', default='')

    # multi-frame
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')

    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='afp-finetuning', type=str,
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', default='', type=str,
                        help='W&B run name (optional)')
    parser.add_argument('--wandb_num_viz', default=8, type=int,
                        help='Number of random training-frame visualizations to log')
    parser.add_argument('--wandb_viz_every', default=1, type=int,
                        help='Log visualizations every N epochs')

    # Text conditioning (optional)
    parser.add_argument('--use_text_conditioning', action='store_true',
                        help='Enable frozen CLIP text conditioning for AFP decoder states.')
    parser.add_argument('--text_clip_model', default='ViT-B/32', type=str,
                        help='OpenAI CLIP model name for text encoder.')
    parser.add_argument('--text_clip_device', default='cpu', type=str,
                        help='Device for CLIP text encoding (cpu recommended for compatibility).')
    parser.add_argument('--text_cache_size', default=4096, type=int,
                        help='Episode/task text embedding cache size.')
    parser.add_argument('--text_default_prompt', default='A generic robotic manipulation task.', type=str,
                        help='Fallback prompt when task_description is absent.')
    parser.add_argument('--text_condition_dropout', default=0.1, type=float,
                        help='Dropout used in text FiLM adapter.')
    return parser


def _denormalize_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0.0, 1.0)


def _to_uint8_rgb(img):
    img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img_np


def _to_uint8_gray(mask):
    mask_np = (mask.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([mask_np, mask_np, mask_np], axis=-1)


def _resize_for_viz(img_np, size=256):
    return np.array(Image.fromarray(img_np).resize((size, size), resample=Image.BILINEAR))


def _build_viz_panel(input_img, pred_alpha, gt_alpha, viz_size=256):
    input_vis = _resize_for_viz(_to_uint8_rgb(_denormalize_image(input_img)), size=viz_size)
    pred_vis = _resize_for_viz(_to_uint8_gray(pred_alpha), size=viz_size)
    gt_vis = _resize_for_viz(_to_uint8_gray(gt_alpha), size=viz_size)
    diff_vis = _resize_for_viz(_to_uint8_gray((pred_alpha - gt_alpha).abs()), size=viz_size)
    return np.concatenate([input_vis, pred_vis, gt_vis, diff_vis], axis=1)


def log_wandb_random_visualizations(model, dataset, device, epoch, num_viz=8, wandb_step=None):
    if wandb is None or num_viz <= 0:
        return

    model_was_training = model.training
    model.eval()
    vis_images = []

    num_viz = min(num_viz, len(dataset))
    sampled_indices = random.sample(range(len(dataset)), k=num_viz)

    with torch.no_grad():
        for sample_idx in sampled_indices:
            sample_imgs, target = dataset[sample_idx]
            sample_imgs = sample_imgs.to(device)

            num_frames = sample_imgs.shape[0] // 3
            sample_imgs_4d = sample_imgs.reshape(num_frames, 3, sample_imgs.shape[-2], sample_imgs.shape[-1])

            model.num_frames = num_frames
            task_text = target.get('task_description', '') if isinstance(target, dict) else ''
            outputs = model.inference(sample_imgs_4d, sample_imgs_4d.shape[-1], sample_imgs_4d.shape[-2], task_text=task_text)

            gt_masks = target['masks']
            if gt_masks.ndim == 2:
                gt_masks = gt_masks.unsqueeze(0)

            frame_idx = random.randrange(num_frames)
            pred_mask = outputs[frame_idx]
            pred_mask = F.interpolate(pred_mask, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
            pred_alpha = pred_mask[0, 0].sigmoid().detach().cpu()
            gt_alpha = gt_masks[frame_idx].detach().cpu().float().clamp(0.0, 1.0)

            panel = _build_viz_panel(sample_imgs_4d[frame_idx].detach().cpu(), pred_alpha, gt_alpha)
            caption = f'epoch={epoch} sample={sample_idx} frame={frame_idx} | input | pred_alpha | gt_alpha | abs_error'
            vis_images.append(wandb.Image(panel, caption=caption))

    if vis_images:
        wandb.log({'train/random_matte_viz': vis_images}, step=wandb_step)

    if model_was_training:
        model.train()

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)

    rank_num = utils.get_rank()

    use_wandb = args.wandb and rank_num == 0
    if args.wandb and wandb is None and rank_num == 0:
        print('Warning: --wandb is set but wandb is not installed. Skipping W&B logging.')
    if use_wandb:
        run_name = args.wandb_run_name if args.wandb_run_name else None
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.log({'train/status': 0.0}, step=0)

    dataset_viz = None
    if use_wandb:
        try:
            dataset_viz = build_dataset(image_set='val', args=args)
        except Exception:
            dataset_viz = dataset_train

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_vm, num_workers=args.num_workers,
                                   pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.resume is not None:
        print('resume from ',args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        strict_load = not args.use_text_conditioning
        incompatible = model_without_ddp.load_state_dict(checkpoint['model'], strict=strict_load)
        missing = incompatible.missing_keys if hasattr(incompatible, 'missing_keys') else []
        unexpected = incompatible.unexpected_keys if hasattr(incompatible, 'unexpected_keys') else []
        if (len(missing) > 0 or len(unexpected) > 0) and rank_num == 0:
            print(f'checkpoint load info | missing={len(missing)} unexpected={len(unexpected)}')
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.last_epoch = args.start_epoch
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
    
    elif args.pretrain_weights is not None:
        print('load weigth from pretrain weight:',args.pretrain_weights)
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu')['model']
        strict_load = not args.use_text_conditioning
        incompatible = model_without_ddp.load_state_dict(checkpoint, strict=strict_load)
        missing = incompatible.missing_keys if hasattr(incompatible, 'missing_keys') else []
        unexpected = incompatible.unexpected_keys if hasattr(incompatible, 'unexpected_keys') else []
        if (len(missing) > 0 or len(unexpected) > 0) and rank_num == 0:
            print(f'pretrain load info | missing={len(missing)} unexpected={len(unexpected)}')

    output_dir = Path(args.output_dir)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        wandb_step_logger = None
        if use_wandb:
            step_offset = epoch * max(1, len(data_loader_train))

            def _wandb_step_logger(step_stats, data_iter_step, _step_offset=step_offset, _epoch=epoch):
                log_dict = {f'train_step/{k}': float(v) for k, v in step_stats.items()}
                log_dict['train_step/epoch'] = _epoch
                log_dict['train_step/iter'] = data_iter_step
                wandb.log(log_dict, step=_step_offset + data_iter_step)

            wandb_step_logger = _wandb_step_logger

        train_stats = train_one_epoch_vm(
            model, criterion, data_loader_train, optimizer, device, epoch, rank_num, args.clip_max_norm,
            wandb_log_fn=wandb_step_logger)

        if use_wandb:
            epoch_step = (epoch + 1) * max(1, len(data_loader_train))
            wandb_log_dict = {f'train/{k}': float(v) for k, v in train_stats.items()}
            wandb_log_dict['train/epoch'] = epoch
            wandb_log_dict['train/lr'] = float(optimizer.param_groups[0]['lr'])
            wandb.log(wandb_log_dict, step=epoch_step)

            if (epoch + 1) % max(args.wandb_viz_every, 1) == 0:
                log_wandb_random_visualizations(
                    model_without_ddp,
                    dataset_viz if dataset_viz is not None else dataset_train,
                    device,
                    epoch,
                    num_viz=max(1, min(args.wandb_num_viz, 16)),
                    wandb_step=epoch_step
                )

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AFP training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)