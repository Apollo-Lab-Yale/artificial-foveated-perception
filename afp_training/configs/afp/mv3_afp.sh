#!/usr/bin/env bash

set -x

python -u main_vm.py \
    --dataset_file vm \
    --epochs 100 \
    --lr 2e-4 \
    --lr_drop 18 30 \
    --batch_size 12 \
    --num_workers 12 \
    --vm_path data/post_processed \
    --num_queries 1 \
    --num_frames 5 \
    --backbone mv3 \
    --mask_loss_coef 5 \
    --dice_loss_coef 1 \
    --temporal_loss_coef 1 \
    --enc_layers 1 \
    --dec_layers 1 \
    --hidden_dim 256 \
    --num_feature_levels 3 \
    --version v1 \
    --query_temporal weight_sum \
    --fpn_temporal \
    --use_text_conditioning \
    --text_clip_model ViT-B/32 \
    --text_clip_device cuda \
    --wandb \
    --wandb_project afp-finetuning \
    --wandb_num_viz 8 \
    --wandb_viz_every 1 \
    --output_dir outputs/mv3_afp \
