# ------------------------------------------------------------------------
# AFP data loader
# ------------------------------------------------------------------------
# Modified from RVM (https://github.com/PeterL1n/RobustVideoMatting)
# Copyright (c) 2021 ByteDance Inc. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import datasets.transforms as T
import os
from PIL import Image
from random import randint
import cv2
import random
import math
import time
import numpy as np

class TrainFrameSampler:
    def __init__(self, speed=[0.5, 1, 2, 3, 4, 5]):
        self.speed = speed
    
    def __call__(self, seq_length):
        frames = list(range(seq_length))
        
        # Speed up
        speed = random.choice(self.speed)
        frames = [int(f * speed) for f in frames]
        
        # Shift
        shift = random.choice(range(seq_length))
        frames = [f + shift for f in frames]
        
        # Reverse
        if random.random() < 0.5:
            frames = frames[::-1]

        return frames

def _downsample_if_needed(img, size):
    w, h = img.size
    if min(w, h) > size:
        scale = size / min(w, h)
        w = int(scale * w)
        h = int(scale * h)
        img = img.resize((w, h))
    return img


class OpenXEmbodimentDataset(Dataset):
    def __init__(self, root, size, seq_length, seq_sampler, transform=None):
        self.root = Path(root)
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

        self.episodes = []
        self.indices = []

        if not self.root.exists():
            raise FileNotFoundError(f'Open-X Embodiment root {self.root} does not exist')

        for episode_dir in self._discover_episode_dirs(self.root):

            scenario_name = self._infer_scenario_name(episode_dir)
            task_description, has_task_text = self._resolve_task_description(episode_dir)

            fgr_dir = episode_dir / 'foreground'
            pha_dir = episode_dir / 'alpha'
            img_dir = episode_dir / 'image'

            fgr_frames = sorted(self._list_image_files(fgr_dir))
            pha_frames = sorted(self._list_image_files(pha_dir))
            img_frames = sorted(self._list_image_files(img_dir))

            frame_count = min(len(fgr_frames), len(pha_frames), len(img_frames))
            if frame_count < self.seq_length:
                continue

            episode_info = {
                'episode_dir': episode_dir,
                'scenario_name': scenario_name,
                'task_description': task_description,
                'has_task_text': has_task_text,
                'fgr_dir': fgr_dir,
                'pha_dir': pha_dir,
                'img_dir': img_dir,
                'fgr_frames': fgr_frames[:frame_count],
                'pha_frames': pha_frames[:frame_count],
                'img_frames': img_frames[:frame_count],
                'frame_count': frame_count,
            }
            episode_idx = len(self.episodes)
            self.episodes.append(episode_info)

            for frame_idx in range(0, frame_count, self.seq_length):
                self.indices.append((episode_idx, frame_idx))

        if len(self.indices) == 0:
            raise RuntimeError(f'No valid clips found under {self.root}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, frame_start = self.indices[idx]
        info = self.episodes[episode_idx]
        frame_count = info['frame_count']

        fgrs, phas, bgrs = [], [], []
        sampler_frames = self.seq_sampler(self.seq_length)

        for offset in sampler_frames:
            frame_pos = (frame_start + offset) % frame_count

            fgr = self._load_rgb(info['fgr_dir'] / info['fgr_frames'][frame_pos])
            pha = self._load_alpha(info['pha_dir'] / info['pha_frames'][frame_pos])
            img = self._load_rgb(info['img_dir'] / info['img_frames'][frame_pos])

            fgr = _downsample_if_needed(fgr, self.size)
            pha = _downsample_if_needed(pha, self.size)
            img = _downsample_if_needed(img, self.size)
            bgr = self._estimate_background(fgr, pha, img)

            fgrs.append(fgr)
            phas.append(pha)
            bgrs.append(bgr)

        if self.transform is not None:
            fgrs, phas, bgrs = self.transform(fgrs, phas, bgrs)

        imgs = []
        bgr_phas = []
        for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
            img = fgr * pha + bgr * (1 - pha)
            imgs.append(img)
            bgr_phas.append(1.0 - pha)

        target = {
            'masks': torch.cat(phas, dim=0),
            'bgr_masks': torch.cat(bgr_phas, dim=0),
            'task_description': info['task_description'],
            'has_task_text': torch.tensor(1 if info['has_task_text'] else 0, dtype=torch.long),
            'scenario_name': info['scenario_name'],
            'episode_dir': str(info['episode_dir']),
        }
        return torch.cat(imgs, dim=0), target

    @staticmethod
    def _list_image_files(directory):
        return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    @staticmethod
    def _is_openx_episode_dir(path: Path):
        return (path / 'foreground').exists() and (path / 'alpha').exists() and (path / 'image').exists()

    @classmethod
    def _discover_episode_dirs(cls, root: Path):
        episode_dirs = []
        for dirpath, dirnames, _ in os.walk(root):
            current = Path(dirpath)

            # skip hidden/system folders and avoid descending into them
            if current.name.startswith('.'):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]

            if cls._is_openx_episode_dir(current):
                episode_dirs.append(current)
                # no need to walk inside an episode directory
                dirnames[:] = []

        return sorted(episode_dirs)

    def _infer_scenario_name(self, episode_dir: Path):
        try:
            relative_parts = episode_dir.relative_to(self.root).parts
            if len(relative_parts) > 0:
                return relative_parts[0]
        except Exception:
            pass
        return episode_dir.parent.name

    @staticmethod
    def _read_first_non_empty_line(path: Path):
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
        except Exception:
            return None
        return None

    def _resolve_task_description(self, episode_dir: Path):
        # Priority 1: exact per-episode text
        episode_text_path = episode_dir / 'task_description.txt'
        episode_text = self._read_first_non_empty_line(episode_text_path)
        if episode_text is not None:
            return episode_text, True

        # Priority 2: scenario-level candidates
        scenario_text_path = episode_dir.parent / 'possible_task_descriptions.txt'
        scenario_text = self._read_first_non_empty_line(scenario_text_path)
        scenario_name = self._infer_scenario_name(episode_dir)
        if scenario_text is not None:
            fallback = f'In {scenario_name}, perform: {scenario_text}'
            return fallback, False

        # Priority 3: robust generic fallback (e.g. droid episodes)
        fallback = f'Robot manipulation in scenario {scenario_name}'
        return fallback, False

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as img:
            return img.convert('RGB')

    @staticmethod
    def _load_alpha(path):
        with Image.open(path) as img:
            return img.convert('L')

    @staticmethod
    def _estimate_background(fgr, pha, img, eps=1e-6):
        fgr_np = np.asarray(fgr).astype(np.float32)
        pha_np = np.asarray(pha).astype(np.float32)[..., None] / 255.0
        img_np = np.asarray(img).astype(np.float32)

        inv_pha = 1.0 - pha_np
        denom = np.where(inv_pha < eps, 1.0, inv_pha)
        bgr_np = (img_np - fgr_np * pha_np) / denom
        bgr_np = np.where(inv_pha < eps, img_np, bgr_np)
        bgr_np = np.clip(bgr_np, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(bgr_np)


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.MotionBlur(prob=0.1),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Keep augmentations within a small resolution budget; cap short side at 224
    scales = [160, 192, 208, 224]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomMotionAffine(prob=0.3),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=224),
                ]),
                T.Compose([
                    T.RandomResize([200, 224]),
                    T.RandomSizeCrop(160, 224),
                    T.RandomResize(scales, max_size=224),
                ])
            ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            T.RandomResize([224], max_size=224),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.vm_path)
    assert root.exists(), f'provided VM path {root} does not exist'

    def has_openx_structure(path: Path):
        if not path.exists():
            return False
        for dirpath, dirnames, _ in os.walk(path):
            current = Path(dirpath)
            if current.name.startswith('.'):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            if (current / 'foreground').exists() and (current / 'alpha').exists() and (current / 'image').exists():
                return True
        return False

    if args.dataset_file == 'vm':
        openx_root = None
        if has_openx_structure(root):
            openx_root = root
        elif has_openx_structure(root / 'open_x_embodiment'):
            openx_root = root / 'open_x_embodiment'

        if openx_root is None:
            raise FileNotFoundError(
                f'No episode folders with image/, foreground/ and alpha/ subfolders found under {root}')
        print(f'use episode dataset from {openx_root}')
        return OpenXEmbodimentDataset(root=openx_root,
                                          size=224,
                                          seq_length=args.num_frames,
                                          seq_sampler=TrainFrameSampler(),
                                          transform=make_coco_transforms(image_set))

    raise ValueError(f'Unsupported dataset_file {args.dataset_file}')
