# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .afp import build_vm

def build_model(args):
    if args.model == 'vm' and args.version == 'v1':
        return build_vm(args)
    raise ValueError(f'Unknown model: {args.model} / {args.version}')

