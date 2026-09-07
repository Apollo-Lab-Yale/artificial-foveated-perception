from .afp_net import build_afp


def build_model(args):
    if args.model == 'afp':
        if args.version == 'v1':
            return build_afp(args)
    raise ValueError(f'Unknown model: {args.model} / {args.version}')
