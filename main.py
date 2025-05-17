import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader

import utils
import warnings

import os
local_rank = int(os.environ['LOCAL_RANK'])

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def get_args():
    parser = argparse.ArgumentParser('PrePrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_preprompt_5e':
        from configs.cifar100_preprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_preprompt_5e', help='Split-CIFAR100 PrePrompt configs')
    elif config == 'imr_preprompt_5e':
        from configs.imr_preprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_preprompt_5e', help='Split-ImageNet-R PrePrompt configs')
    elif config == 'five_datasets_preprompt_5e':
        from configs.five_datasets_preprompt_5e import get_args_parser
        config_parser = subparser.add_parser('five_datasetspreprompt_5e', help='five datasets PrePrompt configs')
    elif config == 'cub_preprompt_5e':
        from configs.cub_preprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_preprompt_5e', help='Split-CUB PrePrompt configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    utils.init_distributed_mode(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        import trainers.pft_trainer as pft_trainer
        pft_trainer.train(args)
    elif 'preprompt' in args.config and not args.train_inference_task_only:
        import trainers.preprompt_trainer as preprompt_trainer
        preprompt_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    main(args)
