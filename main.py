import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
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
        config_parser = subparser.add_parser('five_datasets_preprompt_5e', help='five datasets PrePrompt configs')
    elif config == 'cub_preprompt_5e':
        from configs.cub_preprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_preprompt_5e', help='Split-CUB PrePrompt configs')
    # elif config == 'cars196_preprompt_5e':
    #     from configs.cars196_preprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('cars196_preprompt_5e', help='Split-cars196 PrePrompt configs')
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

    if 'preprompt' in args.config and not args.train_inference_task_only:
        import trainers.preprompt_trainer as preprompt_trainer
        preprompt_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    if args.num_tasks != args.size:
         print(f"num_tasks({args.num_tasks}) is not equal to the Prompt size({args.size})")
         args.size = args.num_tasks
    print(args)
    main(args)