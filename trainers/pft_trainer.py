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
from engines.pft_engine import *
import vits.preprompt_vision_transformer as preprompt_vision_transformer
import utils
import warnings

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def train(args):
    device = torch.device(args.device)

    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    print(f"Creating original model: {args.original_model}")
    model = create_model(
        args.original_model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        mlp_structure=args.original_model_mlp_structure,
    )

    model.to(device)

    if args.freeze:  # 0:'blocks', 1:'patch_embed', 2:'cls_token', 3:'norm', 4:'pos_embed'
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    # print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device,
                                  task_id, class_mask, target_task_map, acc_matrix, args, )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 2440036
    print('number of params:', n_parameters)  # 可训练的参数有 2440036 个，具体是model.mlp中的 2362368 个(norm 768+768, net 768*1536+1536+1536*768+768) 和 model.head中的 76900 (768*100+100)

    if args.unscale_lr:  # True
        global_batch_size = args.batch_size  # 24
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0  # 0.0005 * 24 / 256.0 = 4.6875e-05 = 0.000046875
    args.ca_lr = args.ca_lr * global_batch_size / 256.0  # 0.005 * 24 / 256.0 = 0.00046875

    optimizer = create_optimizer(args, model_without_ddp)  # 1.获取不需要权重衰减的参数名称 no_weight_decay(); 2.根据参数名称设置权重衰减param_groups ; 3.创建优化器torch.optim.SGD
    #optimizer = optim.SGD([p for p in model_without_ddp.parameters() if p.requires_grad], lr=args.lr=0.000046875, momentum=0.9, weight_decay=args.weight_decay)
    # {'pos_embed', 'cls_token', 'dist_token'}
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp,
                       criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler,
                       device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")