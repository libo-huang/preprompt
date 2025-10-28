import torch
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import time, datetime, os, numpy as np
from datasets import build_continual_dataloader
from engines.prepromtp_engine import train_and_evaluate, evaluate_till_now
import vits.preprompt_vision_transformer as preprompt_vision_transformer


def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)
    model = create_model(
        args.model,  # 'vit_base_patch16_224'
        pretrained=args.pretrained,  # True
        num_classes=args.nb_classes,  # 100
        drop_rate=args.drop,  # 0.0
        drop_path_rate=args.drop_path,  # 0.0
        drop_block_rate=None,
        prompt_length=args.length,  # 5
        embedding_key=args.embedding_key,  # 'cls'
        prompt_init=args.prompt_key_init,  # 'uniform'
        prompt_pool=args.prompt_pool,  # True
        prompt_key=args.prompt_key,  # False
        pool_size=args.size,  # 10
        top_k=args.top_k,  # 1
        batchwise_prompt=args.batchwise_prompt,  # False
        prompt_key_init=args.prompt_key_init,  # 'uniform'
        head_type=args.head_type,  # 'token'
        use_prompt_mask=args.use_prompt_mask,  # True
        use_g_prompt=args.use_g_prompt,  # False
        g_prompt_length=args.g_prompt_length,  # 5
        g_prompt_layer_idx=args.g_prompt_layer_idx,  # []
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,  # False
        use_e_prompt=args.use_e_prompt,  # True
        e_prompt_layer_idx=args.e_prompt_layer_idx,  # [0, 1, 2, 3, 4]
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,  # True
        same_key_value=args.same_key_value,  # False
    )
    model.to(device)

    if args.freeze:  # ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']. 460900. e_prompt.prompt:5*2*10*5*12*64=384000; head.weight:768*100=76800; head.bias:100.
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
                
            _ = evaluate_till_now(model,  # original_model,
                                  data_loader, device, task_id, class_mask, target_task_map, acc_matrix, args, )
        
        print('acc_matrix:\n{}'.format(
            np.array2string(acc_matrix, precision=2, separator=', ', suppress_small=True)
        ))
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)  # 460900

    if args.unscale_lr:
        global_batch_size = args.batch_size  # 24
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0  # 0.03*24/256=0.0028125
    args.ca_lr = args.ca_lr * global_batch_size / 256.0  # 0.03*24/256=0.0028125

    if args.larger_prompt_lr:
        base_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' in name and p.requires_grad == True]  # prompt: 384000个参数可训练
        base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'head' in name and p.requires_grad == True]  # head: 76900个参数可训练
        # base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' not in name and p.requires_grad == True]  # head: 76900个参数可训练
        base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = create_optimizer(args, network_params) 
    else:
        optimizer = create_optimizer(args, model_without_ddp)
    route_net_params = [p for name, p in model_without_ddp.named_parameters() if 'route_net' in name and p.requires_grad == True]  # 76900个参数可训练
    route_net_params = {'params': route_net_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
    optimizer1 = create_optimizer(args, [route_net_params])

    lr_scheduler1 = None
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp,  # original_model,
                       criterion, data_loader, data_loader_per_cls,
                       optimizer1, optimizer, lr_scheduler1, lr_scheduler, device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")