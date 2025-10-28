
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from itertools import chain


def train_one_epoch(model: torch.nn.Module,  # original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer1, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)
    # original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'  # 'Train: Epoch[1/1]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 第一轮预测
        # model.eval()  # 固定Dropout/BatchNorm
        use_e_promot, use_prompt_mask, prompt_pool = model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool
        model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool= False, False, False
        # with torch.no_grad():
        output1 = model(input)
        model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool = use_e_promot, use_prompt_mask, prompt_pool
        logits1 = output1['logits']

        # here is the trick to mask out classes of non-current tasks        使用task ID信息
        if args.train_mask and class_mask is not None:  # Ture, [[], [], ..., []]
            mask = class_mask[task_id]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)  # [10, 11, ..., 99]
            logits1 = logits1.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss1 = criterion(logits1, target)  # base criterion (CrossEntropyLoss)
        
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer1.step()       

        prompt_id = torch.full((input.size(0),), task_id, dtype=torch.long, device=device)
        output = model(input, task_id=task_id, prompt_id=prompt_id, train=set_training_mode,  # True
                       prompt_momentum=args.prompt_momentum)  # 0.01

        logits = output['logits']  # [24, 100]
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
        # TODO add contrastive loss
        # loss += orth_loss(output['pre_logits'], target, device, args)  # output['pre_logits']是什么？进入adapter前的768维特征（区别于输入到100分类头前的logits）。
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss1=loss1.item())
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])  # 只输出prompt的学习率
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()  # def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
def evaluate(model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # # 第一轮预测推理prompt [测试策略1]
            use_e_promot, use_prompt_mask, prompt_pool = model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool
            model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool= False, False, False
            output1 = model(input)
            model.module.use_e_prompt, model.module.use_prompt_mask, model.module.prompt_pool = use_e_promot, use_prompt_mask, prompt_pool
            logits1 = output1['logits']
            if args.train_mask and class_mask is not None:
                mask = []  # mask掉没有训练过的分类头
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits1 = logits1.index_fill(dim=1, index=not_mask, value=float('-inf'))
            prompt_id = torch.max(logits1, dim=1)[1]
            prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(-1)

            # 已知task ID用于prompt推理  [测试策略2] TODO
            # prompt_id = torch.tensor([target_task_map[v.item()] for v in target], device=device).unsqueeze(-1)

            output = model(input, task_id=task_id, prompt_id=prompt_id)
            logits = output['logits']  # [24, 100]
            promtp_idx = output['prompt_idx']  # tensor B x topk; [24, 1]

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            task_inference_acc = utils.task_inference_accuracy(promtp_idx, target, target_task_map)

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])  # 所有学过任务的分类头在任务i的验证样本集上的acc@1
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])  # 所有学过任务的分类头在任务i的验证样本集上的acc@5
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])  # 表示task ID(准确来讲是 expert prompt id)推理的结果

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']  # 增量模型在 task i 验证上的准确率
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    # 如果task中的类别个数相等 则avg_stat的结果表示在所有任务上的测试结果
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)  # 沿着列(axis=1)操作，结果就是列(axis=1)向量，是对行(axis=0)进行求和

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],  # Acc@task表示prompt id推理的结果
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer1,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler1,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  # 10,10
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  # 10,10
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    output_file = os.path.join(args.output_dir,  # './output/cifar100_vit_pe_seed42/log_2024_08_28_11_11_stats.txt'
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M')))

    for task_id in range(args.num_tasks):
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task-specific prompt better.
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]  # prompt的参数 384000
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'head' in name and p.requires_grad == True]  # head的参数 76900
                # base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                #                 'prompt' not in name and p.requires_grad == True]  # head的参数 76900
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            route_net_params = [p for name, p in model_without_ddp.named_parameters() if 'route_net' in name and p.requires_grad == True]  # 76900个参数可训练
            route_net_params = {'params': route_net_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
            optimizer1 = create_optimizer(args, [route_net_params])
            lr_scheduler1 = None
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None
        
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:  # 使用历史e_prompt初始化当前要训练的e_prompt
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (
                        slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(None), slice(None),
                        slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()  # [5,2,10,5,12,64]; 10表示task数量
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]  # 拷贝prev的prompt参数导cur的prompt中
                            # optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            # optimizer.param_groups[0]['params'] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:   # （如果有的话）使用历史prompt初始化当前要训练的prompt
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        for epoch in range(args.epochs):  # original_model有没有训练？ 有，在第一阶段训练好的。
            train_stats = train_one_epoch(model=model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer1=optimizer1, optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,  # max_norm=1.0
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,  # task_id=0
                                            target_task_map=target_task_map, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        if args.prompt_momentum > 0 and task_id > 0:
            if args.use_prefix_tune_for_e_prompt:
                with torch.no_grad():
                    # print(model.module.e_prompt.prompt[:, :, task_id].shape)
                    # print(
                    #     model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                    model.module.e_prompt.prompt[:, :, task_id].copy_(  # 将前一个任务的prompt和当前任务的prompt按prompt_momentum加权求和
                        (1 - args.prompt_momentum) * model.module.e_prompt.prompt[:, :, task_id].detach().clone()
                        + args.prompt_momentum * model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                            dim=2))

        pre_ca_test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                                device=device,
                                                task_id=task_id, class_mask=class_mask,
                                                target_task_map=target_task_map,
                                                acc_matrix=pre_ca_acc_matrix, args=args)  # pre_ca_acc_matrix 比 acc_matrix 少了task 0的结果
        # compute mean and variance
        if not args.no_mean_preprompt:
            cls_data = _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)

        if task_id == 0:
            test_stats = pre_ca_test_stats
            acc_matrix[:,task_id] = pre_ca_acc_matrix[:,task_id]
        else:
            if args.no_mean_preprompt:
                test_stats = pre_ca_test_stats
                acc_matrix[:,task_id] = pre_ca_acc_matrix[:,task_id]
            else:                
                train_task_adaptive_prediction(model, args, device, class_mask, task_id, cls_data)  # 用 pass 的思想 fine tuning 全连接层（区别在于一个类用了多个 prompt ）
                test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if hasattr(lr_scheduler, 'state_dict') and args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_pre{k}': v for k, v in pre_ca_test_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                }

        if args.output_dir and utils.is_main_process():
            with open(output_file, 'a') as f:
                f.write(json.dumps(log_stats).replace('"\\t1": "\\n",', '\n').replace('"\\t2": "\\n",', '\n') + '\n\n')
    
    print('pre_ca_acc_matrix:\n{} \n\nacc_matrix:\n{}'.format(
        np.array2string(pre_ca_acc_matrix, precision=2, separator=', ', suppress_small=True),
        np.array2string(acc_matrix, precision=2, separator=', ', suppress_small=True)
    ))


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()
    cls_data = dict()
    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        dist.barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method in {'multi-centroid', 'feat-trans'}:
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=1)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            cluster_datas = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
               cluster_datas.append(torch.tensor(cluster_data, dtype=torch.float64).to(device)) 
            
            cls_mean[cls_id] = cluster_means  # L216, L217 已经将cls_mean和cls_cov global化了
            cls_cov[cls_id] = cluster_vars
            cls_data[cls_id] = cluster_datas

    if task_id>0:
        return cls_data
    else:
        return 

def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1, cls_data=None):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]  # 只有全连接层的参数可训练
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):  # 历史任务的中心
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5 

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):  # 所有学过任务的类中心
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_label = torch.tensor(sampled_label).long().to(device)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
            
            sampled_label = torch.tensor(sampled_label).long().to(device)

        elif args.ca_storage_efficient_method == 'feat-trans':
            for i in range(task_id):  # 为历史任务生成样本
                for c_id in class_mask[i]:  # 为任务中的每个类别生成样本
                    for cluster in range(len(cls_mean[c_id])):  # 为任务中类别的簇中心生成样本 每个簇中心生成一个batch_size的样本
                        mean = cls_mean[c_id][cluster]
                        # 找到与历史中心最接近的当前簇中心
                        min_dist = float('inf')
                        closest_cluster_data = None
                        for j in class_mask[task_id]:
                            for current_cluster_id in range(len(cls_data[j])):
                                current_cluster_data = cls_data[j][current_cluster_id]
                                current_center = cls_mean[j][current_cluster_id]
                                dist = 1 - F.cosine_similarity(mean, current_center, dim=0)  # torch.norm(mean - current_center)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_cluster_data = current_cluster_data - current_center + mean
                                    current_cluster_label = [c_id]*len(closest_cluster_data)
                        sampled_data.append(closest_cluster_data)
                        sampled_label.append(current_cluster_label)

            assert set(class_mask[task_id]) == set(cls_data.keys()), "class_mask[task_id] 和 cls_data.keys() 不相同"
            for i in class_mask[task_id]:  # 为当前任务生成样本
                for cluster in range(len(cls_data[i])):
                    sample = cls_data[i][cluster]
                    sampled_data.append(sample)
                    sampled_label.append([i]*len(sample))
            
            sampled_label = torch.tensor(list(chain.from_iterable(sampled_label))).long().to(device)

            num_sampled_pcls = num_sampled_pcls if args.batch_size * 5 * crct_num <= sampled_label.shape[0] else sampled_label.shape[0] // crct_num
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        # sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()


def orth_loss(features, targets, device, args):  # 为什么要让batch内的特征正交？（类别间的特征正交可以理解）
    if cls_mean:  # 没有使用targets构造SupCon Loss TODO
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)  # 将历史存储的特征中心拼成m*768的矩阵（每个历史类含有10个特征中心）
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8  # 
        loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(device))
        return args.reg * loss
        # return 0.