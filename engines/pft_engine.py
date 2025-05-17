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
from itertools import chain
import torch.nn.functional as F
import warnings


def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, ):
    model.train(set_training_mode)  # True

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'  # 'Train: Epoch[ 1/20]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):  # 单个任务的loader, 10, 'Train: Epoch[ 1/20]'。 data_loader的样本是32，input的样本是224。已在utils.py L134中解决。
        input = input.to(device, non_blocking=True)  # [24, 3, 224, 224]
        target = target.to(device, non_blocking=True)  # [24]

        output = model(input)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks        使用task ID信息
        if args.train_mask and class_mask is not None:  # Ture, [[], [], ..., []]
            mask = class_mask[task_id]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)  # [10, 11, ..., 99]
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            # print("Loss is {}, stopping training".format(loss.item()))
            # sys.exit(1)
            warnings.warn(f"NaN loss encountered. Skipping update.", RuntimeWarning)
            optimizer.zero_grad()  # 清除可能存在的错误梯度
            continue  # 跳过当前batch

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)  # 'Test: [Task 1]'

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(input)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device, task_id=i, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']  # global average results; task_id上训完的模型在task-i上的准确率,i<=task_id
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']  # 行表示test task, 列表示train task,的准确率矩阵。

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)  # 任务学完后，在所有学过任务上的结果。

    diagonal = np.diag(acc_matrix)  # 每学完一个任务在该任务自身上的准确率。

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[0],  # 学完一个新任务后，在所有学过任务上的top-1准确率。
        avg_stat[1],  # 学完一个新任务后，在所有学过任务上的top-5准确率。
        avg_stat[2])  # 学完一个新任务后，在所有学过任务上的训练loss。
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])  # 每学完一个任务后，模型相对以往学过任务上最佳性能的的平均遗忘率。
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])  # 每学完一个任务后，模型相对刚学历史任务时的平均遗忘率。

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, class_mask=None, task_id=None, args=None, ):
    model.eval()
    cls_data = dict()
    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)  # [500, 768]
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
        if args.ca_storage_efficient_method == 'multi-centroid':  # 将一个类的特征聚成多个簇，通过多个簇的mean和cov表示类别（而不是将一个类别表示成一个mean和cov）
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
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
          
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars
            cls_data[cls_id] = cluster_datas
    
    if task_id>0:
        return cls_data
    else:
        return 


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  # [10,10] 
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  # [10,10]
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    output_file = os.path.join(args.output_dir,
                           '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M')))

    for task_id in range(args.num_tasks):  # 10

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:  # True
            optimizer = create_optimizer(args, model)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        for epoch in range(args.epochs):  # 20
            train_stats = train_one_epoch(model=model, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,  # args.clip_grad=1.0
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, )  # 根据data 1训练adapter和head 1

            if lr_scheduler:  # None
                lr_scheduler.step(epoch)

        test_stats_pre_ca = evaluate_till_now(model=model, data_loader=data_loader,
                                              device=device,
                                              task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                              acc_matrix=pre_ca_acc_matrix, args=args)
        
        # compute mean and variance
        if not args.no_mean_tii:
            cls_data = _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, class_mask=class_mask[task_id], task_id=task_id,
                                    args=args)

        # classifier alignment
        if task_id == 0:
            test_stats = test_stats_pre_ca
            acc_matrix[:,task_id] = pre_ca_acc_matrix[:,task_id]
        else:
            if args.no_mean_tii:
                test_stats = test_stats_pre_ca
                acc_matrix[:,task_id] = pre_ca_acc_matrix[:,task_id]
            else:
                train_task_adaptive_prediction(model, args, device, class_mask, task_id, cls_data)  # 根据新旧特征中心生成新旧特征训练adapter和heads 0-1. [args.ca_lr=0.00046875, momentum=0.9, weight_decay=5e-4]
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
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_pre{k}': v for k, v in test_stats_pre_ca.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process():
            # with open(os.path.join(args.output_dir,
            #                        '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
            with open(output_file , 'a') as f:
                f.write(json.dumps(log_stats) + '\n\n')

    print('pre_ca_acc_matrix:\n{} \n\nacc_matrix:\n{}'.format(
        np.array2string(pre_ca_acc_matrix, precision=2, separator=', ', suppress_small=True),
        np.array2string(acc_matrix, precision=2, separator=', ', suppress_small=True)
    ))


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1, cls_data=None):
    model.train()
    run_epochs = args.crct_epochs  # 30
    crct_num = 0
    param_list = [p for p in model.parameters() if p.requires_grad]
    # for name, p in model.named_parameters():  # model.module.mlp 和 model.module.head 两个模块是可训练参数; 参数量为2440036
    #     if p.requires_grad:
    #         # print(name)
    #         pass
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]  # 保存在 param_groups 中
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)  # 重复设置lr和weight_decay 但network_params的优先级更高 所以保持network_params设置的参数。保存在defaults中

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):  # 累加历史学过的class数
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
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
        else:
            raise NotImplementedError

        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(list(chain.from_iterable(sampled_label))).long().to(device)
        # print(sampled_data.shape)  # 4776 768, 4776/24=199

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))  # 因为随机化 + L301 epoch 所以训练过程中用了所有学过的class簇特征。 
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        #print(targets)

        for _iter in range(crct_num):  # 遍历历史学过的class, 不包含刚学的; 跑10个iterations, 但因为对L353做了随机化, 所以实际训练时Finetuning所有的分类头。
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:  # mask掉还未学过的分类头
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                # print("Loss is {}, stopping training".format(loss.item()))
                # sys.exit(1)
                warnings.warn(f"NaN loss encountered at epoch {epoch}, batch {_iter}. Skipping update.", RuntimeWarning)
                optimizer.zero_grad()  # 清除可能存在的错误梯度
                continue  # 跳过当前batch

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
        # print("Averaged stats:", metric_logger)
        scheduler.step()