"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):   # '{avg:.4f}'
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 双端队列：.append头进；头出.pop; .popleft尾出
        self.total = 0.0
        self.count = 0
        self.fmt = fmt  # format

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)  # 创建默认字典类的类型 但不存在键时创建 当存在时自动检索 方便处理大量键值对
        self.delimiter = delimiter  # "  "

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):  # 以task分割的loader, 10, 'Train: Epoch[ 1/20]'
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'  # ':3d'       
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]  # ['Train: Epoch[ 1/20]', '[{0:3d}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']          ['Test: [Task 1]', '[{0:2d}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')  # ['Train: Epoch[ 1/20]', '[{0:3d}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}', 'max mem: {memory:.0f}']       ['Test: [Task 1]', '[{0:2d}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}', 'max mem: {memory:.0f}']
        log_msg = self.delimiter.join(log_msg)  # 'Train: Epoch[ 1/20]  [{0:3d}/{1}]  eta: {eta}  {meters}  time: {time}  data: {data}  max mem: {memory:.0f}'          'Test: [Task 1]  [{0:2d}/{1}]  eta: {eta}  {meters}  time: {time}  data: {data}  max mem: {memory:.0f}'
        MB = 1024.0 * 1024.0
        for obj in iterable:  # 数据在这里由3*32*32变成了3*224*224: .vscode/launch.json "justMyCode": false, 正确环境下的__getitem__函数
            data_time.update(time.time() - end)
            yield obj  # engines/hide_tii_engine.py 跑完一个epoch后开始执行以下内容
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,  # 0, 209, '2 days, 0:17:18'
                        meters=str(self),  # 'Lr: 0.000047  Loss: 2.2224  Acc@1: 16.6667 (16.6667)  Acc@5: 62.5000 (62.5000)'
                        time=str(iter_time), data=str(data_time),  # '831.7610', '0.6281'
                        memory=torch.cuda.max_memory_allocated() / MB))  # 673.53564453125
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # '0:22:18'
        print('{} Total time: {} ({:.4f} s / it)'.format(  # 'Train: Epoch[ 1/20] Total time: 0:22:18 (6.4052 s / it)'
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):  # TODO：不开并行先，会和tsne函数中的UMAP冲突。注释掉会导致终端跑实验时一个print重复输出8次
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)  # TODO：不开并行先，会和tsne中的UMAP冲突


def task_inference_accuracy(prompt_idx, target, target_task_map):  # 计算target的推理任务ID,prompt_idx,相较于任务ID gt, target_task_map,的准确率。
    target_2_task = torch.tensor([target_task_map[v.item()] for v in target]).to(prompt_idx.device)
    batch_size = target.size(0)
    prompt_idx = prompt_idx.t()
    correct = prompt_idx.eq(target_2_task.reshape(1, -1).expand_as(prompt_idx))
    return correct.reshape(-1).float().sum(0) * 100. / batch_size

