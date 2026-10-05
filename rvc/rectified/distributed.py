import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


def parse_devices(value):
    if str(value).strip().lower() == 'auto':
        return [f'cuda:{index}' for index in range(torch.cuda.device_count())] or ['cpu']
    items = [item.strip().lower() for item in str(value).split(',')]
    if items == ['cpu']:
        return items
    if not items or any(not item.startswith('cuda:') or not item[5:].isdigit() for item in items):
        raise ValueError('Use cpu, cuda:0, or a comma-separated GPU list such as cuda:0,cuda:1.')
    devices = [f'cuda:{int(item[5:])}' for item in items]
    if len(devices) != len(set(devices)):
        raise ValueError('Each selected GPU must be unique.')
    return devices


def _host_allreduce(world, bucket):
    buffer = bucket.buffer()
    host = buffer.detach().to('cpu', copy=True)
    dist.all_reduce(host)
    host.div_(world)
    future = torch.futures.Future()
    future.set_result(host.to(buffer.device))
    return future


class Ranks:
    def __init__(self, rank, devices, init_method=None):
        self.rank = rank
        self.world = len(devices)
        self.main = rank == 0
        self.device = torch.device(devices[rank])
        self.init_method = init_method
        self.control_group = None
        self.backend = 'nccl' if self.device.type == 'cuda' and sys.platform != 'win32' else 'gloo'
        self.collective_device = self.device if self.backend == 'nccl' else torch.device('cpu')

    def setup(self):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group(self.backend, init_method=self.init_method,
                                    rank=self.rank, world_size=self.world, timeout=timedelta(minutes=10),
                                    device_id=self.device if self.backend == 'nccl' else None)
            self.control_group = dist.new_group(backend='gloo', timeout=timedelta(hours=1))

    def close(self):
        if self.world > 1 and dist.is_initialized():
            if self.control_group is not None:
                dist.destroy_process_group(self.control_group)
                self.control_group = None
            dist.destroy_process_group()

    def barrier(self):
        if self.world > 1:
            dist.monitored_barrier(group=self.control_group, timeout=timedelta(hours=1),
                                   wait_all_ranks=True)

    @contextmanager
    def main_work(self, label):
        if self.world == 1:
            yield True
            return
        self.barrier()
        failure = None
        try:
            yield self.main
        except Exception as error:
            failure = error
        self.barrier()
        result = [f'{type(failure).__name__}: {failure}' if failure is not None else None]
        dist.broadcast_object_list(result, src=0, group=self.control_group)
        if failure is not None:
            raise failure
        if result[0] is not None:
            raise RuntimeError(f'Rank 0 failed during {label}: {result[0]}')

    def sum(self, value):
        if self.world == 1:
            return value
        reduced = value.detach().to(self.collective_device, copy=True)
        dist.all_reduce(reduced)
        return reduced.to(value.device)

    def all_true(self, value):
        flag = torch.tensor(float(value), device=self.collective_device)
        return bool(self.sum(flag).item() == self.world)

    def wrap(self, model):
        if self.world == 1:
            return model
        device_ids = [self.device.index] if self.device.type == 'cuda' else None
        host_communication = self.backend == 'gloo' and self.device.type == 'cuda'
        if host_communication:
            tensors = list(model.parameters()) + list(model.buffers())
            shapes = [(tuple(value.shape), str(value.dtype)) for value in tensors]
            gathered = [None] * self.world
            dist.all_gather_object(gathered, shapes)
            if any(value != shapes for value in gathered):
                raise ValueError('Distributed model shapes or dtypes differ between ranks.')
            with torch.no_grad():
                for tensor in tensors:
                    host = tensor.detach().to('cpu', copy=True)
                    dist.broadcast(host, src=0)
                    tensor.copy_(host)
        wrapped = DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False,
                                          init_sync=not host_communication)
        if host_communication:
            wrapped.register_comm_hook(self.world, _host_allreduce)
        return wrapped

    def sampler(self, dataset, seed):
        if self.world == 1:
            return None
        return DistributedSampler(dataset, num_replicas=self.world, rank=self.rank,
                                  shuffle=True, seed=seed, drop_last=True)


def _run_rank(rank, target, args, devices, init_method):
    ranks = Ranks(rank, devices, init_method)
    try:
        ranks.setup()
        target(args, ranks)
    finally:
        ranks.close()


def launch(target, args):
    devices = parse_devices(args.device)
    if devices[0] != 'cpu':
        count = torch.cuda.device_count()
        if any(int(item[5:]) >= count for item in devices):
            raise ValueError(f'Selected GPU is unavailable; this machine has {count} CUDA GPU(s).')
    if len(devices) == 1:
        _run_rank(0, target, args, devices, None)
        return
    with tempfile.TemporaryDirectory(prefix='rectified-ddp-') as folder:
        init_method = (Path(folder) / 'rendezvous').resolve().as_uri()
        torch.multiprocessing.spawn(_run_rank, args=(target, args, devices, init_method),
                                    nprocs=len(devices), join=True)
