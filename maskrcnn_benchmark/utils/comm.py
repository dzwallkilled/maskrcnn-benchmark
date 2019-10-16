"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time
import subprocess
import os

import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_available_device(num_gpus=1, memory_used=100, memory_available=-1, gpus=[]):
    """
    Retrives the resources and return the available devices according to the requirement
    :param num_gpus: int, num of gpus to be used
    :param memory_used: int Mb, gpus with memory used less than this value, negative value ignores this option
    :param memory_available: int Mb, gpus with memory available larger than this value, negative value ignores this option
    :param gpus: list(int), the gpu range constrained, e.g. gpus=[0, 1, 2], allocating memory only to these gpus
    :return: torch.device('cuda') or torch.device('cpu')
    """
    print(f'Requirement: {num_gpus} GPUs with >{memory_available}M available and <{memory_used}M used')
    if memory_used < 0:
        memory_used = 1e10
    if memory_available < 0:
        memory_available = 1e10
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory_used = [int(x) for x in result.strip().split('\n')]

    result2 = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory_free = [int(x) for x in result2.strip().split('\n')]

    free_gpus = []
    for gpu, (mem_used, mem_free) in enumerate(zip(gpu_memory_used, gpu_memory_free)):
        if mem_used < memory_used or mem_free > memory_available:
            free_gpus.append(gpu)

    if gpus:
        free_gpus = [gpu for gpu in free_gpus if gpu in gpus]

    if num_gpus == 0:
        print(f'Allocating memory into CPU.')
        return torch.device('cpu')
    elif len(free_gpus) < num_gpus:
        print(f"Not enough GPUs available. {num_gpus} required but {len(free_gpus)} available.")
        print(f"Allocating memory into CPU.")
        return torch.device('cpu')
    else:
        gpus = ','.join(str(gpu) for gpu in free_gpus[:num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in free_gpus[:num_gpus])
        print(f'Allocating memory into GPU: {gpus}')
        return torch.device('cuda'), list(range(num_gpus))

