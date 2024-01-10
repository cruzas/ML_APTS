# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os

import torch, time
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor
import torch.multiprocessing as mp
from utils.utility import prepare_distributed_environment
from torch import distributed as dist

# Create a mesh topology with the available devices:
# 1. We can directly create the mesh using elastic launcher, (recommended)
# 2. If using mp.spawn, one need to initialize the world process_group first and set device
#   i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)

def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def test_for_loop_parallel(rank, mesh):
    big_tensor = torch.randn(int(2e4), int(2e4), device="cpu")
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            multiplier = my_dtensor @ my_dtensor
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Device {rank}. Elapsed time: {elapsed_time/1000} seconds")


def my_prep(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if master_addr is None:
        master_addr = os.environ["MASTER_ADDR"]
    if master_port is None:
        master_port = os.environ["MASTER_PORT"]
    if world_size is None: 
        world_size = dist.get_world_size() if dist.is_initialized() else 1


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    my_prep(rank, master_addr, master_port, world_size)

    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Mesh with Nodes 0 and 1
    if rank in [0, 1]:
        mesh1 = init_device_mesh(f"cuda:0", (2,))
        test_for_loop_parallel(rank, mesh1)

    # Mesh with Nodes 2 and 3
    if rank in [2, 3]:
        mesh2 = init_device_mesh(f"cuda:0", (2,))
        test_for_loop_parallel(rank, mesh2)


def main2(rank=None, master_addr=None, master_port=None, world_size=None):
    my_prep(rank, master_addr, master_port, world_size)
    big_tensor = torch.randn(int(2e4), int(2e4), device=f"cuda:0")
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            multiplier = big_tensor @ big_tensor
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Device {rank}. Elapsed time: {elapsed_time/1000} seconds")



if __name__ == '__main__':
    main()


