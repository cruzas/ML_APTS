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

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    # mesh = init_device_mesh(f"cuda:{rank}", (int(world_size),))
    mesh = init_device_mesh(["cuda:0", "cuda:1"])

    big_tensor = torch.randn(int(2e4), int(2e4), device="cpu")
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            torch.cuda.set_device(rank)
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            multiplier = my_dtensor @ my_dtensor
            print(f'Device {rank} matrix norm: {multiplier.norm()}')
            end_event.record()
            torch.cuda.synchronize()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)


            if rank == 0:
                print(f"Elapsed time: {elapsed_time/1000} seconds")

            

def main2(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    big_tensor = torch.randn(int(2e4), int(2e4), device=f"cuda:{rank}")
    
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            torch.cuda.set_device(rank)
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
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size == 0:
        print("No CUDA device(s) detected.")
        exit(0)
    master_addr = 'localhost'
    master_port = '12345'  
    mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
