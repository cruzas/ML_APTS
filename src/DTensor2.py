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

def print_tensor_weight_in_gb(tensor):
    """
    Print the weight of a given tensor in gigabytes (GB).
    
    Args:
    - tensor: A PyTorch tensor.
    """
    # Get the number of elements in the tensor
    num_elements = tensor.numel()
    
    # Assuming the tensor dtype is float32, each element takes 4 bytes
    # Adjust this value if using a different data type
    element_size_bytes = tensor.element_size()
    
    # Calculate the total weight in bytes
    total_bytes = num_elements * element_size_bytes
    
    # Convert bytes to gigabytes (1 GB = 2^30 bytes)
    total_gb = total_bytes / (1024**3)
    
    print(f"The tensor weight is approximately {total_gb:.6f} GB.")
    
    
def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    mesh = init_device_mesh(f"cuda:{rank}", (int(world_size),))
    # mesh = init_device_mesh(["cuda:0", "cuda:1"], mesh_shape=(2, 1) )

    big_tensor = torch.randn(int(2e8),1,  device="cpu")
    print_tensor_weight_in_gb(big_tensor)
    big_tensor = big_tensor.to(f"cuda:{rank}")
    tic = time.time()
    multiplier = big_tensor.squeeze() @ big_tensor.squeeze()
    torch.cuda.set_device(rank)
    toc = time.time()
    print(f"rank{rank} Elapsed time: {toc-tic:2f} seconds")
    allocated = bytes_to_gb(torch.cuda.memory_allocated())
    reserved = bytes_to_gb(torch.cuda.memory_reserved())
    print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
    
    #remove big_tensor from CUDA
    big_tensor = big_tensor.to("cpu")
    torch.cuda.empty_cache()
    
    allocated = bytes_to_gb(torch.cuda.memory_allocated())
    reserved = bytes_to_gb(torch.cuda.memory_reserved())
    print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
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
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            tic = time.time()
            multiplier = my_dtensor.squeeze() @ my_dtensor.squeeze()
            toc = time.time()
            print(f"---rank{rank} Elapsed time: {toc-tic:2f} seconds")
            del multiplier
            torch.cuda.empty_cache()
            # print(f'Device {rank} matrix norm: {multiplier.norm()}')
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
