# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os
import subprocess
import math
import torch, time
from torch.distributed._tensor import init_device_mesh, Shard, Replicate, distribute_tensor
import torch.multiprocessing as mp
# from utils.utility import prepare_distributed_environment
from torch import distributed as dist
# from utils.new_mesh import *

# Create a mesh topology with the available devices:
# 1. We can directly create the mesh using elastic launcher, (recommended)
# 2. If using mp.spawn, one need to initialize the world process_group first and set device
#   i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    device_id = 0
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
        ## Execute code on a cluster
        os.environ["MASTER_PORT"] = "29501"
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = os.environ["SLURM_NODEID"]
        node_list = os.environ["SLURM_NODELIST"]
        master_node = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = master_node
        print(f"Dist initialized before process group? {dist.is_initialized()}")
        dist.init_process_group(backend="nccl")
        print(f"Dist initialized after init process group? {dist.is_initialized()} with world size {dist.get_world_size()}")
    else: # we are on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port # A free port on the master node
        # os.environ['WORLD_SIZE'] = str(world_size) # The total number of GPUs in the distributed job
        # os.environ['RANK'] = '0' # The unique identifier for this process (0-indexed)
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # "nccl" or "gloo"
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device_id = dist.get_rank()
    print(f"Device id: {device_id}")

# new "init_device_mesh" function


def main2(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
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






def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)



def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Define which ranks belong to each group
    if rank < 2:
        # First group of ranks 0 and 1
        group = dist.new_group(ranks=[0, 1])
    else:
        # Second group of ranks 2 and 3
        group = dist.new_group(ranks=[2, 3])
        
    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0

    if dist.get_backend() == "nccl":
        mesh = init_device_mesh(f"cuda:0", (2,))
    else:
        mesh = init_device_mesh(f"cuda:0", (2,))
        # mesh = init_device_mesh(f"cuda", (2,))

     # Set GPU based on rank; assuming each rank gets a separate GPU for this example
    gpu_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:0")
    group_rank = dist.get_rank(group)
    print(f"Global rank: {rank}, Group rank: {group_rank}, Using GPU: {gpu_id}")



    # Set seed
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    big_tensor = torch.eye(4) # torch.randn(int(2e4), int(2e4), device="cpu")
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    # big_tensor = torch.randn(1,100000000, dtype=torch.float32, device="cpu") # torch.ones(4,1) #torch.randn(int(2e4),1, device=f"cpu")
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])

    # vector = torch.tensor([i for i in range(10000000)], dtype=torch.float32) # torch.ones(4,1) #torch.randn(int(2e4),1, device=f"cpu")
    # vector = torch.randn(100000000,1, dtype=torch.float32, device="cpu") # torch.ones(4,1) #torch.randn(int(2e4),1, device=f"cpu")
    # vector = torch.randn(4,100000000, dtype=torch.float32, device="cpu") # torch.ones(4,1) #torch.randn(int(2e4),1, device=f"cpu")
    vector = torch.tensor([1, 2, 3, 4], dtype=torch.float32).unsqueeze(1)
    # vector = torch.eye(4)
    # my_vector = distribute_tensor(vector, mesh, [Shard(dim=0)]).squeeze()
    my_vector = distribute_tensor(vector, mesh, [Shard(dim=1)])
    # my_vector.to(f"cuda:{rank}")
    
    


    with torch.no_grad():
        for i in range(3):
            if rank == 0:
                print(i)

            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            # asd = my_dtensor @ my_vector
            # multiply the distributed tensor with the distributed vector
            multiplier = my_dtensor @ my_vector
            dist.barrier()
            
            # Here, I want to unshard multiplier and combine the top and bottom part into one tensor
            # full_multiplier = dist.gather([multiplier, None, None], dst=0)
            # print("Full multiplier: ", full_multiplier)
            # full_multiplier = distribute_tensor(multiplier, device_mesh=mesh, placements=[Replicate()])
            # print("Full multiplier: ", full_multiplier)

            # print(f"Device {rank} multiplier@multiplier: {multiplier.T@multiplier}")
            end_event.record()
            torch.cuda.synchronize()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Device {rank}. Elapsed time: {elapsed_time/1000} seconds")

    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    if "snx" in os.getcwd():
        main()
    else:
        # world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        nthreads = 4
        world_size = nthreads
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)
        master_addr = 'localhost'
        master_port = '12345'
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=nthreads, join=True)
        