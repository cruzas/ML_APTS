import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# Add the path to the sys.path
import os
import sys
# Make the following work on Windows and MacOS
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.utility import prepare_distributed_environment, create_dataloaders

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    # Initialize two process groups, one with ranks 0 and 1, one with ranks 2 and 3
    pg0 = dist.new_group([0, 1])
    pg1 = dist.new_group([2, 3])

    if rank in [0, 1]:
        print(f"Rank {rank} group rank {torch.distributed.get_group_rank(pg0, rank)}")
    else:
        print(f"Rank {rank} group rank {torch.distributed.get_group_rank(pg1, rank)}")
        
    if rank in [2,3]:
        # set the process group and rename the ranks to 0 and 1


if __name__ == '__main__':
    torch.manual_seed(1)
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'   
        world_size = 4
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)