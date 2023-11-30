import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
def main(rank, size):
    #MASTER_ADDR #
    os.environ['MASTER_ADDR'] = 'localhost'
    #MASTER_PORT #
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=size)

    # print what backend is being used
    print("Using backend: ", dist.get_backend())
    
    device_count = torch.cuda.device_count()
    print(f"Hello from rank {rank} out of {size}. I have {device_count} GPUs available.")

    if rank == 0:
        # Rank 0 creates a tensor and sends it to Rank 1
        tensor = torch.tensor([1, 2, 3], dtype=torch.int).to("cuda:0").to('cpu')
        dist.send(tensor, dst=1)
    elif rank == 1:
        # Rank 1 receives the tensor from Rank 0 and concatenates it with its tensor
        received_tensor = torch.empty(3, dtype=torch.int)
        dist.recv(received_tensor, src=0)
        own_tensor = torch.tensor([4, 5, 6], dtype=torch.int)
        combined_tensor = torch.cat((received_tensor, own_tensor), 0)
        print(f"Combined tensor on rank 1: {combined_tensor}")

    dist.destroy_process_group()

def run_demo(world_size):
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = 2  # Number of processes
    run_demo(world_size)