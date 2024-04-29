import torch
from mpi4py import MPI 

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print the rank and size of the MPI communicator (CPU core)
    print(f"Hello from rank {rank} out of {size}")

    if rank == 0:
        # Rank 0 creates a tensor and sends it to Rank 1
        tensor = torch.tensor([1, 2, 3], dtype=torch.int).to("cuda:0")
        comm.send(tensor, dest=1)
    elif rank == 1:
        # Rank 1 receives the tensor from Rank 0 and concatenates it with its tensor
        received_tensor = comm.recv(source=0)
        own_tensor = torch.tensor([4, 5, 6], dtype=torch.int).to("cuda:1")
        combined_tensor = torch.cat((received_tensor.to('cuda:1'), own_tensor), 0)
        print(f"Combined tensor on rank 1: {combined_tensor}")

if __name__ == "__main__":
    main()
