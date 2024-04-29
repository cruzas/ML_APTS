import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os, pickle
from optimizers.TR import TR
from models.neural_networks import MNIST_CNN
from utils.utility import *


def serialize_dict(dict_obj):
    """ Serialize a dictionary into a byte tensor. """
    buffer = pickle.dumps(dict_obj)
    return torch.ByteTensor(list(buffer))

def deserialize_dict(tensor_obj):
    """ Deserialize a byte tensor back into a dictionary. """
    buffer = bytes(tensor_obj.tolist())
    return pickle.loads(buffer)

def broadcast_dict(rank, dict_obj=None):
    """ Broadcast a dictionary from rank 0 to all other ranks. """
    if rank == 0:
        # Serialize the dictionary into a tensor
        tensor = serialize_dict(dict_obj)
    else:
        tensor = torch.tensor([0], dtype=torch.long)

    # Broadcast the tensor
    dist.broadcast(tensor, src=0)

    # Deserialize the tensor back into a dictionary
    dict_obj = deserialize_dict(tensor)
    return dict_obj

    
def main(rank, world_size):
    args = parse_args()

    #MASTER_ADDR #
    os.environ['MASTER_ADDR'] = 'localhost'
    #MASTER_PORT #
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # Example usage
    rank = dist.get_rank()  # Get the rank of the current process
    print(f'Rank {rank} has been initialized.')

    model = MNIST_FCNN()
    # Get the shape
    shape = None
    if rank == 0:
        # print('a')
        # some_dict = TR(model.parameters())
        # print('b')
        # buffer = pickle.dumps(some_dict)  # Serialize the dictionary
        # print('c')
        # some_dict = torch.ByteTensor(list(buffer))  # Convert to byte tensor, to convert it back use pickle.loads(bytes(tensor.tolist()))
        # lis = [some_dict]
        lis = torch.tensor([torch.tensor([1,1,1]),torch.tensor([0,0,0])])
    else:
        lis = torch.tensor([torch.tensor([1,2,3]),torch.tensor([1,2,3])])

    # dist.broadcast_object_list(lis, 0)  # Broadcast the tensor from rank 0
    dist.all_reduce(lis, op=dist.ReduceOp.SUM)

    # for i in range(len(lis)):
    #     # lisi = pickle.loads(bytes(lis[i].tolist()))
    #     # print(f"Rank {rank} some_dict: {lisi}")
    #     print(f"Rank {rank} some_dict: {lis[i]}")
    print(f"Rank {rank} some_dict: {lis}")



if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    mp.spawn(main,
            args=(world_size,),
            nprocs=world_size,
            join=True)
