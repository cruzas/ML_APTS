import math
import time 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
from utils import *

class Data_Parallelized_Model(nn.Module):
    def __init__(self, model, rank_list, device):
        super(Data_Parallelized_Model, self).__init__()
        self.rank_list = rank_list
        self.model = model
        self.rank = dist.get_rank()
        self.backend = dist.get_backend()
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        self.backend_device = 'cpu' if self.backend == 'gloo' else self.tensor_device
        # Group containing all the ranks of a replica
        self.replica_group = dist.new_group(ranks=[r for r in rank_list if self.rank in r][0], use_local_synchronization=True) 
        # Make a list of lists where each list is a group of ranks that are responsible for a layer in that position (e.g. [ [0, 3, 6], [1, 4, 7], [2, 5, 8] ] for 9 ranks)
        layer_ranks = [[ranks[i] for ranks in rank_list] for i in range(len(rank_list[0]))]
        # Group containing all the ranks corresponding to copies of a layer across all replicas
        for lr in layer_ranks:
            if self.rank in lr:
                self.layer_group = dist.new_group(ranks=lr, use_local_synchronization=True) 
        # Group containing all the ranks of the final layer of each replica
        self.final_layer_group = dist.new_group(ranks=[r[-1] for r in rank_list], use_local_synchronization=True)
        
    def subdomain_forward(self):
        # Forward pass on the subdomain in weights for the current subdomain
        return self.model.subdomain.forward()
    
    def subdomain_backward(self):
        self.model.subdomain.backward()
        if len(self.rank_list) > 1:
            # Synchronize the gradients across the subdomains in weights (due to pure data parallel approach inside the subdomain)
            for param in self.model.subdomain.parameters():
                dist.all_reduce(tensor=param.grad.to(self.backend_device), group=self.layer_group, op=dist.ReduceOp.SUM)
                param.grad /= len(self.rank_list)
                param.grad.to(self.tensor_device)

    def forward(self, x, chunks_amount, reset_grad, compute_grad):
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
        
    def backward(self, losses):
        self.model.backward(losses)
        if len(self.rank_list) > 1: # No need for synchronization if there is only one replica within the data subdomain
            for param in self.model.parameters():
                dist.all_reduce(tensor=param.grad.to(self.backend_device), group=self.layer_group, op=dist.ReduceOp.SUM)
                param.grad /= len(self.rank_list) # NOTE: we also don't have this variable
                param.grad.to(self.tensor_device)

    def parameters(self, clone=False):
        return self.model.parameters(clone=clone)
