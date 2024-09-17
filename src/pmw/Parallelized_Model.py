import math
import time 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
from utils import *
from pmw.Data_Parallelized_Model import Data_Parallelized_Model
from pmw.Weight_Parallelized_Model import Weight_Parallelized_Model

class Parallelized_Model(nn.Module):
    """
    Data parallel and weight parallel model wrapper.
    Meaning of the parameters:
    - stage_list: List of tuples. Each tuple contains a layer class and its parameters. The layer class is a function that returns a layer (e.g. nn.Linear, nn.Conv2d, nn.Sequential, etc.). The parameters are the parameters of the layer class.
    """
    def __init__(self, stage_list, sample, num_replicas_per_dsd=1, num_dsd=1, device=None):
        super(Parallelized_Model, self).__init__()
        
        self.rank = dist.get_rank()
        self.num_replicas_per_dsd = num_replicas_per_dsd
        self.num_dsd = num_dsd
        self.tot_replicas = num_replicas_per_dsd * num_dsd
        self.world_size = dist.get_world_size()
        self.backend_device = 'cpu' if dist.get_backend() == 'gloo' else 'cuda:0' # TODO: In the future, change this 'cuda:0' in case using different GPUs
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        if self.tot_replicas*len(stage_list) != self.world_size:
            raise ValueError(f"The number of total replicas times the number of layers ({self.tot_replicas}*{len(stage_list)}={self.tot_replicas*len(stage_list)}) must be equal to the world size ({self.world_size}).")
        
        last_layers = [r for r in range(self.world_size) if r%len(stage_list) == len(stage_list)-1]
        # Split last layers into num_dsd pieces
        last_layers = [last_layers[i:i+len(stage_list)] for i in range(0, len(last_layers), len(stage_list))] # [ [1, 3, 5] , [7, 9, 11] ]
        temp = 0
        for l in last_layers:
            if self.rank >= temp and self.rank <= max(l):   # rank >= 0 and rank <= 5
                self.dsd_final_layer_group = dist.new_group(ranks=l, use_local_synchronization=True)
            temp = max(l)+1

        # Model copy rank list
        self.stage_list = stage_list
        self.model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(self.tot_replicas)] 
        for model_rank in self.model_ranks:
            if self.rank in model_rank:
                self.rank_list = model_rank # Rank list of the current model replica
                # self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
                # self.dsd_master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
                # break
        self.final_layer_group = dist.new_group(ranks=[r for r in range(self.world_size) if r%len(stage_list) == len(stage_list)-1], use_local_synchronization=True)

        # Split the model ranks into num_dsd pieces
        self.dsd_model_ranks = [self.model_ranks[i:i+num_replicas_per_dsd] for i in range(0, len(self.model_ranks), num_replicas_per_dsd)] 
        # Creates the models and defines subdomains to handle the models more easily
        for dsd_ranks in self.dsd_model_ranks: # Loops over the set of ranks of the models in different data subdomain, e.g. [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] for 2 replicas and 2 data subdomains each one with pipeline of 3 layers -> 4 total replicas
            for ranks in dsd_ranks: # dsd_ranks = [[0, 1, 2], [3, 4, 5]] for 2 replicas within the first data subdomain, each one with pipeline of 3 layers -> 2 replicas in the first subdomain
                if self.rank in ranks: # ranks = [0, 1, 2] for the first replica in the first data subdomain -> this is a model replica
                    wp_model = Weight_Parallelized_Model(stage_list=stage_list, rank_list=ranks, sample=sample, device=device)
                    self.model = Data_Parallelized_Model(model=wp_model, rank_list=dsd_ranks, device=device)
                    self.subdomain = self.model.model.subdomain
        
        # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
        for layer_idx in range(len(self.stage_list)):
            # Collect ranks that are responsible for this layer across all replicas
            ranks = [r + layer_idx for r in range(0, self.world_size, len(self.stage_list))]
            # Create a new group containing these ranks
            if self.rank in ranks:
                self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
        self.sync_params()
    
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses):
        self.model.backward(losses=losses)
        self.sync_global_grads()

    def subdomain_forward(self):
        self.model.subdomain_forward()
    
    def subdomain_backward(self):
        self.model.subdomain_backward()

    def subdomain_params(self):
        return self.model.subdomain.parameters()
        
    def subdomain_grad(self):
        return self.model.subdomain.grad()
    
    def subdomain_grad_norm(self, p=2):
        return self.model.subdomain.grad_norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        return self.model.parameters(clone=clone)
    
    def parameters_norm(self, p=2): # Returns the global parameters norm of the model
        return self.model.parameters().norm(p=p)
    
    def grad(self, clone=False): # Returns the global gradient of the model
        return self.model.model.grad(clone=clone)

    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.model.grad_norm(p=p)
            
    def sync_params(self, method='average'):
        if len(dist.get_process_group_ranks(self.layer_copies_group)) > 1:
            for param in self.model.parameters():
                dist.all_reduce(tensor=param.data.to(self.backend_device), group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                if method == 'average':
                    param.data /= self.num_replicas_per_dsd
                elif method == 'sum':
                    pass # nothing to do since we already summed the parameters through all_reduce
                else:
                    raise ValueError(f"Method {method} is not supported.")
                param.data.to(self.tensor_device)

    def sync_global_grads(self):
        if self.num_dsd > 1: # No need for synchronization if there is only one replica within the data subdomain
            # Sync the gradients across subdomains
            for param in self.model.parameters():
                dist.all_reduce(tensor=param.grad.to(self.backend_device), group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.grad /= self.num_replicas_per_dsd
                param.grad.to(self.tensor_device)
