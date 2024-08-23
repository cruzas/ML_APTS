import torch
import torch.distributed as dist
import utils
from pmw.base_model import BaseModel
from pmw.weight_parallelized_model import WeightParallelizedModel

class DataAndWeightParallelizedSubdomain(BaseModel):
    def __init__(self, stage_list, rank_list, sample, num_replicas_per_subdomain=1):
        super(DataAndWeightParallelizedSubdomain, self).__init__()
        '''
        This function defines a subdomain in data. The subdomain has subdomains in weights and can parallelize forward and backward passes in data.
        
        rank_list = list of lists containing the ranks of the model replicas that are responsible for each layer, e.g. [[0, 1], [2, 3], [4, 5]] (3 replicas, 2 layers each replica)
        '''
        self.stage_list = stage_list
        self.rank_list = rank_list
        self.num_replicas_per_subdomain = num_replicas_per_subdomain
        self.create_process_groups()
        # Initialize WeightParallelizedModel replicas 
        for model_ranks in self.rank_list:
            if self.rank in model_ranks:
                self.weight_parallelized_model = WeightParallelizedModel(stage_list=stage_list, rank_list=model_ranks, sample=sample)
        # Synchronize the parameters across all replicas
        self.sync_params() 

    def create_process_groups(self):
        self.final_layer_group = None
        self.current_layer_group = None
        if self.num_replicas_per_subdomain > 1:
            use_local_synchronization = True
            weight_parallelized_model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(self.num_replicas_per_subdomain)]
            for stage_ranks in weight_parallelized_model_ranks:
                if self.rank in stage_ranks:
                    self.current_layer_group = dist.new_group(ranks=stage_ranks, use_local_synchronization=use_local_synchronization) # Create a group for the current layer (across all data parallel replicas)       
                    break
            self.final_layer_group = dist.new_group(ranks=[ranks[-1] for ranks in self.rank_list], use_local_synchronization=use_local_synchronization)
        
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.weight_parallelized_model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses, sync=True):
        self.weight_parallelized_model.backward(losses=losses)
        self.sync_grads() 
    
    def sync_params(self, method='average'):
        for param in self.weight_parallelized_model.parameters():
            dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            if method == 'average':
                param.data /= self.num_replicas_per_subdomain
            elif method == 'sum':
                pass # nothing to do since we already summed the parameters through all_reduce
            else:
                raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        for param in self.weight_parallelized_model.parameters():
            dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            param.grad /= self.num_replicas_per_subdomain