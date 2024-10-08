import torch
import torch.distributed as dist
import utils
from pmw.base_model import BaseModel
from pmw.weight_parallelized_model import WeightParallelizedModel

class DataAndWeightParallelizedSubdomain(BaseModel):
    def __init__(self, stage_list, rank_list, sample):
        super().__init__()
        '''
        This function defines a subdomain in data. The subdomain has subdomains in weights and can parallelize forward and backward passes in data.
        
        rank_list = list of lists containing the ranks of the model replicas that are responsible for each layer, e.g. [[0, 1], [2, 3], [4, 5]] (3 replicas, 2 layers each replica)
        '''
        self.stage_list = stage_list
        self.rank_list = rank_list
        self.num_replicas_per_subdomain = len(rank_list)
        self._create_process_groups()
        # Initialize WeightParallelizedModel replicas 
        for model_replica_ranks in self.rank_list:
            if self.rank in utils.list_flattener(model_replica_ranks):
                self.weight_parallelized_model = WeightParallelizedModel(stage_list=stage_list, rank_list=model_replica_ranks, sample=sample)
        # Synchronize the parameters across all replicas
        self.sync_params() 
        
    # [[None] * len(self.stage_list) for _ in range(num_lists)]
    def _create_process_groups(self):
        self.final_layer_group = None
        self.layer_copies_group = None
        self.final_layer_ranks = None
        if self.num_replicas_per_subdomain > 1:                                                         
            weight_parallelized_model_ranks = [[None]*len(self.rank_list) for _ in range(len(self.stage_list)*len(self.rank_list[0][0]))] 
            for r in self.rank_list: # replica ranks
                for stage in r: # stage ranks
                    for shard in stage:
                        weight_parallelized_model_ranks[r.index(stage)*len(self.rank_list[0][0])+stage.index(shard)][self.rank_list.index(r)] = shard
            for layer_ranks in weight_parallelized_model_ranks:
                if self.rank in layer_ranks:
                    self.layer_copies_group = dist.new_group(ranks=layer_ranks, use_local_synchronization=True)

            # self.final_layer_ranks = [ranks[-1] for ranks in self.rank_list]
            # self.final_layer_group = dist.new_group(ranks=self.final_layer_ranks, use_local_synchronization=True)
        
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        return self.weight_parallelized_model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses, sync=True):
        self.weight_parallelized_model.backward(losses=losses)
        if sync:
            self.sync_grads()
    
    def sync_params(self, method='average'):
        # TODO/NOTE: Use the sharding class 
        if self.num_replicas_per_subdomain > 1:
            for param in self.weight_parallelized_model.subdomain.parameters():
                dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                if method == 'average':
                    param.data /= self.num_replicas_per_subdomain
                elif method == 'sum':
                    pass # nothing to do since we already summed the parameters through all_reduce
                else:
                    raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        if self.num_replicas_per_subdomain > 1:
            for param in self.weight_parallelized_model.subdomain.parameters():
                dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.grad /= self.num_replicas_per_subdomain