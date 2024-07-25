import math
import time 
import torch
import os
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import diffdist.functional as dist
from parallel.utils import *
import logging
from inspect import currentframe, getframeinfo

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def get_filename(file_path):
    return os.path.basename(file_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
def sync_operation(filename=None,line_number=None):
    pass
    # global counter
    # counter += 1
    # logger.info(f"Rank {dist.get_rank()} starting sync operation. Python script: {filename} | Line number: {line_number}")
    # dist.barrier()
    # # torch.cuda.synchronize()
    # logger.info(f"Rank {dist.get_rank()} finished sync operation. Python script: {filename} | Line number: {line_number}")
    
class Parallelized_Model(nn.Module):
    '''
    Data parallel and weight parallel model.
    '''
    def __init__(self, stage_list, sample, num_replicas=1, criterion=None, approximated_gradient=False):
        super(Parallelized_Model, self).__init__()

        self.num_replicas = num_replicas
        self.criterion = criterion
        self.world_size = dist.get_world_size()
        if num_replicas*len(stage_list) != self.world_size:
            raise ValueError(f"The number of replicas times the number of layers ({num_replicas}*{len(stage_list)}={num_replicas*len(stage_list)}) must be equal to the world size ({self.world_size}).")
        self.rank = dist.get_rank()
        
        # Model copy rank list
        self.stage_list = stage_list
        self.model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(num_replicas)] 
        for ranks in self.model_ranks:
            if self.rank in ranks:
                # TODO: Change gpu_id to be more generic for tensor sharding later on...
                sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
                self.model = Weight_Parallelized_Model(stage_list=stage_list, rank_list=ranks, sample=sample, gpu_id=0, criterion=criterion, approximated_gradient=approximated_gradient)
                sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
                self.subdomain = self.model.subdomain
                # self.stage = self.model.stage

        # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
        for layer_idx in range(len(self.stage_list)):
            # Collect ranks that are responsible for this layer across all replicas
            ranks = [r + layer_idx for r in range(0, self.world_size, len(self.stage_list))]
            # Create a new group containing these ranks
            if self.rank in ranks:
                self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
    
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, loss, chunks_amount=1, sync=True):
        self.model.backward(loss=loss, chunks_amount=chunks_amount)
        # update the weights across the replicas
        if sync:
            # average all gradients across the replicas
            for param in self.model.parameters():
                dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
                param.grad /= self.num_replicas
    
    def sync_params(self, method='average'):
        for param in self.model.parameters():
            dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            if method == 'average':
                param.data /= self.num_replicas
            elif method == 'sum':
                pass
            else:
                raise ValueError(f"Method {method} is not supported.")
    
# Global model class
class Weight_Parallelized_Model(nn.Module):
    def __init__(self, stage_list, rank_list, sample, gpu_id=0, device=None):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the stage_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        
        ''' 
        super(Weight_Parallelized_Model, self).__init__()
        self.rank = dist.get_rank()
        self.rank_list = rank_list
        self.rank_index = rank_list.index(self.rank)
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.backend_device = 'cpu' if self.backend == 'gloo' else self.tensor_device
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        self.inputs = []  # Each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = []  # Each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = [] # Each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        (layer_class, params) = stage_list[self.rank_index]
        self.stage = layer_class(**params).to(self.tensor_device) # Initialize the layer with provided parameters
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations
        self.f_time = 0 # Forward pass time
        self.g_time = 0 # Gradient computation time
        # These two are built during forward/backward passes in the initialization phase
        # self.shapes 
        # self.shapes_backward
    
        # Setup phase to compute the shapes of the tensors in the pipeline
        self.setup_phase = True
        loss = None
        out = self.forward(torch.randn(*sample.shape).to(self.tensor_device), chunks_amount=1, reset_grad=True, compute_grad=True)
        if self.rank == rank_list[-1]:
            loss = nn.MSELoss()(out[0], torch.rand_like(out[0]))
        self.backward([loss])
        self.zero_grad()
        self.setup_phase = False
        # End of setup phase
        self.subdomain = Weight_Parallelized_Subdomain(self) # Initialize the subdomain model

    def zero_counters(self):
        self.num_f_evals = 0
        self.num_g_evals = 0
        self.f_time = 0
        self.g_time = 0
        self.subdomain.num_f_evals = 0
        self.subdomain.num_g_evals = 0
        self.subdomain.f_time = 0
        self.subdomain.g_time = 0

    def zero_grad(self):
        self.stage.zero_grad()
    
    def grad(self, clone=False): # Returns the global gradient of the model
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return Weight_Parallelized_Tensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        params = [param.clone() if clone else param for param in self.stage.parameters()]
        return Weight_Parallelized_Tensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2): # Returns the subdomain gradient norm of the model
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        start = time.time()
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = [None]*chunks_amount 
        self.outputs = [None]*chunks_amount  
        compute_grad = False if not torch.is_grad_enabled() else compute_grad # flag to avoid storing the tensors needed to compute the gradients
        self.num_f_evals += 1 
        with torch.set_grad_enabled(compute_grad):
            if reset_grad:
                self.zero_grad() # Reset the gradients of the model before starting to accumulate them again
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32).to(self.tensor_device)
            if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = x.chunk(chunks_amount) # Chunkenize the input tensor
                self.inputs = chunks if compute_grad else [] 
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.backend_device) # Store the batch size of each chunk
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer
            # Go through the pipeline
            for c in range(chunks_amount):
                if self.rank_index == 0: # Beginning of the pipeline (first layer)
                    chunk = chunks[c].to(self.tensor_device)
                    out = self.stage.forward(chunk) 
                    next_rank = self.rank_list[self.rank_index+1]
                    self.outputs[c] = out if compute_grad else None
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(chunks[c].shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device=self.backend_device)
                    dist.send(tensor=out.to(self.backend_device), dst=next_rank) # send the tensor
                elif self.rank_index == len(self.rank_list)-1: # End of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device)
                        temp = torch.empty(*shapes, device=self.backend_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device, requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    self.inputs[c] = temp if compute_grad else None
                    out = self.stage.forward(temp)
                    self.outputs[c] = out 
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                else: # Middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # Wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device)
                        temp = torch.empty(*shapes, device=self.backend_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device, requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    out = self.stage.forward(temp)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = out if compute_grad else None
                    next_rank = self.rank_list[self.rank_index+1]
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device=self.backend_device)
                    dist.send(tensor=out.to(self.backend_device), dst=next_rank) # send the tensor
        self.f_time += time.time() - start
        # If the rank is in the last layer's rank list, return the output, else True (for some reason we can't remember)
        return self.outputs if self.rank == self.rank_list[-1] else True

    def backward(self, losses):
        chunks_amount = len(losses) # Number of chunks
        self.grad_output = [None]*chunks_amount 
        start = time.time()
        self.num_g_evals += 1 
        for c, loss in enumerate(losses): # Chunked loss
            chunk_size = self.outputs[c].shape[0]
            if self.rank == self.rank_list[-1]: # End of the pipeline
                self.grad_output[c] = autograd.grad(loss, self.outputs[c], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
                grad_data = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=grad_data.to(self.backend_device), dst=self.rank_list[-2]) # TODO make this async if possible
                for param in self.stage.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
            elif self.rank == self.rank_list[0]: # Beginning of the pipeline
                self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device, requires_grad=True)
                dist.recv(self.grad_output[c], src=self.rank_list[1])       
                self.grad_output[c] = self.grad_output[c].to(self.tensor_device).detach()
                for param in self.stage.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
            else: # Middle of the pipeline
                self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device, requires_grad=True)
                dist.recv(self.grad_output[c], src=self.rank_list[self.rank_index+1])       
                self.grad_output[c] = self.grad_output[c].to(self.tensor_device).detach()
                data_grad2 = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=data_grad2.to(self.backend_device), dst=self.rank_list[self.rank_index-1]) # TODO make this async if possible
                for param in self.stage.parameters():
                    if param.grad is None: 
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
        # TODO / NOTE: maybe we can delete self.inputs to free memory. It is not used anymore after the backward pass. (even in subdomains)
        self.g_time += time.time() - start
        return None
    

# TODO: We may need a "Parallelized_Subdomain" class which includes data parallelism and weight parallelism. 
    
# Subdomain model class
class Weight_Parallelized_Subdomain(nn.Module):
    def __init__(self, model):
        super(Weight_Parallelized_Subdomain, self).__init__()
        self.model = model
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations   
        self.f_time = 0
        self.g_time = 0
        
    def forward(self):
        start = time.time()
        for k,chunk in enumerate(self.model.inputs):
            self.model.outputs[k] = self.model.stage(chunk)
        self.num_f_evals += 1
        self.f_time += time.time() - start

    def backward(self):
        start = time.time()
        for k in range(len(self.model.outputs)):
            for param in self.model.stage.parameters():
                if param.grad is None:
                    param.grad = autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                else:
                    param.grad += autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                # print the norm of the gradient
                # print(f'Rank {self.model.rank} | Grad norm: {torch.norm(param.grad.flatten(), p=2)}')
        # print(f'Rank {self.model.rank} | Grad norm: {torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2)}')
        self.num_g_evals += 1
        self.g_time += time.time() - start
            
    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()

# TODO: We may need a "Parallelized_Tensor" class which includes data parallelism of tensors (in case we want to do different strategies with the steps in APTS).

# Global gradient class
class Weight_Parallelized_Tensor(nn.Module):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        
    def norm(self, p=2):
        if p == 2:
            return math.sqrt(self @ self)
        else:
            # Implement generic p
            raise NotImplementedError("Only L2 norm is implemented.")
    
    def __iter__(self):
        return iter(self.tensor)
    
    def __repr__(self):
        return f'Rank {self.rank}\nGradient: {self.model.subdomain.grad()}'
    
    def __matmul__(self, a): # self.grad @ a
        return self * a
    
    def __rmatmul__(self, a): # a @ self.grad
        return a * self

    def __rmul__(self, a):  # a * self.grad
        return self.__mul__(a)  # This handles the commutative property of multiplication
    
    def __mul__(self, a):  # self.grad * a
        if isinstance(a, Weight_Parallelized_Tensor):  # When both operands are Weight_Parallelized_Tensor instances
            g1 = torch.cat([p.flatten() for p in self.tensor], dim=0)  # Flatten the gradients
            g2 = torch.cat([p.flatten() for p in a.tensor], dim=0)  # Flatten the gradients
            g3 = g1 @ g2
            g3 = g3.to(f'cpu' if self.backend == 'gloo' else f'cuda:0')
            dist.all_reduce(tensor=g3, group=self.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3.item()
        else:
            return Weight_Parallelized_Tensor([p*a for p in self.tensor], backend=self.backend, master_group=self.master_group, rank=self.rank)   # Multiply model by a scalar or tensor

    def __add__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p+q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
    
    def __sub__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p-q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)

def decide_tensor_device(ws, backend, gpu_id):
    if torch.cuda.is_available():
        if backend == 'gloo':
            if torch.cuda.device_count() < ws:
                return f'cuda:{gpu_id}'
            else:
                return f'cuda:{dist.get_rank()}'
        else:
            return f'cuda:{gpu_id}'
    else:
        return 'cpu'


# NOTE: TO BE DONE
class TensorSharding(nn.Module):
    '''
    manually define sharding strategy for a layer (depending on the NN type).  
    '''
    def __init__(self, module, rank_list, device_mesh, parallelize_plan):
        super(TensorSharding, self).__init__()
        self.module = module
        self.rank_list = rank_list
        self.device_mesh = device_mesh
        self.parallelize_plan = parallelize_plan
        

        for i, (layer, ranks) in enumerate(zip(module, rank_list)):
            if len(ranks) == 1:
                # module[i] = parallelize_module(layer, device_mesh, parallelize_plan)
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
            else:
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
        return module

    def unshard(self, rank=None): # or to_local
        '''
        Send all shards to the rank specified.
        '''
        pass
    
    def send_shards(self, rank_list=[]):
        '''
        Shard and send tensor to the specified rank_list for paralellization.
        '''
        pass
    
    def multiply(self, a, b):
        '''
        Multiply the sharded tensors.
        '''
        pass
    
    def add(self, a, b): # TODO: Check if this overrides the + operator
        '''
        Add the sharded tensors.
        '''
        pass