import math
import time 
import torch
import os
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import diffdist.functional as distops
from parallel.utils import *
import logging
from inspect import currentframe, getframeinfo

# global counter
# counter = -1

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
    
# from pippy import pipeline

# TODO: before send we could reduce the weight of tensor by using the half precision / float16, then we can convert it back to float32 after the recv

class Parallelized_Model(nn.Module):
    '''
    Data parallel and weight parallel model.
    '''
    def __init__(self, layer_list, sample, num_replicas=1):
        super(Parallelized_Model, self).__init__()

        self.num_replicas = num_replicas
        self.world_size = dist.get_world_size()
        if num_replicas*len(layer_list) != self.world_size:
            raise ValueError(f"The number of replicas times the number of layers ({num_replicas}*{len(layer_list)}={num_replicas*len(layer_list)}) must be equal to the world size ({self.world_size}).")
        self.rank = dist.get_rank()
        
        # Model copy rank list
        self.layer_list = layer_list
        self.model_ranks = [[r+k*len(self.layer_list) for r in range(len(self.layer_list))] for k in range(num_replicas)] 
        for ranks in self.model_ranks:
            if self.rank in ranks:
                # TODO: Change gpu_id to be more generic for tensor sharding later on...
                sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
                self.model = Weight_Parallelized_Model(layer_list=layer_list, rank_list=ranks, sample=sample, gpu_id=0)
                sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
                self.subdomain = self.model.subdomain
                # self.layer = self.model.layer

        # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
        for layer_idx in range(len(self.layer_list)):
            # Collect ranks that are responsible for this layer across all replicas
            ranks = [r + layer_idx for r in range(0, self.world_size, len(self.layer_list))]
            # Create a new group containing these ranks
            if self.rank in ranks:
                self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
    
    def forward(self, x, chunks_amount=2, reset_grad = False, compute_grad = True, count_f=True):
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad, count_f=count_f)
    
    def backward(self, loss, count_g=True, chunks_amount=2, sync=True):
        self.model.backward(loss=loss, count_g=count_g, chunks_amount=chunks_amount)
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
    def __init__(self, layer_list, rank_list, sample, gpu_id=0):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the layer_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        
        ''' 

        super(Weight_Parallelized_Model, self).__init__()
        self.rank = dist.get_rank()
        # if self.rank in [0,1]:
        #     print('asd')
        self.rank_list = rank_list
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        print(f'rank {self.rank}')
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.gpu_device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.inputs = []#torch.tensor(()).to(self.gpu_device)  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = []#torch.tensor(()).to(self.gpu_device)  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = []#torch.tensor(()).to(self.gpu_device) # each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        self.shapes = []*len(rank_list)
        
        (layer_class, params) = layer_list[rank_list.index(self.rank)]
        self.layer = layer_class(**params).to(self.gpu_device) # Initialize the layer with provided parameters
        # Forward pass to defined the shapes of the tensors
        def send_shape(shape: list, dst: int):
            for s in shape:
                dist.send(tensor=torch.tensor(s, dtype=torch.int32).to('cpu' if self.backend == 'gloo' else self.gpu_device), dst=dst)
            dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to('cpu' if self.backend == 'gloo' else self.gpu_device), dst=dst)
                
        def receive_shape(src: int):
            shape = []; temp = 0
            while True:
                temp = torch.tensor((0), dtype=torch.int32).to('cpu' if self.backend == 'gloo' else self.gpu_device)
                dist.recv(tensor=temp, src=src)
                if temp == -1:
                    break
                shape.append(temp.item())
            return shape
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        if self.rank == 0:
            print('asd')
        if self.rank == rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
            input_shape = lambda x: [x]+list(sample.shape)[1:]
            out = self.layer(sample.to(self.gpu_device))
            output_shape = lambda x: [x]+list(out.shape)[1:]
            # print(f'0 - Rank {self.rank} | Shape: {output_shape(sample.shape[0])} about to be sent')
            send_shape(output_shape(sample.shape[0]), dst=rank_list[1])
            # print(f'1 - Rank {self.rank} | Shape: {output_shape(sample.shape[0])} sent')
        elif self.rank == rank_list[-1]: # If the rank is in the last layer's rank list, receive the input from the previous device
            # print(f'2 - Rank {self.rank} | Shape about to be received')
            shape = receive_shape(rank_list[rank_list.index(self.rank-1)])
            # print(f'3 - Rank {self.rank} | Shape: {shape} received')
            input_shape = lambda x: [x]+list(shape)[1:]
            # random vector to simulate the output of the previous layer with shape input_shape(1)
            input = torch.randn(*input_shape(1)).to(self.gpu_device)
            out = self.layer(input.to(self.gpu_device))
            output_shape = lambda x: [x]+list(out.shape)[1:]
        else:
            shape = receive_shape(rank_list[rank_list.index(self.rank-1)])
            # print(f'3 - Rank {self.rank} | Shape: {shape} received')
            input_shape = lambda x: [x]+list(shape)[1:]
            input = torch.randn(*input_shape(1)).to(self.gpu_device)
            out = self.layer(input.to(self.gpu_device))
            output_shape = lambda x: [x]+list(out.shape)[1:]
            send_shape(output_shape(1), dst=rank_list[self.rank+1])
            # print(f'4 - Rank {self.rank} | Shape: {output_shape(1)} sent')
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        self.shapes = [input_shape, output_shape]
        self.subdomain = Weight_Parallelized_Subdomain(self) # Initialize the subdomain model
            
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations
        self.f_time = 0 # Forward pass time
        self.g_time = 0 # Gradient computation time

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
        self.layer.zero_grad()
    
    def grad(self, clone=False):
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return Weight_Parallelized_Tensor(gradient, self.backend, self.master_group, self.rank, self.gpu_id)
    
    def parameters(self, clone=False):
        params = [param.clone() if clone else param for param in self.layer.parameters()]
        return Weight_Parallelized_Tensor(params, self.backend, self.master_group, self.rank, self.gpu_id)
    
    def subdomain_grad_norm(self, p=2):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass

    def forward(self, x, chunks_amount=2, reset_grad = False, compute_grad = True, count_f=True):
        start = time.time()
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = [None]*chunks_amount
        self.outputs = [None]*chunks_amount
        # "compute_grad" is a flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        if count_f: self.num_f_evals += 1 
        with torch.set_grad_enabled(compute_grad):
            # Reset the gradients of the model before starting to accumulate them again
            if reset_grad:
                self.zero_grad()
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32).to(self.gpu_device)
            if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = x.chunk(chunks_amount)
                self.inputs = chunks if compute_grad else []
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.gpu_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            # torch.distributed.get_process_group_ranks(self.master_group)
            sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
            shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer
            sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
            if self.rank == 1:
                print('asd')
            # Start the pipeline
            for c in range(chunks_amount):
                i = self.rank_list.index(self.rank)
                if i == 0: # begin of the pipeline (first layer)
                    chunk = chunks[c].to(self.gpu_device)
                    out = self.layer(chunk) # Apply the current layer
                    next_rank = self.rank_list[i + 1]
                    self.outputs[c] = out if compute_grad else None
                    print(f'Rank {self.rank} | i {i} sending...')
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
                    print(f'Rank {self.rank} | i {i} sent')
                    # TODO: delete out to free memory? Remember to avoid storing outputs in case "compute_grad" is False
                elif i == len(self.rank_list)-1: # end of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    print(f'Rank {self.rank} | i {i} receiving...')
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1])
                    print(f'Rank {self.rank} | i {i} received')
                    temp = temp.to(self.gpu_device)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = self.layer(temp)
                else: # middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    print(f'Rank {self.rank} | i {i} receiving...')
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1])
                    print(f'Rank {self.rank} | i {i} received')
                    temp = temp.to(self.gpu_device)
                    out = self.layer(temp)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = out if compute_grad else None
                    next_rank = self.rank_list[i + 1]
                    print(f'Rank {self.rank} | i {i} sending...')
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
                    print(f'Rank {self.rank} | i {i} sent')
        print(f'Rank {self.rank} | Forward pass time: {time.time()-start}')
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        if self.rank == self.rank_list[-1]: # If the rank is in the last layer's rank list, return the output
            self.f_time += time.time() - start
            return self.outputs
        else:
            self.f_time += time.time() - start
            return True

    def backward(self, loss, count_g=True, chunks_amount=2):
        self.grad_output = [None]*chunks_amount
        # We used "diffdist" library: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py
        start = time.time()
        if count_g: self.num_g_evals += 1 
        for k in range(chunks_amount): # chunked loss
            sample_shape = torch.zeros(1, dtype=torch.int32).to(self.gpu_device)
            if self.rank == self.rank_list: # If the rank is in the first layer's rank list, send the input to the next device
                sample_shape = torch.tensor([self.outputs[k].shape[0]], dtype=torch.int32).to(self.gpu_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=sample_shape, src=self.rank_list[0], group=self.master_group, async_op=True) 
            i = self.rank_list.index(self.rank)
            if self.rank == self.rank_list[-1]: # End of the pipeline
                self.grad_output[k] = autograd.grad(loss[k], self.outputs[k], create_graph=True)[0]     
                grad_data = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=self.grad_output[k], retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
                distops.send(tensor=grad_data.cpu() if self.backend == 'gloo' else grad_data, dst=self.rank_list[-2]) # TODO make this async if possible
                for param in self.layer.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount

            elif self.rank == self.rank_list[0]:
                # TODO : remember to add a condition for the master rank to be the ONLY one waiting for the "shape_transfer"
                shape_transfer.wait() # wait for the shape to be broadcasted
                self.grad_output[k] = torch.empty(*self.shapes[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                self.grad_output[k], _ = distops.recv(self.grad_output[k], src=self.rank_list[1])       
                self.grad_output[k] = self.grad_output[k].to(self.gpu_device)
                for param in self.layer.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
            else: # middle of the pipeline
                shape_transfer.wait() # wait for the shape to be broadcasted
                self.grad_output[k] = torch.empty(*self.shapes[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                self.grad_output[k], _ = distops.recv(self.grad_output[k], src=self.rank_list[i+1])       
                self.grad_output[k] = self.grad_output[k].to(self.gpu_device)
                data_grad2 = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=self.grad_output[k], retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
                for param in self.layer.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                distops.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1]) # TODO make this async if possible
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
        
    def forward(self, count_f=True):
        start = time.time()
        for k,chunk in enumerate(self.model.inputs):
            self.model.outputs[k] = self.model.layer(chunk)
        if count_f: self.num_f_evals += 1
        self.f_time += time.time() - start

    def backward(self, count_g=True):
        start = time.time()
        for k in range(len(self.model.outputs)):
            for param in self.model.layer.parameters():
                if param.grad is None:
                    param.grad = autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                else:
                    param.grad += autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                # print the norm of the gradient
                # print(f'Rank {self.model.rank} | Grad norm: {torch.norm(param.grad.flatten(), p=2)}')
        # print(f'Rank {self.model.rank} | Grad norm: {torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2)}')
        if count_g: self.num_g_evals += 1
        self.g_time += time.time() - start
            
    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()

# TODO: We may need a "Parallelized_Tensor" class which includes data parallelism of tensors (in case we want to do different strategies with the steps in APTS).

# Global gradient class
class Weight_Parallelized_Tensor(nn.Module):
    def __init__(self, tensor, backend, master_group, list_of_master_nodes, rank, gpu_id):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        self.gpu_id = gpu_id
        
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
            g3 = g3.to(f'cpu' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
            dist.all_reduce(tensor=g3, group=self.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3.item()
        else:
            return Weight_Parallelized_Tensor([p*a for p in self.tensor], backend=self.backend, master_group=self.master_group, rank=self.rank, gpu_id=self.gpu_id)   # Multiply model by a scalar or tensor

    def __add__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p+q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank, gpu_id=self.gpu_id)
    
    def __sub__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p-q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank, gpu_id=self.gpu_id)

def decide_gpu_device(ws, backend, gpu_id):
    if backend == 'gloo':
        if torch.cuda.device_count() < ws:
            return f'cuda:{gpu_id}'
        else:
            return f'cuda:{dist.get_rank()}'
    else:
        return f'cuda:{gpu_id}'


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