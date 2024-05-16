import math
import time 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import diffdist.functional as distops
from parallel.utils import *

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
        self.model.outputs = self.model.layer(self.model.inputs)
        if count_f: self.num_f_evals += 1
        self.f_time += time.time() - start

    def backward(self, count_g=True):
        start = time.time()
        for param in self.model.layer.parameters():
            param.grad = autograd.grad(self.model.outputs, param, grad_outputs=self.model.grad_output, retain_graph=True)[0]   
        if count_g: self.num_g_evals += 1
        self.g_time += time.time() - start
            
    def grad(self):
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()

# Global gradient class
class Weight_Parallelized_Tensor(nn.Module):
    def __init__(self, tensor, rank_list, backend, master_group, list_of_master_nodes, rank, gpu_id):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.rank_list = rank_list
        self.backend = backend
        self.master_group = master_group
        self.list_of_master_nodes = list_of_master_nodes
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
            if self.rank in self.list_of_master_nodes:
                dist.reduce(tensor=g3, dst=self.rank_list[0][0], group=self.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3.item() # Multiply the internal models
        else:
            return Weight_Parallelized_Tensor([p*a for p in self.tensor], rank_list=self.rank_list, backend=self.backend, master_group=self.master_group, list_of_master_nodes=self.list_of_master_nodes, rank=self.rank, gpu_id=self.gpu_id)   # Multiply model by a scalar or tensor

    def __add__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p+q for p,q in zip(self.tensor, a.tensor)], rank_list=self.rank_list, backend=self.backend, master_group=self.master_group, list_of_master_nodes=self.list_of_master_nodes, rank=self.rank, gpu_id=self.gpu_id)
    
    def __sub__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p-q for p,q in zip(self.tensor, a.tensor)], rank_list=self.rank_list, backend=self.backend, master_group=self.master_group, list_of_master_nodes=self.list_of_master_nodes, rank=self.rank, gpu_id=self.gpu_id)

def decide_gpu_device(ws, backend, gpu_id):
    if backend == 'gloo':
        if torch.cuda.device_count() < ws:
            return f'cuda:{gpu_id}'
        else:
            return f'cuda:{dist.get_rank()}'
    else:
        return f'cuda:{gpu_id}'


class Parallelized_Model(nn.Module):
    '''
    Data parallel and weight parallel model.
    '''
    def __init__(self, gpu_list, layer_list, rank_list):
        super(Parallelized_Model, self).__init__()
        # check that the gpu_list has available GPUs
        self.gpu_list = gpu_list
        gpu_available = torch.cuda.device_count()
        if all([gpu < gpu_available for gpu in gpu_list]):
            raise ValueError("GPUs available less than those in gpu_list")
        
        self.all_ranks = [r for rank in rank_list for r in rank]
        self.rank = dist.get_rank()
        rank_list = [[rank] if type(rank) is not list and type(rank) is not tuple else rank for rank in rank_list]
        self.rank_list = rank_list
        self.list_of_master_nodes = [self.rank_list[i][0] for i in range(len(self.rank_list))]
        self.master_group = dist.new_group(ranks=self.list_of_master_nodes)
        self.backend = dist.get_backend()

    


# Global model class
class Weight_Parallelized_Model(nn.Module):
    def __init__(self, layer_list, rank_list, gpu_id=0):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the layer_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        
        ''' 

        super(Weight_Parallelized_Model, self).__init__()
        self.all_ranks = [r for rank in rank_list for r in rank]
        self.rank = dist.get_rank()
        rank_list = [[rank] if type(rank) is not list and type(rank) is not tuple else rank for rank in rank_list]
        self.rank_list = rank_list
        self.list_of_master_nodes = [self.rank_list[i][0] for i in range(len(self.rank_list))]
        self.master_group = dist.new_group(ranks=self.list_of_master_nodes)
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.shapes = [0]*len(layer_list)
        self.gpu_device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.inputs = torch.tensor(()).to(self.gpu_device)  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = torch.tensor(()).to(self.gpu_device)  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = torch.tensor(()).to(self.gpu_device) # each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        # layer_list = [[layer_class, {}] if type(layer_class) is not list and type(layer_class) is not tuple else [layer_class[0], layer_class[1]] for layer_class in layer_list]
        for i, ((layer_class, params, (input_shape, output_shape)), ranks) in enumerate(zip(layer_list, rank_list)): # Create each layer and assign it to the specified rank (device)
            self.shapes[i] = ((input_shape, output_shape))
            if self.rank in ranks:
                if len(ranks) == 1: # No tensor parallelism (no horizontal slicing)
                    self.layer = layer_class(**params).to(self.gpu_device) # Initialize the layer with provided parameters
                    self.ranks = ranks
                else:
                    raise NotImplementedError("Tensor parallelism is not yet implemented.")
                    # ----> maybe use DeviceMesh -> https://pytorch.org/docs/stable/distributed.html    
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
        return Weight_Parallelized_Tensor(gradient, self.rank_list, self.backend, self.master_group, self.list_of_master_nodes, self.rank, self.gpu_id)
    
    def parameters(self, clone=False):
        params = [param.clone() if clone else param for param in self.layer.parameters()]
        return Weight_Parallelized_Tensor(params, self.rank_list, self.backend, self.master_group, self.list_of_master_nodes, self.rank, self.gpu_id)
    
    def subdomain_grad_norm(self, p=2):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass
    
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True, count_f=True):
        start = time.time()
        assert chunks_amount == 1, 'This is a temporary solution to avoid problems in the backward pass due to concatenation of the tensors'
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = torch.tensor(()).to(self.gpu_device) 
        self.outputs = torch.tensor(()).to(self.gpu_device)
        # "compute_grad" is a flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        if count_f: self.num_f_evals += 1 
        with torch.set_grad_enabled(compute_grad):
            # Reset the gradients of the model before starting to accumulate them again
            if reset_grad:
                self.zero_grad()
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32).to(self.gpu_device)
            if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                self.inputs = x if compute_grad else 0
                chunks = x.chunk(chunks_amount)
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.gpu_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0][0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer
            # Start the pipeline
            for c in range(chunks_amount):
                i = self.rank_list.index(self.ranks)
                if i == 0: # begin of the pipeline (first layer)
                    chunk = chunks[c].to(self.gpu_device)
                    out = self.layer(chunk) # Apply the current layer
                    next_rank = self.rank_list[i + 1]
                    self.outputs = out#torch.cat((self.outputs, out), dim=0) if compute_grad else 0
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank[0]) # send the tensor
                    # TODO: delete out to free memory?
                elif i == len(self.rank_list)-1: # end of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    temp = torch.empty(*self.shapes[i][0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    # print(f'expecting tensor of shape {temp.shape} from rank {self.rank_list[i-1][0]}')
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1][0])
                    temp = temp.to(self.gpu_device)
                    self.inputs = temp#torch.cat((self.inputs, temp), dim=0) if compute_grad else 0
                    self.outputs = self.layer(temp)#torch.cat((self.outputs, self.layer[0](temp)), dim=0)
                else: # middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    temp = torch.empty(*self.shapes[i][0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1][0])
                    temp = temp.to(self.gpu_device)
                    self.inputs = temp #torch.cat((self.inputs, temp), dim=0) if compute_grad else 0 # TODO: test concatenation in the future
                    out = self.layer(temp)
                    self.outputs = out#torch.cat((self.outputs, out), dim=0) if compute_grad else 0
                    next_rank = self.rank_list[i + 1]
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank[0]) # send the tensor

        if self.rank in self.rank_list[-1]: # If the rank is in the last layer's rank list, return the output
            self.f_time += time.time() - start
            return self.outputs
        else:
            self.f_time += time.time() - start
            return True

    def backward(self, loss, count_g=True):
        start = time.time()
        # We used "diffdist" library: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py
        if self.rank in self.list_of_master_nodes:
            sample_shape = torch.zeros(1, dtype=torch.int32).to(self.gpu_device)
            if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                sample_shape = torch.tensor([self.outputs.shape[0]], dtype=torch.int32).to(self.gpu_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=sample_shape, src=self.rank_list[0][0], group=self.master_group, async_op=True) 
        i = self.rank_list.index(self.ranks)
        if count_g: self.num_g_evals += 1 
        if self.rank in self.rank_list[-1]: # End of the pipeline
            if len(self.rank_list[-1])==1:
                self.grad_output = autograd.grad(loss, self.outputs, create_graph=True)[0]               
                grad_data = autograd.grad(self.outputs, self.inputs, grad_outputs=self.grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
                distops.send(tensor=grad_data.cpu() if self.backend == 'gloo' else grad_data, dst=self.rank_list[-2][0]) # TODO make this async if possible
                for param in self.layer.parameters():
                    param.grad = autograd.grad(self.outputs, param, grad_outputs=self.grad_output, retain_graph=True)[0]      
            else:
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
        elif self.rank in self.rank_list[0]:
            # TODO : remember to add a condition for the master rank to be the ONLY one waiting for the "shape_transfer"
            shape_transfer.wait() # wait for the shape to be broadcasted
            self.grad_output = torch.empty(*self.shapes[i+1][0](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
            self.grad_output, _ = distops.recv(self.grad_output, src=self.rank_list[1][0])       
            self.grad_output = self.grad_output.to(self.gpu_device)
            for param in self.layer.parameters():
                param.grad = autograd.grad(self.outputs, param, grad_outputs=self.grad_output, retain_graph=True)[0] 
        else: # middle of the pipeline
            shape_transfer.wait() # wait for the shape to be broadcasted
            self.grad_output = torch.empty(*self.shapes[i+1][0](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
            self.grad_output, _ = distops.recv(self.grad_output, src=self.rank_list[i+1][0])       
            self.grad_output = self.grad_output.to(self.gpu_device)
            data_grad2 = autograd.grad(self.outputs, self.inputs, grad_outputs=self.grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
            for param in self.layer.parameters():
                param.grad = autograd.grad(self.outputs, param, grad_outputs=self.grad_output, retain_graph=True)[0] 
            distops.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1][0]) # TODO make this async if possible
            # TODO / NOTE: maybe we can delete self.inputs to free memory. It is not used anymore after the backward pass. (even in subdomains)
        self.g_time += time.time() - start
        return None