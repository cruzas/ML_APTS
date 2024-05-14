import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
# add the path to the sys.path
import sys
# Make the following work on Windows and MacOS
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.utility import prepare_distributed_environment, create_dataloaders
import torch.autograd as autograd
import torch.distributed.rpc as rpc
import diffdist.functional as distops
from typing import Callable, Optional
from torch import Tensor


# NOTE: TO BE DONE
class TensorSharding(nn.Module):
    '''
    manually define sharding strategy for a layer (depending on the NN type).  
    '''
    def __init__(module, rank_list, device_mesh, parallelize_plan):
        super(TensorSharding, self).__init__()
        self.module = module
        self.rank_list = rank_list
        self.device_mesh = device_mesh
        self.parallelize_plan = parallelize_plan
        

        for i, (layer, ranks) in enumerate(zip(module, rank_list)):
            if len(ranks) == 1:
                module[i] = parallelize_module(layer, device_mesh, parallelize_plan)
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

def Parallelize(LayerList, rank_list, gpu_per_node):
    '''
    LayerList: contains the unloaded layers to avoid memory overflow.
    
    LayerList, e.g. LayerList = [ [ Layer_1_class,{dictionary of parameters} ] , [ Layer_2_class,{dictionary of parameters} ], ...]
    -> Therefore the global NN is NN = sequential( Layer_1_class({params1}), Layer_2_class({params2}), ... )
    
    rank_list, e.g. rank_list = [ [rank_0,rank_1,rank_2], [rank_3,rank_4], [rank_5], [rank_6,rank_7],... ]
    -> giving multiple ranks to each layer will parallelize the layer across the ranks through PyTorch's Dtensor library for horizontal slicing.
    
    Note that data parallelism is only available if you have multiple, say n, GPUs per node (in nccl). In that case you can divide the dataset into n chunks.
    '''
    
    # Here we define the forward function for the pipelined model with "send" and "recv" functions
    
def layer_input_output_shape(layer):
    '''
    This function returns the input shape of a layer.
    '''
    if type(layer) is torch.nn.modules.linear.Linear:
        return layer.in_features, layer.out_features

# Subdomain model class
class Weight_Parallelized_Subdomain(nn.Module):
    def __init__(self, model):
        super(Weight_Parallelized_Subdomain, self).__init__()
        self.model = model

    def forward(self):
        self.model.outputs = self.model.layer(self.model.inputs)
        return self.model.outputs

    def backward(self):
        for param in self.model.layer.parameters():
            param.grad = autograd.grad(self.model.outputs, param, grad_outputs=self.model.grad_output, retain_graph=True)[0]   
            
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


class ParallelizedDataLoader():
    def __init__(self, dataset, batch_size, rank_list, device_mesh, parallelize_plan, overlap_ratio=0.0, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank_list = rank_list
        self.device_mesh = device_mesh
        self.parallelize_plan = parallelize_plan
        self.overlap_ratio = overlap_ratio
        self.shuffle = shuffle


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
        self.shapes = []
        self.gpu_device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.inputs = torch.tensor(()).to(self.gpu_device)  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = torch.tensor(()).to(self.gpu_device)  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = torch.tensor(()).to(self.gpu_device) # each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        # layer_list = [[layer_class, {}] if type(layer_class) is not list and type(layer_class) is not tuple else [layer_class[0], layer_class[1]] for layer_class in layer_list]
        for i, ((layer_class, params, (input_shape, output_shape)), ranks)  in enumerate(zip(layer_list, rank_list)): # Create each layer and assign it to the specified rank (device)
            self.shapes.append((input_shape, output_shape))
            if self.rank in ranks:
                if len(ranks) == 1: # No tensor parallelism (no horizontal slicing)
                    self.layer = layer_class(**params).to(self.gpu_device) # Initialize the layer with provided parameters
                    self.ranks = ranks
                else:
                    raise NotImplementedError("Tensor parallelism is not yet implemented.")
                    # ----> maybe use DeviceMesh -> https://pytorch.org/docs/stable/distributed.html    
                self.subdomain = Weight_Parallelized_Subdomain(self) # Initialize the subdomain model

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
    
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        assert chunks_amount == 1, 'This is a temporary solution to avoid problems in the backward pass due to concatenation of the tensors'
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = torch.tensor(()).to(self.gpu_device) 
        self.outputs = torch.tensor(()).to(self.gpu_device)
        # "compute_grad" is a flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
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
            return self.outputs
        else:
            return True

    def backward(self, loss):
        # We used "diffdist" library: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py
        if self.rank in self.list_of_master_nodes:
            sample_shape = torch.zeros(1, dtype=torch.int32).to(self.gpu_device)
            if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                sample_shape = torch.tensor([self.outputs.shape[0]], dtype=torch.int32).to(self.gpu_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=sample_shape, src=self.rank_list[0][0], group=self.master_group, async_op=True) 
        i = self.rank_list.index(self.ranks)
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
        return None
    
class APTS(torch.optim.Optimizer):
    def __init__(self, model, criterion, subdomain_optimizer, subdomain_optimizer_defaults, global_optimizer, global_optimizer_defaults, lr=0.01, max_subdomain_iter=5, dogleg=False):
        '''
        We use infinity norm for the gradient norm.
        '''
        super(APTS, self).__init__(model.parameters(), {'lr': lr, 'max_subdomain_iter': max_subdomain_iter, 'dogleg': dogleg})
        for key in self.param_groups[0].keys():  
            if key not in ['params']:
                setattr(self, key, self.param_groups[0][key])
        self.model = model # subdomain model
        self.criterion = criterion # loss function
        
        # Throw an error if 'lr' not in subdomain_optimizer_defaults.keys()
        if lr <= 0:
            raise ValueError('The learning rate "lr" must be bigger than 0.')
        subdomain_optimizer_defaults.update({'lr': lr})
        self.subdomain_optimizer = subdomain_optimizer(params=model.subdomain.parameters(), **subdomain_optimizer_defaults) # subdomain optimizer
        if 'TR' in str(global_optimizer):
            self.global_optimizer = global_optimizer(model=model, criterion=criterion, **global_optimizer_defaults) # TR optimizer
        else:
            global_optimizer_defaults.update({'lr': lr})
            self.global_optimizer = global_optimizer(params=model.subdomain.parameters(), **global_optimizer_defaults) # standard PyTorch optimizers
    
    # override of param_group (update)
    def update_param_group(self):
        for key in self.param_groups[0].keys():
            if key not in ['params']:
                self.param_groups[0][key] = getattr(self, key)
            
    def subdomain_steps(self):
        # Set up the learning rate
        self.subdomain_optimizer.param_groups[0]['lr'] = self.lr/self.max_subdomain_iter
        # Do subdomain steps
        for _ in range(self.max_subdomain_iter):
            self.subdomain_optimizer.zero_grad()
            self.model.subdomain.forward()
            self.model.subdomain.backward()
            #normalize gradient to 1 to avoid going out of the trust region
            grad_norm = self.model.subdomain.grad_norm()
            if grad_norm > 1: 
                for param in self.model.subdomain.parameters():
                    param.grad /= grad_norm
            self.subdomain_optimizer.step()
        # TODO: we have the gradient, we can compute its norm, we could use the norm to have an idea of the convergence of the subdomain optimization
        self.update_param_group()

    def step(self, closure):
        # Compute loss
        initial_loss = closure(compute_grad=True, zero_grad=True)
        # Store the initial parameters and gradients
        initial_parameters = self.model.parameters(clone=True)
        initial_grads = self.model.grad(clone=True)
        
        # Do subdomain steps
        self.subdomain_steps()
        with torch.no_grad():
            new_loss = closure(compute_grad=False, zero_grad=True)
            step = self.model.parameters(clone=False) - initial_parameters
            # Compute the dogleg step with the hope that new_loss <= old_loss
            lr = self.lr
            w = 0; c = 0
            while new_loss > initial_loss and self.dogleg and c>=5: 
                c += 1
                # Decrease lr to decrease size of step...
                lr = lr/2
                # ... while moving towards the steepest descent direction (-g)
                w = min(w + 0.2, 1)
                step2 = ((1-w)*step) - (w*initial_grads)
                # The step length is "lr", with   lr <= self.lr (global TR lr)
                step2 = (lr/step2.norm())*step2
                # Update the model with the new params
                for i,p in enumerate(self.model.parameters()):
                    p.copy_(initial_parameters.tensor[i] + step2.tensor[i])
                # Compute new global loss
                new_loss = closure(compute_grad=False, zero_grad=True)
                # Empty cache to avoid memory problems
                torch.cuda.empty_cache()

        # Do global TR step
        self.global_optimizer.step(closure)    
        # Update the learning rate
        self.lr = self.global_optimizer.lr
        self.update_param_group()

class TR(torch.optim.Optimizer):
    def __init__(self, model, criterion, lr=0.01, max_lr=1.0, min_lr=0.0001, nu=0.5, inc_factor=2.0, dec_factor=0.5, nu_1=0.25, nu_2=0.75, max_iter=5, norm_type=2):
        '''
        We use infinity norm for the gradient norm.
        '''
        super().__init__(model.parameters(), {'lr': lr, 'max_lr': max_lr, 'min_lr': min_lr, 'max_iter': max_iter})
        self.model = model
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.criterion = criterion
        self.inc_factor = inc_factor # increase factor for the learning rate
        self.dec_factor = dec_factor # decrease factor for the learning rate
        self.nu_1 = nu_1 # lower bound for the reduction ratio
        self.nu_2 = nu_2 # upper bound for the reduction ratio
        self.nu = min(nu, nu_1) # acceptable reduction ratio (cannot be bigger than nu_1)
        self.max_iter = max_iter # max iterations for the TR optimization
        self.norm_type = norm_type 
    
    def step(self, closure):
        # Compute the loss of the model
        old_loss = closure(compute_grad=True)
        # Retrieve the gradient of the model
        grad = self.model.grad()
        # Compute the norm of the gradient
        grad_norm2 = grad.norm(p=2)
        grad_norm = grad.norm(p=self.norm_type) if self.norm_type != 2 else grad_norm2 
        # Rescale the gradient to e at the edges of the trust region
        grad = grad * (self.lr/grad_norm)
        # Make sure the loss decreases
        new_loss = torch.inf; c = 0

        while old_loss - new_loss < 0 and c < self.max_iter:
            stop = True if abs(self.lr - self.min_lr)/self.min_lr < 1e-6 else False
            # TODO: Adaptive reduction factor -> if large difference in losses maybe divide by 4
            for i,param in enumerate(self.model.parameters()):
                param.data -= grad.tensor[i].data
            new_loss = closure(compute_grad=False)
            old_lr = self.lr
            
            # Compute the ratio of the loss reduction       
            act_red = old_loss - new_loss # actual reduction
            pred_red = self.lr*grad_norm2 # predicted reduction (first order approximation of the loss function)
            red_ratio = act_red / pred_red # reduction ratio
            
            if dist.get_rank() == 0:
                print(f'old loss: {old_loss}, new loss: {new_loss}, act_red: {act_red}, pred_red: {pred_red}, red_ratio: {red_ratio}')
            if red_ratio < self.nu_1: # the reduction ratio is too small -> decrease the learning rate
                self.lr = max(self.min_lr, self.dec_factor*self.lr)
            elif red_ratio > self.nu_2: # the reduction ratio is good enough -> increase the learning rate
                self.lr = min(self.max_lr, self.inc_factor*self.lr)
                break
            
            # Else learning rate remains unchanged
            if stop:
                break
            
            if red_ratio < self.nu:
                # In place of storing initial weights, we go backward from the current position whenever the step gets rejected (This only works with first order approximation of the loss)
                if self.lr != self.min_lr:
                    if c == 0: 
                        grad = grad * (-self.lr/old_lr)
                    else:
                        grad = grad * (self.lr/old_lr)
                else: # self.lr == self.min_lr
                    if c == 0:
                        grad = grad * (-(old_lr-self.lr)/old_lr)
                    else:
                        grad = grad * ((old_lr-self.lr)/old_lr)
                if dist.get_rank() == 0:
                    print(f'old loss: {old_loss}, new loss: {new_loss}, lr: {self.lr}')
                    
                    
            else:
                break
            c += 1

class ParallelLoss():
    def __init__(self, rank_list, criterion, criterion_params={}):
        # check if criterion is a class or a function
        if isinstance(criterion, type):
            self.criterion = criterion(**criterion_params)
        else:
            if len(criterion_params) > 0:
                self.criterion = criterion.__class__(**criterion_params)
            else:
                self.criterion = criterion
        self.rank_list = rank_list
        
    def __call__(self, x, y):
        if dist.get_rank() == self.rank_list[-1][0]:
            print('ciao')
        return self.criterion(x,y) if dist.get_rank() == self.rank_list[-1][0] else None    

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True):
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad):
        if zero_grad:
            model.zero_grad()
        outputs = model(inputs)
        loss = torch.zeros(1).to(model.gpu_device)
        if model.rank == model.rank_list[-1][0]:
            loss = criterion(outputs, targets)
        dist.broadcast(tensor=loss, src=model.rank_list[-1][0], group=model.master_group)
        if compute_grad and torch.is_grad_enabled():
            # Compute gradient
            model.backward(loss)
        return loss.item()
    return closure2 
    
def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    print(f'Rank {rank} is ready.')
    criterion = torch.nn.CrossEntropyLoss()
    

    NN1 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(64, 10), nn.ReLU())
   
    if dist.get_world_size() == 3:
        layer_list = [
            (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
            (NN2, {'in_features': 256, 'out_features': 128},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,128], dtype=torch.int32))),
            (NN3, {'in_features': 128, 'out_features': 64},   (lambda samples: torch.tensor([samples,128], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32)))
        ]
        # rank_list = [[0,1], [2,3], [4,5,6]]
        rank_list = [[0], [1], [2]]
        # rank_list = [[0], [1], [2]]
    elif dist.get_world_size() == 2 or dist.get_world_size() == 4:
        layer_list = [
            (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
            (NN3, {'in_features': 256, 'out_features': 64},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32))),
        ]
        rank_list = [[0], [1]]
        # layer_list = [
        #     (NN2, {'in_features': 100, 'out_features': 100},  (lambda samples: torch.tensor([samples,200], dtype=torch.int32), lambda samples: torch.tensor([samples,50 ], dtype=torch.int32))),
        # ]
        # # rank_list = [[0,1]]
        # rank_list = [[0],[1]]

    list_of_all_ranks = [r for rank in rank_list for r in rank]
    if rank in list_of_all_ranks:
        group = dist.new_group(ranks=list_of_all_ranks)
        torch.manual_seed(3456)
        model = Weight_Parallelized_Model(layer_list, rank_list)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # optimizer1 = TR(model, criterion)
        # optimizer2 = torch.optim.Adam(model.subdomain.parameters(), lr=0.0001)
        optimizer = APTS(model, criterion, torch.optim.SGD, {}, TR, {'lr':0.01, 'max_lr':1.0, 'min_lr':1e-3, 'nu_1':0.25, 'nu_2':0.75})
        # optimizer = torch.optim.SGD(model.subdomain.parameters(), lr=0.01)

        # Data loading TODO: load data only on master master rank
        train_loader, test_loader = create_dataloaders(
            dataset="MNIST",
            data_dir=os.path.abspath("./data"),
            mb_size=60000,
            overlap_ratio=0,
            parameter_decomposition=True,
            device="cuda:0"
        )
        device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)

        for epoch in range(1000):
            for i, (x, y) in enumerate(train_loader):
                # Vectorize x 
                x = x.view(x.size(0), -1)
                # One optimizer step
                optimizer.step(closure(x, y, torch.nn.CrossEntropyLoss(), model))
                
                # Train a subdomain model
                # if rank in rank_list[0]:
                # for asd in range(2):
                #     model.subdomain.zero_grad()
                #     out = model.subdomain.forward()
                #     # loss = criterion(out, y.to(device))
                #     # print(f'Epoch {epoch} subdomain loss {loss}')
                #     model.subdomain.backward()
                #     optimizer2.step()

            # Compute the test accuracy
            for j, (x_test, y_test) in enumerate(test_loader): # NOTE: NO MINIBACHES IN THE TEST SET
                x_test = x_test.view(x_test.size(0), -1)
                output_test = model(x_test.to(device), chunks_amount=1, reset_grad = True, compute_grad = True)
                if rank == rank_list[-1][0]:
                    # Compute the accuracy NOT THE LOSS
                    accuracy = (output_test.argmax(dim=1) == y_test.to(device)).float().mean()
                    print(f'Epoch {epoch} test accuracy {accuracy*100}')

                # dist.barrier(group=group)

        # torch.manual_seed(0)
        # x=torch.randn(10, 100)
        # output = model(x.to('cuda'), chunks_amount=1, reset_grad = True, compute_grad = True)
        # if rank == rank_list[-1][0]:
        #     torch.manual_seed(0)
        #     target = torch.randint(0, 50, (10,50), dtype=torch.float32).to(f'cuda:{rank}' if dist.get_backend() == 'gloo' else f'cuda:{model.gpu_id}')
        #     loss = criterion(output, target)
        # else:
        #     loss = None 

        # model.backward(loss)
        # print('Hurra!')
        # dist.barrier(group=group)
        
        g1 = Weight_Parallelized_Tensor(rank_list, model)
        g2 = Weight_Parallelized_Tensor(rank_list, model)
        g3 = g1 @ g2
        print(g3)

        f = model.subdomain.forward()
        print(f)
        
        model.subdomain.backward()

        # Training loop

        # sequential model check
        model_seq = nn.Sequential(*[layer[0](**layer[1]) for layer in layer_list])
        # update Model params with the one of the models on RANK 0
        for i,param in enumerate(model_seq.parameters()):
            if rank == 0 and i < 4:
                param.data=list(model.parameters())[i].to("cuda")
            elif rank == 1 and i >= 4 and i<8:
                dist.send(tensor=list(model.parameters())[i-4].detach().to('cuda'), dst=0)
            elif rank == 2 and i >= 8:
                dist.send(tensor=list(model.parameters())[i-8].detach().to('cuda'), dst=0)

            if rank == 0 and i >= 4 and i<8:
                temp = torch.empty_like(param).to('cuda')
                dist.recv(tensor=temp, src=1)
                param.data = temp
            if rank == 0 and i >= 8:
                temp = torch.empty_like(param).to('cuda')
                dist.recv(tensor=temp, src=2)
                param.data = temp
        if rank == 0:
            model_seq = model_seq.to('cuda')
            output = model_seq(x.to('cuda'))
            torch.manual_seed(0)
            target = torch.randint(0, 50, (10,50), dtype=torch.float32).to(f'cuda:{rank}' if dist.get_backend() == 'gloo' else f'cuda:{model.gpu_id}')
            loss = criterion(output, target)
            print(f"Loss sequential model: {loss}")
        if rank == dist.get_world_size() - 1:
            print(f"Loss parallel model: {loss}")
                
        # check gradients
        if rank == 0:
            loss.backward()
            print(f"Derivative of sequential model:\n{[param.grad for param in model_seq.parameters()]}")

        print(f"Derivative of PARALLEL model rank {rank}:\n{[param.grad for param in model.parameters()]}")


if __name__ == '__main__':
    torch.manual_seed(1)

    # world_size = torch.cuda.device_count()  
    # master_addr = 'localhost'
    # master_port = '12345'
    # mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        world_size = 3
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)

