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
    
# TODO: before send we could reduce the weight of tensor by using the half precision / float16, then we can convert it back to float32 after the recv
class Sequential_Model(nn.Module):
    def __init__(self, pipe_list):
        super(Sequential_Model, self).__init__()
        self.num_pipes = len(pipe_list)
        for i, pipe in enumerate(pipe_list):
            setattr(self, f'pipe{i}', Stage(pipe))
    
    def forward(self, x):
        for i in range(0, self.num_pipes):
            print(f"(FWD) Going through pipe group {i}")
            x = getattr(self, f'pipe{i}')(x)
            if i > 0:
                output_needed = getattr(self, f'pipe{i}').outputs[0]
                if len(getattr(self, f'pipe{i-1}').outputs) > len(getattr(self, f'pipe{i-1}').layer):
                    print("Removing the last element of the outputs list")
                    getattr(self, f'pipe{i-1}').outputs.pop(-1)
                getattr(self, f'pipe{i-1}').outputs.append(output_needed)
        return x
    
    def backward(self, loss):
        for i in range(self.num_pipes-1, -1, -1):
            print(f"(BWD) Going through pipe group {i}")
            if i == self.num_pipes-1:
                grad_output = getattr(self, f'pipe{i}').backward(loss=loss)
            else:
                grad_output = getattr(self, f'pipe{i}').backward(grad_output=grad_output)

class Parallel_Sequential_Model(nn.Module):
    def __init__(self, stages):
        super(Parallel_Sequential_Model, self).__init__()
        self.num_stages = len(stages)
        for i, stage in enumerate(stages):
            setattr(self, f'stage{i}', Stage(stage))
    
    def forward(self, x):
        for i in range(0, self.num_stages):
            print(f"(FWD) Going through stage group {i}")
            x = getattr(self, f'stage{i}')(x)
            if i > 0:
                output_needed = getattr(self, f'stage{i}').outputs[0]
                if len(getattr(self, f'stage{i-1}').outputs) > len(getattr(self, f'stage{i-1}').layer):
                    print("Removing the last element of the outputs list")
                    getattr(self, f'stage{i-1}').outputs.pop(-1)
                getattr(self, f'stage{i-1}').outputs.append(output_needed)
        return x
    
    def backward(self, loss):
        for i in range(self.num_stages-1, -1, -1):
            print(f"(BWD) Going through stage group {i}")
            if i == self.num_stages-1:
                grad_output = getattr(self, f'stage{i}').backward(loss=loss)
            else:
                grad_output = getattr(self, f'stage{i}').backward(grad_output=grad_output)
                
                
class Stage(nn.Sequential):
    def __init__(self, layer):
        if not isinstance(layer, nn.Sequential):
            raise ValueError('layer must be a nn.Sequential')
        for l in layer:
            if isinstance(l, nn.Sequential):
                raise ValueError('layer must not have nested nn.Sequential')
        super(Stage, self).__init__(layer)
        self.layer = layer
        self.outputs = [0]*len(list(layer))
    
    def forward(self, x, compute_grad=True):

        for i, sublayer in enumerate(self.layer): # Problem enumerate self just yields length 1
            x = sublayer(x)
            if compute_grad:
                self.outputs[i] = x
        return x
    def append_output(self, output):
        if len(self.outputs) > len(self.layer):
            self.outputs.pop(-1)
        self.outputs.append(output)
        
    def zero_grad(self, set_to_none: bool = True) -> None:
        return super().zero_grad(set_to_none)
    
    def backward(self, grad_output=None, loss=None):
        if loss is not None:
            grad_output = autograd.grad(loss, self.outputs[-1], retain_graph=True)[0]
            for param in self[-1].parameters():
                param.grad = autograd.grad(self.outputs[-1], param, grad_outputs=grad_output, retain_graph=True)[0]
        for i in range(len(self.outputs)-2, -1, -1): # I modified this to -1 instead of -2
            if self.outputs[i].grad_fn is not None: # NOTE: nn.Flatten() for instance has no grad_fn if it's the first layer in a pipe
                try:
                    grad_output = autograd.grad(self.outputs[i+1], self.outputs[i], grad_outputs=grad_output, retain_graph=True)[0]
                except Exception as e:
                    print(f"Error 1: {e}")
                    print(f"self.outputs[{i+1}]: {self.outputs[i+1].shape}, self.outputs[{i}]: {self.outputs[i].shape}, grad_output: {grad_output.shape}")
                for param in self.layer[i].parameters():
                    try:
                        param.grad = autograd.grad(self.outputs[i], param, grad_outputs=grad_output, retain_graph=True)[0] # Allow unused means param might not be used in the computation graph        
                    except Exception as e:
                        print(f"Error 2: {e}")
                        print(f"self.outputs[{i}]: {self.outputs[i].shape}, param: {param.shape}, grad_output: {grad_output.shape}")
        return grad_output

class Parallelized_Model(nn.Module):
    '''
    Data parallel and weight parallel model.
    '''
    def __init__(self, layer_list, sample, num_replicas=1, criterion=None, approximated_gradient=False):
        super(Parallelized_Model, self).__init__()

        self.num_replicas = num_replicas
        self.criterion = criterion
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
                self.model = Weight_Parallelized_Model(layer_list=layer_list, rank_list=ranks, sample=sample, gpu_id=0, criterion=criterion, approximated_gradient=approximated_gradient)
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
    def __init__(self, layer_list, rank_list, sample, approximated_gradient=False, gpu_id=0, criterion=None):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the layer_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        
        ''' 

        super(Weight_Parallelized_Model, self).__init__()
        self.rank = dist.get_rank()
        self.criterion = criterion
        # if self.rank in [0,1]:
        #     print('asd')
        self.rank_list = rank_list
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        self.approximated_gradient = approximated_gradient
        # print(f'rank {self.rank}')
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.gpu_device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.inputs = []#torch.tensor(()).to(self.gpu_device)  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = []#torch.tensor(()).to(self.gpu_device)  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = []#torch.tensor(()).to(self.gpu_device) # each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        
        # These two are build during forward/backward passes in the initialization phase
        # self.shapes 
        # self.shapes_backward
        
        (layer_class, params) = layer_list[rank_list.index(self.rank)]
        if self.approximated_gradient:
            self.layer = layer_class(**params).to(self.gpu_device) # Initialize the layer with provided parameters
        else:
            self.layer = Stage(layer_class(**params).to(self.gpu_device)) # Initialize the layer with provided parameters
            
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations
        self.f_time = 0 # Forward pass time
        self.g_time = 0 # Gradient computation time
        
        loss = None
        out = self.forward(torch.randn(*sample.shape).to(self.gpu_device), chunks_amount=1, reset_grad=True, compute_grad=True, count_f=False, send_shapes=True)
        if self.rank == rank_list[-1]:
            loss = criterion(out[0], torch.randn(*out[0].shape).to(self.gpu_device))
        self.backward([loss], count_g=False, chunks_amount=1, send_shapes=True)
        self.zero_grad()

        # ZERO GRAD
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
        self.layer.zero_grad()
    
    def grad(self, clone=False):
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return Weight_Parallelized_Tensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2):
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False):
        params = [param.clone() if clone else param for param in self.layer.parameters()]
        return Weight_Parallelized_Tensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass

    # TODO: Keep the wrongly computed derivative (since it is cheap and still works :D) and make a new approach out of it
    def forward(self, x, chunks_amount=2, reset_grad = False, compute_grad = True, count_f=True, send_shapes=False):
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
            # Start the pipeline
            for c in range(chunks_amount):
                i = self.rank_list.index(self.rank)
                if i == 0: # begin of the pipeline (first layer)
                    chunk = chunks[c].to(self.gpu_device)
                    out = self.layer(chunk, compute_grad=compute_grad) if not self.approximated_gradient else self.layer.forward(chunk)
                    next_rank = self.rank_list[i + 1]
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.gpu_device) # this is a placeholder which is needed to make the backward function work
                    if send_shapes:
                        input_shape = lambda x: [x]+list(chunks[c].shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device='cpu' if self.backend == 'gloo' else self.gpu_device)
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
                    # TODO: delete out to free memory? Remember to avoid storing outputs in case "compute_grad" is False
                elif i == len(self.rank_list)-1: # end of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if send_shapes:
                        shapes = receive_shape(src=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        temp = torch.empty(*shapes, device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1])
                    temp = temp.to(self.gpu_device)
                    self.inputs[c] = temp if compute_grad else None
                    out = self.layer(temp, compute_grad=compute_grad) if not self.approximated_gradient else self.layer.forward(temp)
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.gpu_device) # this is a placeholder which is needed to make the backward function 
                    if send_shapes:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                else: # middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if send_shapes:
                        shapes = receive_shape(src=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        temp = torch.empty(*shapes, device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    temp,_ = distops.recv(tensor=temp, src=self.rank_list[i-1])
                    temp = temp.to(self.gpu_device)
                    out = self.layer(temp, compute_grad=compute_grad) if not self.approximated_gradient else self.layer.forward(temp)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.gpu_device) # this is a placeholder which is needed to make the backward function 
                    next_rank = self.rank_list[i + 1]
                    if send_shapes:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device='cpu' if self.backend == 'gloo' else self.gpu_device)
                    distops.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
        # print(f'Rank {self.rank} | Forward pass time: {time.time()-start}')
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        if self.rank == self.rank_list[-1]: # If the rank is in the last layer's rank list, return the output
            self.f_time += time.time() - start
            return self.outputs
        else:
            self.f_time += time.time() - start
            return True

    def backward(self, loss, count_g=True, chunks_amount=2, send_shapes=False):
        self.grad_output = [None]*chunks_amount
        # We used "diffdist" library: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py
        start = time.time()
        if count_g: self.num_g_evals += 1 
        for k in range(chunks_amount): # chunked loss
            # sample_shape = torch.zeros(1, dtype=torch.int32).to(self.gpu_device)
            # if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
            sample_shape = self.outputs[k].shape[0]
            i = self.rank_list.index(self.rank)
            if self.rank == self.rank_list[-1]: # End of the pipeline
                if self.approximated_gradient:
                    self.grad_output[k] = autograd.grad(loss[k], self.outputs[k], create_graph=True)[0]     # TODO: Update so that it takes into account sequential models
                    grad_data = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=self.grad_output[k], retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
                    distops.send(tensor=grad_data.cpu() if self.backend == 'gloo' else grad_data, dst=self.rank_list[-2]) # TODO make this async if possible
                    for param in self.layer.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                else:
                    self.grad_output[k] = self.layer.backward(loss=loss[k])
                    if send_shapes:
                        print(f'(BACKWARD) Rank {self.rank} | Shape: {self.grad_output[k].shape} about to be sent')
                        send_shape(self.grad_output[k].shape, dst=self.rank_list[-2], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        send_shape(self.layer.outputs[0].shape, dst=self.rank_list[-2], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                    distops.send(tensor=self.grad_output[k].cpu() if self.backend == 'gloo' else self.grad_output[k], dst=self.rank_list[-2]) # TODO make this async if possible
                    distops.send(tensor=self.layer.outputs[0].cpu() if self.backend == 'gloo' else self.layer.outputs[0], dst=self.rank_list[-2])

            elif self.rank == self.rank_list[0]:
                if self.approximated_gradient:
                    self.grad_output[k] = torch.empty(*self.shapes[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    self.grad_output[k], _ = distops.recv(self.grad_output[k], src=self.rank_list[1])       
                    self.grad_output[k] = self.grad_output[k].to(self.gpu_device)
                    for param in self.layer.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                else:
                    if send_shapes:
                        print(f'(BACKWARD) Rank {self.rank} | Shape about to be received')
                        shape1 = receive_shape(src=self.rank_list[1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        shape2 = receive_shape(src=self.rank_list[1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        self.shapes_backward = [lambda x: [x]+list(shape1)[1:],lambda x: [x]+list(shape2)[1:]]
                        print(f'(BACKWARD) Rank {self.rank} | Shape: {self.shapes_backward[0](sample_shape)} received')
                    self.grad_output[k], _ = distops.recv(torch.empty(*self.shapes_backward[0](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True), src=self.rank_list[1])   
                    temp = torch.empty(*self.shapes_backward[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    temp, _ = distops.recv(tensor=temp, src=self.rank_list[1])
                    self.layer.append_output(temp)
                    self.grad_output[k] = self.layer.backward(grad_output=self.grad_output[k]) # TO BE FIXED
            else: # middle of the pipeline
                if self.approximated_gradient:
                    self.grad_output[k] = torch.empty(*self.shapes[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    self.grad_output[k], _ = distops.recv(self.grad_output[k], src=self.rank_list[i+1])       
                    self.grad_output[k] = self.grad_output[k].to(self.gpu_device)
                    data_grad2 = autograd.grad(self.outputs[k], self.inputs[k], grad_outputs=self.grad_output[k], retain_graph=True)[0] # this is needed to compute the derivative at the previous layer
                    distops.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1]) # TODO make this async if possible
                    for param in self.layer.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[k], param, grad_outputs=self.grad_output[k], retain_graph=True)[0]/chunks_amount
                    # distops.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1]) # TODO make this async if possible
                    # TODO / NOTE: maybe we can delete self.inputs to free memory. It is not used anymore after the backward pass. (even in subdomains)
                else:
                    if send_shapes:
                        print(f'(BACKWARD) Rank {self.rank} about to be received')
                        shape1 = receive_shape(src=self.rank_list[i+1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        shape2 = receive_shape(src=self.rank_list[i+1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        self.shapes_backward = [lambda x: [x]+list(shape1)[1:],lambda x: [x]+list(shape2)[1:]]
                        print(f'(BACKWARD) Rank {self.rank} | Shape: {self.shapes_backward[0](sample_shape)} received')
                    self.grad_output[k] = torch.empty(*self.shapes_backward[0](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    temp = torch.empty(*self.shapes_backward[1](sample_shape), device='cpu' if self.backend == 'gloo' else self.gpu_device, requires_grad=True)
                    self.grad_output[k], _ = distops.recv(tensor=self.grad_output[k], src=self.rank_list[i+1])
                    temp, _ = distops.recv(tensor=temp, src=self.rank_list[i+1])
                    self.layer.append_output(temp)
                    self.grad_output[k] = self.layer.backward(grad_output=self.grad_output[k])
                    if send_shapes:
                        print(f'(BACKWARD) Rank {self.rank} | Shape: {self.grad_output[k].shape} about to be sent')
                        send_shape(self.grad_output[k].shape, dst=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                        send_shape(self.layer.outputs[0].shape, dst=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.gpu_device)
                    distops.send(tensor=self.grad_output[k].cpu() if self.backend == 'gloo' else self.grad_output[k], dst=self.rank_list[i-1]) # TODO make this async if possible
                    distops.send(tensor=self.layer.outputs[0].cpu() if self.backend == 'gloo' else self.layer.outputs[0], dst=self.rank_list[i-1])
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