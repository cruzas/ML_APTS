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
    
# TODO: before send we could reduce the weight of tensor by using the half precision / float16, then we can convert it back to float32 after the recv
class Sequential_Model(nn.Module):
    def __init__(self, pipe_list):
        super(Sequential_Model, self).__init__()
        self.num_pipes = len(pipe_list)
        for i, pipe in enumerate(pipe_list):
            setattr(self, f'pipe{i}', Stage(stage=pipe, extra_layer=pipe_list[i+1][0] if i+1 < len(pipe_list) else None))
    
    def grad_norm(self):
        g = [0]*self.num_pipes
        for i in range(self.num_pipes):
            g[i] = torch.norm(torch.cat([param.grad.flatten() for param in getattr(self, f'pipe{i}').parameters()]))
        return g

    def forward(self, x):
        for i in range(0, self.num_pipes):
            x = getattr(self, f'pipe{i}')(x)
            if i > 0:
                output_needed = getattr(self, f'pipe{i}').outputs[0]
                if len(getattr(self, f'pipe{i-1}').outputs) > len(getattr(self, f'pipe{i-1}').stage):
                    getattr(self, f'pipe{i-1}').outputs.pop(-1)
                getattr(self, f'pipe{i-1}').outputs.append(output_needed)
                # print(f'(SEQUENTIAL) Index {i} appended to {i-1} with shape {output_needed.shape}')
        return x
    
    def backward(self, loss):
        for i in range(self.num_pipes-1, -1, -1):
            # print(f"(BWD) Going through pipe group {i}")
            if i == self.num_pipes-1:
                grad_output = getattr(self, f'pipe{i}').backward(loss=loss)
            else:
                # print(f"(SEQUENTIAL) Stage {i} grad_output norm {torch.norm(grad_output.flatten())}")
                grad_output = getattr(self, f'pipe{i}').backward(grad_output=grad_output)

class Parallel_Sequential_Model(nn.Module):
    def __init__(self, stages, rank_list, sample):
        super(Parallel_Sequential_Model, self).__init__()
        if len(stages) != dist.get_world_size():
            raise ValueError(f"The number of stages ({len(stages)}) must be equal to the world size ({dist.get_world_size()}).")
        self.rank = dist.get_rank()
        self.rank_index = rank_list.index(self.rank)
        self.backend = dist.get_backend()
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.send_recv_device = 'cpu' if self.backend == 'gloo' else self.tensor_device
        self.stage = Stage(stage=stages[self.rank_index], extra_layer=stages[self.rank_index+1][0] if self.rank_index+1 < len(stages) else None, device = self.tensor_device)
        self.rank_list = rank_list
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        # Setup phase to compute the shapes of the tensors in the pipeline
        self.setup_phase = True
        outputs = self.forward(sample, chunks_amount=2, reset_grad=True, compute_grad=True)
        losses = []
        for x in outputs:
            losses.append(nn.MSELoss()(x, torch.rand_like(x)))
        self.backward(losses)
        self.setup_phase = False
        self.zero_grad()
        # End of setup phase
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.stage.parameters()]))
        
    def zero_grad(self):
        self.stage.zero_grad()
    
    def forward(self, x, chunks_amount=1, reset_grad=False, compute_grad=True):
        if reset_grad:
            self.zero_grad()
        self.stage.clear_outputs()
        # Send and receive chunk shape
        chunk_size = torch.zeros(chunks_amount, dtype=torch.int32, device=self.send_recv_device)
        if self.rank == self.rank_list[0]:
            chunks = x.chunk(chunks_amount)
            chunk_size = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.send_recv_device)
        shared_shape = dist.broadcast(tensor=chunk_size, src=self.rank_list[0], group=self.master_group, async_op=True)
        rank_index = self.rank_index
        for c in range(chunks_amount):
            if self.rank == self.rank_list[0]: # First stage in pipeline
                x = self.stage(chunks[c], compute_grad=compute_grad)
                if self.setup_phase:
                    send_shape(x.shape, dst=self.rank_list[1], device=self.send_recv_device) 
                dist.send(tensor=x.to(self.send_recv_device), dst=self.rank_list[1])
            elif self.rank == self.rank_list[-1]: # Last stage in pipeline
                shared_shape.wait()
                if self.setup_phase:
                    shape = receive_shape(src=self.rank_list[-2], device=self.send_recv_device)
                    self.shape = lambda x: [x]+shape[1:]
                x = torch.empty(*self.shape(chunk_size[c]), device=self.send_recv_device)
                dist.recv(tensor=x, src=self.rank_list[-2])
                x = self.stage(x, compute_grad=compute_grad)
            else: # Middle stages in pipeline
                shared_shape.wait()
                if self.setup_phase:
                    shape = receive_shape(src=self.rank_list[rank_index-1], device=self.send_recv_device)
                    self.shape = lambda x: [x]+shape[1:]
                x = torch.empty(*self.shape(chunk_size[c]), device=self.send_recv_device)
                dist.recv(tensor=x, src=self.rank_list[rank_index-1])
                x = self.stage(x, compute_grad=compute_grad)
                if self.setup_phase:
                    send_shape(x.shape, dst=self.rank_list[rank_index+1], device=self.send_recv_device)
                dist.send(tensor=x.to(self.send_recv_device), dst=self.rank_list[rank_index+1])
        if self.rank == self.rank_list[-1]:
            return self.stage.outputs[-1]
        return [torch.randn(1, device=self.tensor_device) for _ in range(chunks_amount)]
    
    def backward(self, losses):
        for c, loss in enumerate(losses):
            chunk_size = self.stage.outputs[0][c].shape[0]
            rank_index = self.rank_index
            if rank_index == self.rank_list[-1]:
                grad_output = self.stage.backward(chunk_idx=c, loss=loss)
                # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                if self.setup_phase:
                    send_shape(grad_output.shape, dst=self.rank_list[rank_index-1], device=self.send_recv_device)
                dist.send(tensor=grad_output.to(self.send_recv_device), dst=self.rank_list[rank_index-1])
            elif rank_index != self.rank_list[0]: # Middle stages in pipeline
                if self.setup_phase:
                    shape = receive_shape(src=self.rank_list[rank_index+1], device=self.send_recv_device)
                    self.shape_grad_output = lambda x: [x]+shape[1:]
                grad_output = torch.empty(*self.shape_grad_output(chunk_size), device=self.send_recv_device)
                dist.recv(grad_output, src=self.rank_list[rank_index+1])
                grad_output = self.stage.backward(chunk_idx=c, grad_output=grad_output)
                # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                if self.setup_phase:
                    send_shape(grad_output.shape, dst=self.rank_list[rank_index-1], device=self.send_recv_device)
                dist.send(tensor=grad_output.to(self.send_recv_device), dst=self.rank_list[rank_index-1])
            else:
                if self.setup_phase:
                    shape = receive_shape(src=self.rank_list[rank_index+1], device=self.send_recv_device)
                    self.shape_grad_output = lambda x: [x]+shape[1:]
                grad_output = torch.empty(*self.shape_grad_output(chunk_size), device=self.send_recv_device)
                dist.recv(grad_output, src=self.rank_list[rank_index+1])
                # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                self.stage.backward(chunk_idx=c, grad_output=grad_output)
                
class Stage(nn.Sequential):
    def __init__(self, stage, extra_layer, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        if not isinstance(stage, nn.Sequential):
            raise ValueError('layer must be a nn.Sequential')
        for l in stage:
            if isinstance(l, nn.Sequential):
                raise ValueError('layer must not have nested nn.Sequential')
        super(Stage, self).__init__(stage)
        self.stage = stage.to(device)
        self.extra_layer = extra_layer.to(device) if extra_layer is not None else None
        if self.extra_layer is not None:
            self.outputs = [[] for _ in range(len(self.stage)+1)]
        else:
            self.outputs = [[] for _ in range(len(self.stage))]
        self.device = device

    def clear_outputs(self):
        if self.extra_layer is not None:
            self.outputs = [[] for _ in range(len(self.stage)+1)]
        else:
            self.outputs = [[] for _ in range(len(self.stage))]

    # Override parameters() function
    def parameters(self):
        return [param for param in self.stage.parameters()]

    def forward(self, x, compute_grad=True):
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        # TODO: Efficiency problems -> there is no need to store outputs of layers which do not have parameters (like ReLU, Flatten, etc.) -> this is a waste of memory. 
        #       We should skip them and also adjust the backward to skip them and avoid updating grad_output
        for i, substage in enumerate(self.stage): # Problem enumerate self just yields length 1
            x = substage(x.to(self.device))
            if compute_grad:
                self.outputs[i].append(x)
        if self.extra_layer is not None and compute_grad:
            self.outputs[i+1].append(self.extra_layer(x))
        return x
        
    def zero_grad(self, set_to_none: bool = True) -> None:
        return super().zero_grad(set_to_none)
    
    def backward(self, chunk_idx, grad_output=None, loss=None):
        grad_output = grad_output.to(self.device) if grad_output is not None else None
        loss = loss.to(self.device) if loss is not None else None
        if loss is not None:
            try:
                grad_output = autograd.grad(loss, self.outputs[-1][chunk_idx], retain_graph=True)[0]
            except RuntimeError as e:
                print(f"Error in grad computation: {e}")
            for param in self[-1].parameters():
                param.grad = autograd.grad(self.outputs[-1][chunk_idx], param, grad_outputs=grad_output, retain_graph=True)[0]
        for i in range(len(self.outputs)-2, -1, -1): # I modified this to -1 instead of -2
            if self.outputs[i][chunk_idx].grad_fn is not None: # NOTE: nn.Flatten() for instance has no grad_fn if it's the first stage in a pipe
                grad_output = autograd.grad(self.outputs[i+1][chunk_idx], self.outputs[i][chunk_idx], grad_outputs=grad_output, retain_graph=True)[0]
                for param in self.stage[i].parameters():
                    param.grad = autograd.grad(self.outputs[i][chunk_idx], param, grad_outputs=grad_output, retain_graph=True)[0] # Allow unused means param might not be used in the computation graph        
        return grad_output

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
    def __init__(self, stage_list, extra_layers, rank_list, sample, approximated_gradient=False, gpu_id=0):
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
        self.approximated_gradient = approximated_gradient
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.send_recv_device = 'cpu' if self.backend == 'gloo' else self.tensor_device
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)
        self.inputs = []#torch.tensor(()).to(self.tensor_device)  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = []#torch.tensor(()).to(self.tensor_device)  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = []#torch.tensor(()).to(self.tensor_device) # each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        # These two are built during forward/backward passes in the initialization phase
        # self.shapes 
        # self.shapes_backward
        (layer_class, params) = stage_list[self.rank_index]
        (next_class, next_params) = extra_layers[self.rank_index] if self.rank_index < len(stage_list)-1 else (None, None)
        if self.approximated_gradient:
            self.stage = layer_class(**params).to(self.tensor_device) # Initialize the layer with provided parameters
        else:
            extra_layer = next_class(**next_params) if next_class is not None else None
            self.stage = Stage(stage=layer_class(**params).to(self.tensor_device), extra_layer=extra_layer) # Initialize the layer with provided parameters
            
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations
        self.f_time = 0 # Forward pass time
        self.g_time = 0 # Gradient computation time
        
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
        self.stage.zero_grad()
    
    def grad(self, clone=False):
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return Weight_Parallelized_Tensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2):
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False):
        params = [param.clone() if clone else param for param in self.stage.parameters()]
        return Weight_Parallelized_Tensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass

    # TODO: Keep the wrongly computed derivative (since it is cheap and still works :D) and make a new approach out of it
    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        start = time.time()
        self.stage.clear_outputs()
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = [None]*chunks_amount # Needed for the approximated_gradient case
        self.outputs = [None]*chunks_amount  
        # "compute_grad" is a flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        self.num_f_evals += 1 
        with torch.set_grad_enabled(compute_grad):
            # Reset the gradients of the model before starting to accumulate them again
            if reset_grad:
                self.zero_grad()
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32).to(self.tensor_device)
            if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = x.chunk(chunks_amount)
                self.inputs = chunks if compute_grad else []
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.tensor_device)
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            # torch.distributed.get_process_group_ranks(self.master_group)
            sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
            shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer
            sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
            # Start the pipeline
            for c in range(chunks_amount):
                i = self.rank_index
                if i == 0: # begin of the pipeline (first layer)
                    chunk = chunks[c].to(self.tensor_device)
                    out = self.stage(chunk, compute_grad=compute_grad) if not self.approximated_gradient else self.stage.forward(chunk)
                    next_rank = self.rank_list[i + 1]
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.tensor_device) # this is a placeholder which is needed to make the backward function work
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(chunks[c].shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device='cpu' if self.backend == 'gloo' else self.tensor_device)
                    dist.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
                    # TODO: delete out to free memory? Remember to avoid storing outputs in case "compute_grad" is False
                elif i == len(self.rank_list)-1: # end of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.tensor_device)
                        temp = torch.empty(*shapes, device='cpu' if self.backend == 'gloo' else self.tensor_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.tensor_device, requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[i-1])
                    temp = temp.to(self.tensor_device)
                    self.inputs[c] = temp if compute_grad else None
                    out = self.stage(temp, compute_grad=compute_grad) if not self.approximated_gradient else self.stage.forward(temp)
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.tensor_device) # this is a placeholder which is needed to make the backward function 
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                else: # middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[i-1], device='cpu' if self.backend == 'gloo' else self.tensor_device)
                        temp = torch.empty(*shapes, device='cpu' if self.backend == 'gloo' else self.tensor_device)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else self.tensor_device)
                    dist.recv(tensor=temp, src=self.rank_list[i-1])
                    temp = temp.to(self.tensor_device)
                    out = self.stage(temp, compute_grad=compute_grad) if not self.approximated_gradient else self.stage.forward(temp)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = out if compute_grad else torch.randn(out.shape[0], device=self.tensor_device) # this is a placeholder which is needed to make the backward function 
                    next_rank = self.rank_list[i + 1]
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device='cpu' if self.backend == 'gloo' else self.tensor_device)
                    dist.send(tensor=out.cpu() if self.backend == 'gloo' else out, dst=next_rank) # send the tensor
        # print(f'Rank {self.rank} | Forward pass time: {time.time()-start}')
        sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
        if self.rank == self.rank_list[-1]: # If the rank is in the last layer's rank list, return the output
            self.f_time += time.time() - start
            return self.outputs
        else:
            self.f_time += time.time() - start
            return True

    def backward(self, losses):
        chunks_amount = len(losses)
        self.grad_output = [None]*chunks_amount
        # We used "diffdist" library: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py
        start = time.time()
        self.num_g_evals += 1 
        for c, loss in enumerate(losses): # chunked loss
            # sample_shape = torch.zeros(1, dtype=torch.int32).to(self.tensor_device)
            # if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
            chunk_size = self.outputs[c].shape[0] if self.approximated_gradient else self.stage.outputs[0][c].shape[0]
            rank_index = self.rank_index
            if self.rank == self.rank_list[-1]: # End of the pipeline
                if self.approximated_gradient:
                    self.grad_output[c] = autograd.grad(loss, self.outputs[c], create_graph=True)[0]     # TODO: Update so that it takes into account sequential models
                    grad_data = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                    dist.send(tensor=grad_data.cpu() if self.backend == 'gloo' else grad_data, dst=self.rank_list[-2]) # TODO make this async if possible
                    for param in self.stage.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                else:
                    grad_output = self.stage.backward(chunk_idx=c, loss=loss)
                    # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                    if self.setup_phase:
                        send_shape(grad_output.shape, dst=self.rank_list[rank_index-1], device=self.send_recv_device)
                    dist.send(tensor=grad_output.to(self.send_recv_device), dst=self.rank_list[rank_index-1])

            elif self.rank == self.rank_list[0]:
                if self.approximated_gradient:
                    self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device='cpu' if self.backend == 'gloo' else self.tensor_device, requires_grad=True)
                    dist.recv(self.grad_output[c], src=self.rank_list[1])       
                    self.grad_output[c] = self.grad_output[c].to(self.tensor_device)
                    for param in self.stage.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                else:
                    if self.setup_phase:
                        shape = receive_shape(src=self.rank_list[rank_index+1], device=self.send_recv_device)
                        self.shape_grad_output = lambda x: [x]+shape[1:]
                    grad_output = torch.empty(*self.shape_grad_output(chunk_size), device=self.send_recv_device)
                    dist.recv(grad_output, src=self.rank_list[rank_index+1])
                    # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                    self.stage.backward(chunk_idx=c, grad_output=grad_output)
            else: # middle of the pipeline
                if self.approximated_gradient:
                    self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device='cpu' if self.backend == 'gloo' else self.tensor_device, requires_grad=True)
                    dist.recv(self.grad_output[c], src=self.rank_list[i+1])       
                    self.grad_output[c] = self.grad_output[c].to(self.tensor_device)
                    data_grad2 = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                    dist.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1]) # TODO make this async if possible
                    for param in self.stage.parameters():
                        if param.grad is None:
                            param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                        else:
                            param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    # dist.send(tensor=data_grad2.cpu() if self.backend == 'gloo' else data_grad2, dst=self.rank_list[i-1]) # TODO make this async if possible
                    # TODO / NOTE: maybe we can delete self.inputs to free memory. It is not used anymore after the backward pass. (even in subdomains)
                else:
                    if self.setup_phase:
                        shape = receive_shape(src=self.rank_list[rank_index+1], device=self.send_recv_device)
                        self.shape_grad_output = lambda x: [x]+shape[1:]
                    grad_output = torch.empty(*self.shape_grad_output(chunk_size), device=self.send_recv_device)
                    dist.recv(grad_output, src=self.rank_list[rank_index+1])
                    grad_output = self.stage.backward(chunk_idx=c, grad_output=grad_output)
                    # print(f'(PARALLEL) Stage {dist.get_rank()} RECEIVED grad_output norm {torch.norm(grad_output.flatten())}')
                    if self.setup_phase:
                        send_shape(grad_output.shape, dst=self.rank_list[rank_index-1], device=self.send_recv_device)
                    dist.send(tensor=grad_output.to(self.send_recv_device), dst=self.rank_list[rank_index-1])
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