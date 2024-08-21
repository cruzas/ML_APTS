import time 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
from utils import *
from pmw.Weight_Parallelized_Subdomain import Weight_Parallelized_Subdomain
from pmw.Weight_Parallelized_Tensor import Weight_Parallelized_Tensor

class Weight_Parallelized_Model(nn.Module):
    def __init__(self, stage_list, rank_list, sample, device=None):
        '''
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
        # This is built during forward/backward passes in the initialization phase
        # self.shapes 
    
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