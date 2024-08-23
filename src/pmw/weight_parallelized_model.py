import torch
import time
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import utils
from pmw.weight_parallelized_subdomain import WeightParallelizedSubdomain
from pmw.weight_parallelized_tensor import WeightParallelizedTensor
from pmw.base_model import BaseModel

class WeightParallelizedModel(BaseModel):
    def __init__(self, stage_list, rank_list, sample):        
        '''
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the stage_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        ''' 
        super().__init__()
        self.rank_list = rank_list
        self.rank_index = rank_list.index(self.rank)
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        (layer_class, params) = stage_list[self.rank_index]
        self.stage = layer_class(**params).to(self.tensor_device) # Initialize the layer with provided parameters
        self.subdomain = WeightParallelizedSubdomain(self.stage) # Initialize the subdomain model
        self.do_setup_phase(rank_list, sample)
        
    def do_setup_phase(self,rank_list, sample):
        self.setup_phase = True
        loss = None
        out = self.forward(torch.randn(*sample.shape).to(self.tensor_device), chunks_amount=1, reset_grad=True, compute_grad=True)
        if self.rank == rank_list[-1]:
            loss = nn.MSELoss()(out[0], torch.rand_like(out[0]))
        self.backward([loss])
        self.zero_grad()
        self.setup_phase = False

    def zero_grad(self):
        self.stage.zero_grad()
    
    def grad(self, clone=False): # Returns the global gradient of the model
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return WeightParallelizedTensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        params = [param.clone() if clone else param for param in self.stage.parameters()]
        return WeightParallelizedTensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2): # Returns the subdomain gradient norm of the model
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def forward(self, x, chunks_amount=1, reset_grad=False, compute_grad=True):        
        # Initialize the input and output tensors on the subdomain (needed for the backward pass and to process data on their own)
        self.subdomain.inputs = [None]*chunks_amount
        self.subdomain.outputs = [None]*chunks_amount  
        
        compute_grad = False if not torch.is_grad_enabled() else compute_grad # flag to avoid storing the tensors needed to compute the gradients
        with torch.set_grad_enabled(compute_grad):
            if reset_grad:
                self.zero_grad() # Reset the gradients of the model before starting to accumulate them again
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32)
            if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = list(x.chunk(chunks_amount)) # Chunkenize the input tensor
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32) # Store the batch size of each chunk
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=chunk_shapes.to(self.backend_device(x)), src=self.rank_list[0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer

            # Go through the pipeline
            for c in range(chunks_amount):
                if self.rank_index == 0: # Beginning of the pipeline (first layer)
                    chunk = chunks[c].to(self.tensor_device)
                    out = self.subdomain.forward(chunk) 
                    next_rank = self.rank_list[self.rank_index+1]
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(chunks[c].shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        utils.send_shape(out.shape, dst=next_rank, device=self.backend_device(out))
                    dist.send(tensor=out.to(self.backend_device(out)), dst=next_rank) # send the tensor
                elif self.rank_index == len(self.rank_list)-1: # End of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = utils.receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device())
                        temp = torch.empty(*shapes, device=self.backend_device(), requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device(), requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    out = self.subdomain.forward(temp)
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                else: # Middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # Wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = utils.receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device())
                        temp = torch.empty(*shapes, device=self.backend_device(), requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device(), requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    out = self.subdomain.forward(temp)
                    next_rank = self.rank_list[self.rank_index+1]
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        utils.send_shape(out.shape, dst=next_rank, device=self.backend_device())
                    dist.send(tensor=out.to(self.backend_device(out)), dst=next_rank) # send the tensor
        return self.subdomain.outputs if self.rank == self.rank_list[-1] else True

    def backward(self, losses):
        chunks_amount = len(losses) # Number of chunks 
        self.subdomain.grad_outputs = [None]*chunks_amount 
        for c, loss in enumerate(losses): # Chunked loss
            chunk_size = self.subdomain.outputs[c].shape[0]
            if self.rank == self.rank_list[-1]: # End of the pipeline
                grad_output = autograd.grad(loss, self.subdomain.outputs[c], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
                grad_data = autograd.grad(self.subdomain.outputs[c], self.subdomain.inputs[c], grad_outputs=grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=grad_data.to(self.backend_device(grad_data)), dst=self.rank_list[-2]) # TODO make this async if possible
                
            elif self.rank == self.rank_list[0]: # Beginning of the pipeline
                grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
                dist.recv(grad_output, src=self.rank_list[1])       

                grad_output = grad_output.to(self.tensor_device).detach()
            else: # Middle of the pipeline
                grad_output = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device(), requires_grad=True)
                dist.recv(grad_output, src=self.rank_list[self.rank_index+1])       

                grad_output = grad_output.to(self.tensor_device).detach()
                grad_data = autograd.grad(self.subdomain.outputs[c], self.subdomain.inputs[c], grad_outputs=grad_output, retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=grad_data.to(self.backend_device(grad_data)), dst=self.rank_list[self.rank_index-1]) # TODO make this async if possible
                
            self.subdomain.backward(grad_output)
 