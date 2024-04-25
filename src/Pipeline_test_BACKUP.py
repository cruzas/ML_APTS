import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.utility import prepare_distributed_environment
import torch.autograd as autograd
import torch.distributed.rpc as rpc
import diffdist.functional as distops


class F(nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.layer = nn.Linear(100, 50000)

    def forward(self, x):
        return self.layer(x)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.layer = nn.Linear(50000, 1)

    def forward(self, x):
        return self.layer(x)

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

class Weight_Parallelized_Model(nn.Module):
    def __init__(self, layer_list, rank_list, gpu_id=0):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the layer_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        
        ''' 

        super(Weight_Parallelized_Model, self).__init__()
        self.layers = nn.ModuleList()
        rank_list = [[rank] if type(rank) is not list and type(rank) is not tuple else rank for rank in rank_list]
        self.rank_list = rank_list
        self.rank = dist.get_rank()
        self.gpu_id = gpu_id
        self.backend = dist.get_backend()
        self.shapes = []
        self.inputs = torch.tensor(()).to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')  # each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = torch.tensor(()).to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')  # each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass

        # layer_list = [[layer_class, {}] if type(layer_class) is not list and type(layer_class) is not tuple else [layer_class[0], layer_class[1]] for layer_class in layer_list]
        for i, ((layer_class, params, (input_shape, output_shape)), ranks)  in enumerate(zip(layer_list, rank_list)): # Create each layer and assign it to the specified rank (device)
            self.shapes.append((input_shape, output_shape))
            if self.rank in ranks:
                if len(ranks) == 1: # No tensor parallelism (no horizontal slicing)
                    layer = layer_class(**params) # Initialize the layer with provided parameters
                    layer = layer.to(f'cuda:{ranks[0]}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                    self.layers.append(layer) # Register the layer to the module list
                    self.ranks = ranks
                else:
                    raise NotImplementedError("Tensor parallelism is not yet implemented.")
                    # ----> maybe use DeviceMesh -> https://pytorch.org/docs/stable/distributed.html
               
                    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def grad(self):
        return [param.grad for param in self.parameters()]
    
    
    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass
    
    
    def forward(self, x, chunks_amount=2, reset_grad = False, compute_grad = True):
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = torch.tensor(()).to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}') 
        self.outputs = torch.tensor(()).to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
        # "compute_grad" is a flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        # Reset the gradients of the model before starting to accumulate them again
        if reset_grad:
            self.zero_grad()
            
        # Initialize the chunk_shapes tensor to store the shapes of the chunks
        chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32)
        if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
            self.inputs = x if compute_grad else 0
            chunks = x.chunk(chunks_amount)
            chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32)
        # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
        shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0][0], async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer
        # Start the pipeline
        for c in range(chunks_amount):
            assert len(self.layers) == 1, 'This shouldn\'t happen' # This is a temporary solution to avoid the implementation of the forward pass for multiple layers
            i = self.rank_list.index(self.ranks)
            if 0 in self.ranks: # begin of the pipeline (first layer)
                chunk = chunks[c].to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                out = self.layers[0](chunk) # Apply the current layer
                next_rank = self.rank_list[i + 1]
                self.outputs = torch.cat((self.outputs, out), dim=0) if compute_grad else 0
                if self.backend == 'gloo':
                    dist.send(tensor=out.cpu(), dst=next_rank[0]) # send the tensor
                else:
                    dist.send(tensor=out, dst=next_rank[0])
                # TODO: delete out to free memory?
            elif self.rank_list[-1][0] in self.ranks: # end of the pipeline (last layer)
                shape_transfer.wait() # wait for the shape to be broadcasted
                temp = torch.empty(*self.shapes[i][0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                dist.recv(tensor=temp, src=self.rank_list[i-1][0])
                temp = temp.to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                self.inputs = torch.cat((self.inputs, temp), dim=0) if compute_grad else 0
                self.outputs = torch.cat((self.outputs, self.layers[0](temp)), dim=0)
            else: # middle of the pipeline (between the first and the last layer)
                shape_transfer.wait() # wait for the shape to be broadcasted
                temp = torch.empty(*self.shapes[i][0](chunk_shapes[c]), device='cpu' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                dist.recv(tensor=temp, src=self.rank_list[i-1][0])
                temp = temp.to(f'cuda:{self.rank}' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
                self.inputs = torch.cat((self.inputs, temp), dim=0) if compute_grad else 0
                out = self.layers[0](temp)
                self.outputs = torch.cat((self.outputs, out), dim=0) if compute_grad else 0
                next_rank = self.rank_list[i + 1]
                if self.backend == 'gloo':
                    dist.send(tensor=out.cpu(), dst=next_rank[0]) # send the tensor
                else:
                    dist.send(tensor=out, dst=next_rank[0])

        if self.rank in self.rank_list[-1]: # If the rank is in the last layer's rank list, return the output
            return self.outputs
        else:
            return True
    
    
    
    def backward(self, loss):
        # possible solutions: https://github.com/ag14774/diffdist/blob/master/diffdist/testing.py     (DIFFDIST)
        
        sample_shape = torch.zeros(1, dtype=torch.int32)
        if self.rank in self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
            sample_shape = torch.tensor([self.outputs.shape[0]], dtype=torch.int32)
        # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
        shape_transfer = dist.broadcast(tensor=sample_shape.cpu(), src=self.rank_list[0][0], async_op=True) 
        i = self.rank_list.index(self.ranks)
        if self.rank in self.rank_list[-1]: # End of the pipeline
            if len(self.rank_list[-1])==1:
                grad_output = autograd.grad(loss, self.outputs, create_graph=True)[0]
                print(f"RANK {self.rank} - shape of the grad_output: {grad_output.shape}")
                grad_fn = grad_output.grad_fn
                
                ### TEST BEGIN
                distops.send(grad_output.cpu(), dst=self.rank_list[-2][0])
                ### TEST END

                # dist.send(tensor=grad_output.cpu() if self.backend == 'gloo' else grad_output, dst=self.rank_list[-2][0]) # TODO make this async
                # send the grad_fn too
                # dist.send(tensor=, dst=self.rank_list[-2][0])
                for param in self.layers[-1].parameters():
                    param = autograd.grad(self.outputs, param, grad_outputs=grad_output, retain_graph=True)[0]      
            else:
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
        elif self.rank in self.rank_list[0]:
            shape_transfer.wait() # wait for the shape to be broadcasted
            next_grad_output = torch.empty(*self.shapes[i+1][1](sample_shape), device='cpu' if self.backend == 'gloo' else f'cuda:{self.gpu_id}', requires_grad=True)
            
            ### TEST BEGIN
            next_grad_output, _ = distops.recv(next_grad_output, src=self.rank_list[1][0])
            ### TEST END
            
            # dist.recv(tensor=next_grad_output, src=self.rank_list[1][0])
            grad_output = autograd.grad(next_grad_output, self.outputs, create_graph=True)[0]
            for param in self.layers[0].parameters():
                param = autograd.grad(self.outputs, param, grad_outputs=grad_output, retain_graph=True)[0] 
        else: # middle of the pipeline
            shape_transfer.wait() # wait for the shape to be broadcasted
            grad_output = torch.empty(*self.shapes[i+1][1](sample_shape), device='cpu' if self.backend == 'gloo' else f'cuda:{self.gpu_id}')
            dist.recv(tensor=grad_output, src=self.rank_list[i + 1][0])
            for param in self.layers[i].parameters():
                param = autograd.grad(self.outputs, param, grad_outputs=grad_output, retain_graph=True)[0] 
        return None


def main(rank, master_addr, master_port, world_size):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f'Rank {rank} is ready.')
    criteria = torch.nn.CrossEntropyLoss()

    NN1 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(200, 200), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(100, 50), nn.ReLU())
    layer_list = [
        (NN1, {'in_features': 100, 'out_features': 200}, (lambda samples: torch.tensor([samples,100], dtype=torch.int32), lambda samples: torch.tensor([samples,200], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
        (NN2, {'in_features': 200, 'out_features': 100},  (lambda samples: torch.tensor([samples,200], dtype=torch.int32), lambda samples: torch.tensor([samples,50 ], dtype=torch.int32))),
    ]
    Model = nn.Sequential(*[layer[0](**layer[1]) for layer in layer_list]).to('cuda:1')

    rank_list = [0, 1]
    torch.manual_seed(0)
    model = Weight_Parallelized_Model(layer_list, rank_list)
    # update Model params with the one of the model
    for i,param in enumerate(Model.parameters()):
        if rank == 0 and i < 4:
            param.data=list(model.parameters())[i].to('cuda:1')
        elif rank == 1 and i >= 4:
            dist.send(tensor=list(model.parameters())[i-4].cpu().detach(), dst=0)
        if rank == 0 and i >= 4:
            temp = torch.empty_like(param).to('cpu')
            dist.recv(tensor=temp, src=1)
            param.data=temp.to('cuda:1')
        
    # print(list(model.parameters())[0])
    
    x=torch.randn(10, 100)
    output = model(x.to('cuda:0'), reset_grad = True, compute_grad = True)
    if rank == 0:
        exact_output = Model(x.to('cuda:1'))
        print(torch.norm(exact_output))
    # Every rank needs to enter the backward functi1on to compute the gradients
    if rank == 1:
        loss = criteria(output, torch.randint(0, 50, (10,50), dtype=torch.float32).to('cuda:1'))
        print(torch.norm(output))
        params = [param.flatten() for param in Model.parameters()]
        # print(f'Layer 0 : {list(Model.parameters())[0]}')
        # print(f'Layer 1 : {list(Model.parameters())[1]}')
        # # norm of params
        # print(torch.norm(torch.cat(params).flatten()))
    else:
        loss = None
    model.backward(loss)
    dist.barrier()
    grads = model.grad()
    if rank == 0:
        print(torch.norm(torch.cat(g.flatten() for g in grads) - torch.cat([param.grad.flatten() for param in Model.parameters()])))
    # check the derivative is the same as the one computed by the model. First assemble the model only in rank 0
    if rank == 1:
        torch.manual_seed(0)
        output = Model(torch.randn(10, 100).to('cuda:0'))
        loss = criteria(output, torch.randint(0, 50, (10,50), dtype=torch.float32).to('cuda:0'))
        loss.backward()
        grad = torch.cat([param.grad.flatten() for param in Model.parameters()])
        
        print(torch.norm(grad - grad2))
        




if __name__ == '__main__':
    torch.manual_seed(1)

    world_size = torch.cuda.device_count()  
    master_addr = 'localhost'
    master_port = '12345'
    mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
