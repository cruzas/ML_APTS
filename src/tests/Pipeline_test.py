import os
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

# Global gradient class
class Weight_Parallelized_Gradient(nn.Module):
    def __init__(self, rank_list, w_p_model, gpu_id=0):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.rank_list = rank_list
        self.model = w_p_model

    def __repr__(self):
        return str(self.model)
    
    def __matmul__(self, a): # self.grad @ a
        return self * a
    
    def __rmatmul__(self, a): # a @ self.grad
        return a * self
    
    def __mul__(self, a):  # self.grad * a
        if isinstance(a, Weight_Parallelized_Gradient):  # When both operands are Weight_Parallelized_Gradient instances
            g1 = torch.cat([p.grad.flatten() for p in self.model.parameters()], dim=0)  # Flatten the gradients
            g2 = torch.cat([p.grad.flatten() for p in a.model.parameters()], dim=0)  # Flatten the gradients
            g3 = g1 @ g2
            g3 = g3.to(f'cpu' if self.model.backend == 'gloo' else f'cuda:{self.gpu_id}')
            if self.model.rank in self.model.list_of_master_nodes:
                dist.reduce(tensor=g3, dst=self.rank_list[0][0], group=self.model.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3 # Multiply the internal models
        else:
            return Weight_Parallelized_Gradient(self.rank_list, self.model * a)  # Multiply model by a scalar or tensor

    def __rmul__(self, a):  # a * self.grad
        return self.__mul__(a)  # This handles the commutative property of multiplication



def decide_gpu_device(ws, backend, gpu_id):
    if backend == 'gloo':
        if torch.cuda.device_count() < ws:
            return f'cuda:{gpu_id}'
        else:
            return f'cuda:{dist.get_rank()}'
    else:
        return f'cuda:{gpu_id}'
    


# Global model class
class Weight_Parallelized_Model(nn.Module):
    def __init__(self, layer_list, rank_list, gpu_id=0):
        '''
        - input_shape: Shape of the input tensor. This is used to generate a dummy tensor for the first layer and compute the output shape of every layer to adapt the send/recv in the pipeline.
        
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
    
    def grad(self):
        return [param.grad for param in self.parameters()]
    
    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=2).item()

    def shape_setup(self, x):
        # This should be a setup phase
        # TODO: could be useful to have a function that computes the shapes of the tensors for each rank in the pipeline in and out of the layers in place of the manual input_shape and output_shape
        pass
    

    
    def forward(self, x, chunks_amount=2, reset_grad = False, compute_grad = True):
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


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    print(f'Rank {rank} is ready.')
    criteria = torch.nn.CrossEntropyLoss()

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
        torch.manual_seed(0)
        model = Weight_Parallelized_Model(layer_list, rank_list)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Data loading TODO: load data only on master master rank
        train_loader, test_loader = create_dataloaders(
            dataset="MNIST",
            data_dir=os.path.abspath("./data"),
            mb_size=6000,
            overlap_ratio=0,
            parameter_decomposition=True,
            device="cuda:0"
        )
        device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)

        for epoch in range(100):
            for i, (x, y) in enumerate(train_loader):
                # Vectorize x 
                x = x.view(x.size(0), -1)
                output = model(x.to(device), chunks_amount=1, reset_grad = True, compute_grad = True)
                if rank == rank_list[-1][0]:
                    loss = criteria(output, y.to(device))
                    print(f'Epoch {epoch} loss {loss}')
                else:
                    loss = None

                model.backward(loss)
                # Print norm of the gradient
                # print(f'Rank {rank} grad norm: {model.grad_norm()}')
                # print('Hurra 2!')
                
                # one sgd step
                optimizer.step()

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
        #     loss = criteria(output, target)
        # else:
        #     loss = None 

        # model.backward(loss)
        # print('Hurra!')
        # dist.barrier(group=group)
        
        g1 = Weight_Parallelized_Gradient(rank_list, model)
        g2 = Weight_Parallelized_Gradient(rank_list, model)
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
            loss = criteria(output, target)
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

