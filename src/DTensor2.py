# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_loaders.Power_DL import Power_DL
import torch, time
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor, distribute_module
import torch.multiprocessing as mp
from utils.utility import prepare_distributed_environment
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# Create a mesh topology with the available devices:
# 1. We can directly create the mesh using elastic launcher, (recommended)
# 2. If using mp.spawn, one need to initialize the world process_group first and set device
#   i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)



def TEST_TO_SHOW_DIFFERENCE_WITH_SHARDING(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    mesh = init_device_mesh(f"cuda:{rank}", (world_size,))    
    
    v = distribute_tensor(torch.randn([60000*12000,1]), mesh, [Shard(0)])
    w = torch.randn([60000*12000,1]).to(f'cuda:{rank}')
    t1=[]
    t2=[]
    
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(20):
        if rank == 0:
            print(f'Iteration {i}')
        # torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        vv = v.T @ v
        end_event.record()
        # sync
        # torch.cuda.synchronize()
        dist.barrier()
        t1.append(start_event.elapsed_time(end_event)/1000)
        
        start_event.record()
        ww = w.T @ w
        end_event.record()
        # torch.cuda.synchronize()
        dist.barrier() # the difference between sync and barrier is that sync is a point-to-point synchronization, while barrier is a collective synchronization
        t2.append(start_event.elapsed_time(end_event)/1000)
    del v, w, vv, ww
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    print(f'rank{rank} average time for vv: {sum(t1)/len(t1)}, max time for vv: {max(t1)}, min time for vv: {min(t1)}\nrank{rank} average time for ww: {sum(t2)/len(t2)}, max time for ww: {max(t2)}, min time for ww: {min(t2)}')
    return 


def print_tensor_weight_in_gb(tensor):
    """
    Print the weight of a given tensor in gigabytes (GB).
    
    Args:
    - tensor: A PyTorch tensor.
    """
    # Get the number of elements in the tensor
    num_elements = tensor.numel()
    
    # Assuming the tensor dtype is float32, each element takes 4 bytes
    # Adjust this value if using a different data type
    element_size_bytes = tensor.element_size()
    
    # Calculate the total weight in bytes
    total_bytes = num_elements * element_size_bytes
    
    # Convert bytes to gigabytes (1 GB = 2^30 bytes)
    total_gb = total_bytes / (1024**3)
    
    print(f"The tensor weight is approximately {total_gb:.6f} GB.")
    
    
def bytes_to_gb(bytes):
    return bytes / (1024 ** 3)

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    mesh = init_device_mesh(f"cuda", (int(world_size),))
    # mesh = init_device_mesh(f"cuda:{rank}", (int(world_size),))
    # mesh = init_device_mesh(["cuda:0", "cuda:1"], mesh_shape=(2, 1) )

    big_tensor = torch.randn(int(2e8),1,  device="cpu").squeeze()
    big_tensor2 = torch.randn(int(2e8),1,  device="cpu").squeeze()
    print_tensor_weight_in_gb(big_tensor)
    big_tensor = big_tensor.to(f"cuda:{rank}")
    big_tensor2 = big_tensor2.to(f"cuda:{rank}")
    tic = time.time()
    multiplier = big_tensor @ big_tensor2
    toc = time.time()
    print(f"rank{rank} Elapsed time: {toc-tic:2f} seconds")
    torch.cuda.set_device(rank)
    allocated = bytes_to_gb(torch.cuda.memory_allocated())
    reserved = bytes_to_gb(torch.cuda.memory_reserved())
    print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
    
    #remove big_tensor from CUDA
    big_tensor = big_tensor.to("cpu")
    big_tensor2 = big_tensor2.to("cpu")
    torch.cuda.empty_cache()
    
    allocated = bytes_to_gb(torch.cuda.memory_allocated())
    reserved = bytes_to_gb(torch.cuda.memory_reserved())
    print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    my_dtensor2 = distribute_tensor(big_tensor2, mesh, [Shard(dim=1)])
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            torch.cuda.set_device(rank)
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            tic = time.time()
            multiplier = my_dtensor @ my_dtensor2
            toc = time.time()
            end_event.record()
            print(f"---rank{rank} Elapsed time: {toc-tic:2f} seconds")
            del multiplier
            torch.cuda.empty_cache()
            # print(f'Device {rank} matrix norm: {multiplier.norm()}')
            torch.cuda.synchronize()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)


            if rank == 0:
                print(f"Elapsed time: {elapsed_time/1000} seconds")

            

def main2(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    big_tensor = torch.randn(int(2e4), int(2e4), device=f"cuda:{rank}")
    
    with torch.no_grad():
        for i in range(10):
            if rank == 0:
                print(i)

            torch.cuda.set_device(rank)
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated before: {allocated:.2f} GB. Memory Reserved before: {reserved:.2f} GB")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            multiplier = big_tensor @ big_tensor
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()
            allocated = bytes_to_gb(torch.cuda.memory_allocated())
            reserved = bytes_to_gb(torch.cuda.memory_reserved())
            print(f"Device {rank}. Memory Allocated after: {allocated:.2f} GB. Memory Reserved after: {reserved:.2f} GB")
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Device {rank}. Elapsed time: {elapsed_time/1000} seconds")
    

def main3(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    class MNIST_FCNN_Small(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=[32], output_size=10):
            super(MNIST_FCNN_Small, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_sizes[0])
            self.l2 = nn.Linear(hidden_sizes[0], output_size)
            
        def forward(self, x):
            x = F.relu(self.l1(x))
            x = torch.sigmoid(self.l2(x))
            x = F.log_softmax(x, dim=1)
            return x


    mesh = init_device_mesh(f"cuda:{rank}", (world_size,))

    def shard_params(mod_name, mod, mesh):
        col_linear_placement = [Shard(0)]
        # shard fc1 and fc2
        if isinstance(mod, nn.Linear):
            for name, param in mod.named_parameters():
                dist_param = nn.Parameter(
                    distribute_tensor(param, mesh, col_linear_placement)
                )
                mod.register_parameter(name, dist_param)

    sharded_module = distribute_module(MNIST_FCNN_Small(), mesh, partition_fn=shard_params)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = Power_DL(dataset=train_dataset, shuffle=True, device='cpu', minibatch_size=60000)
    test_loader = Power_DL(dataset=test_dataset, shuffle=False, device='cpu', minibatch_size=10000)

    data = list(train_loader)[0][0]
    data = data[0,:]
    data = data.view(data.size(0), -1)

    sharded_data = distribute_tensor(data, mesh, [Shard(1)])
    target = list(train_loader)[0][1]
    target = target[0]
    sharded_target = target.to(f'cuda:{rank}')
    # sharded_target = distribute_tensor(target, mesh, [Shard(0)])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(sharded_module.parameters(), lr=0.01, momentum=0.9)
    
    v = sharded_data
    v2 = data
    model = MNIST_FCNN_Small()
    model.l1(v2)
    print(v.shape)
    sharded_module.l1(v)
    # print information about cuda:0
    print(f"rank{rank} cuda:0, gpu type {torch.cuda.get_device_properties(0)}")
    
    sharded_module.train() # Set the model to training mode
    for epoch in range(10): # Let's train for 10 epochs
        optimizer.zero_grad()
        # Forward pass
        output = sharded_module(sharded_data)
        loss = criterion(output, sharded_target)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')


def main4(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    class MNIST_FCNN_Small(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=[32], output_size=10):
            super(MNIST_FCNN_Small, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_sizes[0])
            self.l2 = nn.Linear(hidden_sizes[0], output_size)
            
        def forward(self, x):
            x = F.relu(self.l1(x))
            x = torch.sigmoid(self.l2(x))
            x = F.log_softmax(x, dim=1)
            return x
        
    torch.cuda.set_device(f"cuda:{rank}")
    sharded_module = FSDP(MNIST_FCNN_Small)
    optim = torch.optim.Adam(sharded_module.parameters(), lr=0.001)
    x = sharded_module(x, y=3, z=torch.Tensor([1]))
    loss = x.sum()
    loss.backward()
    optim.step()

if __name__ == '__main__':
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size == 0:
        print("No CUDA device(s) detected.")
        exit(0)
    master_addr = 'localhost'
    master_port = '12345'  
    mp.spawn(main4, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
