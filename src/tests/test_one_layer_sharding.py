import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
from torch.distributed.device_mesh import init_device_mesh
import os, subprocess

# Define your model architecture here
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer you want to parallelize
        self.linear = nn.Linear(1024, 1024)
        
    def forward(self, x):
        x = self.linear(x)
        return x



# Initialize the model
model = TransformerBlock()

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2' 
os.environ['MASTER_ADDR'] = 'localhost'
os.environ["MASTER_PORT"] = "12355"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.distributed.init_process_group("gloo", rank=0, world_size=2)
# torch.cuda.set_device(rank)

# Set up your distributed environment and initialize DeviceMesh
# This line should be adapted to your specific environment and may require launching a distributed job
tp_mesh = init_device_mesh("cuda", (2,))

# Define the parallelization plan. Use RowwiseParallel for the linear layer
parallelize_plan = {
    "linear": RowwiseParallel(),
}

# Apply row-wise tensor parallelism to the model
parallelized_model = parallelize_module(model, tp_mesh, parallelize_plan)

# Dummy input for the sake of example
input_tensor = torch.rand(64, 1024)

# Forward pass
output = parallelized_model(input_tensor)









# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributed.tensor.parallel import RowwiseParallel, ColwiseParallel, parallelize_module
# from torch.distributed.device_mesh import init_device_mesh
# import torch.multiprocessing as mp
# import os, subprocess
# import torch.distributed as dist

# something_size = 1024*20

# def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
#     device_id = 0
#     if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
#         print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
#         ## Execute code on a cluster
#         os.environ["MASTER_PORT"] = "29501"
#         os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
#         os.environ["LOCAL_RANK"] = "0"
#         os.environ["RANK"] = os.environ["SLURM_NODEID"]
#         node_list = os.environ["SLURM_NODELIST"]
#         master_node = subprocess.getoutput(
#             f"scontrol show hostname {node_list} | head -n1"
#         )
#         os.environ["MASTER_ADDR"] = master_node
#         print(f"Dist initialized before process group? {dist.is_initialized()}")
#         dist.init_process_group(backend="nccl")
#         print(f"Dist initialized after init process group? {dist.is_initialized()} with world size {dist.get_world_size()}")
#     else: # we are on a PC
#         os.environ['MASTER_ADDR'] = master_addr
#         os.environ['MASTER_PORT'] = master_port # A free port on the master node
#         # os.environ['WORLD_SIZE'] = str(world_size) # The total number of GPUs in the distributed job
#         # os.environ['RANK'] = '0' # The unique identifier for this process (0-indexed)
#         # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # "nccl" or "gloo"
#         dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

#     device_id = dist.get_rank()
#     print(f"Device id: {device_id}")
    
# # Define your model architecture here
# class TransformerBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define a linear layer you want to parallelize
#         self.linear = nn.Linear(something_size, something_size)
        
#     def forward(self, x):
#         x = self.linear(x)
#         return x

# def main(rank=None, master_addr=None, master_port=None, world_size=None):
#     prepare_distributed_environment(rank, master_addr, master_port, world_size)
#     # Initialize the model
#     model = TransformerBlock()

#     # Set up your distributed environment and initialize DeviceMesh
#     # This line should be adapted to your specific environment and may require launching a distributed job
#     tp_mesh = init_device_mesh("cuda", (2,1))

#     # Define the parallelization plan. Use RowwiseParallel for the linear layer
#     parallelize_plan = {
#         # "linear": RowwiseParallel(),
#         "linear": ColwiseParallel(),
#     }

#     # Apply row-wise tensor parallelism to the model
#     parallelized_model = parallelize_module(model, tp_mesh, parallelize_plan)#.to(f'cuda:{rank}')

#     # Dummy input for the sake of example
#     # input_tensor = torch.rand(1024, 1024)
#     torch.manual_seed(10*rank)
#     # input_tensor = torch.rand(something_size, int(something_size/2)).cuda()
#     input_tensor = torch.rand(something_size, int(something_size))#.cuda()

#     # Forward pass
#     for _ in range(100):
#         output = parallelized_model(input_tensor)
#     print(output.shape)


# if __name__ == '__main__':
#     torch.manual_seed(1)

#     # world_size = torch.cuda.device_count()  
#     # master_addr = 'localhost'
#     # master_port = '12345'
#     # mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
#     if 1==1:
#         main()
#     else:
#         world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
#         if world_size == 0:
#             print("No CUDA device(s) detected.")
#             exit(0)

#         master_addr = 'localhost'
#         master_port = '12345'  
#         mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
#         # main()
