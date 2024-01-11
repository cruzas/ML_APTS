import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.utility import prepare_distributed_environment

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)
        self.layer3 = nn.Linear(10, 10, bias=False)
        self.layer4 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

def train(rank, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    # Create model
    model = MyModel().to(f'cuda:{rank}')

    # Dummy input and target
    x = torch.randn(1, 10).to(f'cuda:{rank}')
    target = torch.randn(1, 10).to(f'cuda:{rank}')
    # Compute loss
    loss = torch.nn.functional.mse_loss(x4, target)

    # Backward pass (compute gradients)
    if rank == 1:
        # Compute gradients for layers 4 and 3
        grad_output = autograd.grad(loss, model.x4, create_graph=True)[0]
        grad_x3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
        model.layer4.weight.grad = autograd.grad(model.x4, model.layer4.weight, grad_outputs=grad_output)[0]
        model.layer3.weight.grad = autograd.grad(model.x3, model.layer3.weight, grad_outputs=grad_x3)[0]

        # Send grad_x3 to rank 0
        dist.send(tensor=grad_x3, dst=0)

    elif rank == 0:
        # Receive grad_x3 from rank 1
        grad_x3 = torch.zeros_like(model.x3)
        dist.recv(tensor=grad_x3, src=1)

        # Compute gradients for layers 2 and 1
        grad_x2 = autograd.grad(model.x3, model.x2, grad_outputs=grad_x3, retain_graph=True)[0]
        grad_x1 = autograd.grad(model.x2, model.x1, grad_outputs=grad_x2, retain_graph=True)[0]
        model.layer2.weight.grad = autograd.grad(model.x2, model.layer2.weight, grad_outputs=grad_x2)[0]
        model.layer1.weight.grad = autograd.grad(model.x1, model.layer1.weight, grad_outputs=grad_x1)[0]

    # Gather gradients on rank 0
    all_gradients = []
    if rank == 1:
        # Send gradients to rank 0
        for param in model.parameters():
            dist.send(tensor=param.grad, dst=0)

    elif rank == 0:
        # Aggregate gradients from rank 1 and rank 0
        for param in model.parameters():
            grad_copy = torch.zeros_like(param.grad)
            all_gradients.append(param.grad)  # Rank 0's own gradients

            # Receive gradients from rank 1
            dist.recv(tensor=grad_copy, src=1)
            all_gradients.append(grad_copy)

    # Only rank 0 computes the full gradient norm
    if rank == 0:
        full_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in all_gradients]))
        print("Full Gradient Norm:", full_grad_norm.item())

world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
if world_size == 0:
    print("No CUDA device(s) detected.")
    exit(0)

master_addr = 'localhost'
master_port = '12345'  
mp.spawn(train, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
