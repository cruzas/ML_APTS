import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def train(rank, world_size):
    # Initialize distributed environment
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model
    model = MyModel()

    # Dummy input and output
    x = torch.randn(1, 10, requires_grad=True)
    target = torch.randn(1, 10)

    # Forward pass
    output = model(x)

    if rank == 1:
        # Compute gradients w.r.t. input of the last two layers
        grad_output = torch.ones_like(output)
        partial_derivatives = autograd.grad(outputs=output, inputs=x, grad_outputs=grad_output, only_inputs=True)[0]
        # Send partial derivatives to Rank 0
        dist.send(tensor=partial_derivatives, dst=0)

    if rank == 0:
        # Receive partial derivatives on Rank 0
        partial_derivatives = torch.zeros_like(x)
        dist.recv(tensor=partial_derivatives, src=1)
        # Compute gradients for the first two layers
        output.backward(partial_derivatives)

        # Update weights using optimizer (only for the first two layers)
        optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
        optimizer.step()

# Example usage
train(rank=0, world_size=2)
train(rank=1, world_size=2)
