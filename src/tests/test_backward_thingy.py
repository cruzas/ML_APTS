import torch
import torch.nn as nn
import torch.autograd as autograd

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)
        self.layer3 = nn.Linear(10, 10, bias=False)
        self.layer4 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)
        return self.x4

def print_memory_usage(device_id):
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)    # Convert bytes to GB
    print(f"Device cuda:{device_id} - Allocated Memory: {allocated:.2f} GB, Reserved Memory: {reserved:.2f} GB")

def train_corrected_v2():
    # Create model 
    model = MyModel().to(f'cuda:1')

    # Dummy input and target
    samples = int(1e7)
    x = torch.randn(samples, 10).to(f'cuda:1')
    target = torch.randn(samples, 10).to(f'cuda:1')

    ## 1. Gradient computed as usual
    output_1 = model(x)
    loss_1 = torch.nn.functional.mse_loss(output_1, target)
    print_memory_usage(1)
    loss_1.backward()

    # Gradients of the entire network
    grad_1 = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    print_memory_usage(1)

    # Reset gradients to zero
    model.zero_grad()

    print_memory_usage(1)
    ## 2. Gradient computed sequentially
    output = model(x)
    print_memory_usage(1)
    # Compute output gradients
    loss = torch.nn.functional.mse_loss(output, target)
    grad_output = autograd.grad(loss, model.x4, create_graph=True)[0]

    # Compute gradients for each layer sequentially
    grad_x3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    grad_x2 = autograd.grad(model.x3, model.x2, grad_outputs=grad_x3, retain_graph=True)[0]
    grad_x1 = autograd.grad(model.x2, model.x1, grad_outputs=grad_x2, retain_graph=True)[0]

    # Compute gradients for parameters
    model.layer4.weight.grad = autograd.grad(model.x4, model.layer4.weight, grad_outputs=grad_output)[0]
    model.layer3.weight.grad = autograd.grad(model.x3, model.layer3.weight, grad_outputs=grad_x3)[0]
    model.layer2.weight.grad = autograd.grad(model.x2, model.layer2.weight, grad_outputs=grad_x2)[0]
    model.layer1.weight.grad = autograd.grad(model.x1, model.layer1.weight, grad_outputs=grad_x1)[0]

    # Compare gradients
    grad_2 = [param.grad for param in model.parameters() if param.requires_grad]
    return grad_1, grad_2

# Example usage
grad_1, grad_2 = train_corrected_v2()

# Print norm of entire grad_1
print(torch.norm(torch.cat([g.flatten() for g in grad_1])).item())

# Calculating the difference in norms
norm_diff = [torch.norm(g1 - g2).item() for g1, g2 in zip(grad_1, grad_2)]
print(norm_diff)
