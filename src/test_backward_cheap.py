import torch
import torch.nn as nn
import torch.optim as optim
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

def train():
    # Create model and move it to GPU
    model = MyModel().cuda()

    # Dummy input and target, move input to GPU
    x = torch.randn(1, 10, requires_grad=True).cuda()
    target = torch.randn(1, 10).cuda()

    ## Forward pass 1
    output = model(x)

    # Compute output gradients (e.g., using a loss function)
    loss = torch.nn.functional.mse_loss(output, target)
    grad_output = autograd.grad(loss, output, create_graph=True)[0]

    # Compute gradients of the last two layers
    grad_x3 = autograd.grad(output, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    grad_x2 = autograd.grad(model.x3, model.x2, grad_outputs=grad_x3)[0]
    grad_x1 = autograd.grad(model.x2, model.x1, grad_outputs=grad_x2)[0]
    # grad_x0 = autograd.grad(model.x1, x, grad_outputs=grad_x1)[0]    
    # grad_old = torch.cat([param.grad.flatten() for param in model.parameters()])

    ## Forward pass 2    
    model.zero_grad()
    output = model(x)
    loss_new = torch.nn.functional.mse_loss(output, target)
    loss_new.backward()
    grad_new = [param.grad for param in model.parameters()]
    
    ## Compare gradients
    torch.norm(grad_new - grad_old)
    
# Example usage
train()