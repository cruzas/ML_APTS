import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import fairscale, copy

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(5, 4, bias=False)
        self.layer2 = nn.Linear(4, 3, bias=False)
        self.layer3 = nn.Linear(3, 2, bias=False)
        self.layer4 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        self.f1 = self.layer1(x)
        self.f2 = self.layer2(self.f1)
        self.f3 = self.layer3(self.f2)
        self.f4 = self.layer4(self.f3)
        return self.f4

class SubModel(nn.Module):
    def __init__(self, model, grad_loss_f3):
        super(SubModel, self).__init__()
        self.model = model
        self.layer3 = copy.deepcopy(model.layer3)

        self.f2 = model.f2.detach()
        self.f4 = model.f4
        self.grad_loss_f3 = grad_loss_f3

    def forward(self):
        self.f3 = self.layer3(self.f2)
        return self.f3

    def backward(self):
        self.layer3.weight.grad = autograd.grad(self.f3, self.layer3.weight, grad_outputs=self.grad_loss_f3, retain_graph=True)[0]
    
def train():
    # Create model and move it to GPU
    model = MyModel().cuda()

    # Dummy input and target, move input to GPU
    x = torch.randn(1, 5, requires_grad=True).cuda()
    target = torch.randn(1, 1).cuda()

    ## Forward pass 0    
    model.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward(retain_graph=True)
    # print layer 3 weight grad
    print(model.layer3.weight.grad)
    g4 = autograd.grad(loss, model.f4, create_graph=True)[0]
    grad_loss_f3 = autograd.grad(loss, model.f3, retain_graph=True)[0]
    submodel = SubModel(model, grad_loss_f3)
    submodel.forward()
    submodel.backward()
    print(model.layer3.weight.grad)
    
# Example usage
train()