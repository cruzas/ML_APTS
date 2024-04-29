import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import copy

# model = nn.Sequential(
#             torch.nn.Linear(10, 10),
#             torch.nn.ReLU(),
#             torch.nn.Linear(10, 5)
#         )

# model = fairscale.nn.Pipe(model, balance=[2, 1])


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.layer1 = nn.Linear(10, 10, bias=False)
#         self.layer2 = nn.Linear(10, 10, bias=False)
#         self.layer3 = nn.Linear(10, 10, bias=False)
#         self.layer4 = nn.Linear(10, 10, bias=False)

#     def forward(self, x):
#         self.x1 = self.layer1(x)
#         self.x2 = self.layer2(self.x1)
#         self.x3 = self.layer3(self.x2)
#         self.x4 = self.layer4(self.x3)
#         return self.x4


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 9, bias=True)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(9, 8, bias=False)
        self.layer3 = nn.Linear(8, 7, bias=False)
        self.layer4 = nn.Linear(7, 6, bias=False)

    def forward(self, x):
        self.x1 = self.layer1(x)
        self.x1r = self.activation(self.x1)
        self.x2 = self.layer2(self.x1r)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)
        return self.x4

class SubModel(nn.Module):
    def __init__(self, model):
        super(SubModel, self).__init__()
        self.model = model
        self.layer3 = model.layer3
        self.x4

    def forward(self, x2, x4):
        self.x3 = self.layer3(x2)
        

        return x4
    
    
def train():
    # Create model and move it to GPU
    torch.manual_seed(0)
    model = MyModel().cuda()
    torch.manual_seed(0)
    # Dummy input and target, move input to GPU
    x = torch.randn(1, 10, requires_grad=True).cuda()
    target = torch.randn(1, 6).cuda()

    ## Forward pass 0    
    model.zero_grad()
    output = model(x)
    loss_new = torch.nn.functional.mse_loss(output, target)
    loss_new.backward(retain_graph=True)
    grad_new = [param.grad for param in model.parameters()]
    
    ## Forward pass 1 (GLOBAL)
    model.zero_grad()
    torch.manual_seed(0)
    x = torch.randn(1, 10, requires_grad=True).cuda()
    output = model(x)

    # Compute output gradients (e.g., using a loss function)
    loss = torch.nn.functional.mse_loss(output, target)
    grad_output = autograd.grad(loss, model.x4, create_graph=True)[0]
    model.layer4.weight.grad = autograd.grad(model.x4, model.layer4.weight, grad_outputs=grad_output, retain_graph=True)[0]
    
    grad_data3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    model.layer3.weight.grad = autograd.grad(model.x3, model.layer3.weight, grad_outputs=grad_data3, retain_graph=True)[0]
    
    grad_data2 = autograd.grad(model.x3, model.x2, grad_outputs=grad_data3, retain_graph=True)[0]
    model.layer2.weight.grad = autograd.grad(model.x2, model.layer2.weight, grad_outputs=grad_data2, retain_graph=True)[0]
    
    grad_data1 = autograd.grad(model.x2, model.x1r, grad_outputs=grad_data2, retain_graph=True)[0]
    for param in model.layer1.parameters():
        param.grad  = autograd.grad(model.x1r, param, grad_outputs=grad_data1, retain_graph=True)[0]
    # model.layer1.weight.grad = autograd.grad(model.x1r, model.layer1.weight, grad_outputs=grad_data1, retain_graph=True)[0]
    

    grad_new2 = [param.grad for param in model.parameters()]
    print(torch.norm(torch.cat([g1.flatten()-g2.flatten() for g1,g2 in zip(grad_new,grad_new2)])))
    

    model.zero_grad()
    output_2 = model.x2
    output_3 = model.layer3(output_2)
    
    ## Forward pass 2 (LOCAL) without "output = model(x)"
    model.zero_grad()
    # grad_output = autograd.grad(loss, model.x4, create_graph=True)[0]
    # grad_data3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    # grad_x3_tilde = autograd.grad(model.x3, model.layer3.weight, grad_outputs=grad_data3, retain_graph=False)[0]
    grad_x4_x3 = autograd.grad(grad_output, model.x3, grad_outputs=grad_output, retain_graph=False)[0]
    grad_x3_tilde = torch.mm(torch.transpose(model.x2, 0, 1), grad_x4_x3)
    print(torch.norm(grad_x3_tilde - grad_x3))
    
    # do the step
    model.layer3.weight.add_(-grad_x3_tilde)
    # compute the new gradients
    
    
    print(torch.norm(grad_x3_tilde - grad_x3))
    # modify the layers
    # model.layer1.weight.data.add_(-grad_new2[0])
    # model.layer2.weight.data.add_(-grad_new2[1])
    # model.layer3.weight.data.add_(-grad_new2[2])
    # model.layer4.weight.data.add_(-grad_new2[3])
    
    # gradient of layers without using a global pass
    grad_x3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    
    

    
    
    
    ## Compare gradients
    torch.norm(grad_new - grad_old)
    
# Example usage
train()