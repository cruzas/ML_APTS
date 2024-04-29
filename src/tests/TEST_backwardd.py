import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import copy

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(5, 4, bias=True)
        self.layer2 = nn.Linear(4, 3, bias=True)
        self.layer3 = nn.Linear(3, 2, bias=True)
        self.layer4 = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        self.f1 = torch.relu(self.layer1(x))
        self.f2 = torch.relu(self.layer2(self.f1))
        self.f3 = torch.relu(self.layer3(self.f2))
        self.f4 = torch.relu(self.layer4(self.f3))
        return self.f4

class SubModel_working(nn.Module):
    def __init__(self, model, grad_loss_f4):
        super(SubModel_working, self).__init__()
        self.model = model
        self.layer3 = copy.deepcopy(model.layer3)
        self.layer4 = copy.deepcopy(model.layer4)   

        self.f2 = model.f2.detach()
        self.f4 = model.f4
        self.grad_loss_f4 = grad_loss_f4

    def forward(self):
        # times random number between 0 and 10000000
        self.f3 = torch.sigmoid(self.layer3(self.f2))
        self.f4 = torch.sigmoid(self.layer4(self.f3))
        return self.f3

    def backward(self):
        grad_loss_f3 = autograd.grad(self.f4, self.f3, grad_outputs=self.grad_loss_f4, retain_graph=True)[0]
        self.layer3.weight.grad = autograd.grad(self.f3, self.layer3.weight, grad_outputs=grad_loss_f3, retain_graph=True)[0]

class SubModel(nn.Module):
    def __init__(self, model, grad_loss_f3):
        super(SubModel, self).__init__()
        self.model = model
        self.layer3 = copy.deepcopy(model.layer3)
        self.layer4 = copy.deepcopy(model.layer4)   

        self.f2 = model.f2.detach()
        self.f4 = model.f4
        self.grad_loss_f3 = grad_loss_f3

    def forward(self):
        # times random number between 0 and 10000000
        self.f3 = torch.sigmoid(self.layer3(self.f2))
        return self.f3

    def backward(self):
        self.layer3.weight.grad = autograd.grad(self.f3, self.layer3.weight, grad_outputs=self.grad_loss_f3, retain_graph=True)[0]
        
def train_1():
    torch.random.manual_seed(42)
    # Set default precision to 64 bits
    torch.set_default_dtype(torch.float64)

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
    true_grad = model.layer3.weight.grad
    # print layer 3 weight grad
    g4 = autograd.grad(loss, model.f4, create_graph=True)[0]
    grad_loss_f3 = autograd.grad(loss, model.f3, retain_graph=True)[0]
    submodel = SubModel(model, grad_loss_f3)
    old_f3 = submodel.forward()
    submodel.backward()
    old_grad = submodel.layer3.weight.grad
    print(f"torch.norm(old_grad) = {torch.norm(old_grad)}")
    
    with torch.no_grad():
        s=submodel.layer3.weight.grad
        submodel.layer3.weight.add_(s)
    
    new_f3 = submodel.forward() 
    submodel.backward()
    new_grad = submodel.layer3.weight.grad
    print(f"torch.norm(new_grad) = {torch.norm(new_grad)}")

    # Difference between true_grad and old_grad
    print(f" {torch.norm(true_grad - old_grad,'fro')} - this is the difference between true_grad and old_grad")
    # Print difference between old_f3 and new_f3
    print(torch.norm(new_f3 - old_f3))
    print(torch.norm(new_grad - old_grad,'fro'))
    
    with torch.no_grad():
        model.layer3.weight.add_(s)
    output = model.forward(x)
    loss = nn.MSELoss()(output, target)
    loss.backward(retain_graph=True)
    true_grad = model.layer3.weight.grad
    # print difference between true_grad and old_grad
    print(f"Difference between the exact and approx gradients: {torch.norm(true_grad - new_grad,'fro')}")
    print(true_grad)
    print(new_grad)
    
    
def train_2():
    torch.random.manual_seed(42)
    # Set default precision to 64 bits
    torch.set_default_dtype(torch.float64)

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
    true_grad = model.layer3.weight.grad
    # print layer 3 weight grad
    g4 = autograd.grad(loss, model.f4, create_graph=True)[0]
    grad_loss_f4 = autograd.grad(loss, model.f4, retain_graph=True)[0]
    submodel = SubModel_working(model, grad_loss_f4)
    old_f3 = submodel.forward()
    submodel.backward()
    old_grad = submodel.layer3.weight.grad
    print(f"torch.norm(old_grad) = {torch.norm(old_grad)}")
    
    with torch.no_grad():
        submodel.layer3.weight.add_(submodel.layer3.weight.grad)
    
    new_f3 = submodel.forward() 
    submodel.backward()
    new_grad = submodel.layer3.weight.grad
    print(f"torch.norm(new_grad) = {torch.norm(new_grad)}")

    # Difference between true_grad and old_grad
    print(f" {torch.norm(true_grad - old_grad,'fro')} - this is the difference between true_grad and old_grad")
    # Print difference between old_f3 and new_f3
    print(torch.norm(new_f3 - old_f3))
    print(torch.norm(new_grad - old_grad,'fro'))

# Example usage
train_1()

# Example usage
# train_2()