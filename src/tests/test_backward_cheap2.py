import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import copy

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
    
class SubModel(nn.Sequential):
    def __init__(self, layer):
        if not isinstance(layer, nn.Sequential):
            raise ValueError('layer must be a nn.Sequential')
        for l in layer:
            if isinstance(l, nn.Sequential):
                raise ValueError('layer must not have nested nn.Sequential')
        super(SubModel, self).__init__(layer)
        self.layer = layer
        self.outputs = [0]*len(list(layer))
    
    def forward(self, x):
        for i, sublayer in enumerate(self.layer): # Problem enumerate self just yields length 1
            x = sublayer(x)
            self.outputs[i] = x
        return x
    
    def backward(self, grad_output=None, loss=None):
        if loss is not None:
            print(f"(SUB) Loss: {loss}")
            grad_output = autograd.grad(loss, self.outputs[-1], retain_graph=True)[0]
            print(f"(SUB) Grad output: {grad_output}")
            for param in self[-1].parameters():
                param.grad = autograd.grad(self.outputs[-1], param, grad_outputs=grad_output, retain_graph=True)[0]

        for i in range(len(self.outputs)-2, -1, -1): # I modified this to -1 instead of -2
            grad_output = autograd.grad(self.outputs[i+1], self.outputs[i], grad_outputs=grad_output, retain_graph=True)[0]
            print(f"(SUB) self.outputs[{i+1}]: {self.outputs[i+1]}, self.outputs[{i}]: {self.outputs[i]}, grad_output: {grad_output}")
            for param in self.layer[i].parameters():
                param.grad = autograd.grad(self.outputs[i], param, grad_outputs=grad_output, retain_graph=True)[0] # Allow unused means param might not be used in the computation graph
        return grad_output

class MyModel2(nn.Module):
    def __init__(self, pipe_list):
        super(MyModel2, self).__init__()
        self.num_pipes = len(pipe_list)
        for i, pipe in enumerate(pipe_list):
            setattr(self, f'pipe{i}', SubModel(pipe))
    
    def forward(self, x):
        x = self.pipe0(x)
        for i in range(1, self.num_pipes):
            print(f"(FWD) Going through pipe group {i}")
            x = getattr(self, f'pipe{i}')(x)
            output_needed = getattr(self, f'pipe{i}').outputs[0]
            getattr(self, f'pipe{i-1}').outputs.append(output_needed)
        return x
    
    def backward(self, loss):
        for i in range(self.num_pipes-1, -1, -1):
            print(f"(BWD) Going through pipe group {i}")
            if i == self.num_pipes-1:
                grad_output = getattr(self, f'pipe{i}').backward(loss=loss)
            else:
                grad_output = getattr(self, f'pipe{i}').backward(grad_output=grad_output)

def train():
    # Create model and move it to GPU
    torch.manual_seed(0)
    model = MyModel()
    torch.manual_seed(0)
    # Dummy input and target, move input to GPU
    x = torch.randn(1, 10, requires_grad=True)
    target = torch.randn(1, 6)

    ## Forward pass 0    
    model.zero_grad()
    output = model(x)
    loss_new = torch.nn.functional.mse_loss(output, target)
    loss_new.backward(retain_graph=True)
    grad_new = [param.grad for param in model.parameters()]
    
    ## Forward pass 1 (GLOBAL)
    model.zero_grad()
    torch.manual_seed(0)
    x = torch.randn(1, 10, requires_grad=True)
    output = model(x)
    print(f'Output original: {output}')

    # Compute output gradients (e.g., using a loss function)
    loss = torch.nn.functional.mse_loss(output, target)
    grad_output = autograd.grad(loss, model.x4, retain_graph=True)[0]
    print(f"Loss orig: {loss}")
    print(f"Model.x4: {model.x4}")
    print("Grad output: ", grad_output)
    model.layer4.weight.grad = autograd.grad(model.x4, model.layer4.weight, grad_outputs=grad_output, retain_graph=True)[0]
    # print(torch.norm(model.layer4.weight.grad))
    grad_data3 = autograd.grad(model.x4, model.x3, grad_outputs=grad_output, retain_graph=True)[0]
    print(f"Model.x3: {model.x3}")
    print(f"grad_data3: {grad_data3}")
    model.layer3.weight.grad = autograd.grad(model.x3, model.layer3.weight, grad_outputs=grad_data3, retain_graph=True)[0]
    # print(torch.norm(model.layer3.weight.grad))
    
    grad_data2 = autograd.grad(model.x3, model.x2, grad_outputs=grad_data3, retain_graph=True)[0]
    print(f"Model.x2: {model.x2}")
    print(f"grad_data2: {grad_data2}")
    model.layer2.weight.grad = autograd.grad(model.x2, model.layer2.weight, grad_outputs=grad_data2, retain_graph=True)[0]
    # print(torch.norm(model.layer2.weight.grad))
    
    grad_data1 = autograd.grad(model.x2, model.x1, grad_outputs=grad_data2, retain_graph=True)[0]
    print(f"Model.x1: {model.x1}")
    print(f"grad_data1: {grad_data1}")
    for param in model.layer1.parameters():
        param.grad  = autograd.grad(model.x1, param, grad_outputs=grad_data1, retain_graph=True)[0]
    # model.layer1.weight.grad = autograd.grad(model.x1r, model.layer1.weight, grad_outputs=grad_data1, retain_graph=True)[0]
    
    grad_new2 = [param.grad for param in model.parameters()]
    print(f'This must be close to 0: {torch.norm(torch.cat([g1.flatten()-g2.flatten() for g1,g2 in zip(grad_new,grad_new2)]))}')
    
    model.zero_grad()
   
    seq_NN = [
        nn.Sequential(
            nn.Linear(10, 9, bias=True),
            nn.Sigmoid()
        ),
        nn.Sequential(
            nn.Linear(9, 8, bias=False),
            nn.Linear(8, 7, bias=False)
        ),
        nn.Sequential(
            nn.Linear(7, 6, bias=False)
        )
    ]

    model3 = MyModel2(seq_NN)
    # Synchronize the parameters of model3 with model
    for i, (param, param3) in enumerate(zip(model.parameters(), model3.parameters())):
        param3.data = param.data.clone().detach()
        param3.requires_grad = True

    out3 = model3.forward(x)
    print(f"Output model 3: {out3}, subtraction from original: {torch.norm(out3-output).item()}")
    
    loss = torch.nn.functional.mse_loss(out3, target)
    model3.backward(loss)
    grad_new3 = [param.grad for param in model3.parameters()]
    print(f'This must be close to 0: {torch.norm(torch.cat([g1.flatten()-g3.flatten() for g1,g3 in zip(grad_new,grad_new3)]))}')
        
if __name__ == '__main__':
    train()