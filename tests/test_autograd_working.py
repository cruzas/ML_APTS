import torch
import torch.nn as nn

torch.manual_seed(1334564564)
torch.set_default_tensor_type (torch.DoubleTensor)

# Define NN1
class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)

    def forward(self, x):
        out_10 = torch.relu(self.linear1(x))
        self.out_10 = out_10
        out_11 = torch.relu(self.linear2(out_10))
        out_12 = torch.relu(self.linear3(out_11)) + x  # Skip connection
        return out_11, out_12

# Define NN2
class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, a, b):
        out0 = torch.relu(self.linear1(a)) + (123 * b)
        out2 = torch.relu(self.linear2(out0)) + b
        return out2
    
# Define NN3
class NN3(nn.Module):
    def __init__(self):
        super(NN3, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, b):
        out0 = torch.relu(self.linear1(b))
        out3 = torch.relu(self.linear2(out0)) + b
        return out3
    
# Define NN4
class NN4(nn.Module):
    def __init__(self):
        super(NN4, self).__init__()
        self.linear1 = nn.Linear(2, 2)

    def forward(self, a, b):
        final_out = torch.relu(self.linear1(a)+self.linear1(b))
        return final_out
    
# Define GlobalModel
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.model1 = NN1()
        self.model2 = NN2()
        self.model3 = NN3()
        self.model4 = NN4()

    def forward(self, x):
        a, b = self.model1(x)
        out = self.model2(a, b)
        out3 = self.model3(b)
        final_out = self.model4(out, out3)
        return final_out

for _ in range(200):
    # Create the global model, loss function, and input
    global_model = GlobalModel()
    # Set network parameters to random positive values
    for param in global_model.parameters():
        param.data = torch.randn_like(param.data)
        param.data[param.data < 0] = -param.data[param.data < 0]

    criterion = nn.MSELoss()

    # Sample input and target with requires_grad=False
    x = torch.tensor([[1.0, 2.0]], requires_grad=False)
    y = torch.tensor([[1.1, 2.1]])

    # Forward pass through the global model
    output = global_model(x)
    loss = criterion(output, y)

    # Compute gradients using .backward() on the global model
    loss.backward()

    # Store gradients from the global model
    global_grads = {name: param.grad.clone() for name, param in global_model.named_parameters()}
    # Zero gradients for NN1 and NN2
    global_model.model1.zero_grad()
    global_model.model2.zero_grad()
    global_model.model3.zero_grad()
    global_model.model4.zero_grad()

    # Manually compute gradients for NN1 and NN2
    nn1 = global_model.model1
    nn2 = global_model.model2
    nn3 = global_model.model3
    nn4 = global_model.model4

    # Forward pass through NN1 and NN2
    out_11, out_12 = nn1(x)
    out2 = nn2(out_11, out_12)
    out3 = nn3(out_12)
    final_out = nn4(out2, out3)
    loss = criterion(final_out, y)

    # OUTPUT GRADIENTS FROM NN4 
    # Compute gradients w.r.t. NN4 parameters
    nn4_grads = torch.autograd.grad(outputs=loss, inputs=nn4.parameters(), retain_graph=True)
    
    # Compute gradients of loss w.r.t. out2 and out3
    grad_out2, grad_out3 = torch.autograd.grad(outputs=loss, inputs=[out2, out3], retain_graph=True, create_graph=False)
    
    # Compute gradients w.r.t. NN3 parameters
    nn3_grads = torch.autograd.grad(outputs=out3, inputs=nn3.parameters(), grad_outputs=grad_out3, retain_graph=True)
    
    # Compute gradients w.r.t. NN2 parameters
    nn2_grads = torch.autograd.grad(outputs=out2, inputs=nn2.parameters(), grad_outputs=grad_out2, retain_graph=True)

    # FROM HERE THINGS GO WRONG
    # OUTPUT GRADIENTS FROM NN3 - Compute gradients of out3 w.r.t out_12
    grad_12 = torch.autograd.grad(outputs=out3, inputs=out_12, grad_outputs=grad_out3, retain_graph=True, create_graph=False)[0]
    
    # OUTPUT GRADIENTS FROM NN2 - Compute gradients of out2 w.r.t out_11
    grad_11 = torch.autograd.grad(outputs=out2, inputs=out_11, grad_outputs=grad_out2, retain_graph=True, create_graph=False)[0]
    grad_120 = torch.autograd.grad(outputs=out2, inputs=out_12, grad_outputs=grad_out2, retain_graph=True, create_graph=False)[0]

    # Compute gradients w.r.t. NN1 parameters using grad_out1 and grad_out3
    for param in nn1.linear3.parameters():
        param.grad = torch.autograd.grad(
            outputs=out_12,
            inputs=param,
            grad_outputs=grad_12+grad_120,
            retain_graph=True,
        )[0]
        
    
    for param in nn1.linear2.parameters():
        param.grad = torch.autograd.grad(
            outputs=out_11,
            inputs=param,
            grad_outputs=grad_11,
            retain_graph=True,
        )[0]
        param.grad += torch.autograd.grad(
            outputs=out_12,
            inputs=param,
            grad_outputs=grad_12,
            retain_graph=True,
        )[0]
        
    for param in nn1.linear1.parameters():
        param.grad = torch.autograd.grad(
            outputs=out_11,
            inputs=param,
            grad_outputs=grad_11,
            retain_graph=True,
        )[0]
        param.grad += torch.autograd.grad(
            outputs=out_12,
            inputs=param,
            grad_outputs=grad_12,
            retain_graph=True,
        )[0]
        
    nn1_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in nn1.parameters()]

    # Compare manually computed gradients with global model gradients
    diff = False
    for name2 in ['nn1_grads', 'nn2_grads', 'nn3_grads', 'nn4_grads']:
        nn_name = name2.split('_')[0]
        nr = nn_name[2]
        for (name, param), grad in eval(f"zip({nn_name}.named_parameters(), {name2})"):
            difference = (grad - global_grads[f'model{nr}.{name}']).abs().max()
            if f"model{nr}.{name}" in global_grads and difference > 1e-6:
                print(f"Gradient from global model for {nn_name.upper()} parameter {name}:")
                print(f"Difference: {difference}")
                diff = True


    if not diff:
        print("The manually computed gradients match the gradients from .backward() for both NN1 and NN2!")
    else:
        print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")
