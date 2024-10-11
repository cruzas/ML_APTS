import torch
import torch.nn as nn

torch.manual_seed(134)
# Define NN1
class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)

    def forward(self, x):
        out1 = torch.relu(self.linear1(x))
        out2 = torch.relu(self.linear2(out1))
        out3 = torch.relu(self.linear3(out2)) + x  # Skip connection
        return out2, out3

# Define NN2
class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, a, b):
        out4 = torch.relu(self.linear1(b))
        out = torch.relu(self.linear2(out4)) + b
        return out
    
# Define NN3
class NN3(nn.Module):
    def __init__(self):
        super(NN3, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, b):
        out4 = torch.relu(self.linear1(b))
        out = torch.relu(self.linear2(out4)) + b
        return out
    
# Define NN4
class NN4(nn.Module):
    def __init__(self):
        super(NN4, self).__init__()
        self.linear1 = nn.Linear(2, 2)

    def forward(self, a, b):
        out = torch.relu(self.linear1(a) + self.linear1(b))
        return out
    
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
    # Zero gradients for all models
    global_model.zero_grad()

    # Manually compute gradients for NN1, NN2, NN3, NN4
    nn1 = global_model.model1
    nn2 = global_model.model2
    nn3 = global_model.model3
    nn4 = global_model.model4

    # Forward pass through NN1, NN2, NN3, NN4
    out_11, out_12 = nn1(x)
    out2 = nn2(out_11, out_12)
    out3 = nn3(out_12)
    final_out = nn4(out2, out3)
    loss = criterion(final_out, y)

    # Manually compute gradients for NN4
    nn4_grads = torch.autograd.grad(loss, nn4.parameters(), retain_graph=True)

    # Compute gradients of loss w.r.t. out2 and out3
    grad_out2, grad_out3 = torch.autograd.grad(loss, [out2, out3], retain_graph=True)

    # Manually compute gradients for NN3
    nn3_grads = torch.autograd.grad(out3, nn3.parameters(), grad_outputs=grad_out3, retain_graph=True)

    # Compute gradients of out_12 w.r.t. out3
    grad_out12_from_out3 = torch.autograd.grad(out3, out_12, grad_outputs=grad_out3, retain_graph=True)[0]

    # Manually compute gradients for NN2
    nn2_grads = torch.autograd.grad(out2, nn2.parameters(), grad_outputs=grad_out2, retain_graph=True)

    # Compute gradients of out_11 and out_12 w.r.t. out2
    grad_out11, grad_out12_from_out2 = torch.autograd.grad(out2, [out_11, out_12], grad_outputs=grad_out2, retain_graph=True)

    # Combine gradients for out_12 from out2 and out3
    grad_out12 = grad_out12_from_out2 + grad_out12_from_out3

    # Manually compute gradients for NN1
    nn1_grads_linear3 = torch.autograd.grad(out_12, nn1.linear3.parameters(), grad_outputs=grad_out12, retain_graph=True)
    nn1_grads_linear2 = torch.autograd.grad(out_11, nn1.linear2.parameters(), grad_outputs=grad_out11, retain_graph=True)
    nn1_grads_linear1 = torch.autograd.grad(out_11, nn1.linear1.parameters(), grad_outputs=grad_out11)

    manual_grads = list(nn1_grads_linear1) + list(nn1_grads_linear2) + list(nn1_grads_linear3) + list(nn2_grads) + list(nn3_grads) + list(nn4_grads)

    # Compare manually computed gradients with global model gradients
    diff = False
    idx = 0
    for name, param in global_model.named_parameters():
        manual_grad = manual_grads[idx]
        difference = (manual_grad - global_grads[name]).abs().max()
        if difference > 1e-6:
            print(f"Gradient from global model for {name}:")
            print(f"Difference: {difference}")
            diff = True
        idx += 1

    if not diff:
        print("The manually computed gradients match the gradients from .backward() for all models!")
    else:
        print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")