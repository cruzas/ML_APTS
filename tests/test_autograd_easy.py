import torch
import torch.nn as nn

torch.manual_seed(133464)
torch.set_default_tensor_type (torch.DoubleTensor)

class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.layer0 = nn.Linear(2, 2)
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 2)
        self.layer4 = nn.Linear(2, 2)
        
    def forward(self, x):
        self.out0 = torch.relu(self.layer0(x))
        self.out1 = torch.relu(self.layer1(self.out0))
        self.out2 = torch.relu(self.layer2(self.out1))
        self.out3 = torch.relu(self.layer3(self.out2))
        self.out4 = torch.relu(self.layer4(self.out3)) + self.out1 + self.out2
        return self.out4
    

for _ in range(200):
    # Create the global model, loss function, and input
    global_model = GlobalModel()
    
    # Loss function
    criterion = nn.MSELoss()

    # Sample input and target with requires_grad=False
    x = torch.tensor([[1.0, 2.0]], requires_grad=False)
    y = torch.tensor([[1.1, 2.1]])

    # Forward pass through the global model
    output = global_model(x)
    loss = criterion(output, y)
    
    # Compute gradients using .backward() on the global model
    loss.backward(retain_graph=True)

    # Store gradients from the global model
    global_grads = {name: param.grad.clone() for name, param in global_model.named_parameters()}
    # Zero gradients for NN1 and NN2
    global_model.zero_grad()

    def update_params(model, grads):
        for param, grad in zip(model.parameters(), grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    # MANUAL GRADIENT COMPUTATION
    # Layer 4 gradients
    grad_out4 = torch.autograd.grad(outputs=loss, inputs=output, retain_graph=True)[0]
    layer4_grads = torch.autograd.grad(outputs=loss, inputs=global_model.layer4.parameters(), retain_graph=True)
    update_params(global_model.layer4, layer4_grads)
    # Layer 3 gradients
    grad_out3 = torch.autograd.grad(outputs=global_model.out4, inputs=global_model.out3, grad_outputs=grad_out4, retain_graph=True)[0]
    layer3_grads = torch.autograd.grad(outputs=global_model.out3, inputs=global_model.layer3.parameters(), grad_outputs=grad_out3, retain_graph=True)
    update_params(global_model.layer3, layer3_grads)
    # Layer 2 gradients
    grad_out2 = torch.autograd.grad(outputs=global_model.out3, inputs=global_model.out2, grad_outputs=grad_out3, retain_graph=True)[0]
    layer2_grads = torch.autograd.grad(outputs=global_model.out2, inputs=global_model.layer2.parameters(), grad_outputs=grad_out2 + grad_out4, retain_graph=True)
    update_params(global_model.layer2, layer2_grads)
    # Layer 1 gradients
    grad_out1 = torch.autograd.grad(outputs=global_model.out2, inputs=global_model.out1, grad_outputs=grad_out2 + grad_out4, retain_graph=True)[0]
    layer1_grads = torch.autograd.grad(outputs=global_model.out1, inputs=global_model.layer1.parameters(), grad_outputs=grad_out1 + grad_out4, retain_graph=True)
    update_params(global_model.layer1, layer1_grads)
    
    grad_out0 = torch.autograd.grad(outputs=global_model.out1, inputs=global_model.out0, grad_outputs=grad_out1 + grad_out4, retain_graph=True)[0]
    layer0_grads = torch.autograd.grad(outputs=global_model.out0, inputs=global_model.layer0.parameters(), grad_outputs=grad_out0, retain_graph=True)
    update_params(global_model.layer0, layer0_grads)

    manual_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in global_model.named_parameters()}

    # Compare manually computed gradients with global model gradients
    diff = False
    for name in global_grads.keys():
        difference = (manual_grads[name] - global_grads[name]).abs().max()
        print(f"{name} norm {torch.norm(global_grads[name])} and difference: {difference}")
        if difference > 1e-6:
            diff = True

    if not diff:
        print("The manually computed gradients match the gradients from .backward() for both NN1 and NN2!")
    else:
        print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")
