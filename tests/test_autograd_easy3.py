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
        self.layer5 = nn.Linear(2, 2)
        
    def forward(self, x):
        self.out0 = torch.relu(self.layer0(x))
        self.out1 = torch.relu(self.layer1(self.out0))
        self.out2 = torch.relu(self.layer2(self.out1))
        self.out3 = torch.relu(self.layer3(self.out2)) + self.out0
        self.out4 = torch.relu(self.layer4(self.out3)) + self.out1 + self.out2
        self.out5 = torch.relu(self.layer5(self.out4)) + self.out3
        return self.out5
    
class GlobalModelTruncated(nn.Module):
    def __init__(self):
        super(GlobalModelTruncated, self).__init__()
        self.layer0 = nn.Linear(2, 2)
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 2)
        self.layer4 = nn.Linear(2, 2)
        self.layer5 = nn.Linear(2, 2)
        
    def forward(self, x):
        # SD0
        self.sd0_out0 = torch.relu(self.layer0(x))
        self.sd0_out1 = torch.relu(self.layer1(self.sd0_out0))
        # SD1
        self.sd1_out0 = self.sd0_out0.detach().clone().requires_grad_(True)
        self.sd1_out1 = self.sd0_out1.detach().clone().requires_grad_(True)
        self.sd1_out2 = torch.relu(self.layer2(self.sd1_out1))
        self.sd1_out3 = torch.relu(self.layer3(self.sd1_out2)) + self.sd1_out0
        self.sd1_out4 = torch.relu(self.layer4(self.sd1_out3)) + self.sd1_out1 + self.sd1_out2
        # SD2
        self.sd2_out4 = self.sd1_out4.detach().clone().requires_grad_(True)
        self.sd2_out3 = self.sd1_out3.detach().clone().requires_grad_(True)
        self.sd2_out5 = torch.relu(self.layer5(self.sd2_out4)) + self.sd2_out3
        return self.sd2_out5

for _ in range(200):
    # Create the global model, loss function, and input
    gb = GlobalModel()
    gb_trunc = GlobalModelTruncated()
    gb_trunc.load_state_dict(gb.state_dict())
    
    # Loss function
    criterion = nn.MSELoss()

    # Sample input and target with requires_grad=False
    x = torch.tensor([[1.0, 2.0]], requires_grad=False)
    y = torch.tensor([[1.1, 2.1]])

    # Forward pass through the global model
    output = gb(x)
    loss = criterion(output, y)
    loss.backward(retain_graph=True)
    global_grads = {name: param.grad.clone() for name, param in gb.named_parameters()}
    
    # Zero gradients for gb_trunc
    gb_trunc.zero_grad()
    final_out = gb_trunc(x)
    loss_trunc = criterion(final_out, y)
    def update_params(model, grads):
        for param, grad in zip(model.parameters(), grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    # MANUAL GRADIENT COMPUTATION
    # SD2
    grad_loss_sd2_out4 = torch.autograd.grad(outputs=loss_trunc, inputs=gb_trunc.sd2_out4, retain_graph=True)[0]
    grad_loss_sd2_out3 = torch.autograd.grad(outputs=loss_trunc, inputs=gb_trunc.sd2_out3, retain_graph=True)[0]
    grad_loss_sd2_out5 = torch.autograd.grad(outputs=loss_trunc, inputs=gb_trunc.sd2_out5, retain_graph=True)[0]
    layer5_grads = torch.autograd.grad(outputs=gb_trunc.sd2_out5, inputs=gb_trunc.layer5.parameters(), grad_outputs=grad_loss_sd2_out5, retain_graph=True)
    update_params(gb_trunc.layer5, layer5_grads)

    # SD1
    layer4_grads = torch.autograd.grad(outputs=gb_trunc.sd1_out4, inputs=gb_trunc.layer4.parameters(), grad_outputs=grad_loss_sd2_out4, retain_graph=True)
    update_params(gb_trunc.layer4, layer4_grads)
    
    grad_data_through_layer4_sd1_out0 = torch.autograd.grad(outputs=gb_trunc.sd1_out4, inputs=gb_trunc.sd1_out0, grad_outputs=grad_loss_sd2_out4, retain_graph=True)[0]
    grad_data_through_layer4_sd1_out1 = torch.autograd.grad(outputs=gb_trunc.sd1_out4, inputs=gb_trunc.sd1_out1, grad_outputs=grad_loss_sd2_out4, retain_graph=True)[0]
    
    layer3_grads = torch.autograd.grad(outputs=gb_trunc.sd1_out3, inputs=gb_trunc.layer3.parameters(), grad_outputs=grad_loss_sd2_out3 + grad_data_through_layer4_sd1_out0, retain_graph=True)
    update_params(gb_trunc.layer3, layer3_grads)
    
    grad_data_through_layer3_sd1_out0 = torch.autograd.grad(outputs=gb_trunc.sd1_out3, inputs=gb_trunc.sd1_out0, grad_outputs=grad_loss_sd2_out3, retain_graph=True)[0]
    grad_data_through_layer3_sd1_out1 = torch.autograd.grad(outputs=gb_trunc.sd1_out3, inputs=gb_trunc.sd1_out1, grad_outputs=grad_loss_sd2_out3, retain_graph=True)[0]
    
    layer2_grads = torch.autograd.grad(outputs=gb_trunc.sd1_out2, inputs=gb_trunc.layer2.parameters(), grad_outputs=grad_data_through_layer3_sd1_out1, retain_graph=True)
    update_params(gb_trunc.layer2, layer2_grads)

    

    
    manual_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in gb_trunc.named_parameters()}

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
