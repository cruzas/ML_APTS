import torch
import torch.nn as nn

torch.manual_seed(133464)
torch.set_default_tensor_type(torch.DoubleTensor)

class SD0Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        ctx.save_for_backward(x)
        out0 = torch.relu(module.layer0(x))
        out1 = torch.relu(module.layer1(out0))
        # Simulate sending over network by detaching
        return out0.detach(), out1.detach()

    @staticmethod
    def backward(ctx, grad_out0, grad_out1):
        x, = ctx.saved_tensors
        module = ctx.module

        # Recompute forward pass
        x.requires_grad_(True)
        out0 = torch.relu(module.layer0(x))
        out1 = torch.relu(module.layer1(out0))

        # Compute gradients
        with torch.enable_grad():
            grads = torch.autograd.grad(
                outputs=[out0, out1],
                inputs=[x] + list(module.parameters()),
                grad_outputs=[grad_out0, grad_out1],
                retain_graph=True
            )

        # Extract gradients
        grad_x = grads[0]
        param_grads = grads[1:]

        # Set gradients for module parameters
        for param, grad in zip(module.parameters(), param_grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad.clone()

        return grad_x, None

class SD0(nn.Module):
    def __init__(self):
        super(SD0, self).__init__()
        self.layer0 = nn.Linear(2, 2)
        self.layer1 = nn.Linear(2, 2)
            
    def forward(self, x):
        out0, out1 = SD0Function.apply(x, self)
        return out0, out1

class SD1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out0, out1, module):
        ctx.module = module
        ctx.save_for_backward(out0, out1)
        out2 = torch.relu(module.layer2(out1))
        out3 = torch.relu(module.layer3(out2)) + out0
        out4 = torch.relu(module.layer4(out3)) + out1 + out2
        # Simulate sending over network by detaching
        return out4.detach(), out3.detach()

    @staticmethod
    def backward(ctx, grad_out4, grad_out3):
        out0, out1 = ctx.saved_tensors
        module = ctx.module

        # Recompute forward pass
        out0.requires_grad_(True)
        out1.requires_grad_(True)
        out2 = torch.relu(module.layer2(out1))
        out3 = torch.relu(module.layer3(out2)) + out0
        out4 = torch.relu(module.layer4(out3)) + out1 + out2

        # Compute gradients
        with torch.enable_grad():
            grads = torch.autograd.grad(
                outputs=[out4, out3],
                inputs=[out0, out1] + list(module.parameters()),
                grad_outputs=[grad_out4, grad_out3],
                retain_graph=True
            )

        # Extract gradients
        grad_out0 = grads[0]
        grad_out1 = grads[1]
        param_grads = grads[2:]

        # Set gradients for module parameters
        for param, grad in zip(module.parameters(), param_grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad.clone()

        return grad_out0, grad_out1, None

class SD1(nn.Module):
    def __init__(self):
        super(SD1, self).__init__()
        self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 2)
        self.layer4 = nn.Linear(2, 2)
            
    def forward(self, out0, out1):
        out4, out3 = SD1Function.apply(out0, out1, self)
        return out4, out3

class SD2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out4, out3, module):
        ctx.module = module
        ctx.save_for_backward(out4, out3)
        out5 = torch.relu(module.layer5(out4)) + out3
        return out5  # Final output, no need to detach

    @staticmethod
    def backward(ctx, grad_out5):
        out4, out3 = ctx.saved_tensors
        module = ctx.module

        # Recompute forward pass
        out4.requires_grad_(True)
        out3.requires_grad_(True)
        out5 = torch.relu(module.layer5(out4)) + out3

        # Compute gradients
        with torch.enable_grad():
            grads = torch.autograd.grad(
                outputs=out5,
                inputs=[out4, out3] + list(module.parameters()),
                grad_outputs=grad_out5,
                retain_graph=True
            )

        # Extract gradients
        grad_out4 = grads[0]
        grad_out3 = grads[1]
        param_grads = grads[2:]

        # Set gradients for module parameters
        for param, grad in zip(module.parameters(), param_grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad.clone()

        return grad_out4, grad_out3, None

class SD2(nn.Module):
    def __init__(self):
        super(SD2, self).__init__()
        self.layer5 = nn.Linear(2, 2)
            
    def forward(self, out4, out3):
        out5 = SD2Function.apply(out4, out3, self)
        return out5

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
        out0 = torch.relu(self.layer0(x))
        out1 = torch.relu(self.layer1(out0))
        out2 = torch.relu(self.layer2(out1))
        out3 = torch.relu(self.layer3(out2)) + out0
        out4 = torch.relu(self.layer4(out3)) + out1 + out2
        out5 = torch.relu(self.layer5(out4)) + out3
        return out5

class GlobalModelTruncated(nn.Module):
    def __init__(self):
        super(GlobalModelTruncated, self).__init__()
        self.sd0 = SD0()
        self.sd1 = SD1()
        self.sd2 = SD2()
            
    def forward(self, x):
        out0, out1 = self.sd0(x)
        out4, out3 = self.sd1(out0, out1)
        out5 = self.sd2(out4, out3)
        return out5

# Function to remap the state dict from GlobalModel to GlobalModelTruncated
def remap_state_dict(global_model_state_dict):
    remapped_state_dict = {}
    for key, value in global_model_state_dict.items():
        if 'layer0' in key or 'layer1' in key:
            remapped_key = 'sd0.' + key
        elif 'layer2' in key or 'layer3' in key or 'layer4' in key:
            remapped_key = 'sd1.' + key
        elif 'layer5' in key:
            remapped_key = 'sd2.' + key
        else:
            remapped_key = key
        remapped_state_dict[remapped_key] = value
    return remapped_state_dict

# Create the global model, loss function, and input
gb = GlobalModel()
gb_trunc = GlobalModelTruncated()
gb_trunc.load_state_dict(remap_state_dict(gb.state_dict()))

# Loss function
criterion = nn.MSELoss()

# Sample input and target with requires_grad=False
x = torch.tensor([[1.0, 2.0]], requires_grad=False)
y = torch.tensor([[1.1, 2.1]])

# Forward pass through the global model
output = gb(x)
loss = criterion(output, y)
loss.backward()
global_grads = {name: param.grad.clone() for name, param in gb.named_parameters()}

# Zero gradients for gb_trunc
gb_trunc.zero_grad()
final_out = gb_trunc(x)
loss_trunc = criterion(final_out, y)
loss_trunc.backward()
manual_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in gb_trunc.named_parameters()}

# Compare manually computed gradients with global model gradients
diff = False
for name in global_grads.keys():
    # Map the parameter name in gb to the corresponding name in gb_trunc
    if 'layer0' in name or 'layer1' in name:
        trunc_name = 'sd0.' + name
    elif 'layer2' in name or 'layer3' in name or 'layer4' in name:
        trunc_name = 'sd1.' + name
    elif 'layer5' in name:
        trunc_name = 'sd2.' + name
    else:
        trunc_name = name
    difference = (manual_grads[trunc_name] - global_grads[name]).abs().max()
    print(f"{name} norm {torch.norm(global_grads[name])} and difference: {difference}")
    if difference > 1e-6:
        diff = True

if not diff:
    print("The manually computed gradients match the gradients from .backward() for both models!")
else:
    print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")
