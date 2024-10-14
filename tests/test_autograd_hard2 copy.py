import torch
import torch.nn as nn

torch.manual_seed(1334564564)
torch.set_default_tensor_type(torch.DoubleTensor)

# Define NN1
class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)

    def forward(self, x):
        out_10 = torch.relu(self.linear1(x))
        self.out_10 = out_10  # Save for later use
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
        out0 = torch.relu(self.linear1(a)) + (2 * b)
        out2 = torch.relu(self.linear2(out0)) + b
        return out2

# Define NN3
class NN3(nn.Module):
    def __init__(self):
        super(NN3, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, a, b):
        out0 = torch.relu(self.linear1(b)) + a
        out3 = torch.relu(self.linear2(out0)) + b
        return out3

# Define NN4
class NN4(nn.Module):
    def __init__(self):
        super(NN4, self).__init__()
        self.linear1 = nn.Linear(2, 2)

    def forward(self, a, b):
        final_out = torch.relu(self.linear1(a) + self.linear1(b))
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
        out3 = self.model3(a, b)
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
    # Zero gradients for all models
    global_model.model1.zero_grad()
    global_model.model2.zero_grad()
    global_model.model3.zero_grad()
    global_model.model4.zero_grad()

    # Manually compute gradients for all models
    nn1 = global_model.model1
    nn2 = global_model.model2
    nn3 = global_model.model3
    nn4 = global_model.model4

    # Forward pass through NN1, NN2, and NN3
    out_11, out_12 = nn1(x)
    out2 = nn2(out_11, out_12)
    out3 = nn3(out_11, out_12)
    final_out = nn4(out2, out3)
    loss = criterion(final_out, y)

    # Compute gradients w.r.t. NN4 parameters
    nn4_grads = torch.autograd.grad(outputs=loss, inputs=nn4.parameters(), retain_graph=True)

    # Assign gradients to NN4 parameters
    for param, grad in zip(nn4.parameters(), nn4_grads):
        param.grad = grad

    # Compute gradients of loss w.r.t. out2 and out3
    grad_out2, grad_out3 = torch.autograd.grad(outputs=loss, inputs=[out2, out3], retain_graph=True)

    # Compute gradients w.r.t. NN3 parameters and its inputs
    nn3_params = list(nn3.parameters())
    nn3_grads_and_inputs = torch.autograd.grad(outputs=out3, inputs=nn3_params + [out_11, out_12], grad_outputs=grad_out3, retain_graph=True)
    nn3_grads = nn3_grads_and_inputs[:len(nn3_params)]
    grad_out3_a, grad_out3_b = nn3_grads_and_inputs[-2], nn3_grads_and_inputs[-1]

    # Assign gradients to NN3 parameters
    for param, grad in zip(nn3_params, nn3_grads):
        param.grad = grad

    # Compute gradients w.r.t. NN2 parameters and its inputs
    nn2_params = list(nn2.parameters())
    nn2_grads_and_inputs = torch.autograd.grad(outputs=out2, inputs=nn2_params + [out_11, out_12], grad_outputs=grad_out2, retain_graph=True)
    nn2_grads = nn2_grads_and_inputs[:len(nn2_params)]
    grad_out2_a, grad_out2_b = nn2_grads_and_inputs[-2], nn2_grads_and_inputs[-1]

    # Assign gradients to NN2 parameters
    for param, grad in zip(nn2_params, nn2_grads):
        param.grad = grad

    # Sum the gradients w.r.t. out_11 and out_12 from both NN2 and NN3
    grad_11 = grad_out2_a + grad_out3_a
    grad_12 = grad_out2_b + grad_out3_b

    # Compute additional gradient contribution from out_12 back to out_11
    grad_out12_wrt_out11 = torch.autograd.grad(outputs=out_12, inputs=out_11, grad_outputs=grad_12, retain_graph=True)[0]

    # Update grad_11 with the contribution from out_12
    grad_11_total = grad_11 + grad_out12_wrt_out11

    # Compute gradients w.r.t. nn1.linear3 parameters
    nn1_linear3_params = list(nn1.linear3.parameters())
    nn1_linear3_grads = torch.autograd.grad(outputs=out_12, inputs=nn1_linear3_params, grad_outputs=grad_12, retain_graph=True)

    # Assign gradients to nn1.linear3 parameters
    for param, grad in zip(nn1_linear3_params, nn1_linear3_grads):
        param.grad = grad

    # Compute gradients w.r.t. nn1.linear2 parameters
    nn1_linear2_params = list(nn1.linear2.parameters())
    nn1_linear2_grads = torch.autograd.grad(outputs=out_11, inputs=nn1_linear2_params, grad_outputs=grad_11_total, retain_graph=True)

    # Assign gradients to nn1.linear2 parameters
    for param, grad in zip(nn1_linear2_params, nn1_linear2_grads):
        param.grad = grad

    # Compute gradients w.r.t. nn1.out_10
    grad_out11_wrt_out10 = torch.autograd.grad(outputs=out_11, inputs=nn1.out_10, grad_outputs=grad_11_total, retain_graph=True)[0]

    # Compute gradients w.r.t. nn1.linear1 parameters
    nn1_linear1_params = list(nn1.linear1.parameters())
    nn1_linear1_grads = torch.autograd.grad(outputs=nn1.out_10, inputs=nn1_linear1_params, grad_outputs=grad_out11_wrt_out10, retain_graph=True)

    # Assign gradients to nn1.linear1 parameters
    for param, grad in zip(nn1_linear1_params, nn1_linear1_grads):
        param.grad = grad

    # Collect all gradients for NN1
    nn1_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in nn1.parameters()]

    # Compare manually computed gradients with global model gradients
    diff = False
    for name2 in ['nn1_grads', 'nn2_grads', 'nn3_grads', 'nn4_grads']:
        nn_name = name2.split('_')[0]
        nr = nn_name[2]
        model = getattr(global_model, f"model{nr}")
        for (name, param) in model.named_parameters():
            manual_grad = param.grad
            global_grad_name = f'model{nr}.{name}'
            if global_grad_name in global_grads:
                auto_grad = global_grads[global_grad_name]
                difference = (manual_grad - auto_grad).abs().max()
                if difference > 1e-6:
                    print(f"Gradient mismatch in {nn_name.upper()} parameter {name}: Difference = {difference}")
                    diff = True

    if not diff:
        print("The manually computed gradients match the gradients from .backward() for all models!")
    else:
        print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")
