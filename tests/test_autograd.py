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
        out4 = torch.relu(self.linear1(a)) ** (2 * b)
        out = torch.relu(self.linear2(out4))
        return out

# Define GlobalModel
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.model1 = NN1()
        self.model2 = NN2()

    def forward(self, x):
        a, b = self.model1(x)
        out = self.model2(a, b)
        return out

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
    # Zero gradients for NN1 and NN2
    global_model.model1.zero_grad()
    global_model.model2.zero_grad()

    # Manually compute gradients for NN1 and NN2
    nn1 = global_model.model1
    nn2 = global_model.model2

    # Forward pass through NN1 and NN2
    out1, out2 = nn1(x)
    out = nn2(out1, out2)
    loss_nn2 = criterion(out, y)

    # Compute gradients w.r.t. NN2 parameters
    nn2_grads = torch.autograd.grad(
        outputs=loss_nn2,
        inputs=nn2.parameters(),
        retain_graph=True,
    )

    # Temporarily set requires_grad=False for NN2's parameters to compute gradients w.r.t. NN1 parameters only
    orig_requires_grad = []
    for param in nn2.parameters():
        orig_requires_grad.append(param.requires_grad)
        param.requires_grad = False

    # Compute gradients of loss w.r.t. out1 and out2
    grad_out1,grad_out2 = torch.autograd.grad(
        outputs=loss_nn2,
        inputs=[out1,out2],
        retain_graph=True,
        create_graph=False
    )
    # SAME AS DOING THIS:
    # grad_out1 = torch.autograd.grad(
    #     outputs=loss_nn2,
    #     inputs=out1,
    #     retain_graph=True,
    #     create_graph=False
    # )
    # grad_out2 = torch.autograd.grad(
    #     outputs=loss_nn2,
    #     inputs=out2,
    #     retain_graph=True,
    #     create_graph=False
    # )
    
    # Restore the original requires_grad values for NN2's parameters
    for param, req_grad in zip(nn2.parameters(), orig_requires_grad):
        param.requires_grad = req_grad
        
    nn1_grads = torch.autograd.grad(
        outputs=[out1, out2],
        inputs=nn1.parameters(),
        grad_outputs=[grad_out1, grad_out2],
        retain_graph=True,
    )
    
    
    # Compute gradients w.r.t. NN1 parameters using grad_out1 and grad_out2
    for param in nn1.linear3.parameters():
        param.grad = torch.autograd.grad(
            outputs=out2,
            inputs=param,
            grad_outputs=grad_out2,
            retain_graph=True,
        )[0]
        
    for param in nn1.linear2.parameters():
        param.grad = torch.autograd.grad(
            outputs=out1,
            inputs=param,
            grad_outputs=grad_out1,
            retain_graph=True,
        )[0]
        
    for param in nn1.linear1.parameters():
        param.grad = torch.autograd.grad(
            outputs=out1,
            inputs=param,
            grad_outputs=grad_out1,
            retain_graph=True,
        )[0]
        # param.grad += torch.autograd.grad(
        #     outputs=out2,
        #     inputs=param,
        #     grad_outputs=grad_out2,
        #     retain_graph=True,
        # )[0]
        
    # Backpropagate through NN1 using grad_out1 and grad_out2
    # out1.backward(grad_out1, retain_graph=True, inputs=list(nn1.parameters()))
    # # now backprop only through the first layer of NN1
    # out2.backward(grad_out2, retain_graph=True, inputs=list(nn1.parameters())) 
    nn1_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in nn1.parameters()]
    

    # Compare manually computed gradients with global model gradients
    print("\nComparing gradients for NN2:")
    for (name, param), grad in zip(nn2.named_parameters(), nn2_grads):
        print(f"\nGradient computed manually for NN2 parameter {name}:")
        # print(grad)
        if f"model2.{name}" in global_grads:
            print(f"Gradient from global model for NN2 parameter {name}:")
            # print(global_grads[f'model2.{name}'])
            difference = (grad - global_grads[f'model2.{name}']).abs().max()
            if difference > 1e-6:
                print(f"Gradient computed manually for NN2 parameter {name}:")
                print(grad)
                print(f"Gradient from global model for NN2 parameter {name}:")
                print(global_grads[f'model2.{name}'])
            print(f"Difference: {difference}\n")

    print("\nComparing gradients for NN1:")
    for (name, param), grad in zip(nn1.named_parameters(), nn1_grads):
        if grad is not None:
            # print(f"\nGradient computed manually for NN1 parameter {name}:")
            # print(grad)
            if f"model1.{name}" in global_grads:
                print(f"Gradient from global model for NN1 parameter {name}:")
                # print(global_grads[f'model1.{name}'])
                difference = (grad - global_grads[f'model1.{name}']).abs().max()
                if difference > 1e-6:
                    print(f"Gradient computed manually for NN1 parameter {name}:")
                    print(grad)
                    print(f"Gradient from global model for NN1 parameter {name}:")
                    print(global_grads[f'model1.{name}'])
                print(f"Difference: {difference}\n")

    # Check if gradients match
    matching_nn2 = all(
        torch.allclose(grad, global_grads[f"model2.{name}"], atol=1e-6)
        for (name, _), grad in zip(nn2.named_parameters(), nn2_grads)
    )

    matching_nn1 = all(
        grad is not None and torch.allclose(grad, global_grads[f"model1.{name}"], atol=1e-6)
        for (name, _), grad in zip(nn1.named_parameters(), nn1_grads)
        if grad is not None
    )

    if matching_nn2 and matching_nn1:
        print("The manually computed gradients match the gradients from .backward() for both NN1 and NN2!")
    else:
        print("There is a mismatch between the manually computed gradients and the gradients from .backward()!")
