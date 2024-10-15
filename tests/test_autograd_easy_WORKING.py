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
        self.out3 = torch.relu(self.layer3(self.out2)) + self.out0/2 + 1
        self.out4 = torch.relu(self.layer4(self.out3)) + torch.sigmoid(self.out1 + self.out2)
        self.out5 = torch.relu(self.layer5(self.out4)) + 2*self.out3
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
        
    def forward(self, x, chunk_id):
        if chunk_id == 0:
            self.sd0_out0 = []
            self.sd0_out1 = []
            self.sd1_out0 = []
            self.sd1_out1 = []
            self.sd1_out3 = []
            self.sd1_out4 = []
            self.sd2_out3 = []
            self.sd2_out4 = []
            self.sd2_out5 = []
        # SD0
        self.sd0_out0.append(torch.relu(self.layer0(x)))
        self.sd0_out1.append(torch.relu(self.layer1(self.sd0_out0[-1])))
        # SD1
        self.sd1_out0.append(self.sd0_out0[-1].detach().clone().requires_grad_(True))
        self.sd1_out1.append(self.sd0_out1[-1].detach().clone().requires_grad_(True))
        sd1_out2 = torch.relu(self.layer2(self.sd1_out1[-1]))
        self.sd1_out3.append(torch.relu(self.layer3(sd1_out2)) + self.sd1_out0[-1]/2 + 1)
        self.sd1_out4.append(torch.relu(self.layer4(self.sd1_out3[-1])) + torch.sigmoid(self.sd1_out1[-1] + sd1_out2))
        # SD2
        self.sd2_out4.append(self.sd1_out4[-1].detach().clone().requires_grad_(True))
        self.sd2_out3.append(self.sd1_out3[-1].detach().clone().requires_grad_(True))
        self.sd2_out5.append(torch.relu(self.layer5(self.sd2_out4[-1])) + 2*self.sd2_out3[-1])
        return self.sd2_out5[-1]

for _ in range(1):
    # Create the global model, loss function, and input
    gb = GlobalModel()
    gb_trunc = GlobalModelTruncated()
    gb_trunc.load_state_dict(gb.state_dict())
    
    # Loss function
    criterion = nn.MSELoss()

    # Sample input and target with requires_grad=False
    X = torch.randn(10, 2, requires_grad=False)
    Y = torch.randn(10, 2)
    chunk_amount = 5

    output = gb(X)
    loss = criterion(output, Y)
    loss.backward(retain_graph=True)
    global_grads = {name: param.grad.clone() for name, param in gb.named_parameters()}
    
    loss_trunc = []
    for chunk_id, (x, y) in enumerate(zip(X.chunk(chunk_amount), Y.chunk(chunk_amount))):
        # Zero gradients for gb_trunc
        gb_trunc.zero_grad()
        final_out = gb_trunc(x, chunk_id)
        loss_trunc.append(criterion(final_out, y))

    grad_loss_sd2_out4 = [0]*chunk_amount
    grad_loss_sd2_out3 = [0]*chunk_amount
    grad_sd1_out43_sd1_out0 = [0]*chunk_amount
    grad_sd1_out43_sd1_out1 = [0]*chunk_amount
    
    for chunk_id in range(chunk_amount-1, -1, -1):
        # PATTERN:
        # 1. Compute the backward of every output in the subdomain
        # 2. Compute the autograd.grad of every output in the subdomain with respect to each input in a for-loop
        # SD2
        loss_trunc[chunk_id].backward(retain_graph=True)
        grad_loss_sd2_out4[chunk_id] = torch.autograd.grad(outputs=loss_trunc[chunk_id], inputs=gb_trunc.sd2_out4[chunk_id], retain_graph=True)[0]
        grad_loss_sd2_out3[chunk_id] = torch.autograd.grad(outputs=loss_trunc[chunk_id], inputs=gb_trunc.sd2_out3[chunk_id], retain_graph=True)[0]

        # SD1
        gb_trunc.sd1_out4[chunk_id].backward(grad_loss_sd2_out4[chunk_id], retain_graph=True)
        gb_trunc.sd1_out3[chunk_id].backward(grad_loss_sd2_out3[chunk_id], retain_graph=True)
        grad_sd1_out43_sd1_out0[chunk_id] = torch.autograd.grad(outputs=[gb_trunc.sd1_out4[chunk_id], gb_trunc.sd1_out3[chunk_id]], inputs=gb_trunc.sd1_out0[chunk_id], grad_outputs=[grad_loss_sd2_out4[chunk_id], grad_loss_sd2_out3[chunk_id]], retain_graph=True)
        grad_sd1_out43_sd1_out1[chunk_id] = torch.autograd.grad(outputs=[gb_trunc.sd1_out4[chunk_id], gb_trunc.sd1_out3[chunk_id]], inputs=gb_trunc.sd1_out1[chunk_id], grad_outputs=[grad_loss_sd2_out4[chunk_id], grad_loss_sd2_out3[chunk_id]], retain_graph=True)
        
        # SD0
        gb_trunc.sd0_out1[chunk_id].backward(grad_sd1_out43_sd1_out1[chunk_id], retain_graph=True)
        gb_trunc.sd0_out0[chunk_id].backward(grad_sd1_out43_sd1_out0[chunk_id], retain_graph=True)
                                
    manual_grads = {name: param.grad.clone()/chunk_amount if param.grad is not None else torch.zeros_like(param) for name, param in gb_trunc.named_parameters()}

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
