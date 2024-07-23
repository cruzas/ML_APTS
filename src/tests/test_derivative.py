import torch
import torch.nn as nn  # Add this line to import the nn module
import torch.distributed as dist
import torch.multiprocessing as mp
# Add the path to the sys.path
import os
import sys
import diffdist.functional as distops

# Make the following work on Windows and MacOS
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.utility import prepare_distributed_environment, create_dataloaders

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.layer1(x)

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    if dist.get_rank() == 0:
        torch.manual_seed(0)
        print("SEND/RECV TEST\n")
        x = torch.tensor([3.], requires_grad=True)
        net = SimpleModel()
        # Initialize all net weights to 1 and bias to 0
        for p in net.parameters():
            p.data.fill_(3)
        y = net(x)

        print("Before sending y:", y)
        connector = distops.send(y, dst=1)
        # Computation happens in process 1
        buffer = torch.tensor(0.)
        z, _ = distops.recv(buffer, src=1, next_backprop=connector)
        print("After receiving:", z)

        z.backward(retain_graph=True)
        flat_grad = torch.cat([p.grad.flatten() for p in net.parameters()])
        print(f"\n(RANK 0) Gradient: {flat_grad}")

    elif dist.get_rank() == 1:
        buffer = torch.tensor(0., requires_grad=True)
        y, _ = distops.recv(buffer, src=0)

        torch.manual_seed(10)
        net2 = SimpleModel()
        for p in net2.parameters():
            p.data.fill_(2)
        y2 = net2(y.unsqueeze(0))
        
        criterion = nn.MSELoss()
        loss = criterion(y2, torch.tensor([10.]))

        connector = distops.send(loss, dst=0)
        connector.backward(torch.tensor([]), retain_graph=True)
        
        # local gradient
        grad_flatten = torch.cat([p.grad.flatten() for p in net2.parameters()])
        print(f"\n(RANK 1) Gradient: {grad_flatten}")
        
        # HERE WE DO A STEP
        optimizer = torch.optim.SGD(net2.parameters(), lr=0.1)
        optimizer.step()
        print(f"\n(RANK 1) Updated weights: {[p for p in net2.parameters()]}")
        # forward pass and backward
        y2 = net2(y.unsqueeze(0))
        # loss = criterion(y2, torch.tensor([10.]))
        # new_connector = distops.send(loss, dst=1)
        # loss, _ = dist.recv(loss, src=1)
        connector.backward(torch.tensor([]), retain_graph=True)
        print(f"\n(RANK 1) Updated Gradient: {[p.grad for p in net2.parameters()]}")

    if dist.get_rank() == 0:
        # make sequential model and compare gradients
        model = nn.Sequential(
                nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        )
        for i in range(2):
            model[i].weight.data.fill_(i+1)
            model[i].weight.requires_grad = True
            model[i].bias = None
        
        x = torch.tensor([3.], requires_grad=True)
        y = model(x)
        criterion = nn.MSELoss()
        loss = criterion(y, torch.tensor([10.]))
        loss.backward()
        flat_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        print(f"\n(RANK 0) Sequential Model Gradient: {flat_grad}")
        
if __name__ == '__main__':
    torch.manual_seed(1)
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'   
        world_size = 2
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
        

# def main(rank=None, master_addr=None, master_port=None, world_size=None):
#     prepare_distributed_environment(rank, master_addr, master_port, world_size)
#     if dist.get_rank() == 0:
#         print("SEND/RECV TEST\n")
#         x = torch.tensor(3., requires_grad=True)
#         y = 2 * x

#         print("Before sending y:", y)
#         connector = distops.send(y, dst=1)
#         # Computation happens in process 1
#         buffer = torch.tensor(0.)
#         z, _ = distops.recv(buffer, src=1, next_backprop=connector)
#         print("After receiving:", z)

#         k = 4 * z
#         k.backward()
#         print("Gradient with MPI:", x.grad)

#     elif dist.get_rank() == 1:
#         buffer = torch.tensor(0., requires_grad=True)
#         y, _ = distops.recv(buffer, src=0)

#         l = y * 10

#         connector = distops.send(l, dst=0)
#         connector.backward(torch.tensor([]))

#         l = y * 2
        
