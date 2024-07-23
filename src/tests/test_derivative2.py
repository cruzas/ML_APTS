import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn
from torch.autograd import Variable
import torch.multiprocessing as mp

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.fc1(x)

def prepare_distributed_environment(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
    rpc.init_rpc(f"worker{rank}", backend='gloo', rank=rank, world_size=world_size)

def rpc_send(tensor, dst):
    return rpc.rpc_sync(f"worker{dst}", torch.add, args=(tensor, 0))

def rpc_recv(src):
    return rpc.rpc_sync(f"worker{src}", torch.add, args=(torch.tensor([0.]), 0))

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    
    if dist.get_rank() == 0:
        torch.manual_seed(0)
        print("SEND/RECV TEST\n")
        x = torch.tensor([3.], requires_grad=True)
        net = SimpleModel()
        for p in net.parameters():
            p.data.fill_(1)
        y = net(x)

        print("Before sending y:", y)
        rpc_send(y, dst=1)
        buffer = rpc_recv(src=1)
        print("After receiving:", buffer)

        buffer.backward(retain_graph=True)
        flat_grad = torch.cat([p.grad.flatten() for p in net.parameters()])
        print(f"\n(RANK 0) Gradient: {flat_grad}")

    elif dist.get_rank() == 1:
        buffer = rpc_recv(src=0)
        buffer.requires_grad = True

        torch.manual_seed(10)
        net2 = SimpleModel()
        for p in net2.parameters():
            p.data.fill_(2)
        y2 = net2(buffer.unsqueeze(0))
        
        criterion = nn.MSELoss()
        loss = criterion(y2, torch.tensor([10.]))
        rpc_send(loss, dst=0)
        
        loss.backward(retain_graph=True)
        grad_flatten = torch.cat([p.grad.flatten() for p in net2.parameters()])
        print(f"\n(RANK 1) Gradient: {grad_flatten}")

        optimizer = torch.optim.SGD(net2.parameters(), lr=0.1)
        optimizer.step()
        print(f"\n(RANK 1) Updated weights: {[p for p in net2.parameters()]}")

        # y2 = net2(buffer.unsqueeze(0))
        # loss = criterion(y2, torch.tensor([10.]))
        # rpc_send(loss, dst=0)

    if dist.get_rank() == 0:
        model = nn.Sequential(
            nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        )
        for i in range(2):
            model[i].weight.data.fill_(i + 1)
            model[i].weight.requires_grad = True
            model[i].bias = None
        
        x = torch.tensor([3.], requires_grad=True)
        y = model(x)
        criterion = nn.MSELoss()
        loss = criterion(y, torch.tensor([10.]))
        loss.backward()
        flat_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        print(f"\n(RANK 0) Sequential Model Gradient: {flat_grad}")

    rpc.shutdown()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size == 0:
        print("No CUDA device(s) detected.")
        exit(0)

    master_addr = 'localhost'
    master_port = '12345'   
    world_size = 2
    mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)