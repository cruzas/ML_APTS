import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP # pip install fairscale
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net1 = []
        for k in range(4):
            self.net1.append(nn.Linear(10,10))
            self.net1[-1].weight.data = torch.ones_like(self.net1[-1].weight.data)
            self.net1[-1].bias.data = torch.ones_like(self.net1[-1].bias.data)
            self.net1.append(nn.LeakyReLU(negative_slope=0.1))
        self.net1 = nn.Sequential(*self.net1)


    def forward(self, x):
        return self.net1(x)

class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.net1 = []
        for k in range(4):
            self.net1.append(nn.Linear(10, 10))
            self.net1[-1].weight.data = torch.ones_like(self.net1[-1].weight.data)
            self.net1[-1].bias.data = torch.ones_like(self.net1[-1].bias.data)
            self.net1.append(nn.LeakyReLU(negative_slope=0.1))
        self.net1.append(nn.Linear(10, 1))
        self.net1[-1].weight.data = torch.ones_like(self.net1[-1].weight.data)
        self.net1[-1].bias.data = torch.ones_like(self.net1[-1].bias.data)
        self.net1 = nn.Sequential(*self.net1)

    def forward(self, x):
        return self.net1(x)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353'

    # initialize the process group
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    # mode = "ddp"
    mode = "sddp"
    # mode = "fsdp"

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    G = Generator()
    if os.path.exists("state_G.ckpt"):
        dist.barrier()
        print("Loading G weights")
        G.load_state_dict(torch.load("state_G.ckpt"))
    G.to(rank)

    D = Discrim()
    if os.path.exists("state_D.ckpt"):
        dist.barrier()
        D.load_state_dict(torch.load("state_D.ckpt"))
    D.to(rank)


    if mode == "ddp":
        # Optimizer after DDP (follow pytorch tutorial)
        ddp_model_G = DDP(G, device_ids=[rank])
        optimizer_G = OSS(params=G.parameters(), optim=torch.optim.Adam, **{"lr": 1e-4})

        ddp_model_D = DDP(D, device_ids=[rank])
        optimizer_D = OSS(params=D.parameters(), optim=torch.optim.Adam, **{"lr": 1e-4})

    elif mode == "sddp":
        # Optimizer before SDDP
        optimizer_G = OSS(params=G.parameters(), optim=torch.optim.Adam, **{"lr": 1e-4})
        if os.path.exists("optim_G.ckpt"):
            print("loading optimizer")
            dist.barrier()
            cur_state_dict = torch.load("optim_G.ckpt")
            optimizer_G.load_state_dict(cur_state_dict)

        ddp_model_G = ShardedDDP(G, [optimizer_G])
        optimizer_D = OSS(params=D.parameters(), optim=torch.optim.Adam, **{"lr": 1e-4})
        ddp_model_D = ShardedDDP(D, [optimizer_D])

    elif mode == "fsdp":
        # Optimizer after FSDP
        ddp_model_G = FSDP(G)
        optimizer_G = torch.optim.Adam(params=ddp_model_G.parameters(), lr=1e-4)
        if os.path.exists("optim_G.ckpt"):
            print("loading optimizer")
            dist.barrier()
            cur_state_dict = torch.load("optim_G.ckpt")
            optim_shard_dict = ddp_model_G.get_shard_from_optim_state_dict(cur_state_dict)
            optimizer_G.load_state_dict(optim_shard_dict)

        ddp_model_D = FSDP(D)
        optimizer_D = torch.optim.Adam(params=ddp_model_D.parameters(), lr=1e-4)


    max_iter = 1000
    for iter_idx in range(max_iter):
        ddp_model_G.zero_grad(set_to_none=True)
        loss_G = torch.sum(ddp_model_G(torch.ones(20, 10).to(rank)))
        loss_G.backward()
        optimizer_G.step()

        ddp_model_D.zero_grad(set_to_none=True)
        loss_D = torch.sum(ddp_model_D(torch.ones(5, 10).to(rank)))
        loss_D.backward()
        optimizer_D.step()


        if mode == "ddp":
            if rank==0 and iter_idx==max_iter:
                state = ddp_model_G.module.state_dict()
                for k in state:
                    state[k] = state[k].cpu()
                print(state)
        elif mode == "sddp":
            # Call on all ranks
            optimizer_G.consolidate_state_dict(recipient_rank=0)
            if rank==0 and iter_idx==max_iter:
                state = ddp_model_G.module.state_dict()
                optim_state = optimizer_G.state_dict()
                for k in state:
                    state[k] = state[k].cpu()
                torch.save(state, "state_G.ckpt")
                torch.save(optim_state, "optim_G.ckpt")
        elif mode == "fsdp":
            if iter_idx==max_iter:
                # Must call on all devices - otherwise fails
                state = ddp_model_G.state_dict()

                # Must call on all devices - otherwise hangs
                optim_state = ddp_model_G.gather_full_optim_state_dict(optimizer_G)
                if rank == 0:
                    # Save on single device
                    for k in state:
                        state[k] = state[k].cpu()
                    torch.save(state, "state_G.ckpt")
                    torch.save(optim_state, "optim_G.ckpt")

        if iter_idx == max_iter and rank == 0:
            print("Counting unique values, should be size one in each matrix")
            for k in state:
                print(torch.unique(state[k]))
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print(world_size)
    run_demo(demo_basic, world_size)
    
