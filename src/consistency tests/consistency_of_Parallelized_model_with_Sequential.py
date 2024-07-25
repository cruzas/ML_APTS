import os
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add the path to the sys.path
import sys
# Make the following work on Windows and MacOS
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.utility import prepare_distributed_environment, create_dataloaders
from parallel.models import *
from parallel.optimizers import *
from parallel.utils import *
from parallel.dataloaders import ParallelizedDataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from parallel.dataloaders import GeneralizedDistributedSampler, GeneralizedDistributedDataLoader
from data_loaders.OverlappingDistributedSampler import *
import dill

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    torch.manual_seed(0)
        
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    NN1 = lambda in_features,out_features: nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 10), nn.ReLU())

    stage_list = [
        (NN1, {'in_features': 784, 'out_features': 256}), # Stage 1
        (NN2, {'in_features': 256, 'out_features': 128}), # Stage 2
        (NN3, {'in_features': 128, 'out_features': 64})   # Stage 3
    ]

    num_replicas = 1
    device = 'cpu' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # train_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # test_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
     
    torch.manual_seed(155405)
    layers = [layer[0](**layer[1]) for layer in stage_list]
    random_input = torch.randn(10, 1, 784, device=device)
    target = torch.randn(10, 10, device=device)
    learning_rage = 10
    if rank == 0:
        torch.manual_seed(0)
        seq_model = nn.Sequential(*layers).to(device)
        param_norm = lambda seq_model: [torch.norm(torch.cat([p.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        grad_norm = lambda seq_model: [torch.norm(torch.cat([p.grad.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        seq_optimizer = torch.optim.SGD(seq_model.parameters(), lr=learning_rage)

        print(f'START (SEQ) param norm -> {param_norm(seq_model)}')

    dist.barrier()
    
    weight_par_model = Weight_Parallelized_Model(stage_list=stage_list, rank_list=list(range(0,len(stage_list))), sample=random_input, gpu_id=0, device=device) 
    for i, p1 in enumerate(weight_par_model.stage.parameters()):
        p1.data = list(layers[rank].parameters())[i].data.clone().detach().to(device)
        p1.requires_grad = True

    par_optimizer = torch.optim.SGD(weight_par_model.parameters(), lr=learning_rage)
    data_chunks_amount = 10

    sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    for epoch in range(40):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            if rank == 0:
                print('____________ NEW ITERATION ____________')
                # Gather sequential model norm
                seq_optimizer.zero_grad()
                out = seq_model(x)
                seq_loss = criterion(out, y)
                seq_loss.backward()
                seq_optimizer.step()

                print(f'(SEQ) param norm -> {param_norm(seq_model)}, grad norm -> {grad_norm(seq_model)}')

            tic = time.time()
            par_optimizer.zero_grad()
            c = closure(x, y, torch.nn.CrossEntropyLoss(), weight_par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)
            par_loss = c()
            par_optimizer.step()  
            print(f"(ACTUAL PARALLEL) stage {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in weight_par_model.stage.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in weight_par_model.stage.parameters()]))}")

            if rank == 0:
                print(f'Epoch {epoch}, Iteration {i}, SEQ loss: {seq_loss}, PAR loss: {par_loss}, diff: {seq_loss - par_loss}')     


        
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
        world_size = 3
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)

