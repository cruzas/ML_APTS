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

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    torch.manual_seed(0)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print(f"Memory consumption of some float: {sys.getsizeof(float(1.0))}")
    print(f"Memory consumption of training dataset: {sys.getsizeof(train_dataset)}")
    print(f"Memory consumption of test dataset: {sys.getsizeof(test_dataset)}")

    print(f'Rank {rank} is ready.')
    criterion = torch.nn.CrossEntropyLoss()
    
    NN1 = lambda in_features,out_features: nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 10), nn.ReLU())

    layer_list = [
        (NN1, {'in_features': 784, 'out_features': 256}),
        (NN2, {'in_features': 256, 'out_features': 128}),
        (NN3, {'in_features': 128, 'out_features': 64})
    ]

    num_replicas = 1
    
    # train_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # test_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
     
    x = torch.randn(2, 1, 784, device='cuda:0') # NOTE: The first dimension is the batch size, the second is the channel size, the third is the image size
    torch.manual_seed(0)
    layers = [layer[0](**layer[1]) for layer in layer_list]
    seq_model = nn.Sequential(*layers)

    random_input = torch.randn(10, 1, 784, device='cuda:0')

    torch.manual_seed(0)
    seq_model_2 = Parallel_Sequential_Model(stages=copy.deepcopy(layers), rank_list=[0,1,2]) # NOTE: It is important to do a copy.deepcopy here! Otherwise, the two models will be linked directly.

    if rank == 0:
        seq_optimizer = torch.optim.SGD(seq_model.parameters(), lr=10)
    seq_optimizer_2 = torch.optim.SGD(seq_model_2.parameters(), lr=10)

    sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    for epoch in range(4):
        for i, (x, y) in enumerate(train_loader):
            seq_optimizer_2.zero_grad()
            # out_2 = seq_model_2(x.to('cuda:0'))
            c = closure(x, y, torch.nn.CrossEntropyLoss(), seq_model_2, data_chunks_amount=1, compute_grad=False)
            seq_loss_2 = c()
            print(f'(PARALLEL) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss_2}')
            # seq_loss_2 = criterion(out_2, y.to('cuda:0'))

            # print(f"Epoch {epoch} - Iteration {i} - Out and out2 difference {torch.norm(out.flatten().double()-out_2.flatten().double())} - Loss difference {seq_loss-seq_loss_2}")
            # seq_model_2.backward(seq_loss_2)
            # seq_optimizer_2.step()
            
            if rank == 0:
                # Gather sequential model norm
                seq_optimizer.zero_grad()
                out = seq_model(x)

                seq_loss = criterion(out, y)
                print(f'(SEQ) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss}')
        
    if rank == 0:
        for j in range(len(layer_list)):
            seq_grad = torch.cat([layer.grad.flatten() for layer in seq_model[j].parameters()])
            seq_grad_2 = torch.cat([layer.grad.flatten() for layer in getattr(seq_model_2, f'stage{j}').parameters()])
            print(f'(SEQ) grad norm {j} -> {torch.norm(seq_grad)}')
            print(f'(SEQ2) grad norm {j} -> {torch.norm(seq_grad_2)}')

        dist.barrier()
        print(f'Rank {rank}, epoch {epoch} is done.')

    random_input = torch.randn(10, 1, 784, device='cuda:0')
    if rank == 0:
        seq_output = seq_model(random_input)
        print(f'Rank {rank}, norm {torch.norm(seq_output.flatten().double())}')
    
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

