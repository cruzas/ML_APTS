import os
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
# TODO: before send we could reduce the weight of tensor by using the half precision / float16, then we can convert it back to float32 after the recv
from data_loaders.OverlappingDistributedSampler import *

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    torch.manual_seed(0)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.flatten(x))
        transforms.Lambda(lambda x: x.view(x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print(f"Memory consumption of some float: {sys.getsizeof(float(1.0))}")
    print(f"Memory consumption of training dataset: {sys.getsizeof(train_dataset)}")
    print(f"Memory consumption of test dataset: {sys.getsizeof(test_dataset)}")

    # training_sampler_parallel = GeneralizedDistributedSampler([0,2], train_dataset, shuffle=True, drop_last=True)
    # test_sampler_parallel = GeneralizedDistributedSampler([0,2], test_dataset, shuffle=False, drop_last=True)

    print(f'Rank {rank} is ready.')
    criterion = torch.nn.CrossEntropyLoss()
    
    NN1 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(64, 10), nn.ReLU())
   
    # if dist.get_world_size() == 3:
    # layer_list = [
    #     (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
    #     (NN2, {'in_features': 256, 'out_features': 128},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,128], dtype=torch.int32))),
    #     (NN3, {'in_features': 128, 'out_features': 64},   (lambda samples: torch.tensor([samples,128], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32)))
    # ]

    layer_list = [
        (NN1, {'in_features': 784, 'out_features': 256}),
        (NN2, {'in_features': 256, 'out_features': 128}),
        (NN3, {'in_features': 128, 'out_features': 64})
    ]

    #     # rank_list = [[0], [1], [2]]
    #     rank_list = [[0, 1], [1, 2]]
    # elif dist.get_world_size() == 2 or dist.get_world_size() == 4:
    # layer_list = [
    #     (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
    #     (NN3, {'in_features': 256, 'out_features': 64},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32))),
    # ]
    # # rank_list = [[0], [1]]
    # layer_list = [
    #     (NN1, {'in_features': 784, 'out_features': 256}), # samples <- is the sample amount of the input tensor
    #     (NN3, {'in_features': 256, 'out_features': 64}),
    # ]

    num_replicas = 2
    
    train_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)

    # WE ARE HERE <------------------
    #  - set the amount of model copies as "amount of subdomains in data TIMES replicas per model" -> apts will take care of synchronizing the models accordingly
     
    # model = Weight_Parallelized_Model(layer_list, rank_list, sample=x)
    x = torch.randn(1, 784)
    # sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    # model = Parallelized_Model(layer_list=layer_list, sample=x, num_replicas=num_replicas)

    # optimizer1 = TR(model, criterion)
    # optimizer2 = torch.optim.Adam(model.subdomain.parameters(), lr=0.0001)
    # optimizer3 = torch.optim.SGD(model.subdomain.parameters(), lr=0.01)
    lr = 0.001                         # torch.optim.SGD TRAdam
    # sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    # optimizer = APTS(model, criterion, subdomain_optimizer=TRAdam, global_optimizer=TR, subdomain_optimizer_defaults={'lr':lr},
    #                     global_optimizer_defaults={'lr':lr, 'max_lr':1.0, 'min_lr':1e-5, 'nu_1':0.25, 'nu_2':0.75}, 
    #                     max_subdomain_iter=3, dogleg=True, lr=lr)
    
    sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    for epoch in range(1000):
        for i, (x, y) in enumerate(train_loader):
            print(f'Rank {rank}, norm x {torch.norm(x.flatten().double())}, norm y {torch.norm(y.flatten().double())}')
            # a = model(x, chunks_amount=1, reset_grad=True, compute_grad=True)
            # print(f'Rank {rank}, epoch {epoch}, iteration {i}, output {a}')
            # loss = optimizer.step(closure(x, y, torch.nn.CrossEntropyLoss(), model, data_chunks_amount=1))
        dist.barrier()
        print(f'Rank {rank}, epoch {epoch} is done.')
        # print(f'Epoch {epoch}, loss {loss}')

        # Compute the test accuracy
        # accuracy = []
        # for j, (x_test, y_test) in enumerate(test_loader): # NOTE: NO MINIBACHES IN THE TEST SET
        #     output_test = model(x_test.to(device), chunks_amount=1, reset_grad = True, compute_grad = False)
        #     if rank == rank_list[-1][0]:
        #         output_test = output_test[0]
        #         # Compute the accuracy NOT THE LOSS
        #         accuracy.append((output_test.argmax(dim=1) == y_test.to(device)).float().mean()) 
        # if rank == rank_list[-1][0]:
        #     print(f'Epoch {epoch}, loss {loss}, test accuracy {sum(accuracy)/len(accuracy)*100}')

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
        world_size = 6
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)

