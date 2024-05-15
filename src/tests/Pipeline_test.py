import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# add the path to the sys.path
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

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # here we print the memory consumption of train_dataset and test_dataset
    print(f"Memory consumption of some float: {sys.getsizeof(float(1.0))}")
    print(f"Memory consumption of train_dataset: {sys.getsizeof(train_dataset)}")

    rank_list = [[0], [1]]
    train_loader = DataLoader(train_dataset, batch_size=6000, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=6000, shuffle=True, num_workers=2, pin_memory=True)
    # train_loader = ParallelizedDataLoader(dataset=train_dataset, device_list=[0,1], rank_list=rank_list, batch_size=30000, shuffle=True, num_workers=2, pin_memory=True)
    # test_loader = ParallelizedDataLoader(dataset=test_dataset, device_list=[0,1], rank_list=rank_list, batch_size=10000, shuffle=False)

    # if rank == 0:
    #     for i, (x, y) in enumerate(train_loader):
    #         print(f"Sample {i}: {x}, {y}")

    print(f'Rank {rank} is ready.')
    criterion = torch.nn.CrossEntropyLoss()
    
    NN1 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(64, 10), nn.ReLU())
   
    if dist.get_world_size() == 3:
        layer_list = [
            (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
            (NN2, {'in_features': 256, 'out_features': 128},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,128], dtype=torch.int32))),
            (NN3, {'in_features': 128, 'out_features': 64},   (lambda samples: torch.tensor([samples,128], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32)))
        ]
        # rank_list = [[0,1], [2,3], [4,5,6]]
        rank_list = [[0], [1], [2]]
        # rank_list = [[0], [1], [2]]
    elif dist.get_world_size() == 2 or dist.get_world_size() == 4:
        layer_list = [
            (NN1, {'in_features': 784, 'out_features': 256}, (lambda samples: torch.tensor([samples,784], dtype=torch.int32), lambda samples: torch.tensor([samples,256], dtype=torch.int32))), # samples <- is the sample amount of the input tensor
            (NN3, {'in_features': 256, 'out_features': 64},  (lambda samples: torch.tensor([samples,256], dtype=torch.int32), lambda samples: torch.tensor([samples,10], dtype=torch.int32))),
        ]
        rank_list = [[0], [1]]
        # layer_list = [
        #     (NN2, {'in_features': 100, 'out_features': 100},  (lambda samples: torch.tensor([samples,200], dtype=torch.int32), lambda samples: torch.tensor([samples,50 ], dtype=torch.int32))),
        # ]
        # # rank_list = [[0,1]]
        # rank_list = [[0],[1]]

    list_of_all_ranks = [r for rank in rank_list for r in rank]
    if rank in list_of_all_ranks:
        group = dist.new_group(ranks=list_of_all_ranks)
        torch.manual_seed(3456)
        model = Weight_Parallelized_Model(layer_list, rank_list)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # optimizer1 = TR(model, criterion)
        # optimizer2 = torch.optim.Adam(model.subdomain.parameters(), lr=0.0001)
        optimizer = APTS(model, criterion, torch.optim.SGD, {}, TR, {'lr':0.01, 'max_lr':1.0, 'min_lr':1e-3, 'nu_1':0.25, 'nu_2':0.75})
        # optimizer = torch.optim.SGD(model.subdomain.parameters(), lr=0.01)

        # Data loading TODO: load data only on master master rank
        train_loader, test_loader = create_dataloaders(
            dataset="MNIST",
            data_dir=os.path.abspath("./data"),
            mb_size=60000,
            overlap_ratio=0,
            parameter_decomposition=True,
            device="cuda:0"
        )
        device = decide_gpu_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0)

        for epoch in range(1000):
            for i, (x, y) in enumerate(train_loader):
                # Vectorize x 
                x = x.view(x.size(0), -1)
                # One optimizer step
                optimizer.step(closure(x, y, torch.nn.CrossEntropyLoss(), model))
                
                # Train a subdomain model
                # if rank in rank_list[0]:
                # for asd in range(2):
                #     model.subdomain.zero_grad()
                #     out = model.subdomain.forward()
                #     # loss = criterion(out, y.to(device))
                #     # print(f'Epoch {epoch} subdomain loss {loss}')
                #     model.subdomain.backward()
                #     optimizer2.step()

            # Compute the test accuracy
            for j, (x_test, y_test) in enumerate(test_loader): # NOTE: NO MINIBACHES IN THE TEST SET
                x_test = x_test.view(x_test.size(0), -1)
                output_test = model(x_test.to(device), chunks_amount=1, reset_grad = True, compute_grad = False)
                if rank == rank_list[-1][0]:
                    # Compute the accuracy NOT THE LOSS
                    accuracy = (output_test.argmax(dim=1) == y_test.to(device)).float().mean()
                    print(f'Epoch {epoch} test accuracy {accuracy*100}')

                # dist.barrier(group=group)

        # torch.manual_seed(0)
        # x=torch.randn(10, 100)
        # output = model(x.to('cuda'), chunks_amount=1, reset_grad = True, compute_grad = True)
        # if rank == rank_list[-1][0]:
        #     torch.manual_seed(0)
        #     target = torch.randint(0, 50, (10,50), dtype=torch.float32).to(f'cuda:{rank}' if dist.get_backend() == 'gloo' else f'cuda:{model.gpu_id}')
        #     loss = criterion(output, target)
        # else:
        #     loss = None 

        # model.backward(loss)
        # print('Hurra!')
        # dist.barrier(group=group)
        
        g1 = Weight_Parallelized_Tensor(rank_list, model)
        g2 = Weight_Parallelized_Tensor(rank_list, model)
        g3 = g1 @ g2
        print(g3)

        f = model.subdomain.forward()
        print(f)
        
        model.subdomain.backward()

        # Training loop

        # sequential model check
        model_seq = nn.Sequential(*[layer[0](**layer[1]) for layer in layer_list])
        # update Model params with the one of the models on RANK 0
        for i,param in enumerate(model_seq.parameters()):
            if rank == 0 and i < 4:
                param.data=list(model.parameters())[i].to("cuda")
            elif rank == 1 and i >= 4 and i<8:
                dist.send(tensor=list(model.parameters())[i-4].detach().to('cuda'), dst=0)
            elif rank == 2 and i >= 8:
                dist.send(tensor=list(model.parameters())[i-8].detach().to('cuda'), dst=0)

            if rank == 0 and i >= 4 and i<8:
                temp = torch.empty_like(param).to('cuda')
                dist.recv(tensor=temp, src=1)
                param.data = temp
            if rank == 0 and i >= 8:
                temp = torch.empty_like(param).to('cuda')
                dist.recv(tensor=temp, src=2)
                param.data = temp
        if rank == 0:
            model_seq = model_seq.to('cuda')
            output = model_seq(x.to('cuda'))
            torch.manual_seed(0)
            target = torch.randint(0, 50, (10,50), dtype=torch.float32).to(f'cuda:{rank}' if dist.get_backend() == 'gloo' else f'cuda:{model.gpu_id}')
            loss = criterion(output, target)
            print(f"Loss sequential model: {loss}")
        if rank == dist.get_world_size() - 1:
            print(f"Loss parallel model: {loss}")
                
        # check gradients
        if rank == 0:
            loss.backward()
            print(f"Derivative of sequential model:\n{[param.grad for param in model_seq.parameters()]}")

        print(f"Derivative of PARALLEL model rank {rank}:\n{[param.grad for param in model.parameters()]}")


if __name__ == '__main__':
    torch.manual_seed(1)

    # world_size = torch.cuda.device_count()  
    # master_addr = 'localhost'
    # master_port = '12345'
    # mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)
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

