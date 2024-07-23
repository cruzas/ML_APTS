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
    
    NN1 = lambda in_features,out_features: nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(64, 10), nn.ReLU())
    # NN1 = lambda in_features,out_features: nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(784, 256))
    # NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(256,128), nn.ReLU())
    # NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(128,10), nn.ReLU())
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

    # Make a sequential model with this layer list
    
    
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

    num_replicas = 1
    
    # train_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # test_loader = GeneralizedDistributedDataLoader(layer_list=layer_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)

    # WE ARE HERE <------------------
    #  - set the amount of model copies as "amount of subdomains in data TIMES replicas per model" -> apts will take care of synchronizing the models accordingly
     
    # model = Weight_Parallelized_Model(layer_list, rank_list, sample=x)
    x = torch.randn(2, 1, 784, device='cuda:0') # NOTE: The first dimension is the batch size, the second is the channel size, the third is the image size
    # x = torch.randn(1, 784) # NOTE: The first dimension is the batch size, the second is the channel size, the third is the image size
    # sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    model = Parallelized_Model(layer_list=layer_list, sample=x, num_replicas=num_replicas, criterion=criterion, approximated_gradient=True)

    # share parameters from model with seq_model
    if rank == 0:
        torch.manual_seed(0)
        layers = [layer[0](**layer[1]) for layer in layer_list]
        seq_model = nn.Sequential(*layers)
        # receive the parameters from the other ranks
        for l_idx in [1,2]:
            for seq_layer in seq_model[l_idx].parameters():
                tensor = torch.zeros_like(seq_layer.data, device='cpu')
                # print(f'Rank {rank}, receiving from rank {l_idx} with shape {tensor.shape}')
                dist.recv(tensor=tensor, src=l_idx)
                # print(f"Rank {rank} received tensor from rank {l_idx}")
                seq_layer.data = tensor.to('cuda:0')
        
        for i, (layer,seq_layer) in enumerate(zip(model.parameters(), seq_model[0].parameters())):
            seq_layer.data = layer.data
    else:
        for i, layer in enumerate(model.parameters()):
            # print(f'Rank {rank}, sending layer {i} with shape {layer.shape}')
            dist.send(tensor=layer.data.cpu(), dst=0)
            
    # Check if the parameters are the same through norm
    # for r in [0,1,2]:
    #     if rank == r:
    #         print(f'Rank {rank}, difference in norm: {torch.norm(torch.cat([layer.flatten() for layer in model.parameters()]))-torch.norm(torch.cat([layer.flatten() for layer in seq_model[r].parameters()]))}')
    # check if the parameters are the same through the output
    random_input = torch.randn(10, 1, 784, device='cuda:0')
    output = model(random_input, chunks_amount=1, reset_grad=True, compute_grad=True)
    
    if rank == 0:
        torch.manual_seed(0)
        seq_model_2 = Parallel_Sequential_Model(copy.deepcopy(layers)).to('cuda:0') # NOTE: It is important to do a copy.deepcopy here! Otherwise, the two models will be linked directly.
        # synchronize the parameters
        # for l_idx in [0,1,2]:
        #     for seq_layer, seq_layer_2 in zip(seq_model[l_idx].parameters(), seq_model_2.layers[l_idx].parameters()):
        #         seq_layer_2.data = seq_layer.data

        # seq_output = seq_model(random_input)
        # seq_output_2 = seq_model_2(random_input)
        # print(f'Rank {rank}, output norm {torch.norm(seq_output.flatten().double())} - output norm 2 {torch.norm(seq_output_2.flatten().double())}')

    if rank == 2:
        print(f'Rank {rank}, parallel output norm {torch.norm(output[0].flatten().double())}')
    
    # optimizer1 = TR(model, criterion)
    # optimizer2 = torch.optim.Adam(model.subdomain.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.subdomain.parameters(), lr=1)
    if rank == 0:
        seq_optimizer = torch.optim.SGD(seq_model.parameters(), lr=10)
        seq_optimizer_2 = torch.optim.SGD(seq_model_2.parameters(), lr=10)
                        # torch.optim.SGD TRAdam
    # sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    # optimizer = APTS(model, criterion, subdomain_optimizer=TRAdam, global_optimizer=TR, subdomain_optimizer_defaults={'lr':lr},
    #                     global_optimizer_defaults={'lr':lr, 'max_lr':1.0, 'min_lr':1e-5, 'nu_1':0.25, 'nu_2':0.75}, 
    #                     max_subdomain_iter=3, dogleg=True, lr=lr)
    
    sync_operation(filename=get_filename(__file__), line_number=get_linenumber())
    for epoch in range(4):
        for i, (x, y) in enumerate(train_loader):
            # print(f'Rank {rank}, norm x {torch.norm(x.flatten().double())}, norm y {torch.norm(y.flatten().double())}')
            # a = model(x, chunks_amount=1, reset_grad=True, compute_grad=True)
            # print(f'Rank {rank}, epoch {epoch}, iteration {i}, output {a}')
            # c = closure(x, y, torch.nn.CrossEntropyLoss(), model, data_chunks_amount=1)
            # l = c()
            # print(l)
            if rank == 0:
                # Gather sequential model norm
                seq_optimizer.zero_grad()
                out = seq_model(x.to('cuda:0'))

                seq_loss = criterion(out, y.to('cuda:0'))
                seq_loss.backward()
                seq_optimizer.step()

                # 2
                seq_optimizer_2.zero_grad()
                out_2 = seq_model_2(x.to('cuda:0'))
                seq_loss_2 = criterion(out_2, y.to('cuda:0'))

                print(f"Epoch {epoch} - Iteration {i} - Out and out2 difference {torch.norm(out.flatten().double()-out_2.flatten().double())} - Loss difference {seq_loss-seq_loss_2}")
                seq_model_2.backward(seq_loss_2)
                seq_optimizer_2.step()
    if rank == 0:
        for j in range(len(layer_list)):
            seq_grad = torch.cat([layer.grad.flatten() for layer in seq_model[j].parameters()])
            seq_grad_2 = torch.cat([layer.grad.flatten() for layer in getattr(seq_model_2, f'stage{j}').parameters()])
            print(f'(SEQ) grad norm {j} -> {torch.norm(seq_grad)}')
            print(f'(SEQ2) grad norm {j} -> {torch.norm(seq_grad_2)}')

            # loss = optimizer.step(closure(x, y, torch.nn.CrossEntropyLoss(), model, data_chunks_amount=1))

            # print(f'(PAR) grad norm {rank} -> {model.model.subdomain_grad_norm()}')
            # print(f"Parallel grad norm {model.model.grad_norm()}")

            # if rank == 0:
            #     print(f'Rank {rank}, epoch {epoch}, iteration {i}, loss {loss}, seq_loss {seq_loss}')

            # print(loss)
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
    random_input = torch.randn(10, 1, 784, device='cuda:0')
    output = model(random_input, chunks_amount=1, reset_grad=True, compute_grad=True)
    if rank == 0:
        seq_output = seq_model(random_input)
        print(f'Rank {rank}, norm {torch.norm(seq_output.flatten().double())}')

    if rank == 2:
        print(f'Rank {rank}, parallel norm {torch.norm(output[0].flatten().double())}')
        
    print('asd')
    
    
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

