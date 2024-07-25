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
    extra_layers = [
        (lambda in_features,out_features: nn.Linear(in_features,out_features), {'in_features': 256, 'out_features': 128}), # First layer of the second stage
        (lambda in_features,out_features: nn.Linear(in_features,out_features), {'in_features': 128, 'out_features': 64}),  # First layer of the third stage
    ]

    num_replicas = 1
    device = 'cpu' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # train_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # test_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
     
    torch.manual_seed(150105405)
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

        # Add 10 to every seq_model parameter
        # with torch.no_grad():
        #     for p in seq_model.parameters():
        #         p.data += 10

        seq_model_3 = Sequential_Model(copy.deepcopy(layers))

        print(f'START (SEQ) stages param norm [{torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe0.parameters()]))}, {torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe1.parameters()]))}, {torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe2.parameters()]))}]')

        # seq_model_3(random_input) # Initalization of shapes
        seq_optimizer3 = torch.optim.SGD(seq_model_3.parameters(), lr=learning_rage)
        # out3 = seq_model_3(random_input)
        # seq_loss_3 = nn.MSELoss()(out3.to('cuda:0'), target.to('cuda:0'))
        # print("START SEQUENTIAL BACKWARD")
        # seq_model_3.backward(seq_loss_3)
        # print("END SEQUENTIAL BACKWARD")
    dist.barrier()
    
    weight_par_model = Weight_Parallelized_Model(stage_list=stage_list, extra_layers=extra_layers, rank_list=list(range(0,len(stage_list))), sample=random_input, approximated_gradient=False, gpu_id=0) 
    weight_par_model_approx = Weight_Parallelized_Model(stage_list=stage_list, extra_layers=extra_layers, rank_list=list(range(0,len(stage_list))), sample=random_input, approximated_gradient=True, gpu_id=0) 

    # # Synchronize the models
    with torch.no_grad():
        for i,(p1,p2) in enumerate(zip(weight_par_model.parameters(), weight_par_model_approx.parameters())):
            p2.data = list(layers[rank].parameters())[i].data.clone().detach().to(device)
            p1.data = list(layers[rank].parameters())[i].data.clone().detach().to(device)
            p1.requires_grad = True
            p2.requires_grad = True
    par=1
    if par==1:
        # torch.manual_seed(0)
        # print(f"Rank {rank} START PARALLEL BACKWARD")
        seq_model_2 = Parallel_Sequential_Model(stages=copy.deepcopy(layers), rank_list=[0,1,2], sample=random_input) # NOTE: It is important to do a copy.deepcopy here! Otherwise, the two models will be linked directly.
        
        # print(f"Rank {rank} END PARALLEL BACKWARD")
        seq_optimizer_2 = torch.optim.SGD(seq_model_2.parameters(), lr=learning_rage)
        print(f'START (PARALLEL SEQ) Stage {rank} param norm {torch.norm(torch.cat([p.flatten() for p in seq_model_2.stage.parameters()]))}')
        
        # synchronize weights with the weight_par_model
        with torch.no_grad():
            for p1,p2 in zip(weight_par_model.parameters(), seq_model_2.stage.parameters()):
                p2.data = p1.data.clone().detach()
                p2.requires_grad = True

    par_optimizer = torch.optim.SGD(weight_par_model.parameters(), lr=learning_rage)
    par_optimizer_approx = torch.optim.SGD(weight_par_model_approx.parameters(), lr=learning_rage)
    data_chunks_amount = 1

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

                seq_grad = torch.cat([layer.grad.flatten() for layer in seq_model.parameters()])
                print(f'(SEQ) param norm -> {param_norm(seq_model)}, grad norm -> {grad_norm(seq_model)}')

                seq_optimizer3.zero_grad()
                out3 = seq_model_3(x)
                seq_loss_3 = criterion(out3.to(device), y.to(device))
                seq_model_3.backward(seq_loss_3)
                print(f'(SEQ 1 RANK) stages param norm [{torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe0.parameters()]))}, {torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe1.parameters()]))}, {torch.norm(torch.cat([p.flatten() for p in seq_model_3.pipe2.parameters()]))}]')
                print(f'(SEQ 1 RANK) stages grad norm [{torch.norm(torch.cat([p.grad.flatten() for p in seq_model_3.pipe0.parameters()]))}, {torch.norm(torch.cat([p.grad.flatten() for p in seq_model_3.pipe1.parameters()]))}, {torch.norm(torch.cat([p.grad.flatten() for p in seq_model_3.pipe2.parameters()]))}]')

                seq_optimizer3.step()
                # print(f'(SEQ) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss} - loss3 {seq_loss_3} - difference {torch.norm(seq_loss-seq_loss_3)/torch.norm(seq_loss)*100}%')

            if par==1:
                seq_optimizer_2.zero_grad()
                c = closure(x, y, torch.nn.CrossEntropyLoss(), seq_model_2, data_chunks_amount=data_chunks_amount, compute_grad=True)
                seq_loss_2 = c()
                grad2 = seq_model_2.grad_norm()
                weights_norm = torch.norm(torch.cat([p.flatten() for p in seq_model_2.parameters()]))
                seq_optimizer_2.step()
            
                print(f"(PARALLEL SEQ) Stage {dist.get_rank()} weights norm {weights_norm.item()}")
                # if rank == 0:
                #     print(f'(PARALLEL - SEQ) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss} - loss2 {seq_loss_2} - difference {torch.norm(seq_loss-seq_loss_2)/torch.norm(seq_loss)*100}%')
                
                tic = time.time()
                # for _ in range(10):
                par_optimizer.zero_grad()
                c = closure(x, y, torch.nn.CrossEntropyLoss(), weight_par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)
                par_loss = c()
                par_optimizer.step()
                # if rank == 0:
                #     print(f'(ACTUAL PARALLEL) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss} - loss2 {par_loss} - difference {torch.norm(seq_loss-par_loss)/torch.norm(seq_loss)*100}% - time {time.time()-tic}')
                    
                tic = time.time()
                # for _ in range(10):
                par_optimizer_approx.zero_grad()
                c = closure(x, y, torch.nn.CrossEntropyLoss(), weight_par_model_approx, data_chunks_amount=data_chunks_amount, compute_grad=True)
                par_loss_approx = c()
                par_optimizer_approx.step()
                
                print(f"(ACTUAL PARALLEL) stage {rank} grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in weight_par_model.stage.parameters()]))}")
                dist.barrier()
                print(f"(APPROX PARALLEL) stage {rank} grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in weight_par_model_approx.stage.parameters()]))}")

                # print(f"(PARALLELS DIFF GRAD NORM) Rank {rank}, epoch {epoch}, iteration {i}, act grad norm {act_parallel_grad_norm} - approx grad norm {approx_parallel_grad_norm} - difference {abs(act_parallel_grad_norm-approx_parallel_grad_norm)/abs(act_parallel_grad_norm)*100}%")

                # if rank == 0:
                #     print(f'(APPROX PARALLEL) Rank {rank}, epoch {epoch}, iteration {i}, loss {seq_loss} - loss2 {par_loss_approx} - difference {torch.norm(seq_loss-par_loss_approx)/torch.norm(seq_loss)*100} - time {time.time()-tic}%')
                #     print(f'(PARALLELS DIFF) Rank {rank}, epoch {epoch}, iteration {i}, par loss {par_loss} - par loss approx {par_loss_approx} - difference {abs(par_loss-par_loss_approx)/abs(par_loss)*100} - time {time.time()-tic}%')
                
    if rank == 0:
        for j in range(len(stage_list)):
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
   
   
   


def main2(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        raise ValueError("World size must be at least 2 for this example")

    if rank == 0:
        # Define a lambda function
        my_lambda = lambda x: x**2
        print(f"Node {rank}: Sending lambda function...")
        send_lambda(my_lambda, dst=1)
        print(f"Node {rank}: Sent lambda function...")
    elif rank == 1:
        print(f"Node {rank}: Receiving lambda function...")
        received_lambda = recv_lambda(src=0)
        print(f"Node {rank}: Received lambda function...")
        # Test the received lambda function
        result = received_lambda(5)
        print(f"Node {rank}: Result of lambda function(5) = {result}")
        
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

