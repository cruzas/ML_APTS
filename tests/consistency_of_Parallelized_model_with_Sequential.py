import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from pmw.parallelized_model import ParallelizedModel
from optimizers import *
import utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataloaders import GeneralizedDistributedDataLoader

# TODO: return dummy variables in the generalized dataloader for first and last ranks
def main(rank=None, master_addr=None, master_port=None, world_size=None):
    utils.prepare_distributed_environment(rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    '''
    This test is to check the consistency of the data parallel model with the sequential model.
    '''
    # _________ Some parameters __________
    PARALLEL = True
    SEQUENTIAL = True
    is_sharded = False
    num_replicas_per_subdomain = 2
    num_subdomains = 1
    batch_size = 10000 # 17234, 24894
    data_chunks_amount = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 0
    torch.manual_seed(seed)
    learning_rate = 1
    # ____________________________________
    tot_replicas = num_subdomains * num_replicas_per_subdomain
        
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # NOTE: Normalization makes a huge difference
        transforms.Lambda(lambda x: x.view(x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset_seq = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset_seq = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_dataset_par = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset_par = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    criterion = torch.nn.CrossEntropyLoss()

    NN1 = [nn.Flatten, nn.Linear, nn.ReLU,nn.Linear, nn.ReLU]
    NN1_dict_list = [{'start_dim': 1}, 
                    {'in_features': 784, 'out_features': 256}, 
                    {}, 
                    {'in_features': 256, 'out_features': 256}, 
                    {}]

    NN2 = [nn.Linear, nn.ReLU]
    NN2_dict_list = [{'in_features': 256, 'out_features': 128}, {}]
    
    NN3 = [nn.Linear, nn.Sigmoid, nn.LogSoftmax]
    NN3_dict_list = [{'in_features': 128, 'out_features': 10}, {}, {'dim': 1}]
    
    stage_list = [
        (NN1, NN1_dict_list), # Stage 1
        (NN2, NN2_dict_list), # Stage 2
        (NN3, NN3_dict_list)  # Stage 3
    ]
    
    train_loader = GeneralizedDistributedDataLoader(len_stage_list=len(stage_list), num_replicas=tot_replicas, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dist.barrier()
    test_loader = GeneralizedDistributedDataLoader(len_stage_list=len(stage_list), num_replicas=tot_replicas, dataset=test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)
    dist.barrier()
    train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    test_loader_seq = DataLoader(test_dataset_seq, batch_size=len(test_dataset_seq), shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    random_input = torch.randn(10, 1, 784, device=device)
    
    layers = []
    for stage in stage_list:
        layer = []
        for i in range(len(stage[0])):
            layer.append(stage[0][i](**stage[1][i]))
        layers.append(nn.Sequential(*layer))

    if rank == 0 and SEQUENTIAL:
        torch.manual_seed(seed)
        seq_model = nn.Sequential(*layers).to(device)#MNIST_FCNN().to(device)

        param_norm = lambda seq_model: [torch.norm(torch.cat([p.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        grad_norm = lambda seq_model: [torch.norm(torch.cat([p.grad.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        seq_optimizer = torch.optim.SGD(seq_model.parameters(), lr=learning_rate)
        print(f'START (SEQ) param norm -> {param_norm(seq_model)}')

    dist.barrier()
    if PARALLEL:
        torch.manual_seed(seed)
        par_model = ParallelizedModel(stage_list=stage_list, sample=random_input, num_replicas_per_subdomain=num_replicas_per_subdomain, num_subdomains=num_subdomains, is_sharded=is_sharded) 

    if PARALLEL and SEQUENTIAL:
        with torch.no_grad():
            for i, p1 in enumerate(par_model.subdomain.weight_parallelized_model.subdomain.parameters()):
                layer_number = par_model.from_rank_structure_to_layer_number()
                p1.data = list(layers[layer_number].parameters())[i].data.clone().detach().to(par_model.tensor_device)
                p1.requires_grad = True
    if PARALLEL:
        print(f'START (PAR) Rank {rank} param norm -> {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}')
        par_optimizer = torch.optim.SGD(par_model.parameters(), lr=learning_rate)
        
    for epoch in range(40):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')  
        if rank == 0 and SEQUENTIAL:
            loss_total_seq = 0
            counter_seq = 0
            # Sequential training loop
            for i, (x, y) in enumerate(train_loader_seq):
                x = x.to(device)
                # print(f'(SEQ) Norm of input -> {torch.norm(x.flatten())}')
                y = y.to(device)
                # Gather sequential model norm
                seq_optimizer.zero_grad()
                out = seq_model(x)
                seq_loss = criterion(out, y)
                loss_total_seq += seq_loss
                counter_seq += 1
                seq_loss.backward()
                seq_optimizer.step()
                # print(f'(SEQ) param norm -> {param_norm(seq_model)}, grad norm -> {grad_norm(seq_model)}')
        
            # Sequential testing loop
            total = 0
            correct = 0
            with torch.no_grad():
                for data in test_loader_seq:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    test_outputs = seq_model(images)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(predicted.device)).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch {epoch}, Sequential accuracy: {accuracy}')
            
        dist.barrier()
        if PARALLEL:
            loss_total_par = 0
            counter_par = 0
            # Parallel training loop
            ds = train_loader
            # print(f'Rank {rank} amount of batches {len(ds)}, shape of batches {[[d[0].shape,d[1].shape] for d in ds]}')
            for i, (x, y) in enumerate(train_loader):
                # print(f'Rank {rank} batch {i} of shape {x.shape}')
                dist.barrier()
                # dist.barrier()
                x = x.to(device)
                y = y.to(device)
        
                tic = time.time()
                # Gather parallel model norm
                par_optimizer.zero_grad()
                c = utils.closure(x, y, torch.nn.CrossEntropyLoss(), par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)
                par_loss = c()
                loss_total_par += par_loss
                counter_par += 1
                par_optimizer.step()  
                par_model.sync_params()
                # print(f"(ACTUAL PARALLEL) stage {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in par_model.parameters()]))}")   
                        
            # Parallel testing loop
            with torch.no_grad(): # TODO: Make this work also with NCCL
                correct = 0
                total = 0
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    closuree = utils.closure(images, labels, criterion, par_model, compute_grad=False, zero_grad=True, return_output=True)       
                    _, test_outputs = closuree()
                    if dist.get_rank() in par_model.all_final_stages_main_rank:
                        test_outputs = torch.cat(test_outputs)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted ==
                                    labels.to(predicted.device)).sum().item()

                if dist.get_rank() in par_model.all_final_stages_main_rank:
                    accuracy = 100 * correct / total
                    # here we dist all reduce the accuracy
                    accuracy = torch.tensor(accuracy).to(device)
                    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM, group=par_model.all_final_stages_main_rank_group)
                    accuracy /= len(par_model.all_final_stages_main_rank)
                    print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')
 
        
        if rank == 0 and SEQUENTIAL:
            print(f'Epoch {epoch}, Sequential avg loss: {loss_total_seq/counter_seq}')
        if PARALLEL and rank == 0:
            print(f'Epoch {epoch}, Parallel avg loss: {loss_total_par/counter_par}')            

if __name__ == '__main__':
    torch.manual_seed(0)
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
