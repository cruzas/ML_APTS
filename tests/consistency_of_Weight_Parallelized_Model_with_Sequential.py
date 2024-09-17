import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import prepare_distributed_environment
from pmw import *
from optimizers import *
from utils import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataloaders import GeneralizedDistributedDataLoader

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    '''
    This test is to check the consistency of the Weight_Parallelized_Model with the Sequential model.
    '''
    # _________ Some parameters __________
    PARALLEL = True
    SEQUENTIAL = True
    num_replicas = 1
    batch_size = 30000
    data_chunks_amount = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    learning_rage = 1
    # ____________________________________
        
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # NOTE: Normalization makes a huge difference
        transforms.Lambda(lambda x: x.view(x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    criterion = torch.nn.CrossEntropyLoss()

    NN1 = lambda in_features,out_features: nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_features,out_features), nn.ReLU(), nn.Linear(out_features, 256), nn.ReLU())
    NN2 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.ReLU())
    NN3 = lambda in_features,out_features: nn.Sequential(nn.Linear(in_features,out_features), nn.Sigmoid(), nn.LogSoftmax(dim=1))
    stage_list = [
        (NN1, {'in_features': 784, 'out_features': 256}), # Stage 1
        (NN2, {'in_features': 256, 'out_features': 128}), # Stage 2
        (NN3, {'in_features': 128, 'out_features': 10})   # Stage 3
    ]

    train_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = GeneralizedDistributedDataLoader(stage_list=stage_list, num_replicas=num_replicas, dataset=test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)

    train_loader_seq = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader_seq = DataLoader(test_dataset, batch_size=50000, shuffle=False, num_workers=0, pin_memory=True)
     
    layers = [layer[0](**layer[1]) for layer in stage_list]
    random_input = torch.randn(10, 1, 784, device=device)
    
    if rank == 0 and SEQUENTIAL:
        torch.manual_seed(0)
        seq_model = nn.Sequential(*layers).to(device)#MNIST_FCNN().to(device)

        param_norm = lambda seq_model: [torch.norm(torch.cat([p.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        grad_norm = lambda seq_model: [torch.norm(torch.cat([p.grad.flatten() for p in stage.parameters()])).item() for stage in seq_model]
        seq_optimizer = torch.optim.SGD(seq_model.parameters(), lr=learning_rage)
        print(f'START (SEQ) param norm -> {param_norm(seq_model)}')

    dist.barrier()
    if PARALLEL:
        weight_par_model = Weight_Parallelized_Model(stage_list=stage_list, rank_list=list(range(0,len(stage_list))), sample=random_input) 
    if PARALLEL and SEQUENTIAL:
        for i, p1 in enumerate(weight_par_model.stage.parameters()):
            p1.data = list(layers[rank].parameters())[i].data.clone().detach().to(weight_par_model.tensor_device)
            p1.requires_grad = True
    if PARALLEL:
        print(f'START (PAR) Rank {rank} param norm -> {torch.norm(torch.cat([p.flatten() for p in weight_par_model.stage.parameters()]))}')
        par_optimizer = torch.optim.SGD(weight_par_model.parameters(), lr=learning_rage)
        
    for epoch in range(40):
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')  
        if rank == 0 and SEQUENTIAL:
            loss_total_seq = 0
            counter_seq = 0
            # Sequential training loop
            for i, (x, y) in enumerate(train_loader_seq):
                x = x.to(device)
                y = y.to(device)
                # Gather sequential model norm
                seq_optimizer.zero_grad()
                out = seq_model(x)
                seq_loss = criterion(out, y)
                loss_total_seq += seq_loss
                counter_seq += 1
                seq_loss.backward()
                seq_optimizer.step()
                print(f'(SEQ) param norm -> {param_norm(seq_model)}, grad norm -> {grad_norm(seq_model)}')

            # Sequential testing loop
            total = 0
            correct = 0
            with torch.no_grad():
                # seq_model.eval()
                for data in test_loader_seq:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    test_outputs = seq_model(images)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(predicted.device)).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch {epoch}, Sequential accuracy: {accuracy}')

        if PARALLEL:
            loss_total_par = 0
            counter_par = 0
            # Parallel training loop
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                tic = time.time()
                # Gather parallel model norm
                par_optimizer.zero_grad()
                c = closure(x, y, torch.nn.CrossEntropyLoss(), weight_par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)
                par_loss = c()
                loss_total_par += par_loss
                counter_par += 1
                par_optimizer.step()  
                print(f"(ACTUAL PARALLEL) stage {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in weight_par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in weight_par_model.parameters()]))}")   
                        
            # Parallel testing loop
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    closuree = closure(images, labels, criterion, weight_par_model, compute_grad=False, zero_grad=True, return_output=True)       
                    _, test_outputs = closuree()
                    if dist.get_rank() == weight_par_model.rank_list[-1]:
                        test_outputs = torch.cat(test_outputs)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(predicted.device)).sum().item()

                if dist.get_rank() == weight_par_model.rank_list[-1]:
                    accuracy = 100 * correct / total
                    print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')

        if rank == 0 and SEQUENTIAL:
            print(f'Epoch {epoch}, Sequential avg loss: {loss_total_seq/counter_seq}')
            if PARALLEL:
                print(f'Epoch {epoch}, Parallel avg loss: {loss_total_par/counter_par}')            

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

