import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import datasets, transforms

from dataloaders import GeneralizedDistributedDataLoader
from optimizers import APTS, TR
from models.model_dict import *
from pmw.parallelized_model import ParallelizedModel
from pmw.model_handler import *
import utils

num_subdomains = 1
num_replicas_per_subdomain = 1
num_stages = 1 # 1 or 2
num_shards = 1

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()
    # _________ Some parameters __________
    batch_size = 60000 # NOTE: Setting a bach size lower than the dataset size will cause the two dataloader (sequential and parallel) to have different batches, hence different losses and accuracies
    data_chunks_amount = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 2456456
    torch.manual_seed(seed)
    learning_rage = 0.1
    # ____________________________________

    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        # NOTE: Normalization makes a huge difference
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(
            x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset_par = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset_par = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    criterion = torch.nn.CrossEntropyLoss()
    model_dict = get_model_dict()
    if num_stages == 1:
        # Go through every key in dictionary and set the field stage to 0
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0
    model_handler = ModelHandler(model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)

    # Sharded first layer into [0,1] -> dataloader should upload batch to 0 only
    random_input = torch.randn(10, 1, 784, device=device)

    par_model = ParallelizedModel(model_handler=model_handler, sample=random_input)
    # Save the par_model state_dict
    dst_rank = 0
    par_dict = par_model.state_dict(dst_rank=dst_rank)
    if rank == dst_rank:
        global_model = GlobalModel()
        global_model.load_state_dict(par_dict)
        global_model = global_model.to(device)
        sequential_train_loader = torch.utils.data.DataLoader(train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        sequential_test_loader = torch.utils.data.DataLoader(test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)
        global_optimizer = torch.optim.SGD(global_model.parameters(), lr=learning_rage)
    
    train_loader = GeneralizedDistributedDataLoader(model_handler=model_handler, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = GeneralizedDistributedDataLoader(model_handler=model_handler, dataset=test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)

    par_optimizer = torch.optim.SGD(par_model.parameters(), lr=learning_rage)
    for epoch in range(40):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        loss_total_par = 0
        counter_par = 0
        # Parallel training loop
        for _, (x, y) in enumerate(train_loader):
            dist.barrier()
            # dist.barrier()
            x = x.to(device)
            y = y.to(device)

            # Gather parallel model norm
            par_optimizer.zero_grad()
            counter_par += 1

            par_loss = par_optimizer.step(closure=utils.closure(
                x, y, criterion=criterion, model=par_model, data_chunks_amount=data_chunks_amount, compute_grad=True)) 

            loss_total_par += par_loss
            par_model.sync_params()
            # print(f"(ACTUAL PARALLEL) {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in par_model.parameters()]))}")

        if rank == 0:
            print(f'Epoch {epoch}, Parallel avg loss: {loss_total_par/counter_par}')
            
        # Parallel testing loop
        with torch.no_grad():  # TODO: Make this work also with NCCL
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                closuree = utils.closure(
                    images, labels, criterion, par_model, compute_grad=False, zero_grad=True, return_output=True)
                _, test_outputs = closuree()
                if model_handler.is_last_stage():
                    test_outputs = torch.cat(test_outputs)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted ==
                                labels.to(predicted.device)).sum().item()

            if model_handler.is_last_stage():
                accuracy = 100 * correct / total
                # here we dist all reduce the accuracy
                accuracy = torch.tensor(accuracy).to(device)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM, group=model_handler.get_layers_copy_group(mode='global'))
                accuracy /= len(model_handler.get_stage_ranks(stage_name='last', mode='global'))
                print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')

        # -------------------------------------- Sequential training loop --------------------------------------
        if rank == dst_rank:
            loss = 0; counter = 0
            for _, (x, y) in enumerate(sequential_train_loader):
                counter += 1
                x = x.to(device)
                y = y.to(device)

                def closure():
                    global_optimizer.zero_grad()
                    global_outputs = global_model(x)
                    global_loss = criterion(global_outputs, y)
                    global_loss.backward()
                    return global_loss

                loss += global_optimizer.step(closure)
                
            print(f'Epoch {epoch}, Sequential loss: {loss/counter}')
            # params = [param for param in global_model.parameters()]
            # print(f"(ACTUAL SEQUENTIAL) param norm: {torch.norm(torch.cat([p.flatten() for p in global_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in global_model.parameters()]))}")

            # -------------------------------------- Sequential testing loop --------------------------------------
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in sequential_test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'Epoch {epoch}, Sequential accuracy: {accuracy}')

if __name__ == '__main__':
    torch.manual_seed(0)
    if 1 == 2:
        main()
    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        WORLD_SIZE = num_subdomains*num_replicas_per_subdomain*num_stages*num_shards
        mp.spawn(main, args=(MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)