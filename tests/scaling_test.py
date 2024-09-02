import argparse
import time 
import torch
import torch.distributed as dist
import os
import pandas as pd
import sys
import torch.multiprocessing as mp

# Make ../src visible for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import networks
import utils

from torchvision import datasets, transforms
from dataloaders import GeneralizedDistributedDataLoader
from optimizers import APTS, TR
from pmw.parallelized_model import ParallelizedModel

def parse_cmd_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="APTS", required=True)
    parser.add_argument("--dataset", type=str, default="mnist", required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--model", type=str,
                        default="feedforward", required=True)
    parser.add_argument("--num_subdomains", type=int, default=1, required=True)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1, required=True)
    parser.add_argument("--num_stages_per_replica",
                        type=int, default=1, required=True)
    parser.add_argument("--epochs", type=int, default=10, required=True)
    parser.add_argument("--trial", type=int, default=1, required=True)
    parser.add_argument("--lr", type=float, default=1.0, required=False)
    parser.add_argument("--is_sharded", type=bool, default=False, required=False)
    parser.add_argument("--data_chunks_amount", type=int, default=1, required=False)
    return parser.parse_args(args)


def main(args, rank=None, master_addr=None, master_port=None, world_size=None):
    # Parse command line arguments
    parsed_args = parse_cmd_args(args)

    # Initialize distributed environment
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    # Make sure all ranks have the same number of GPUs
    utils.check_gpus_per_rank()
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set seed    
    seed = 123456789 * parsed_args.trial
    torch.manual_seed(seed)

    tot_replicas = parsed_args.num_subdomains * \
        parsed_args.num_replicas_per_subdomain 

    APTS_in_data_sync_strategy = 'average'  # 'sum' or 'average'
    criterion = torch.nn.CrossEntropyLoss()    

    # Create training and testing datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        # NOTE: Normalization makes a huge difference
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(
            x.size(0), -1))  # Reshape the tensor
    ])

    print("Creating datasets")
    train_dataset_par = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset_par = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    print("Datasets created")

    # Create stage list
    print("Creating stage list")
    stage_list = networks.construct_stage_list(
        parsed_args.model, parsed_args.num_stages_per_replica)
    print("Stage list: ", stage_list)
    print("Stage list created")

    # Create data loaders
    print("Creating data loaders")
    train_loader = GeneralizedDistributedDataLoader(len_stage_list=len(
        stage_list), num_replicas=tot_replicas, dataset=train_dataset_par, batch_size=parsed_args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = GeneralizedDistributedDataLoader(len_stage_list=len(stage_list), num_replicas=tot_replicas, dataset=test_dataset_par, batch_size=len(
        test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)
    print("Data loaders created")


    if parsed_args.dataset.lower() == "mnist":
        input_size = 784
        # random_input = torch.randn(10, 1, 784, device=device)
        random_input = torch.randn(parsed_args.batch_size, 1, input_size, device=device)
    elif parsed_args.dataset.lower() == "cifar10":
        # TODO: Verify whether this works or not
        raise NotImplementedError("CIFAR-10 dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {parsed_args.dataset}")
    
    # Create model
    print("Creating model")
    par_model = ParallelizedModel(stage_list=stage_list, sample=random_input,
                                  num_replicas_per_subdomain=parsed_args.num_replicas_per_subdomain, 
                                  num_subdomains=parsed_args.num_subdomains,
                                  is_sharded=parsed_args.is_sharded)
    print("Model created")

    # Create optimizer
    print("Creating optimizer")
    subdomain_optimizer = torch.optim.SGD
    glob_opt_params = {
        'lr': 0.01,
        'max_lr': 1.0,
        'min_lr': 0.0001,
        'nu': 0.5,
        'inc_factor': 2.0,
        'dec_factor': 0.5,
        'nu_1': 0.25,
        'nu_2': 0.75,
        'max_iter': 5,
        'norm_type': 2
    }
    par_optimizer = APTS(model=par_model, criterion=criterion, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults={'lr': parsed_args.lr},
                         global_optimizer=TR, global_optimizer_defaults=glob_opt_params, lr=parsed_args.lr, max_subdomain_iter=5, dogleg=True, APTS_in_data_sync_strategy=APTS_in_data_sync_strategy)

    print("Optimizer created")
    loss_per_epoch = [None] * parsed_args.epochs
    acc_per_epoch = [None] * parsed_args.epochs
    time_per_epoch = [None] * parsed_args.epochs
    for epoch in range(parsed_args.epochs):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        dist.barrier()
        loss_total_par = 0
        counter_par = 0
        # Parallel training loop
        for i, (x, y) in enumerate(train_loader):
            dist.barrier()
            x = x.to(device)
            y = y.to(device)

            print(f"Rank {dist.get_rank()} train loader x shape: {x.shape}, y shape: {y.shape}")

            # Gather parallel model norm
            par_optimizer.zero_grad()
            counter_par += 1
            step_start_time = time.time()  # Start time for the optimizer step
            par_loss = par_optimizer.step(closure=utils.closure(
                x, y, criterion=criterion, model=par_model, data_chunks_amount=parsed_args.data_chunks_amount, compute_grad=True))
            par_model.sync_params()
            time_per_epoch[epoch] = time.time() - step_start_time

            loss_total_par += par_loss

        loss_per_epoch[epoch] = loss_total_par / counter_par
        if rank == 0:
            print(
                f'Epoch {epoch}, Parallel avg loss: {loss_per_epoch[epoch]}')

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
                if rank in par_model.all_stage_ranks[-1]:
                    test_outputs = torch.cat(test_outputs)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted ==
                                labels.to(predicted.device)).sum().item()

            if rank in par_model.all_stage_ranks[-1]:
                accuracy = 100 * correct / total
                if rank in par_model.all_stage_ranks[-1][1:]:
                    dist.send(tensor=torch.tensor(accuracy).to(
                        'cpu'), dst=par_model.all_stage_ranks[-1][0])
                if rank == par_model.all_stage_ranks[-1][0]:
                    for i in range(len(par_model.all_stage_ranks[-1][1:])):
                        temp = torch.zeros(1)
                        dist.recv(
                            tensor=temp, src=par_model.all_stage_ranks[-1][i+1])
                        accuracy += temp.item()
                    accuracy /= len(par_model.all_stage_ranks[-1])
                    print(f'Epoch {epoch}, Parallel accuracy: {accuracy}')
                    acc_per_epoch[epoch] = accuracy

    if rank == par_model.all_stage_ranks[-1][0]:
        print(f"Loss per epoch: {loss_per_epoch}")
        print(f"Accuracy per epoch: {acc_per_epoch}")
        print(f"Time per epoch: {time_per_epoch}")

        # Save results to a CSV file. You can use pandas.
        results = pd.DataFrame(
            {'loss': loss_per_epoch, 'accuracy': acc_per_epoch, 'time': time_per_epoch})
        
        results.to_csv(f"results_{parsed_args.optimizer}_{parsed_args.dataset}_{parsed_args.model}_{parsed_args.num_subdomains}_{parsed_args.num_replicas_per_subdomain}_{parsed_args.num_stages_per_replica}_{parsed_args.epochs}_{parsed_args.trial}.csv", index=False)
    

if __name__ == '__main__':
    if 1 == 2:
        main(sys.argv[1:])
    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        WORLD_SIZE = 15
        mp.spawn(main, args=(MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)
