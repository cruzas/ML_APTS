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
    parser.add_argument("--optimizer", type=str, default="APTS", required=False)
    parser.add_argument("--lr", type=float, default=1.0, required=False)
    parser.add_argument("--dataset", type=str, default="mnist", required=False)
    parser.add_argument("--batch_size", type=int, default=28000, required=False)
    parser.add_argument("--model", type=str,
                        default="feedforward", required=False)
    parser.add_argument("--num_subdomains", type=int, default=2, required=False)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1, required=False)
    parser.add_argument("--num_stages_per_replica",
                        type=int, default=2, required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--trial", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--is_sharded", type=bool, default=False, required=False)
    parser.add_argument("--data_chunks_amount", type=int, default=10, required=False)
    return parser.parse_args(args)


def main(rank=None, cmd_args=None, master_addr=None, master_port=None, world_size=None):
    # Parse command line arguments
    parsed_cmd_args = parse_cmd_args(cmd_args)

    # Initialize distributed environment
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    # Make sure all ranks have the same number of GPUs
    utils.check_gpus_per_rank()

    # Rank and main computing device
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # CSV filename
    results_dir = "results"
    csv_filename = os.path.join(results_dir, f"{parsed_cmd_args.optimizer}_{parsed_cmd_args.lr}_{parsed_cmd_args.dataset}_{parsed_cmd_args.batch_size}_{parsed_cmd_args.model}_{parsed_cmd_args.num_subdomains}_{parsed_cmd_args.num_replicas_per_subdomain}_{parsed_cmd_args.num_stages_per_replica}_{parsed_cmd_args.epochs}_{parsed_cmd_args.seed}_t{parsed_cmd_args.trial}.csv")

    # If csv_filename exists, exit 0
    if os.path.exists(csv_filename):
        print(f"File {csv_filename} already exists. Exiting.")
        exit(0)

    # Set seed    
    seed = parsed_cmd_args.seed
    torch.manual_seed(seed)

    # Total number of model replicas
    tot_replicas = parsed_cmd_args.num_subdomains * \
        parsed_cmd_args.num_replicas_per_subdomain 

    APTS_in_data_sync_strategy = 'average'  # 'sum' or 'average'

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()    

    # Create training and testing datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        # NOTE: Normalization makes a huge difference
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(
            x.size(0), -1))  # Reshape the tensor
    ])

    # Training and test sets
    train_dataset_par = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset_par = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    # Create stage list
    stage_list = networks.construct_stage_list(
        parsed_cmd_args.model, parsed_cmd_args.num_stages_per_replica)

    # Create model
    if parsed_cmd_args.dataset.lower() == "mnist":
        input_size = 784
        random_input = torch.randn(parsed_cmd_args.batch_size, 1, input_size, device=device)
    elif parsed_cmd_args.dataset.lower() == "cifar10":
        # TODO: Verify whether this works or not
        raise NotImplementedError("CIFAR-10 dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {parsed_cmd_args.dataset}")

    par_model = ParallelizedModel(stage_list=stage_list, 
                                  sample=random_input,
                                  num_replicas_per_subdomain=parsed_cmd_args.num_replicas_per_subdomain, 
                                  num_subdomains=parsed_cmd_args.num_subdomains,
                                  is_sharded=parsed_cmd_args.is_sharded)
    
    # Create data loaders
    train_loader = GeneralizedDistributedDataLoader(model_structure=par_model.all_model_ranks, dataset=train_dataset_par, batch_size=parsed_cmd_args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = GeneralizedDistributedDataLoader(model_structure=par_model.all_model_ranks, dataset=test_dataset_par, batch_size=len(test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)

    # Create optimizer
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
    par_optimizer = APTS(model=par_model, 
                         criterion=criterion, 
                         subdomain_optimizer=subdomain_optimizer, 
                         subdomain_optimizer_defaults={'lr': parsed_cmd_args.lr},
                         global_optimizer=TR, 
                         global_optimizer_defaults=glob_opt_params, 
                         lr=parsed_cmd_args.lr, 
                         max_subdomain_iter=5, 
                         dogleg=True, 
                         APTS_in_data_sync_strategy=APTS_in_data_sync_strategy)

    loss_per_epoch = [None] * (parsed_cmd_args.epochs + 1)
    acc_per_epoch = [None] * (parsed_cmd_args.epochs + 1)
    time_per_epoch = [None] * (parsed_cmd_args.epochs + 1)
    time_per_epoch[0] = 0
    for epoch in range(parsed_cmd_args.epochs + 1):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        dist.barrier()
        loss_total_par = 0
        counter_par = 0
        # Parallel training loop
        step_start_time = time.time()  # Start time for the optimizer step
        for i, (x, y) in enumerate(train_loader):
            dist.barrier()
            counter_par += 1
            x = x.to(device)
            y = y.to(device)

            par_optimizer.zero_grad()
            if epoch == 0:
                # Compute initial loss using closure only
                closure = utils.closure(
                    x, y, criterion=criterion, model=par_model, data_chunks_amount=parsed_cmd_args.data_chunks_amount, compute_grad=False)
                par_loss = closure()
            else:
                # Gather parallel model norm
                par_loss = par_optimizer.step(closure=utils.closure(
                    x, y, criterion=criterion, model=par_model, data_chunks_amount=parsed_cmd_args.data_chunks_amount, compute_grad=True))
            
            par_model.sync_params()
            loss_total_par += par_loss

        dist.barrier() # To ensure a more accurate measure of time per epoch
        if epoch > 0:
            time_per_epoch[epoch] = time.time() - step_start_time
            # Average the time per epoch across all ranks
            time_per_epoch[epoch] = torch.tensor(time_per_epoch[epoch]).to(device)
            dist.all_reduce(tensor=time_per_epoch[epoch], op=dist.ReduceOp.SUM)
            time_per_epoch[epoch] /= dist.get_world_size()
            time_per_epoch[epoch] = time_per_epoch[epoch].item()

        loss_per_epoch[epoch] = loss_total_par / counter_par
        if rank == 0:
            print(f'Epoch {epoch}, Parallel avg loss: {loss_per_epoch[epoch]}')

        # Parallel testing loop
        with torch.no_grad():  # TODO: Make this work also with NCCL
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                test_closure = utils.closure(images, labels, criterion, par_model, compute_grad=False, zero_grad=True, return_output=True)
                _, test_outputs = test_closure()
                if rank in par_model.all_final_stages_main_rank:
                    test_outputs = torch.cat(test_outputs)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(predicted.device)).sum().item()

            if rank in par_model.all_final_stages_main_rank:
                accuracy = 100 * correct / total
                accuracy = torch.tensor(accuracy).to(device)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM, group=par_model.all_final_stages_main_rank_group)
                accuracy /= len(par_model.all_final_stages_main_rank)
                acc_per_epoch[epoch] = accuracy.item()

                print(f'Epoch {epoch}, Parallel accuracy: {acc_per_epoch[epoch]}')
    
    if rank == par_model.all_final_stages_main_rank[0]:
        print(f"Loss per epoch: {loss_per_epoch}")
        print(f"Accuracy per epoch: {acc_per_epoch}")
        print(f"Time per epoch: {time_per_epoch}")

        # Save results to a CSV file. You can use pandas.
        results = pd.DataFrame({'loss': loss_per_epoch, 'accuracy': acc_per_epoch, 'time': time_per_epoch})
        
        print(f"Saving results to {csv_filename}")
        results.to_csv(csv_filename, index=False)
    

if __name__ == '__main__':
    if 1 == 1:
        main(cmd_args=sys.argv[1:])
    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        WORLD_SIZE = 4
        # Also add command-line arguments to main
        # --optimizer "APTS" --dataset "MNIST" --batch_size "60000" --model "feedforward" --num_subdomains "2" --num_replicas_per_subdomain "1" --num_stages_per_replica "2" --trial "0" --epochs "2"

        mp.spawn(main, args=(None, MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)
