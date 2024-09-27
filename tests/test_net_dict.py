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

num_subdomains = 2
num_replicas_per_subdomain = 2
num_stages = 2 # 1 or 3

#TODO: Make sure that even in case layers are set to be non trainable, the code doesn't crash

# TODO: return dummy variables in the generalized dataloader for first and last ranks
def main(rank=None, master_addr=None, master_port=None, world_size=None):
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    utils.check_gpus_per_rank()
    # _________ Some parameters __________
    batch_size = 28000
    data_chunks_amount = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 0
    torch.manual_seed(seed)
    learning_rage = 1
    APTS_in_data_sync_strategy = 'average'  # 'sum' or 'average'
    is_sharded = False
    # ____________________________________

    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    transform = transforms.Compose([
        transforms.ToTensor(),
        # NOTE: Normalization makes a huge difference
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(
            x.size(0), -1))  # Reshape the tensor
    ])

    train_dataset_par = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset_par = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    criterion = torch.nn.CrossEntropyLoss()

    model_dict = get_model_dict()
    model_handler = ModelHandler(model_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None)

    if num_stages == 1:
        # Go through every key in dictionary and set the field stage to 0
        for key in model_dict.keys():
            model_dict[key]['stage'] = 0

    #  sharded first layer into [0,1] -> dataloader should upload batch to 0 only
    random_input = torch.randn(10, 1, 784, device=device)
    par_model = ParallelizedModel(stage_list=model_dict, sample=random_input,
                                  num_replicas_per_subdomain=num_replicas_per_subdomain, num_subdomains=num_subdomains, is_sharded=is_sharded)

    train_loader = GeneralizedDistributedDataLoader(model_structure=par_model.all_model_ranks, dataset=train_dataset_par, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = GeneralizedDistributedDataLoader(model_structure=par_model.all_model_ranks, dataset=test_dataset_par, batch_size=len(
        test_dataset_par), shuffle=False, num_workers=0, pin_memory=True)


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
    par_optimizer = APTS(model=par_model, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults={'lr': learning_rage},
                         global_optimizer=TR, global_optimizer_defaults=glob_opt_params, lr=learning_rage, max_subdomain_iter=5, dogleg=True, APTS_in_data_sync_strategy=APTS_in_data_sync_strategy)

    for epoch in range(40):
        dist.barrier()
        if rank == 0:
            print(f'____________ EPOCH {epoch} ____________')
        dist.barrier()
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
            # print(f"(ACTUAL PARALLEL) stage {rank} param norm: {torch.norm(torch.cat([p.flatten() for p in par_model.parameters()]))}, grad norm: {torch.norm(torch.cat([p.grad.flatten() for p in par_model.parameters()]))}")

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

        if rank == 0:
            print(
                f'Epoch {epoch}, Parallel avg loss: {loss_total_par/counter_par}')


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
        WORLD_SIZE = num_subdomains*num_replicas_per_subdomain*num_stages*2
        mp.spawn(main, args=(MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)