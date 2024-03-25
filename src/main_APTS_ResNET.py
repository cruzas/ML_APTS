# External libraries
import os
import pprint,copy, torch
import torch.multiprocessing as mp
# User libraries
from optimizers.APTS_W import APTS_W
from utils.utility import *
from matplotlib import pyplot as plt
import torch.distributed as dist
from utils import *
from torch import nn
import pandas as pd
import torchvision

def get_apts_w_params(momentum=False, second_order=False, nr_models=2, max_iter=5, fdl=False, global_pass=True, device=None):
    TR_APTS_W_PARAMS_GLOBAL = {
        "radius": 0.01, #0.1,
        "max_radius": 0.1, #4.0
        "min_radius": 0.0001,
        "decrease_factor": 0.5,
        "increase_factor": 2.0,
        "second_order": second_order,
        "delayed_second_order": 0,
        "device": device,
        "accept_all": False,
        "acceptance_ratio": 0.75,
        "reduction_ratio": 0.25,
        "history_size": 5,
        "momentum": momentum,
        "beta1": 0.9,
        "beta2": 0.999,
        "norm_type": torch.inf,
    }

    TR_APTS_W_PARAMS_LOCAL = {
        "radius": 0.01,
        "max_radius": 0.1,
        "min_radius": 0,  # based on APTS class
        "decrease_factor": 0.5,
        "increase_factor": 2.0,
        "second_order": second_order,
        "delayed_second_order": 0,
        "device": device,
        "accept_all": False,
        "acceptance_ratio": 0.75,
        "reduction_ratio": 0.25,
        "history_size": 5,
        "momentum": momentum,
        "beta1": 0.9,
        "beta2": 0.999,
        "norm_type": torch.inf,
    }

    APTS_W_PARAMS = {
        "max_iter": max_iter,
        "nr_models": nr_models,
        "global_opt": TR,
        "global_opt_params": TR_APTS_W_PARAMS_GLOBAL,
        "local_opt": TR,
        "local_opt_params": TR_APTS_W_PARAMS_LOCAL,
        "global_pass": global_pass,
        "forced_decreasing_loss": fdl,
    }

    return APTS_W_PARAMS
    
def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if master_addr is None:
        master_addr = os.environ["MASTER_ADDR"]
    if master_port is None:
        master_port = os.environ["MASTER_PORT"]
    if world_size is None: 
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    parameter_decomposition = True # TODO: Implement data decomposition
    # args = parse_args() # TODO: make this is easier to execute on a personal computer

    torch.random.manual_seed(0)

    # Device    
    backend = dist.get_backend()
    device = torch.device(f"cuda:{rank if backend == 'gloo' else 0}")
    # torch.set_default_device(device) # TODO/NOTE: This line creates problems and inconsistencies (compared to the cluster), when uncommented.
    # args.device = device
    # args.nr_models = dist.get_world_size()

    # Training settings
    trials = 10  # number of trials
    epochs = 50  # number of epochs to run per trial
    # net_nr = 4  # model number to choose
    dataset = 'MNIST'  # name of the dataset
    minibatch_size = int(2500)  # size of the mini-batches
    overlap_ratio = 0.01  # overlap ratio between mini-batches
    # optimizer_name = 'APTS_W'  # name of the optimizer
    nr_models = world_size # amount of subdomains to use in the optimizer

    loss_function = nn.CrossEntropyLoss
    loss_fn = loss_function
    optimizer_params = get_apts_w_params(momentum=False, second_order=False, nr_models=nr_models, max_iter=5, fdl=False, global_pass=True, device=None)
    
    net_fun, net_params = torchvision.models.resnet18 #MNIST_FCNN_Small, {}#models.resnet18, {'weights':None, 'progress':True}#get_net_fun_and_params(dataset, net_nr)
        
    # print parameters norm:
    # torch.random.manual_seed(0)
    # print("Parameters norm: ", torch.norm(torch.cat([torch.flatten(p) for p in net_fun(**net_params).parameters()])))
    
    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        data_dir=os.path.abspath("./data"),
        mb_size=minibatch_size,
        overlap_ratio=overlap_ratio,
        parameter_decomposition=parameter_decomposition,
        device=device
    )
    
    # Training loop
    losses=[None]*trials
    accuracies=[None]*trials
    cum_times=[None]*trials
    for trial in range(trials):
        network = net_fun(**net_params).to(device)
        optimizer = APTS_W(network.parameters(), model=network, loss_fn=loss_fn, **optimizer_params)
        # optimizer = optim.Adam(network.parameters())
        losses[trial], accuracies[trial], cum_times[trial] = do_one_optimizer_test(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            net=network,
            num_epochs=epochs,
            criterion=loss_function(),
            desired_accuracy=100,
            device=device
        )

    if dist.get_rank() == 0:
       # print(
        #    f"Rank {rank}. Trial {trial + 1}/{trials} finished. Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}. Cum. time: {[round(t, 2) for t in cum_times]}"
        #)
        # here we plot the accuracy after the training through matplotlib
        # plt.plot(losses)
        # hold on:
        # Here we plot the test accuracies vs cum times
        # plt.plot(cum_times, accuracies)
        # plt.show()
        
        # Save losses, accuracies, and cum times to a CSV file using Pandas
        df = pd.DataFrame({"losses": losses, "accuracies": accuracies, "cum_times": cum_times})
        df.to_csv(f"results_APTS_W_{nr_models}.csv", index=False)
        print("Successfully saved to file.")

    # print(f"Rank {rank}: Plot successful.")

if __name__ == "__main__":    
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)

            
