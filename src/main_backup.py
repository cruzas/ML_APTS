# External libraries
import pprint,copy, torch
import torch.multiprocessing as mp
# User libraries
from utils.utility import *
from matplotlib import pyplot as plt

# a=torch.randn(1000000000, device='cuda:0')
# time.sleep(2)
# a=torch.randn(1000000000, device='cuda:0')
# time.sleep(1)
# a = 0
# gc.collect()
# torch.cuda.set_device('cuda:1')
# 
# torch.cuda.set_device('cuda:1')
# torch.cuda.empty_cache()

def get_apts_w_params(momentum=False, second_order=False, nr_models=2, max_iter=5, fdl=False, global_pass=True, device=None):
    TR_APTS_W_PARAMS_GLOBAL = {
        "radius": 0.01, #0.1,
        "max_radius": 0.1, #4.0
        "min_radius": 0.0001,
        "decrease_factor": 0.5,
        "increase_factor": 2.0,
        "is_adaptive": False,
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
        "is_adaptive": False,
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
        "device": device,
        "max_iter": max_iter,
        "nr_models": nr_models,
        "global_opt": TR,
        "global_opt_params": TR_APTS_W_PARAMS_GLOBAL,
        "local_opt": TR,
        "local_opt_params": TR_APTS_W_PARAMS_LOCAL,
        "global_pass": global_pass,
        "forced_decreasing_loss": fdl,
        "loss_fn": nn.CrossEntropyLoss(),
    }

    return APTS_W_PARAMS
    
def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    parameter_decomposition = True # TODO: Implement data decomposition
    args = parse_args() # TODO: make this is easier to execute on a personal computer

    torch.random.manual_seed(0)

    # Device    
    backend = dist.get_backend()
    device = torch.device(f"cuda:{rank if backend == 'gloo' else 0}")
    # torch.set_default_device(device) # TODO/NOTE: This line creates problems and inconsistencies (compared to the cluster), when uncommented.
    args.device = device
    args.nr_models = dist.get_world_size()

    # Training settings
    trials = args.trials  # number of trials
    epochs = args.epochs  # number of epochs to run per trial
    net_nr = args.net_nr  # model number to choose
    dataset = args.dataset  # name of the dataset
    minibatch_size = args.minibatch_size  # size of the mini-batches
    overlap_ratio = args.overlap_ratio  # overlap ratio between mini-batches
    optimizer_name = args.optimizer_name  # name of the optimizer

    loss_function = nn.CrossEntropyLoss()
    args.loss_fn = loss_function
    optimizer_params = get_apts_w_params(momentum=False, second_order=False, nr_models=2, max_iter=5, fdl=True, global_pass=True, device=None)
    opt_fun = get_optimizer_fun(optimizer_name)
    net_fun, net_params = get_net_fun_and_params(dataset, net_nr)

    network = net_fun(**net_params).to(device)
    optimizer = opt_fun(network.parameters(), model=network, **optimizer_params)
    
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
    for trial in range(trials):
        losses, accuracies, = do_one_optimizer_test(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            net=network,
            num_epochs=epochs,
            criterion=loss_function,
            desired_accuracy=100,
            device=device
        )

        if dist.get_rank() == 0:
            print(
                f"Rank {rank}. Trial {trial + 1}/{trials} finished. Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}"
            )
            # here we plot the accuracy after the training through matplotlib
            plt.plot(losses)
            # hold on:

        plt.show()
        print(f"Rank {rank}: Plot successful.")

if __name__ == "__main__":    
    cmd_args = parse_args()
    if cmd_args.on_cluster.lower() == "y" or cmd_args.on_cluster.lower() == "yes":
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = find_free_port()
        mp.spawn(main, args=(master_addr,master_port, world_size), nprocs=world_size, join=True)

            