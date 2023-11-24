# External libraries
import pprint
import torch.multiprocessing as mp
# User libraries
from utils.utility import *


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    sequential = (
        True if (dist.is_initialized() and dist.get_world_size() == 1) else False
    )

    use_default_args = input("Do you want to use the default arguments? (y/n): ")
    if use_default_args == "y":
        # NOTE: This will set the arguments to the default ones.
        args = parse_args()
    else:
        print("To be implemented.")
        exit(0)

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    args.nr_models = dist.get_world_size() if dist.is_initialized() else args.nr_models

    # Training settings
    trials = args.trials  # number of trials
    epochs = args.epochs  # number of epochs to run per trial
    net_nr = args.net_nr  # model number to choose
    dataset = args.dataset  # name of the dataset
    minibatch_size = args.minibatch_size  # size of the mini-batches
    overlap_ratio = args.overlap_ratio  # overlap ratio between mini-batches
    optimizer_name = args.optimizer_name  # name of the optimizer
    if "MNIST" in dataset or "CIFAR" in dataset:
        loss_function = nn.CrossEntropyLoss()
    args.loss_fn = loss_function
    optimizer_params = get_optimizer_params(args)
    opt_fun = get_optimizer_fun(optimizer_name)
    net_fun, net_params = get_net_fun_and_params(dataset, net_nr)

    optimizer = opt_fun(**optimizer_params)
    network = net_fun(**net_params).to(device)

    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        data_dir=os.path.abspath("./data"),
        mb_size=minibatch_size,
        overlap_ratio=overlap_ratio,
        sequential=sequential,
    )

    # Training loop
    for trial in range(trials):
        loss, accuracy = do_one_optimizer_test(
            train_loader,
            test_loader,
            optimizer,
            net=network,
            num_epochs=epochs,
            criterion=loss_function,
            desired_accuracy=100,
            device=device,
        )

        print(
            f"Trial {trial + 1}/{trials} finished. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA device(s) detected.")
        exit(0)
    
    master_addr = 'localhost'
    master_port = find_free_port()
    
    on_cluster=False
    if not on_cluster:
        mp.spawn(main, args=(master_addr,master_port,world_size), nprocs=world_size, join=True)
    else:
        main()