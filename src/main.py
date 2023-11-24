# External libraries
import pprint,copy
import torch.multiprocessing as mp
# User libraries
from utils.utility import *

def create_sub_model(main_model, rank, world_size):
    '''
    Creates a submodel with trainable layers distributed across ranks.
    Raises an error if the number of ranks exceeds the number of parameter groups.
    '''
    local_model = copy.deepcopy(main_model)
    total_params = len(list(main_model.parameters()))
    
    # Raise an error if there are more ranks than parameters
    # TODO: For efficiency, change the following (E.g. choose a world size that is just enough, i.e. with every GPU needed filled)
    if world_size >= total_params:
        # NOTE: This is here because otherwise we might be wasting computational time on Daint.
        # However, if the number of GPUs is greater than the number of total_params, perhaps we could simply exit for those processes
        # and keep going on the other ones.
        raise ValueError(f"Number of ranks ({world_size}) cannot exceed the number of parameter groups ({total_params}).")
    
        # TODO: If enough processes are availalb and each layer is not big enough, distribute even more layers per process...or something like that

    layers_per_rank = total_params // world_size
    start_layer = rank * layers_per_rank
    end_layer = total_params if rank == world_size - 1 else (rank + 1) * layers_per_rank
    trainable_layers = list(range(start_layer, end_layer))

    print(f"Rank {dist.get_rank()}. Start layer {start_layer}, End layer {end_layer}. Trainable layers {trainable_layers}.")

    for index, param in enumerate(local_model.parameters()):
        param.requires_grad = index in trainable_layers

    return local_model


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    parameter_decomposition = True # TODO: Implement data decomposition
    args = parse_args() # TODO: make this is easier to execute on a personal computer

    # torch.random.manual_seed(rank)

    # Device
    device = torch.device("cuda")
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
    optimizer_params = get_optimizer_params(args)
    opt_fun = get_optimizer_fun(optimizer_name)
    net_fun, net_params = get_net_fun_and_params(dataset, net_nr)

    network = net_fun(**net_params).to(device)
    local_network = create_sub_model(network, rank, args.nr_models)

    optimizer = opt_fun(network.parameters(), **optimizer_params)
    
    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        data_dir=os.path.abspath("./data"),
        mb_size=minibatch_size,
        overlap_ratio=overlap_ratio,
        parameter_decomposition=parameter_decomposition,
    )

    # Training loop
    for trial in range(trials):
        loss, accuracy = do_one_optimizer_test(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            net=local_network,
            num_epochs=epochs,
            criterion=loss_function,
            desired_accuracy=100,
            device=device,
        )

        print(
            f"Rank {rank}. Trial {trial + 1}/{trials} finished. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )

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

            