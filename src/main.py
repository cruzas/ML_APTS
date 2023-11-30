# External libraries
import pprint,copy, torch
import torch.multiprocessing as mp
# User libraries
from utils.utility import *

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


def create_sub_model(local_model, rank, world_size):
    '''
    Creates a submodel with trainable layers distributed across ranks.
    Raises an error if the number of ranks exceeds the number of parameter groups.
    '''
    # TODO: For efficiency, change the following (E.g. choose a world size that is just enough, i.e. with every GPU needed filled)
    # TODO: If enough processes are availalb and each layer is not big enough, distribute even more layers per process...or something like that

    tot_layers = len(list(local_model.parameters()))
    # tot_params = sum([p.numel() for p in local_model.parameters()])
    if tot_layers < world_size:
        raise ValueError(f"Too many GPUs ({world_size}) for {tot_layers} subdomains. Crashing the code to avoid wasting computational time.")
    
    # index_shuffled = torch.randperm(tot_layers)
    index_shuffled = list(range(tot_layers))

    trainable_layers = []
    params_per_subset = [0]*world_size
    for i in range(world_size):
        params_per_subset[i] = int(tot_layers / (world_size-i))
        if i == world_size - 1:
            trainable_layers.append(index_shuffled[sum(params_per_subset[:i]):])
        else:
            trainable_layers.append(index_shuffled[sum(params_per_subset[:i]):sum(params_per_subset[:i+1])])
        tot_layers -= params_per_subset[i]

    for index, param in enumerate(local_model.parameters()):
        param.requires_grad = index in trainable_layers[rank]

    dict = {k: v for k, v in enumerate(trainable_layers)}
    del trainable_layers # TODO: do this in a more memory-efficient way next time
    layer2rank = {v: k for k, lst in dict.items() for v in lst}

    return local_model, layer2rank


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    parameter_decomposition = True # TODO: Implement data decomposition
    args = parse_args() # TODO: make this is easier to execute on a personal computer

    torch.random.manual_seed(0)

    # Device
    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
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
    local_network,layer2rank = create_sub_model(network, rank, args.nr_models)

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
        loss, accuracy = do_one_optimizer_test(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            net=local_network,
            num_epochs=epochs,
            criterion=loss_function,
            desired_accuracy=100,
            device=device
        )

        print(
            f"Rank {rank}. Trial {trial + 1}/{trials} finished. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        
         

        
        #  # USE CUDA STREAMS???
        # dist.barrier()
        # print(f"Rank {rank}: Barrier successful.")
        # if rank > 0:
        #     for i,param in enumerate(local_network.parameters()):
        #         if param.requires_grad is not False:
        #             # print(f"Rank {rank} norm of layer parameters before sending: {torch.norm([param.flatten() for param in local_network.parameters()][i])}")
        #             print(f"Rank {rank} sent: {param.data.cpu()}")
        #             dist.send(param.data.cpu(), dst=0)
        
        # if rank == 0:
           
        #     for i,param in enumerate(local_network.parameters()):
        #         if param.requires_grad is False:
        #             # print(f"Rank {0} norm of layer parameters before receiving: {torch.norm([param.flatten() for param in local_network.parameters()][i])}")
        #             p = torch.zeros_like(param).cpu()
        #             dist.recv(p, src=layer2rank[i])
        #             print(f"Rank {rank} received: {p.data}")
        #             param.data = p.to(device)
        #             # print(f"Rank {0} norm of layer parameters after receiving: {torch.norm([param.flatten() for param in local_network.parameters()][i])}")
        
        # print(f"Rank {rank}: Receive successful.")
        # dist.barrier()


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

            