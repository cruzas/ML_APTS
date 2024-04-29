# External libraries
import pprint,copy
import torch.multiprocessing as mp
# User libraries
from utils.utility import *
from optimizers.APTS_W import APTS_W


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

    # torch.random.manual_seed(rank)

    # Device
    device = torch.device(f"cuda:{rank}")
    args.device = device
    args.nr_models = dist.get_world_size() if dist.is_initialized() else torch.cuda.device_count()

    if torch.cuda.is_available():
        if dist.get_backend() == 'gloo':
            torch.set_default_device(f'cuda:{rank}')
        else:
            torch.set_default_device(f'cuda:0')

    global_model = MNIST_FCNN()
    print(f"Global model device: {next(global_model.parameters()).device}")

    local_model, layer2rank = create_sub_model(global_model, rank, args.nr_models)
    print(f"Local model device: {next(local_model.parameters()).device}")
    
    optimizer = APTS_W(local_model.parameters(), model=local_model, loss_fn=nn.CrossEntropyLoss(), device=device, max_iter=2, nr_models=args.nr_models, global_opt=TR, global_opt_params={'radius': 1e-2}, local_opt=optim.SGD, local_opt_params={'radius': 1e-2}, global_pass=True, R=None, forced_decreasing_loss=False)


    print(f"Rank {rank}: Receive successful.")
    dist.barrier()


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

            