# External libraries
import torch
import torch.nn as nn
import pprint

# User libraries
from utils.utility import *
from utils.Power_Dataframe_SAM import *


def main():
    # Parse arguments from command line
    args = parse_args()

    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        # Convert the args namespace to a dictionary
        args_dict = vars(args)
        # Pretty print the dictionary
        pprint.pprint(f"Printing all args...")
        # Print each key value pair in args one by one
        for key, value in args_dict.items():
            print(f"{key}: {value}")
        print(f"Printing all args...done.")

    # Set environment (in case we are doing distributed training).
    # If not distributed, the function will do nothing.
    prepare_distributed_environment()

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cuda')
    args.device = device
    # Number of processes in the environment (in case executing in parallel)
    if dist.is_initialized() and "APTS" in args.optimizer_name and "D" in args.optimizer_name:
        if args.nr_models != dist.get_world_size():
            raise ValueError(f"Number of models ({args.nr_models}) is different from the number of processes ({dist.get_world_size()}).")
    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    # args.rank = rank

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
        regression = False
    else:
        loss_function = nn.MSELoss()
        regression = True

    args.loss_fn = loss_function
    optimizer_params = get_optimizer_params(args)
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print(f"Printing optimizer params...")
        for key, value in optimizer_params.items():
            print(f"{key}: {value}")
        print(f"Printing optimizer params...done.")

    opt_fun = get_optimizer_fun(optimizer_name)

    net_fun, net_params = get_net_fun_and_params(dataset, net_nr) # TODO: should we also send the network to device?
    ignore_optimizer_params = get_ignore_params(args.optimizer_name)
    
    # Replace the ".db" in db_path with f"_{args.counter}.db"
    args.db_path = args.db_path.replace(".db", f"_{args.counter}.db")

    pdf = Power_Dataframe(results_filename=args.db_path, sequential=True, regression=regression)
    df = pdf.get(
        dataset=dataset,
        mb_size=minibatch_size,
        opt_fun=opt_fun,
        optimizer_params=optimizer_params,
        ignore_optimizer_params=ignore_optimizer_params,
        network_fun=net_fun,
        network_params=net_params,
        loss_function=loss_function,
        trials=trials,
        epochs=epochs,
        pretraining_status=0,
        overlap_ratio=overlap_ratio
    )

    pdf.plot(dataset=dataset,
             mb_size=minibatch_size,
             opt_fun=opt_fun,
             optimizer_params=optimizer_params,
             network_fun=net_fun,
             loss_fun=loss_function,
             ignore_optimizer_params=ignore_optimizer_params,
             loss_params={},
             network_params={},
             overlap_ratio=overlap_ratio,
             trials=args.trials,
             epochs=args.epochs,
             pretraining_status=0,
             mode="mean",
             SAVE=False,
             OPT_NAME=None,
             text_size=14,
             legend_text_size=14,
             title_text_size=14,
             linewidth=2,
             show_variance=False,
             plot_type=[['loss']],
             IGNORED_FIELDS=["loss_class_str"])

    print("Done")


if __name__ == "__main__":
    main()
