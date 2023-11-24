# External libraries
import pprint

# User libraries
from utils.utility import *


def main():
    # Set up the distributed environment.
    prepare_distributed_environment()
    # Rank ID
    rank = dist.get_rank() if dist.is_initialized() else 0
    args = parse_args()

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
    loss_function = nn.CrossEntropyLoss()
    args.loss_fn = loss_function
    
    net_fun, net_params = get_net_fun_and_params(dataset, net_nr)
    network = net_fun(**net_params).to(device)
    args.model = network

    optimizer_params = get_optimizer_params(args)
    opt_fun = get_optimizer_fun(optimizer_name)

    if optimizer_name == "APTS_W":
        optimizer = opt_fun(network.parameters(), **optimizer_params)
    else:
        optimizer = opt_fun(network.parameters(), **optimizer_params)


    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        data_dir=os.path.abspath("./data"),
        mb_size=minibatch_size,
        overlap_ratio=overlap_ratio,
        sequential=True,
    )

    # Training loop
    for trial in range(trials):
        loss, accuracy = do_one_optimizer_test(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
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
    main()
