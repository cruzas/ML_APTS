import torch 
import time 
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
from utils.utility import prepare_distributed_environment, create_dataloaders
import torchvision
from torchvision import transforms
from parallel.networks import *
from parallel.models import *
from parallel.utils import *
import pandas as pd

# Make a function that reads arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='Parallel test')
    # We will have optimizer name, batch size, learning rate, number of trials, number of epochs
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    return parser.parse_args()

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    args = parse_args()
    # Optimizer 
    optimizer_name = args.optimizer
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    trials = args.trials
    epochs = args.epochs
    dataset = args.dataset
    
    torch.manual_seed(1)
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    rank = dist.get_rank() if dist.get_backend() != 'nccl' else rank
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the dataset 
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset} not supported')  

    # Create the dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    # Instantiate the model, loss function, and optimizer
    net = ResNet().to(device)
    for sample in trainloader:
        break
    
    # Reset trainloader
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    
    sample = sample[0]
    layers = []
    tot_layers = len(net.layer_list)
    if tot_layers < world_size:
        raise ValueError("Number of layers must be greater than or equal to the number of ranks.")
    layers_per_rank = tot_layers // world_size
    for i in range(world_size):
        layers.append(nn.Sequential(*net.layer_list[i*layers_per_rank:(i+1)*layers_per_rank]))
            
    layer_list = [0]*len(layers)
    for i, l in enumerate(layers):
        l = l.to('cuda'); sample = sample.to('cuda')
        input_shape = lambda samples, sample=sample: torch.cat([torch.tensor([samples], dtype=torch.int32), torch.tensor(sample.shape)[1:]])
        sample = l(sample)
        output_shape = lambda samples, sample=sample: torch.cat([torch.tensor([samples], dtype=torch.int32), torch.tensor(sample.shape[1:])])
        layer_list[i] = ([lambda l=l: l, {}, (input_shape, output_shape)])
    
    rank_list = [[r] for r in range(world_size)]
    net = Weight_Parallelized_Model(layer_list, rank_list)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Training function
    def train(epoch):
        # net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            closuree = closure(inputs, labels, criterion, net, compute_grad=True, zero_grad=True)        
            optimizer.step(closuree)
            running_loss += closuree(compute_grad=False)
            # Print every 100 epochs
            if i % 100 == 99:
                if dist.get_rank() == net.rank_list[-1][0]:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        running_loss = 0.0

    # Testing function
    def test():
        # net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                closuree = closure(images, labels, criterion, net, compute_grad=False, zero_grad=True, output=True, counter=False)       
                test_loss, test_outputs = closuree()
                if dist.get_rank() == net.rank_list[-1][0]:
                    # print(f'Loss: {test_loss:.4f}')
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(predicted.device)).sum().item()
        if dist.get_rank() == net.rank_list[-1][0]:
            print(f'Accuracy on the 10000 test images: {100 * correct / total:.2f}%')

    # Train the network for a few epochs
    num_epochs = 2
    epoch_times = [0] * num_epochs
    epoch_usage_times = [0] * num_epochs
    epoch_num_f_evals = [0] * num_epochs
    epoch_num_g_evals = [0] * num_epochs
    epoch_num_sf_evals = [0] * num_epochs
    epoch_num_sg_evals = [0] * num_epochs
    for epoch in range(num_epochs):  # Number of epochs can be adjusted
        epoch_start = time.time()
        net.num_f_evals = 0
        net.num_g_evals = 0
        net.subdomain.num_f_evals = 0 
        net.subdomain.num_g_evals = 0 
        net.subdomain.f_time = 0
        net.subdomain.g_time = 0
        net.f_time = 0
        net.g_time = 0
        train(epoch)
        epoch_times[epoch] = time.time() - epoch_start
        epoch_usage_times[epoch] = net.f_time + net.g_time + net.subdomain.f_time + net.subdomain.g_time
        epoch_num_f_evals[epoch] = net.num_f_evals 
        epoch_num_g_evals[epoch] = net.num_g_evals
        epoch_num_sf_evals[epoch] = net.subdomain.num_f_evals
        epoch_num_sg_evals[epoch] = net.subdomain.num_g_evals
        test()
    infos = [epoch_times, epoch_usage_times, epoch_num_f_evals, epoch_num_g_evals, epoch_num_sf_evals, epoch_num_sg_evals]
    # save this information to a CSV file through pandas
    df = pd.DataFrame(infos).T
    df.columns = ['Epoch Time', 'Epoch Usage Time', 'Epoch Num f evals', 'Epoch Num g evals', 'Epoch Num sf evals', 'Epoch Num sg evals']
    # make the CSV file depend on the optimizer
    df.to_csv(f'epoch_info_{optimizer.__class__.__name__}.csv', index=False)
    
if __name__ == '__main__':
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        world_size = 2
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)







