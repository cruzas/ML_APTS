import torch, os, time
import numpy as np
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
from parallel.optimizers import *
from data_loaders.Power_DL import *

import pandas as pd
import numpy as np

# Make a function that reads arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='Parallel test')
    # We will have optimizer name, batch size, learning rate, number of trials, number of epochs
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer name')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--trial_number', type=int, default=0, help='Trial number')
    return parser.parse_args()

def get_dataset(dataset):
    if dataset.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), \
               torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == 'mnist':
        # Transform for MNIST
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform), \
               torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset} not supported')

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    args = parse_args()
    print(f'args: {args}')
    optimizer_name, batch_size, learning_rate, trial_number, epochs, dataset = \
        args.optimizer, args.batch_size, args.lr, args.trial_number, args.epochs, args.dataset
    
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    world_size = dist.get_world_size() if dist.is_initialized() else world_size

    filename = f'{optimizer_name}_nl6_{dataset}_{batch_size}_{learning_rate}_{epochs}_{world_size}_t{trial_number}.npz'
    # Do experiment only if the filename doesn't already exist
    if not os.path.exists(filename):
        rank = dist.get_rank() if dist.get_backend() != 'nccl' else rank
    
        # Assuming 'epochs' is already defined
        trial_data = {
            'epoch_loss': np.zeros(epochs, dtype=np.float32),
            'epoch_accuracy': np.zeros(epochs, dtype=np.float32),
            'epoch_times': np.zeros(epochs, dtype=np.float64),  # Assuming times might need higher precision
            'epoch_usage_times': np.zeros(epochs, dtype=np.float64),
            'epoch_num_f_evals': np.zeros(epochs, dtype=np.int32),
            'epoch_num_g_evals': np.zeros(epochs, dtype=np.int32),
            'epoch_num_sf_evals': np.zeros(epochs, dtype=np.int32),
            'epoch_num_sg_evals': np.zeros(epochs, dtype=np.int32)
        }
                
        trainset, testset = get_dataset(dataset)
        torch.manual_seed(1000*trial_number + 1)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        # Define the device 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        print(f"Device is {device}")

        # Instantiate the model, loss function, and optimizer
        if dataset.lower() == 'cifar10':
            # Number of layers is the number of hidden layers, and excludes the input and output layers
            net = ResNet(num_layers=6).to(device) # Equivalent to ResNet-18 if num_layers=2
            for sample in trainloader:
                break
            sample = x = sample[0]
                
            # Reset trainloader
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

            layers = []
            tot_layers = len(net.layer_list)
            if tot_layers < world_size:
                raise ValueError("Number of layers must be greater than or equal to the number of ranks.")
            layers_per_rank = tot_layers // world_size
            for i in range(world_size):
                if i < world_size-1:
                    layers.append(nn.Sequential(*net.layer_list[i*layers_per_rank:(i+1)*layers_per_rank]))
                else:
                    layers.append(nn.Sequential(*net.layer_list[i*layers_per_rank:]))
                    
            layer_list = [0]*len(layers)
            for i, l in enumerate(layers):
                l = l.to('cuda'); sample = sample.to('cuda')
                input_shape = lambda samples, sample=sample: torch.cat([torch.tensor([samples], dtype=torch.int32), torch.tensor(sample.shape)[1:]])
                sample = l(sample)
                output_shape = lambda samples, sample=sample: torch.cat([torch.tensor([samples], dtype=torch.int32), torch.tensor(sample.shape[1:])])
                # layer_list[i] = ([lambda l=l: l, {}, (input_shape, output_shape)])
                layer_list[i] = ([lambda l=l: l, {}])

        elif dataset.lower() == 'mnist':
            # layer_list = [
            # (CNNPart1, {}, ((1, 28, 28), (64, 5, 5))),  # Adjust input and output shapes according to the actual sizes
            # (CNNPart2, {}, ((64, 5, 5), (128,))),        # Adjust input and output shapes
            # (CNNPart3, {}, ((128,), (10,)))              # Adjust input and output shapes
            # ]
            for sample in trainloader:
                break
            x = sample[0]
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            layer_list = [
            (FCNNPart1, {}),  # Adjust input and output shapes according to the actual sizes
            (FCNNPart2, {}),        # Adjust input and output shapes
            (FCNNPart3, {})              # Adjust input and output shapes
            ]
        rank_list = [r for r in range(world_size)]
        net = Weight_Parallelized_Model(layer_list, rank_list, sample=x)
        criterion = nn.CrossEntropyLoss()

        # Check if optimizer_name in lower case is 'sgd'
        if optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) # According to https://arxiv.org/pdf/1512.03385
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        elif optimizer_name.lower() == 'tradam':
            optimizer = TRAdam(net.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'apts':
            subdomain_optimizer = optim.SGD#TRAdam
            subdomain_optimizer_defaults = {'lr': 1e99}
            global_optimizer = TR
            global_optimizer_defaults = {'lr': learning_rate, 'max_iter': 3}    
            optimizer = APTS(model=net, lr=learning_rate, dogleg=True, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults=subdomain_optimizer_defaults, criterion=criterion, global_optimizer=global_optimizer, global_optimizer_defaults=global_optimizer_defaults)
        else:
            raise ValueError(f'Optimizer {optimizer_name} not supported')
        
        # Define the learning rate scheduler
        if optimizer_name.lower() == 'sgd':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

        # TODO: Save the initial accuracy and loss too

        # Training function
        def train(epoch):
            running_loss = 0.0
            count = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                closuree = closure(inputs, labels, criterion, net, compute_grad=True, zero_grad=True)     
                loss = optimizer.step(closuree) 
                running_loss += loss if loss is not None else 0
                count += 1
                if i % 100 == 99: # Print every 100 epochs
                    if dist.get_rank() == net.rank_list[-1] and optimizer_name.lower() == 'apts':
                        optimizer.display_avg_timers()
            running_loss = running_loss/count
            return running_loss

        # Testing function
        def test():
            correct = 0
            total = 0
            accuracy = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    closuree = closure(images, labels, criterion, net, compute_grad=False, zero_grad=True, return_output=True)       
                    _, test_outputs = closuree()
                    if dist.get_rank() == net.rank_list[-1]:
                        test_outputs = torch.cat(test_outputs)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(predicted.device)).sum().item()
            if dist.get_rank() == net.rank_list[-1]:
                accuracy = 100 * correct / total
            return accuracy

        # Train and test the model
        for epoch in range(epochs):  # Number of epochs can be adjusted
            epoch_start = time.time()
            net.zero_counters()
            if optimizer_name.lower() == 'apts':
                optimizer.zero_timers()
                
            trial_data['epoch_loss'][epoch] = train(epoch)
            trial_data['epoch_times'][epoch] = time.time() - epoch_start

            if optimizer_name.lower() == 'sgd': 
                scheduler.step()

            trial_data['epoch_accuracy'][epoch] = test()
            if dist.get_rank() == net.rank_list[-1]:
                print(f"Epoch {epoch}, loss {trial_data['epoch_loss'][epoch]}, accuracy {trial_data['epoch_accuracy'][epoch]}")

            if dist.get_rank() == net.rank_list[-1]:
                print(f'Epoch {epoch} loss: {trial_data["epoch_loss"][epoch]} accuracy: {trial_data["epoch_accuracy"][epoch]}')

                trial_data['epoch_usage_times'][epoch] = net.f_time + net.g_time + net.subdomain.f_time + net.subdomain.g_time
                trial_data['epoch_num_f_evals'][epoch] = net.num_f_evals
                trial_data['epoch_num_g_evals'][epoch] = net.num_g_evals
                trial_data['epoch_num_sf_evals'][epoch] = net.subdomain.num_f_evals
                trial_data['epoch_num_sg_evals'][epoch] = net.subdomain.num_g_evals

        # if dist.get_rank() == net.rank_list[-1][0]:
        #     np.savez(filename, **trial_data) # Save trial as an npz file
    
if __name__ == '__main__':
    try:
        operative_system = os.environ['OS']
    except:
        operative_system = 'Linux'
    if 'Windows' not in operative_system:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        world_size = 3
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)







