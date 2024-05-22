import torch, os, time
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
import pandas as pd
import numpy as np

# Make a function that reads arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description='Parallel test')
    # We will have optimizer name, batch size, learning rate, number of trials, number of epochs
    parser.add_argument('--optimizer', type=str, default='APTS', help='Optimizer name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    return parser.parse_args()

def get_dataset(dataset):
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), \
               torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'mnist':
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
    optimizer_name, batch_size, learning_rate, trials, epochs, dataset = \
        args.optimizer, args.batch_size, args.lr, args.trials, args.epochs, args.dataset
    filename = f'{optimizer_name}_{dataset}_{batch_size}_{learning_rate}_{epochs}_{trials}.npz'

    # Do experiment only if the filename doesn't already exist
    if not os.path.exists(filename):
        prepare_distributed_environment(rank, master_addr, master_port, world_size)
        rank = dist.get_rank() if dist.get_backend() != 'nccl' else rank
        world_size = dist.get_world_size() if dist.is_initialized() else world_size
        all_trials = {key: np.empty((trials, epochs)) for key in 
                    ['epoch_loss', 'epoch_accuracy', 'epoch_times', 'epoch_usage_times', 
                    'epoch_num_f_evals', 'epoch_num_g_evals', 'epoch_num_sf_evals', 'epoch_num_sg_evals']}
        for trial in range(trials):
            torch.manual_seed(1000*trial + 1)
            trainset, testset = get_dataset(dataset)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

            # Define the device 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

            # Instantiate the model, loss function, and optimizer
            if dataset == 'cifar10':
                net = ResNet(num_layers=2).to(device) # Equivalent to ResNet-18
                for sample in trainloader:
                    break
                
                # Reset trainloader
                trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
                
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
            elif dataset == 'mnist':
                layer_list = [
                (CNNPart1, {}, ((1, 28, 28), (64, 5, 5))),  # Adjust input and output shapes according to the actual sizes
                (CNNPart2, {}, ((64, 5, 5), (128,))),        # Adjust input and output shapes
                (CNNPart3, {}, ((128,), (10,)))              # Adjust input and output shapes
]
            rank_list = [[r] for r in range(world_size)]
            net = Weight_Parallelized_Model(layer_list, rank_list)
            criterion = nn.CrossEntropyLoss()

            # Check if optimizer_name in lower case is 'sgd'
            if optimizer_name.lower() == 'sgd':
                # SGD from torch optim
                optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) # According to https://arxiv.org/pdf/1512.03385
            elif optimizer_name.lower() == 'adam':
                # Adam from torch optim
                optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
            elif optimizer_name.lower() == 'tradam':
                optimizer = TRAdam(net.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == 'apts':
                subdomain_optimizer = TRAdam
                subdomain_optimizer_defaults = {'lr': 1e99}
                global_optimizer = TR
                global_optimizer_defaults = {'lr': learning_rate, 'max_iter': 3}    
                optimizer = APTS(model=net, lr=learning_rate, dogleg=False, subdomain_optimizer=subdomain_optimizer, subdomain_optimizer_defaults=subdomain_optimizer_defaults, criterion=criterion, global_optimizer=global_optimizer, global_optimizer_defaults=global_optimizer_defaults)
            else:
                raise ValueError(f'Optimizer {optimizer_name} not supported')
            
            # Define the learning rate scheduler
            if optimizer_name.lower() == 'sgd':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

            # Training function
            def train(epoch):
                # net.train()
                running_loss = 0.0
                count = 0
                tic = time.time()
                tot_time_load_data = 0
                time_spent_in_optimizer = 0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    tot_time_load_data += time.time() - tic
                    tic = time.time()
                    closuree = closure(inputs, labels, criterion, net, compute_grad=True, zero_grad=True)     
                    loss = optimizer.step(closuree) 
                    time_spent_in_optimizer += time.time() - tic
                    running_loss += loss if loss is not None else 0
                    count += 1
                    # Print every 100 epochs
                    if i % 100 == 99:
                        if dist.get_rank() == net.rank_list[-1][0] and optimizer_name.lower() == 'apts':
                            tot_time = tot_time_load_data + time_spent_in_optimizer
                            print(f'''Epoch: {epoch + 1}, Batch: {i+1} 
                                MODEL avg f_time     = {net.f_time/count:.3f},   MODEL avg g_time     = {net.g_time/count:.3f}, 
                                SUBDOMAIN avg f_time = {net.subdomain.f_time/count:.3f},   SUBDOMAIN avg g_time = {net.subdomain.g_time/count:.3f}, 
                                avg time in optimizer= {time_spent_in_optimizer/count:.3f},   avg time loading data= {tot_time_load_data/count:.3f},
                                avg time per iter    = {tot_time/count:.3f},      Loss: {running_loss / 100:.4f}''')
                            timings = {k : round(v/count,3) for k, v in optimizer.timings.items()}
                            print(timings)
                            timings2 = optimizer.subdomain_optimizer.timings
                            print({k : round(v/count,3) for k, v in timings2.items()})
                            # print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}')
                        # running_loss = 0.0
                    tic = time.time()
                running_loss = running_loss/count
                return running_loss

            # Testing function
            def test():
                # net.eval()
                correct = 0
                total = 0
                accuracy = 0
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
                    accuracy = 100 * correct / total
                    print(f'Accuracy on the 10000 test images: {accuracy:.2f}%')
                return accuracy

            # Train and test the model
            for epoch in range(epochs):  # Number of epochs can be adjusted
                epoch_start = time.time()
                net.zero_counters()
                if optimizer_name.lower() == 'apts':
                    optimizer.zero_timers()
                all_trials['epoch_loss'][trial, epoch] = train(epoch)
                all_trials['epoch_times'][trial, epoch] = time.time() - epoch_start
                
                if dist.get_rank() == net.rank_list[-1][0]:
                    print(f'Epoch {epoch} loss: {all_trials["epoch_loss"][trial, epoch]} accuracy: {all_trials["epoch_accuracy"][trial, epoch]}')

                if optimizer_name.lower() == 'sgd': 
                    scheduler.step()
                
                all_trials['epoch_accuracy'][trial, epoch] = test()
                if dist.get_rank() == net.rank_list[-1][0]:
                    all_trials['epoch_usage_times'][trial, epoch] = net.f_time + net.g_time + net.subdomain.f_time + net.subdomain.g_time
                    all_trials['epoch_num_f_evals'][trial, epoch] = net.num_f_evals
                    all_trials['epoch_num_g_evals'][trial, epoch] = net.num_g_evals
                    all_trials['epoch_num_sf_evals'][trial, epoch] = net.subdomain.num_f_evals
                    all_trials['epoch_num_sg_evals'][trial, epoch] = net.subdomain.num_g_evals
        
        if dist.get_rank() == net.rank_list[-1][0]:
            # Save all trials as an npz file
            np.savez(filename, **all_trials)    
    
if __name__ == '__main__':
    if 1==1:
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







