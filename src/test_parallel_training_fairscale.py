import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from utils.utility import prepare_distributed_environment

# Define your neural network architectures
class ModelEngine1(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(ModelEngine1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out

class ModelEngine2(nn.Module):
    def __init__(self, num_classes=10):
        super(ModelEngine2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512) # Adjust the dimensions according to your input size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16) # Flatten and adjust dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Set up the distributed environment.
    prepare_distributed_environment(rank, master_addr, master_port, world_size)

    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if master_addr is None:
        master_addr = os.environ["MASTER_ADDR"]
    if master_port is None:
        master_port = os.environ["MASTER_PORT"]
    if world_size is None: 
        world_size = dist.get_world_size() if dist.is_initialized() else 1

    model_engine1 = ModelEngine1()
    model_engine2 = ModelEngine2()

    # Define process groups
    group1 = dist.new_group([0, 1]) 
    if dist.get_world_size() > 2 or torch.cuda.device_count() > 2:
        group2 = dist.new_group([2, 3])

    # Wrap models with Fairscale's FullyShardedDataParallel
    fsdp_model1 = FSDP(model_engine1, process_group=group1)
    if dist.get_world_size() > 2 or torch.cuda.device_count() > 2:
        fsdp_model2 = FSDP(model_engine2, process_group=group2)

    # For ModelEngine1 (e.g., MNIST)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST(root='./data', train=True, download=True, transform=transform1)
    dataset1_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform1)
    train_loader1 = DataLoader(dataset1, batch_size=64, shuffle=True)
    val_loader1 = DataLoader(dataset1_test, batch_size=64, shuffle=False)

    if dist.get_world_size() > 2 or torch.cuda.device_count() > 2:
        # For ModelEngine2 (e.g., CIFAR-10)
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset2 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform2)
        dataset2_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform2)
        train_loader2 = DataLoader(dataset2, batch_size=64, shuffle=True)
        val_loader2 = DataLoader(dataset2_test, batch_size=64, shuffle=False)


    # Define optimizers
    optimizer1 = optim.Adam(fsdp_model1.parameters(), lr=0.001)
    criterion1 = nn.CrossEntropyLoss()

    if dist.get_world_size() > 2 or torch.cuda.device_count() > 2:
        optimizer2 = optim.Adam(fsdp_model2.parameters(), lr=0.001)
        criterion2 = nn.CrossEntropyLoss()

    # Training loop
    num_epochs=10
    # Assuming you have defined optimizers (optimizer1 for model_engine1, optimizer2 for model_engine2)
    # and loss functions (criterion1 for model_engine1, criterion2 for model_engine2) elsewhere in your code

    for epoch in range(num_epochs):
        
        # Training steps for model_engine1 in ranks 0 and 1
        if dist.get_rank() in [0, 1]:
            fsdp_model1.train()
            for data, target in train_loader1:  # Assuming train_loader1 is your DataLoader for fsdp_model1
                # Transfer data to the appropriate device (e.g., GPU)
                if dist.get_backend() == "nccl":
                    data, target = data.to('cuda:0'), target.to('cuda:0')
                else:
                    data, target = data.to('cuda:{}'.format(dist.get_rank())), target.to('cuda:{}'.format(dist.get_rank()))
                
                optimizer1.zero_grad()  # Zero the gradient buffers
                output = fsdp_model1(data)  # Forward pass
                loss = criterion1(output, target)  # Compute loss
                loss.backward()  # Backward pass
                optimizer1.step()  # Update weights

                # Validation step for fsdp_model1
                fsdp_model1.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader1:
                        data, target = data.to('cuda:0'), target.to('cuda:0')
                        outputs = fsdp_model1(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                accuracy = 100 * correct / total
                print(f'MNIST - Epoch {epoch+1}, Accuracy: {accuracy}%')
            


        # Training steps for model_engine2 in ranks 2 and 3
        if dist.get_rank() in [2, 3]:
            fsdp_model2.train()
            for data, target in train_loader2:  # Assuming train_loader2 is your DataLoader for fsdp_model2
                # Transfer data to the appropriate device
                if dist.get_backend() == "nccl":
                    data, target = data.to('cuda:0'), target.to('cuda:0')
                else:
                    data, target = data.to('cuda:{}'.format(dist.get_rank())), target.to('cuda:{}'.format(dist.get_rank()))
                
                optimizer2.zero_grad()  # Zero the gradient buffers
                output = fsdp_model2(data)  # Forward pass
                loss = criterion2(output, target)  # Compute loss
                loss.backward()  # Backward pass
                optimizer2.step()  # Update weights

                # Validation step for model_engine1
                fsdp_model2.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader2:
                        if dist.get_backend() == "nccl":
                            data, target = data.to('cuda:0'), target.to('cuda:0')
                        else:
                            data, target = data.to('cuda:{}'.format(dist.get_rank())), target.to('cuda:{}'.format(dist.get_rank()))
                        outputs = fsdp_model2(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                accuracy = 100 * correct / total
                print(f'CIFAR - Epoch {epoch+1}, Accuracy: {accuracy}%')


    # Clean up
    dist.destroy_process_group()



if __name__ == "__main__":  
    if 1==2:
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        master_addr = 'localhost'
        master_port = '12345'  
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)