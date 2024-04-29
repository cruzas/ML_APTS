import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.utility import *
import multiprocessing
import os

# Define the training function
def train(model, device_id, train_loader, optimizer, criterion, num_epochs):
    torch.cuda.set_device(device_id)
    stream = torch.cuda.Stream(device=device_id)
    with torch.cuda.stream(stream):
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device_id), labels.to(device_id)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Device: {device_id}, Loss: {loss.item()}")
    torch.cuda.synchronize(device_id)


if __name__ == "__main__":
    # Check if two GPUs are available
    # if torch.cuda.device_count() < 2:
    #     raise RuntimeError("This code requires at least 2 GPUs")

    # Neural Network, Data Loaders, and other setups go here...
    # Simple Neural Network Definition
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)


    # Initialize two models and move them to separate GPUs
    model1 = SimpleNet().cuda(0)
    model2 = SimpleNet().cuda(1)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    # Load Data (e.g., MNIST)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Data loading for GPU 0
    train_loader1, test_loader1 = create_dataloaders(
        dataset='MNIST',
        data_dir=os.path.abspath("./data"),
        mb_size=64,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:0'
    )

    # Data loading for GPU 1
    train_loader2, test_loader2 = create_dataloaders(
        dataset='MNIST',
        data_dir=os.path.abspath("./data"),
        mb_size=64,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:1'
    )

    num_epochs = 100

    # Create and start multiprocessing processes
    processes = []
    process1 = multiprocessing.Process(target=train, args=(model1, 0, train_loader1, optimizer1, criterion, num_epochs))
    process2 = multiprocessing.Process(target=train, args=(model2, 1, train_loader2, optimizer2, criterion, num_epochs))
    processes.append(process1)
    processes.append(process2)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('Finished Training')

