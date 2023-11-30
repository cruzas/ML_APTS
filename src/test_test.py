import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.utility import *

if __name__ == "__main__":
    # Check if two GPUs are available
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This code requires at least 2 GPUs")

    # Simple Neural Network Definition
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Load Data (e.g., MNIST)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Data loading
    train_loader1, test_loader1 = create_dataloaders(
        dataset='MNIST',
        data_dir=os.path.abspath("./data"),
        mb_size=64,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:0'
    )
    # Data loading
    train_loader2, test_loader2 = create_dataloaders(
        dataset='MNIST',
        data_dir=os.path.abspath("./data"),
        mb_size=64,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:1'
    )

    # Initialize two models and move them to separate GPUs
    model1 = SimpleNet().cuda(0)
    model2 = SimpleNet().cuda(1)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    stream1 = torch.cuda.Stream(device=0)
    stream2 = torch.cuda.Stream(device=1)

    num_epochs = 100
    # Train on GPU 1
    with torch.cuda.stream(stream1):
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            for i, (inputs, labels) in enumerate(train_loader1):
                # Split data between the two GPUs
                inputs1, labels1 = inputs.cuda(0), labels.cuda(0)
                optimizer1.zero_grad()
                outputs1 = model1(inputs1)
                loss1 = criterion(outputs1, labels1)
                loss1.backward()
                optimizer1.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader1.indices)}], Loss1: {loss1.item()}")

    with torch.cuda.stream(stream2):
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            for i, (inputs, labels) in enumerate(train_loader2):
                # Split data between the two GPUs
                inputs2, labels2 = inputs.cuda(1), labels.cuda(1)
                optimizer2.zero_grad()
                outputs2 = model2(inputs2)
                loss2 = criterion(outputs2, labels2)
                loss2.backward()
                optimizer2.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader2.indices)}], Loss2: {loss2.item()}")

    # Wait for all computations to complete
    torch.cuda.synchronize()



    # print(f"Epoch [{epoch+1}/{2}], Step [{i+1}/{len(train_loader)}], Loss1: {loss1.item()}, Loss2: {loss2.item()}")

    print('Finished Training')