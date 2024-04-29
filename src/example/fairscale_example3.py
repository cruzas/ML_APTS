import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributed.pipeline.sync import Pipe
import torch.multiprocessing as mp
from utils.utility import prepare_distributed_environment

# Assuming CUDA devices are set up and torch.distributed is initialized

import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

# ModelParallelLinear implementation
class ModelParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, split_size, device_ids):
        super(ModelParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_size = split_size
        self.device_ids = device_ids

        # Assuming out_features is divisible by split_size for simplicity
        self.split_weights = nn.ModuleList([
            nn.Linear(in_features, out_features // split_size, bias=False).to(device)
            for device in device_ids
        ])

    def forward(self, x):
        # Split the input and output across devices
        splitted_outputs = [
            layer(x.to(layer.weight.device))
            for layer in self.split_weights
        ]

        # Concatenate the outputs back to the original device
        y = torch.cat([o.to(x.device) for o in splitted_outputs], dim=1)
        return y

# PipelineStage implementation
class PipelineStage(nn.Module):
    def __init__(self, in_features, out_features, split_size, device_ids):
        super(PipelineStage, self).__init__()
        self.model_parallel_linear = ModelParallelLinear(in_features, out_features, split_size, device_ids)

    def forward(self, x):
        return self.model_parallel_linear(x)

# MNISTPipelinedModel implementation
class MNISTPipelinedModel(nn.Sequential):
    def __init__(self):
        super(MNISTPipelinedModel, self).__init__(
            PipelineStage(28*28, 512, split_size=4, device_ids=['cuda:0','cuda:1']),
            nn.ReLU(),
            PipelineStage(512, 10, split_size=2, device_ids=['cuda:2','cuda:3'])
        )

# Instantiate the pipelined model
pipelined_model = MNISTPipelinedModel()

# Wrap the model with Pipe to enable pipelining
pipelined_model = Pipe(pipelined_model, chunks=8)

# MNIST Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pipelined_model.parameters(), lr=0.01, momentum=0.9)

# Training loop
def train(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            # Forward pass through the pipeline
            output = model(data)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def main(rank, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment()
    # Execute the training loop
    train(pipelined_model, train_loader, criterion, optimizer, epochs=5)


if __name__ == "__main__":
    if "snx" in os.getcwd():
        main()
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if world_size == 0:
            print("No CUDA device(s) detected.")
            exit(0)
        master_addr = 'localhost'
        master_port = '12345'
        mp.spawn(main, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)