import os
import torch
import torch.nn as nn
import deepspeed
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

deepspeed_args = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 2
}

class SyntheticDataset(Dataset):
    def __init__(self, size, input_dim, output_dim):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, output_dim, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def main(rank, master_addr, master_port, world_size):
    model = SimpleNN()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=deepspeed_args,  # assuming deepspeed CLI args are parsed
        model=model,
        model_parameters=model.parameters()
    )

    # Parameters for dataset
    dataset_size = 1000  # number of data points in the dataset
    input_dim = 10       # input dimension
    output_dim = 10      # output dimension (for classification, number of classes)

    # Create the dataset
    synthetic_dataset = SyntheticDataset(dataset_size, input_dim, output_dim)

    # DataLoader
    batch_size = 32
    dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)

    # Loss function is l2 error
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    
    for data, labels in dataloader:
        data, labels = data.to(model_engine.local_rank), labels.to(model_engine.local_rank)
        optimizer.zero_grad()  # Reset gradients
        output = model_engine(data)
        loss = loss_fn(output, labels)  # Define your loss function (e.g., nn.CrossEntropyLoss)
        model_engine.backward(loss)
        model_engine.step()

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