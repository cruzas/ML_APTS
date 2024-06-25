import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
# Add the path to the sys.path
import sys
# Make the following work on Windows and MacOS
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.utility import prepare_distributed_environment, create_dataloaders


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    print(f"World size: {dist.get_world_size()}")
    rank = dist.get_rank() if dist.get_backend() == 'nccl' else rank

    torch.manual_seed(0)
    model = ConvNet()
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=2,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            # print(f"Rank {rank} - Epoch {epoch} - Images {images} - Labels {labels}")
            images = images
            labels = labels
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Flatten the model gradien locally
            

            print(f"EPOCH {epoch} Rank {rank} - Layer 1 - Conv2d - Weight norm: {torch.norm(model.layer1[0].weight.grad.data.flatten())}")


            if (i + 1) % 100 == 0 and rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 3, i + 1, total_step,
                                                                        loss.item()))
    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    torch.manual_seed(1)
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