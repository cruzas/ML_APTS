
import os
import argparse
import torch
import torch.nn as nn
import deepspeed
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from utils.utility import prepare_distributed_environment

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

def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    parser.add_argument(
        '--dtype',
        default='fp16',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help=
        'Datatype used for training'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    deepspeed_args = {
    "train_micro_batch_size_per_gpu": 8,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 2
    }

    args.update(deepspeed_args)
    return args

def create_moe_param_groups(model):
    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)


def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # prepare_distributed_environment(rank, master_addr, master_port, world_size)
    deepspeed.init_distributed()
    ds_config = {
    "train_batch_size": 16,
    }

    deepspeed_args = add_argument()

    model = SimpleNN()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if deepspeed_args.moe_param_group:
        parameters = create_moe_param_groups(model)

    # Parameters for dataset
    dataset_size = 1000  # number of data points in the dataset
    input_dim = 10       # input dimension
    output_dim = 10      # output dimension (for classification, number of classes)

    # Create the dataset
    synthetic_dataset = SyntheticDataset(dataset_size, input_dim, output_dim)


    # Initialize DeepSpeed
    model_engine, optimizer, dataloader, __ = deepspeed.initialize(args=deepspeed_args, 
                                                          model=model, 
                                                          model_parameters=parameters, 
                                                          training_data=synthetic_dataset, 
                                                          config=ds_config)

    # Loss function is l2 error
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()

    for data, labels in dataloader:
        data, labels = data.to(model_engine.local_rank), labels.to(model_engine.local_rank)
        optimizer.zero_grad()  # Reset gradients
        output = model_engine(data)
        loss = loss_fn(output, labels)  # Define your loss function (e.g., nn.CrossEntropyLoss)
        print(f"Rank {dist.get_rank()}. Loss: {loss}")
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
