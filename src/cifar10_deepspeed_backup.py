import os
import subprocess
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

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
                        default=10000,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=1,
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

    return args



args = add_argument()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if args.moe:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                # create moe layers based on the number of experts
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=fc3,
                        num_experts=n_e,
                        ep_size=args.ep_world_size,
                        use_residual=args.mlp_type == 'residual',
                        k=args.top_k,
                        min_capacity=args.min_capacity,
                        noisy_gate_policy=args.noisy_gate_policy))
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if args.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x



def create_moe_param_groups(model):
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

        parameters = {
            'params': [p for p in model.parameters()],
            'name': 'parameters'
        }

        return split_params_into_different_moe_groups_for_optimizer(parameters)



def trial():
    # Set seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2)
    net = Net()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    if args.moe_param_group:
        parameters = create_moe_param_groups(net)
    

    ds_config = {
    "train_batch_size": 256,  
    "steps_per_print": 100,  
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [0.8, 0.999],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    "wall_clock_breakdown": False,
    "zero_optimization": {
        "stage": 0,  
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": False,  
        "contiguous_gradients": True,
        "cpu_offload": False
    }
    }
    model_engine, _, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config)

    local_device = get_accelerator().device_name(model_engine.local_rank)

    # For float32, target_dtype will be None so no datatype conversion needed
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype=torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype=torch.half

    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    cumulative_times_s = []  # Array for cumulative times
    memory_usage_gb = []  # Array for memory usage
    for epoch in range(1, args.epochs + 1):  # loop over the dataset multiple times
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as prof:
            running_loss = 0.0
            count = 0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(local_device), data[1].to(local_device)
                if target_dtype != None:
                    inputs = inputs.to(target_dtype)
                outputs = model_engine(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                count += 1

                model_engine.backward(loss)
                model_engine.step()

            avg_loss = running_loss / count

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if target_dtype != None:
                    images = images.to(target_dtype)
                outputs = net(images.to(local_device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(local_device)).sum().item()
            accuracy = 100 * correct / total

        # Append to arrays
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Extract profiling metrics
        total_cuda_time_us = prof.total_average().cuda_time_total # in micro-seconds
        total_cuda_memory_usage_bytes = prof.total_average().self_cuda_memory_usage # in bytes
        # Convert and accumulate
        total_cuda_time_s = total_cuda_time_us / 1e6 # in seconds
        total_cuda_memory_usage_gb = total_cuda_memory_usage_bytes / 1e9 # in gigabytes

        # If it's the first epoch, initialize cumulative time
        if epoch == 1:
            cumulative_cuda_time_s = total_cuda_time_s
        else:
            cumulative_cuda_time_s += total_cuda_time_s

        # Append to arrays
        cumulative_times_s.append(cumulative_cuda_time_s)
        memory_usage_gb.append(total_cuda_memory_usage_gb)

    torch.distributed.barrier()
    return losses, accuracies, cumulative_times_s, memory_usage_gb



def main():
    num_trials = 3
    all_losses = []
    all_accuracies = []
    all_cumulatime_times_s = []
    all_memory_usage_gb = []
    for trial in range(1, num_trials+1):
        print(f"Trial {trial} of {num_trials}")
        losses, accuracies, cumulative_times_s, memory_usage_gb = trial()

        if torch.distributed.get_rank() == 0:
            all_losses.append(losses)
            all_accuracies.append(accuracies)
            all_cumulatime_times_s.append(cumulative_times_s)
            all_memory_usage_gb.append(memory_usage_gb)
            for i in range(len(losses)):
                print(f"Trial {trial}. Epoch {i + 1}: {losses[i]:.5f} {accuracies[i]:.2f}% {cumulative_times_s[i]:.3f} {memory_usage_gb[i]:.5f}")

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        # Average all_losses element-wise
        losses = np.mean(all_losses, axis=0)
        accuracies = np.mean(all_accuracies, axis=0)
        cumulative_times_s = np.mean(all_cumulatime_times_s, axis=0)
        memory_usage_gb = np.mean(all_memory_usage_gb, axis=0)

        # Save to CSV    
        print("Saving results to CSV...")
        results_df = pd.DataFrame({
            'Epoch': range(1, args.epochs + 1),
            'Loss': losses,
            'Accuracy': accuracies,
            'Cumulative CUDA Time (s)': cumulative_times_s,
            'Total CUDA Memory Usage (GB)': memory_usage_gb
        })

        csv_file_name = f"cifar10_deepspeed_{torch.distributed.get_world_size()}.csv"
        results_df.to_csv(csv_file_name, index=False)
        print(f"Results saved to {csv_file_name}")

    torch.distributed.barrier()
    print(f"Rank {torch.distributed.get_rank()} should be exiting now")
    exit(0)




if __name__ == "__main__":
    print("Initializing deepspeed...")
    deepspeed.init_distributed()
    print("Finished initializing deepspeed...")
    main()
