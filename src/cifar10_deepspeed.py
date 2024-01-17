import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.profiler import profile, ProfilerActivity
import pandas as pd

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


deepspeed.init_distributed()

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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F

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


net = Net()


def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
ds_config = {
  "train_batch_size": 16,
  "steps_per_print": 10000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
#   "scheduler": {
#     "type": "WarmupLR",
#     "params": {
#       "warmup_min_lr": 0,
#       "warmup_max_lr": 0.001,
#       "warmup_num_steps": 1
#     }
#   },
#   "gradient_clipping": 1.0,
#   "prescale_gradients": False,
  "bf16": {
      "enabled": args.dtype == "bf16"
  },
  "fp16": {
      "enabled": args.dtype == "fp16",
      "fp16_master_weights_and_grads": False,
      "loss_scale": 0,
      "loss_scale_window": 10000,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": args.stage,
      "allgather_partitions": True,
      "reduce_scatter": True,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "cpu_offload": False
  }
}

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config)

local_device = get_accelerator().device_name(model_engine.local_rank)
local_rank = model_engine.local_rank

# For float32, target_dtype will be None so no datatype conversion needed
target_dtype = None
if model_engine.bfloat16_enabled():
    target_dtype=torch.bfloat16
elif model_engine.fp16_enabled():
    target_dtype=torch.half

criterion = nn.CrossEntropyLoss()
profiling_results = []
losses = []
accuracies = []
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

        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Add profiling result of the current epoch to the list
        profiling_results.append(prof.key_averages())

# After all epochs, print profiling results
if local_rank == 0:
    print('Finished Training')
    print('Printing training results...')
    for i in range(len(losses)):
        # Print loss to 5 decimal places and accuracy as a percentage
        print(f"Epoch {i+1} Loss: {losses[i]:.5f} Accuracy: {accuracies[i]:.2f}%")

    print('Printing profiling results...')
    for epoch, prof in enumerate(profiling_results, 1):
        print(f"Epoch {epoch} Profiling Results:")
        print(prof.table(sort_by="cuda_time_total", row_limit=-1))
        print(prof.table(sort_by="self_cuda_memory_usage", row_limit=-1))
        print("\n")

    # Save results to CSV
    print("Saving results to CSV...")
    # Create a DataFrame for losses and accuracies
    results_df = pd.DataFrame({
        'Epoch': range(1, args.epochs + 1),
        'Loss': losses,
        'Accuracy': accuracies
    })

    # Extract key metrics from profiling data
    for epoch, prof in enumerate(profiling_results, 1):
        results_df.loc[epoch - 1, 'Total CUDA Time (s)'] = prof.total_average().cuda_time_total
        results_df.loc[epoch - 1, 'Total CUDA Memory Usage (Bytes)'] = prof.total_average().self_cuda_memory_usage

    # Save to CSV
    world_size = torch.distributed.get_world_size()
    csv_file_name = f"cifar10_deepspeed_{world_size}.csv"
    results_df.to_csv(csv_file_name, index=False)
    print(f"Results saved to {csv_file_name}")

