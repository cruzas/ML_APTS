import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
                        default=50000,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=200,
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
        default=3,
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

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

args = add_argument()
mb_size = args.batch_size

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()


torch.manual_seed(0)
torch.cuda.manual_seed(0)
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=1,
                                          drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=int(10000),
                                         shuffle=False,
                                         num_workers=1,
                                         drop_last=True)

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).




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

# Set seed
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

# net = Net()
print(f"Rank {torch.distributed.get_rank()} creating ResNet18...")
net = models.resnet18(pretrained=False)
print("Rank {torch.distributed.get_rank()} ResNet18 created")


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
  "train_batch_size": args.batch_size,
  "steps_per_print": 100,
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
  "bf16": {
      "enabled": args.dtype == "bf16"
  },
  "fp16": {
      "enabled": args.dtype == "fp16",
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
rank = torch.distributed.get_rank()

print(f"Rank: {rank}. Local device: {local_device}")

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
memory_allocated_gb = []  # Array for memory allocated
print(f"Rank {rank} started training...") 
for epoch in range(1, args.epochs+1):  # loop over the dataset multiple times
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as prof:
        epoch_loss = 0.0
        count = 0
        avg_mem_allocated_gb = 0
        print(f"Epoch {epoch}. Rank {rank}. Training...")
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            if target_dtype != None:
                inputs = inputs.to(target_dtype)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            avg_mem_allocated_gb += (torch.cuda.memory_allocated() / 1e9)

            model_engine.backward(loss)
            model_engine.step()
            
            # print statistics
            epoch_loss += loss.item()
            count += 1
        epoch_loss /= count 
        avg_mem_allocated_gb /= count
    
    print(f"Epoch {epoch}. Rank {rank}. Computing accuracy...")
    # Compute accuracy after epoch has concluded
    correct = 0
    total = 0
    with torch.no_grad():
        counter2 = 0
        for data in testloader:
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 1")
            images, labels = data
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 2")
            if target_dtype != None:
                print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 2a")
                images = images.to(target_dtype)
                print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 2b")
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 3")
            outputs = net(images.to(local_device))
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 4")
            _, predicted = torch.max(outputs.data, 1)
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 5")
            total += labels.size(0)
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 6")
            correct += (predicted == labels.to(local_device)).sum().item()
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 7")
            counter2 += 1
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 8")
        accuracy = correct / total * 100
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 9")
        
        print(f"Epoch {epoch}, rank {rank}, loss {epoch_loss:.4f}, accuracy {accuracy:.2f}%")

        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 9a")
        torch.distributed.barrier()
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 9b")

        # Append to arrays
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 9c")
        losses.append(epoch_loss)
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 9d")
        accuracies.append(accuracy)

        # Extract profiling metrics
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 10")
        # total_cuda_time_s = sum([event.self_cuda_time_total / 1e6 for event in prof.key_averages() if event.device_type == "cuda"])
        total_cuda_time_s = sum([event.cuda_time_total / 1e6 for event in prof.key_averages()])
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 11")
        total_cuda_memory_usage_gb = sum([event.self_cuda_memory_usage / 1e9 for event in prof.key_averages()])
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 12a")

        # If it's the first epoch, initialize cumulative time
        if epoch == 1:
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 12b1")
            cumulative_cuda_time_s = total_cuda_time_s
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 12b2")
        else:
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 12c1")
            cumulative_cuda_time_s += total_cuda_time_s
            print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 12c2")

        # Append to arrays
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 13")            
        cumulative_times_s.append(cumulative_cuda_time_s)
        print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 14")            
        memory_usage_gb.append(total_cuda_memory_usage_gb)
        memory_allocated_gb.append(avg_mem_allocated_gb)
        torch.distributed.barrier()

print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 15")            
torch.distributed.barrier()
print(f'Rank {rank} finished Training')

results_df = pd.DataFrame({
        'Epoch': range(1, args.epochs + 1),
        'Loss': losses,
        'Accuracy': accuracies,
        'Time': cumulative_times_s,
        'Memory_Profiler': memory_usage_gb,
        'Memory_Allocated': memory_allocated_gb,
    })

print(f"Rank {rank} results_df: {results_df}")

# Save to CSV    
if rank == 0:
    print("Saving results to CSV...")
    # TODO: add batch size to filename
    print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 16")            
    csv_file_name = f"cifar10_DS_ws_{torch.distributed.get_world_size()}_mbs_{args.batch_size}_zero_{args.stage}.csv"
    print(f"Epoch {epoch}. Counter {counter2}. Rank {rank}. Print 17")            
    results_df.to_csv(csv_file_name, index=False)
    print(f"Results saved to {csv_file_name}")
