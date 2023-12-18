from torch.multiprocessing import Process, set_start_method, Queue
import torch, time, os, copy, pickle
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils.utility import *
from models.neural_networks import *
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # Residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def calculate_model_weight_in_gb(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 4  # Assuming 32-bit floats
    total_bytes = total_params * bytes_per_param
    return total_bytes / 1e9  # Convert to GB

def from_params_amount_to_weight_in_gb(params):
    bytes_per_param = 4  # Assuming 32-bit floats
    total_bytes = params * bytes_per_param
    return total_bytes / 1e9  # Convert to GB







class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(CustomSGD, self).__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov))
        self._momentum_buffer = list()
        for group in self.param_groups:
            for p in group['params']:
                self._momentum_buffer.append(torch.zeros_like(p.data))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group, momentum_buffer in zip(self.param_groups, self._momentum_buffer):
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p = d_p.add(group['weight_decay'], p.data)
                if group['momentum'] != 0:
                    momentum_buffer.mul_(group['momentum']).add_(d_p)
                    d_p = momentum_buffer
                p.data.add_(-group['lr'], d_p)
        return loss



def sequential_gradient_computation():
    net = ResNet18()#MNIST_FCNN()
    optimizer = CustomSGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset="MNIST",
        data_dir=os.path.abspath("./data"),
        mb_size=1024,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:1'
    )

    for epoch in range(num_epochs):
        for _, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # grad norm:
            time.sleep(2)
            torch.cuda.empty_cache()
            time.sleep(2)
            print(torch.norm(torch.cat([param.grad.flatten() for param in net.parameters()])))
            # print(inputs.shape, labels.shape)
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = net(inputs)
            time.sleep(2)

            #divide inputs and labels into two batches
            optimizer.zero_grad()
            for i in range(0,2):
                inputs2 = inputs[i*512:(i+1)*512]
                labels2 = labels[i*512:(i+1)*512]
                # print(inputs2.shape, labels2.shape)
                outputs = net(inputs2)
                loss = criterion(outputs, labels2)
                # optimizer.zero_grad()
                loss.backward()
                # grad norm:
                if i==0:
                    g = torch.cat([param.grad.flatten() for param in net.parameters()])
                else:
                    g += torch.cat([param.grad.flatten() for param in net.parameters()])
            torch.cuda.empty_cache()
            time.sleep(2)
            print(torch.norm(g/2))
            print('--------------')





def memory_check(trainable_layer):

    # torch.cuda.empty_cache()
    # time.sleep(0.1)
    net = ResNet18()#MNIST_FCNN()
    # print the weight of the NN
    print(f'Average memory allocated: {torch.cuda.memory_allocated(1)/1e9} GB')
    # print(f'Amount of layers: {len([param for param in net.parameters()])}')
    for i,p in enumerate(net.parameters()):
        p.requires_grad = False
        if i==trainable_layer:
            p.requires_grad = True
            params = torch.prod(torch.tensor(p.shape))

    optimizer = CustomSGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset="MNIST",
        data_dir=os.path.abspath("./data"),
        mb_size=1500,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:1'
    )

    for _, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        mem = torch.cuda.memory_allocated(1)/1e9
        a = time.time()
        loss.backward()
        cpu_time = time.time()-a
        # print(torch.cuda.memory_allocated(1)/1e9)
        # mem=torch.cuda.memory_reserved(1)/1e9
        break

    return mem, cpu_time, params.item()


def memory_needed_for_backward():
    net = ResNet18()#MNIST_FCNN()
    model_weight_gb = calculate_model_weight_in_gb(net)
    optimizer = CustomSGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    train_loader, test_loader = create_dataloaders(
        dataset="MNIST",
        data_dir=os.path.abspath("./data"),
        mb_size=1500,
        overlap_ratio=0,
        parameter_decomposition=True,
        device='cuda:1'
    )
    for _, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        mem=torch.cuda.memory_allocated(1)/1e9
        loss.backward()
        break

    print(from_params_amount_to_weight_in_gb(net.parameters()))
    print('asd')





if __name__ == "__main__":
    num_epochs = 100

    torch.set_default_device(1)
    torch.random.manual_seed(0)

    # sequential_gradient_computation()



    #print the total memory available on the GPU in gigabytes
    print(f'Average memory available: {torch.cuda.get_device_properties(1).total_memory/1e9} GB')
    
    # memory_needed_for_backward()

    M=[];C=[];P=[]
    for i in range(0, 62): #[5]:#
        memory,cpu_time,params = memory_check(i)
        # print(f'Layer {i}: {memory} GB - {cpu_time} s')
        C.append(cpu_time)
        M.append(memory)
        P.append(params)
        
        # print('------------------')
    C=C[1:]
    M=M[1:]
    P=P[1:]
    #plot the results on the same plot with two y-axes
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Memory reserved (GB)', color=color)
    ax1.plot(M, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('CPU time (s)', color=color)  # we already handled the x-label with ax1
    # ax2.plot(C, color=color)# in log scale: 
    ax2.semilogy(C, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.set_ylabel('Number of parameters', color=color)  # we already handled the x-label with ax1
    ax3.semilogy(P, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()






    # sequential_gradient_computation()
