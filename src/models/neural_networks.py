import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MNIST_FCNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[32, 32], output_size=10):
        super(MNIST_FCNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        # x = torch.sigmoid(self.l2(x))
        x = F.log_softmax(x, dim=1)
        return x

class MNIST_FCNN_Small(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[32], output_size=10):
        super(MNIST_FCNN_Small, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = F.log_softmax(x, dim=1)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # 1 input channel, 8 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 window
        self.fc1 = nn.Linear(8 * 13 * 13, 32)  # 8 channels * 12x12 image dimensions
        self.fc2 = nn.Linear(32, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 8 * 13 * 13)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)










class CIFAR_FCNN(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[128, 64], output_size=10):
        super(CIFAR_FCNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Define the ResNet-8 architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AvgPool2d(8)(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
# Source: https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=5)),
            ('batch1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=5)),
            ('batch2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1600, 384)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(384, 192)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(192, 10))
        ]))

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)
    



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self):
        super(SmallResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = BasicBlock(16, 16, 1)
        self.layer2 = BasicBlock(16, 16, 1)
        self.layer3 = BasicBlock(16, 16, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Added global average pooling layer
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)  # Apply global average pooling
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out
