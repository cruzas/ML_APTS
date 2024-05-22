import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from parallel.optimizers import OFFO_TR, ASTR1, TRAdam
from data_loaders.Power_DL import Power_DL
from torch.optim import adagrad


# from parallel.utils import closure

# Define transformations for the training set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
torch.manual_seed(0)
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = Power_DL(train_set, minibatch_size=60000, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = Power_DL(test_set, minibatch_size=10000, shuffle=False)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network, loss function, and optimizer

model = SimpleNN().to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = TRAdam(model.parameters(), lr=0.1)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# optimizer = TRAdam(model.parameters(), lr=0.01)
#optim.Adagrad(model.parameters())#adagrad.Adagrad(model.parameters()) 

# ASTR1(model.parameters())

def closure(inputs,labels):
    optimizer.zero_grad()
    outputs = model(inputs.to('cuda'))
    loss = criterion(outputs, labels.to('cuda'))
    loss.backward()
    return loss

# Training loop
for epoch in range(50):  # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        Closure = lambda x=0: closure(inputs, labels)
        #closure(criterion=criterion, inputs=inputs, targets=labels, model=model)
        
        # Forward pass
        optimizer.step(Closure)
        running_loss  += Closure()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
            # running_loss = 0.0
    print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f} \nTimers: {optimizer.display_avg_timers()}")
    running_loss = 0.0

    # Compute the test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs.to('cuda'))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to('cuda')).sum().item()
    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

print("Finished Training")
