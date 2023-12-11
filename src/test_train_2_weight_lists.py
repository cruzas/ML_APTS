import torch,os
from utils.utility import create_dataloaders
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network for MNIST
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network
torch.set_default_device('cuda:1')
torch.random.manual_seed(0)
model = SimpleNN().to('cuda:1')
# torch.random.manual_seed(0)
model2 = SimpleNN().to('cuda:1')
# torch.random.manual_seed(0)

# Define two sets of parameters
params_set1 = [param.clone().detach() for param in model.parameters()]
print(f'Epochs {-1} - Model 1 - {torch.norm(torch.cat([param.flatten() for param in params_set1]))}')
params_set2 = [param.clone().detach() for param in model2.parameters()]
print(f'Epochs {-1} - Model 2 - {torch.norm(torch.cat([param.flatten() for param in params_set2]))}')

# params_set2 = [torch.randn_like(param)/100 for param in model.parameters()]
# print(f'Epochs {-1} - Model 2 - {torch.norm(torch.cat([param.flatten() for param in params_set]))}')
# params_set1 = [torch.randn_like(param)/100 for param in model.parameters()]
# print(f'Epochs {-1} - Model 2 - {torch.norm(torch.cat([param.flatten() for param in params_set]))}')


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
train_loader, test_loader = create_dataloaders(
    dataset="MNIST",
    data_dir=os.path.abspath("./data"),
    mb_size=60000,
    overlap_ratio=0,
    parameter_decomposition=True,
    device='cuda:1'
)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Function to update model parameters
def update_model_parameters(model, new_params):
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), new_params):
            param.copy_(new_param)
            param.requires_grad = True
# def update_optimizer(optimizer,new_params):
#     with torch.no_grad():
#         for param, new_param in zip(optimizer.param_groups[0]['params'], new_params):
#             param.copy_(new_param)
#             param.requires_grad = True
# Training loop
epochs = 100
for epoch in range(epochs):
    for data, target in train_loader:
        # Train with parameters from set 1
        update_model_parameters(model, params_set1)
        # update_optimizer(optimizer, params_set1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        params_set1 = [param.clone().detach() for param in model.parameters()]
        print(f'Epochs {epoch} - Model 1 - loss {loss} - param norm {torch.norm(torch.cat([param.flatten() for param in model.parameters()]))}')

        # Train with parameters from set 2
        update_model_parameters(model, params_set2)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        params_set2 = [param.clone().detach() for param in model.parameters()]
        print(f'Epochs {epoch} - Model 2 - loss {loss} - param norm {torch.norm(torch.cat([param.flatten() for param in model.parameters()]))}')

# # Print final parameters from both sets
# print("Final parameters from set 1:")
# for param in params_set1:
#     print(param)

# print("\nFinal parameters from set 2:")
# for param in params_set2:
#     print(param)