import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.autograd as autograd

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Adjust input size to 784
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 10)    # Adjust output size to 10

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the image
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def train():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    # Create model
    model = MyModel()
    if torch.cuda.is_available():
        model = model.cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(1):  # Number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # Flatten inputs
            inputs = inputs.view(-1, 784)

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss_old = criterion(outputs, labels)
            loss_old.backward()
            grad0 = [param.grad for param in model.parameters()]
            grad = torch.cat([param.grad.flatten() for param in model.parameters()])
            # optimizer.step()
            
            model.zero_grad()
            output = model(inputs)
            loss = torch.nn.functional.cross_entropy(output, labels)
            
            # grad2 = autograd.grad(outputs=loss, inputs=model.parameters())
            grad2 = autograd.grad(outputs=loss, inputs=model.parameters(), grad_outputs=loss_old)
            concat_grad2 = torch.cat([grad.flatten() for grad in grad2])
            # partial_derivatives = autograd.grad(outputs=output, inputs=model.parameters(), grad_outputs=grad, only_inputs=True)[0]

            print(torch.norm(concat_grad2 - grad))



            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')


# Example usage
train()
