import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_loaders.Power_DL import Power_DL

class ASTR(torch.optim.Optimizer):
    def __init__(self, params, lr=1, epsilon=1e-2, mu=1/2): # mu = 1/2 for Adagrad
        super(ASTR, self).__init__(params, {})
        self.lr = lr
        self.epsilon = epsilon
        self.state_sum = [torch.zeros_like(p) for p in self.param_groups[0]['params']]
        self.mu = mu
    def step(self):
        for i,param in enumerate(self.param_groups[0]['params']):
            if param.grad is None:
                continue
            grad = param.grad.data
            # print(torch.norm(grad))
            self.state_sum[i] += grad.pow(2)
            std = self.state_sum[i].add_(self.epsilon).pow(self.mu)
            param.data -= self.lr * grad / std


def accuracy(output, target):
    preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = preds.eq(target.view_as(preds)).sum().item()
    return correct / len(output)

# Example usage
if __name__ == "__main__":
    example = 2
    if example == 1: # 2D with plot
        pass
    else: # MNIST
        # MNIST Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        train_loader = Power_DL(dataset=train_dataset, shuffle=True, device='cuda', minibatch_size=60000)
        test_loader = Power_DL(dataset=test_dataset, shuffle=False, device='cuda', minibatch_size=10000)


        # Model definition
        class MNISTModel(torch.nn.Module):
            def __init__(self):
                super(MNISTModel, self).__init__()
                self.flatten = torch.nn.Flatten()
                self.fc1 = torch.nn.Linear(28*28, 128)
                self.fc2 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = self.flatten(x)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
            
        model = MNISTModel().cuda()
        optimizer = ASTR(model.parameters(), lr=0.01, mu=1/2)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        epochs = 50; losses = []
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    correct += accuracy(output, target) * len(data)
            print(f"Epoch: {epoch}, Accuracy: {correct / len(test_loader.dataset) *100:.2f}%")
            
        # Plotting
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()