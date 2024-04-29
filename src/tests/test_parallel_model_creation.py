import torch
import threading
import torch.nn as nn
import torch.nn.functional as F

class MNIST_FCNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[1000, 1000, 1000, 1000], output_size=10):
        super(MNIST_FCNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.l5 = nn.Linear(hidden_sizes[3], output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = torch.sigmoid(self.l5(x))
        # x = torch.sigmoid(self.l2(x))
        x = F.log_softmax(x, dim=1)
        return x

def init_model_on_gpu(models, gpu_id):
    torch.cuda.set_device(gpu_id)
    torch.set_default_device(f'cuda:{gpu_id}')
    with torch.cuda.device(gpu_id):
        model = MNIST_FCNN().to(f'cuda')
    models.append((gpu_id, model))

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Ensure you have GPUs available.")

    models = []
    threads = []

    for i in range(num_gpus):
        thread = threading.Thread(target=init_model_on_gpu, args=(models, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    models.sort(key=lambda x: x[0])  # Sort by GPU ID
    models = [model for _, model in models]  # Extract only model references

    return models

models = main()
for i in range(len(models)):
    print(f"Model {i} is on GPU {next(models[i].parameters()).device}")
print(models)  # This will print the list of neural network references
