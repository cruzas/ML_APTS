from torch.multiprocessing import Process, set_start_method, Queue
import torch, time, os, copy, pickle
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils.utility import *
from models.neural_networks import *

devices = torch.cuda.device_count()
streams = [None]*devices
models = [None]*devices
for device_id in range(devices):
    streams[device_id] = torch.cuda.Stream(device_id)
torch.cuda.synchronize()
    
def create_sub_model(local_model, rank, world_size):
    '''
    Creates a submodel with trainable layers distributed across ranks.
    Raises an error if the number of ranks exceeds the number of parameter groups.
    '''
    # TODO: For efficiency, change the following (E.g. choose a world size that is just enough, i.e. with every GPU needed filled)
    # TODO: If enough processes are availalb and each layer is not big enough, distribute even more layers per process...or something like that
    local_model = copy.deepcopy(local_model)
    tot_layers = len(list(local_model.parameters()))
    # tot_params = sum([p.numel() for p in local_model.parameters()])
    if tot_layers < world_size:
        raise ValueError(f"Too many GPUs ({world_size}) for {tot_layers} subdomains. Crashing the code to avoid wasting computational time.")
    
    # index_shuffled = torch.randperm(tot_layers)
    index_shuffled = list(range(tot_layers))

    trainable_layers = []
    params_per_subset = [0]*world_size
    for i in range(world_size):
        params_per_subset[i] = int(tot_layers / (world_size-i))
        if i == world_size - 1:
            trainable_layers.append(index_shuffled[sum(params_per_subset[:i]):])
        else:
            trainable_layers.append(index_shuffled[sum(params_per_subset[:i]):sum(params_per_subset[:i+1])])
        tot_layers -= params_per_subset[i]

    for index, param in enumerate(local_model.parameters()):
        param.requires_grad = index in trainable_layers[rank]

    dict = {k: v for k, v in enumerate(trainable_layers)}
    del trainable_layers # TODO: do this in a more memory-efficient way next time
    layer2rank = {v: k for k, lst in dict.items() for v in lst}

    return local_model, layer2rank

def process(device_id, queue, model, optimizer, train_loader, test_loader, num_epochs):
    global streams
    print(f'IM HERE {device_id} -> model device {next(model.parameters()).get_device()}\n')
    
    stream = streams[device_id]

    with torch.cuda.stream(stream):
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device_id), labels.to(device_id)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # print(f"Epoch [{epoch+1}/{num_epochs}], Device: {device_id}, Loss: {loss.item()}")
    
    queue.put([device_id, pickle.dumps(model), num_epochs])



def process2(device_id):
    global streams, models   
    stream = streams[device_id]

    with torch.cuda.stream(stream):
        models[device_id] = MNIST_FCNN()
        # print(f"Device before: {models[device_id].get_device()}")
        models[device_id] = models[device_id].cuda(device_id)
        # print(f"Device after: {models[device_id].get_device()}")

                
if __name__ == "__main__":
    # global models
    
    torch.cuda.synchronize()
    num_epochs=2

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
    # Data loading
    train_loaders = [None]*devices
    test_loaders = [None]*devices
    models  = [None]*devices
    optimizers = [None]*devices
    layers2ranks = [None]*devices
    main_model = MNIST_FCNN()

    for device_id in range(devices):
        train_loaders[device_id], test_loaders[device_id] = create_dataloaders(dataset='MNIST',
        data_dir=os.path.abspath("./data"),
        mb_size=64,
        overlap_ratio=0,
        parameter_decomposition=True,
        device=f'cuda:{device_id}')
        models[device_id], layers2ranks[device_id] = create_sub_model(main_model, device_id, devices)
        models[device_id] = models[device_id].cuda(device_id)
        optimizers[device_id] = optim.Adam(models[device_id].parameters(), lr=0.001)

    set_start_method('spawn', force = True)
    start = time.time()
    
    queue = Queue()
    p1 = Process(target = process, args = (0, queue, models[0], optimizers[0], train_loaders[0], test_loaders[0], num_epochs))
    p2 = Process(target = process, args = (1, queue, models[1], optimizers[1], train_loaders[1], test_loaders[1], num_epochs))
            
    # p1 = Process(target = process2, args = (0,))
    # p2 = Process(target = process2, args = (1,))
    
    p1.start()
    p2.start()

    results = {}
    for k in range(devices):
        device_id,result,num = queue.get()
        # print(result)
        results[device_id] = pickle.loads(result)

        print(f"I'm here again {device_id} -> model device {next(results[device_id].parameters()).get_device()}\n")

    p1.join()
    p2.join()



    torch.cuda.synchronize()
    
    # print(next(models[0].parameters()).get_device())
    # print(next(models[1].parameters()).get_device())


    # Vectorize the main_model parameters and print the 2 norm of the vector
    print("Global norm before: ", torch.norm(torch.tensor(torch.cat([p.data.flatten() for p in main_model.parameters()]))))
    
    p=[list(models[0].parameters()), list(models[1].parameters())]
    
    for i in range(devices):
        with torch.cuda.stream(streams[i]):
            for ii, param in enumerate(main_model.parameters()):
                if p[i][ii].requires_grad is True:
                    param.data.copy_(p[i][ii].data.clone().detach().to(param.device))

    print("Global norm after: ", torch.norm(torch.tensor(torch.cat([p.data.flatten() for p in main_model.parameters()]))))

    print(f'IM HERE {next(models[0].parameters()).get_device()}\n')
    print(f'IM HERE {next(models[1].parameters()).get_device()}\n')
    

            