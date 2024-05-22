import torch,torchvision,warnings,time
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms



def change_channel_position(tensor):
    # TODO: change this so that it's adaptive to "modified" datasets
    flattened_tensor = tensor
    if len(tensor.shape)==3: #image Black and white
        flattened_tensor = tensor.unsqueeze(1) # Adds a dimension to the tensor
    elif 3 in tensor.shape[1:] and len(tensor.shape[1:])==3: #image RGB
        flattened_tensor = tensor.permute(0, 3, 1, 2) # Changes the position of the channels
    elif 4 in tensor.shape[1:]: #TODO: video?
        raise ValueError('TODO')
    # Now the shape is correct, check with ---> plt.imshow(flattened_tensor[0,1,:,:].cpu())
    return flattened_tensor

def normalize_dataset(data, mean=[], std=[]):
    if not mean:
        mean = torch.mean(data, dtype=torch.float32) # Calculate the mean and standard deviation of the dataset
    if not std:
        std = torch.std(data)
    data_normalized = (data - mean) / std # Normalize the dataset
    return data_normalized


class Power_DL():
    def __init__(self, 
                 dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                 precision=torch.get_default_dtype(), 
                 overlapping_samples=0, 
                 SHARED_OVERLAP=False, # if True: overlap samples are shared between minibatches
                 mean=[], 
                 std=[]):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.iter = 0
        self.epoch = 0
        self.precision = int(''.join([c for c in str(precision) if c.isdigit()]))
        self.overlap = overlapping_samples
        self.SHARED_OVERLAP = SHARED_OVERLAP
        self.minibatch_amount = int(np.ceil(len(self.dataset)/self.batch_size))
        if self.minibatch_amount == 1:
            if self.overlap !=0:
                print('(Power_DL) Warning: overlap is not used. Only 1 minibatch (full dataset).')
            self.overlap = 0

        if "Subset" in str(dataset.__class__):
            self.dataset = dataset.dataset

        if 'numpy' in str(self.dataset.data.__class__):
            self.dataset.data = torch.from_numpy(self.dataset.data)

        if round(self.overlap) != self.overlap and self.overlap<1 and self.overlap>0: #overlap is a percentage
            self.overlap = int(self.overlap*self.batch_size)
        if self.overlap == self.batch_size:
            raise ValueError('Overlap cannot be equal to the minibatch size, this will generate "mini"batches with the entire dataframe each.')
        elif self.overlap > self.batch_size:
            raise ValueError('Overlap cannot be higher than minibatch size.')
        
        # if 'torch' in str(self.dataset.data.__class__):
            # TODO: Fix this stuff. It's nice for further optimization, but not required for now.
            # number = int(''.join([c for c in str(self.dataset.data.dtype) if c.isdigit()]))
            # type = ''.join([c for c in str(self.dataset.data.dtype) if not c.isdigit()])

            # if 'int' in type and number*2 > self.precision:
            #     warnings.warn(f'\nSpecified precision is too low (for PyTorch). Forcing {min(number*2, 64)}-bit precision.')
            #     self.precision = min(number*2, 64) # precision cannot be higher than 64 

            # if type is not 'float':
            #     if number*2 < self.precision:
            #         warnings.warn(f'\nSpecified precision is too high. Forcing {number*2}-precision.')
            #         self.precision = number
            # self.dataset.data = self.dataset.data.to(self.device)
        # elif 'numpy' in str(self.dataset.data.__class__):
        #     self.dataset.data = torch.from_numpy(self.dataset.data)
        # else:
        #     raise ValueError('Invalid dataset type. Please use a torch or numpy dataset.')
        
        assert 'torch' in str(self.dataset.data.__class__)
        self.dataset.data = self.dataset.data.to(self.device)
        number = int(''.join([c for c in str(self.dataset.data.dtype) if c.isdigit()]))
        if self.precision != number:
            exec(f"self.dataset.data = self.dataset.data.to(torch.float{self.precision})")

        self.dataset.data = change_channel_position(self.dataset.data)
        # self.dataset.data = normalize_dataset(self.dataset.data, mean, std) # Data normalization

        dtype = torch.LongTensor
        if torch.cuda.is_available() and ('MNIST' in str(dataset.__class__) or 'CIFAR' in str(dataset.__class__)):
            dtype = torch.cuda.LongTensor
        elif torch.cuda.is_available() and 'Sine' in str(dataset.__class__):
            dtype = torch.cuda.FloatTensor
        elif not torch.cuda.is_available() and ('MNIST' in str(dataset.__class__) or 'CIFAR' in str(dataset.__class__)):
            dtype = torch.LongTensor
        elif not torch.cuda.is_available() and 'Sine' in str(dataset.__class__):
            if self.precision == 32:
                dtype = torch.float32
            elif self.precision == 64:
                dtype = torch.float64
            elif self.precision == 16:
                dtype = torch.float16

        try:
            # TODO: Copy on cuda to avoid problems with parallelization (and maybe other problems)
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets.cpu())).type(torch.LongTensor).to(self.device) 
            self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets.cpu())).to(self.device).type(dtype)
        except:
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).type(torch.LongTensor).to(self.device)
            # if torch.cuda.is_available():
            #     self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).to(self.device).type(dtype)
            # else:
            self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).to(self.device).type(dtype)



    def __iter__(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(self.epoch * 100)
        self.indices = torch.randperm(len(self.dataset), generator=g, device=self.device) if self.shuffle else torch.arange(len(self.dataset), device=self.device)
        self.epoch += 1
        self.iter = 0
        return self
    

    def __next__(self):
        index_set = self.indices[ self.iter*self.batch_size : self.iter*self.batch_size+self.batch_size ]
        self.iter += 1
        if len(index_set) == 0:
            raise StopIteration()
        
        # This is probably slow, it would be better to generate the overlapping indices in the __init__ method
        if self.overlap > 0:
            overlapping_indices = torch.tensor([], dtype=torch.long, device=self.device)
            for i in range(self.minibatch_amount):
                if i != self.iter:
                    if self.SHARED_OVERLAP:
                        indexes = torch.tensor([range(i*self.batch_size, i*self.batch_size+self.overlap)], device=self.device)
                    else:
                        indexes = torch.randint(i*self.batch_size, i*self.batch_size+self.batch_size, (self.overlap,), device=self.device) # generate "self.overlap" random indeces inside the i-th minibatch
                    overlapping_indices = torch.cat([overlapping_indices, self.indices[indexes]], 0)
            
            index_set = torch.cat([index_set, overlapping_indices], 0) # Combining the original index set with the overlapping indices

        return self.dataset.data[index_set], self.dataset.targets[index_set]
        
