from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Optional
from torch.utils.data import Dataset

class MockDataset(Dataset):
    # NOTE: First=True means that the real input data and mock output data will be provided
    # First=False means that the mock input data and real output data will be provided
    # First=None means that the mock input and output data will be provided
    def __init__(self, dataset, amount_of_batches=None, device=None, first=True):
        """
        Initializes a MockDataset object.

        Args:
            dataset: The dataset to be used.
            amount_of_batches (optional): The number of batches to be used. Defaults to None.
            device (optional): The device to be used. Defaults to None.
            first (optional): A boolean indicating if it is the first dataset. Defaults to True.
        """
        super(MockDataset, self).__init__()
        self.amount_of_batches = amount_of_batches
        self.dataset = dataset
        self.first = first
        self.device = device

    def __len__(self):
        """
        Returns the number of batches in the dataloader.

        :return: The number of batches.
        :rtype: int
        """
        return self.amount_of_batches

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the item at the specified index.
        """
        if self.first == True:
            return (self.dataset[idx][0], 1)
        elif self.first == False:
            return (1, self.dataset[idx][1])
        else:
            return (1, 1)
        
class GeneralizedDistributedDataLoader(DataLoader):
    def __init__(self, stage_list, num_replicas, dataset, batch_size, shuffle, device='cpu' if not torch.cuda.is_available() else 'cuda', num_workers=0, pin_memory=False, seed=0, **kwargs):
        """
        Initializes the GeneralizedDistributedDataLoader object.

        Args:
            stage_list (list): A list of stages.
            num_replicas (int): The number of replicas.
            dataset: The dataset to be loaded.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            device (str): The device to use. Defaults to 'cpu' if torch.cuda.is_available() is False, otherwise 'cuda'.
            num_workers (int): The number of worker processes. Defaults to 0.
            pin_memory (bool): Whether to pin memory. Defaults to False.
            seed (int): The random seed. Defaults to 0.
            **kwargs: Additional keyword arguments.
            
        E.g.: Supppose len(stage_list) = 3 and num_replicas = 2.Then:
        model 0 will be distributed across ranks [0,1,2] with first layer in rank 0 and second layer in rank 1 and so on.
        model 1 will be distributed across ranks [3,4,5] with first layer in rank 3 and second layer in rank 4 and so on.
        """
        if 'drop_last' in kwargs:
            #print a warning 
            print(f"(WARNING) drop_last will always be set to 'True' in the GeneralizedDistributedDataLoader.")
            kwargs.pop('drop_last')
        if batch_size > len(dataset):
            print(f"(WARNING) Batch size {batch_size} is greater than the dataset size {len(dataset)}. Setting batch size to dataset size.")
            batch_size = min(batch_size, len(dataset))
        first_layer_ranks = [0+len(stage_list)*i for i in range(num_replicas)]
        last_layer_ranks = [len(stage_list)-1+len(stage_list)*i for i in range(num_replicas)]

        rank = dist.get_rank()
        self.is_active_rank = True
        if rank not in first_layer_ranks+last_layer_ranks: # rank in the middle does not require any real data
            # Make a mock dataset with the same amount of batches as the original dataset (this is needed to keep iterations consistent across all ranks)
            amount_of_batches = 1 if len(dataset) == batch_size else len(dataset) // batch_size
            dataset = MockDataset(dataset, amount_of_batches, device=device, first=None)     
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)    
        elif rank in first_layer_ranks:
            dataset = MockDataset(dataset, len(dataset), device=device, first=True)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=first_layer_ranks, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=True, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//num_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)
        else:
            dataset = MockDataset(dataset, len(dataset), device=device, first=False)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=last_layer_ranks, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=True, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//num_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)

class GeneralizedDistributedSampler(DistributedSampler):
    def __init__(self, layer_ranks, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, **kwargs): 
        """
        Initializes the GeneralizedDistributedSampler object.

        Parameters:
        - layer_ranks: List of ranks for each layer.
        - dataset: The dataset to sample from.
        - num_replicas: Number of distributed replicas. Defaults to None.
        - rank: Rank of the current process. Defaults to None.
        - shuffle: Whether to shuffle the samples. Defaults to True.
        - seed: Seed value for shuffling. Defaults to 0.
        - drop_last: Whether to drop the last incomplete batch. Defaults to False.
        - **kwargs: Additional keyword arguments.

        Raises:
        - RuntimeError: If the distributed package is not available.
        - ValueError: If num_replicas is not equal to the number of layer_ranks.
        """
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank() if rank is None else rank
        if num_replicas is not None and (len(layer_ranks) != num_replicas):
            raise ValueError("num_replicas should be equal to the number of first_layer_ranks.")
        rank = layer_ranks.index(rank)
        kwargs.update({'dataset': dataset, 'num_replicas': len(layer_ranks), 'rank': rank, 'shuffle': shuffle, 'seed': seed, 'drop_last': drop_last})
        super(GeneralizedDistributedSampler, self).__init__(**kwargs)
        # super(GeneralizedDistributedSampler, self).__init__(dataset=dataset, num_replicas=len(first_layer_ranks), rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last, **kwargs)
