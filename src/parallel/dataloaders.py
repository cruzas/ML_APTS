from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from time import time
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Optional
from torch.utils.data import Dataset
import numpy as np

class MockDataset(Dataset):
    # NOTE: First=True means that the real input data and mock output data will be provided
    # First=False means that the mock input data and real output data will be provided
    # First=None means that the mock input and output data will be provided
    def __init__(self, dataset, amount_of_batches=None, first=True):
        super(MockDataset, self).__init__()
        self.amount_of_batches = amount_of_batches
        self.dataset = dataset
        self.first = first
        # TODO: remove unused variables in dataset to reduce memory requirements

    def __len__(self):
        return self.amount_of_batches

    def __getitem__(self, idx):
        if self.first == True:
            # return (self.dataset[idx][0], self.dataset[idx][1])
            return (self.dataset[idx][0], torch.zeros(1))
        elif self.first == False:
            # return (self.dataset[idx][0], self.dataset[idx][1])
            return (torch.zeros(1), self.dataset[idx][1])
        else:
            return (torch.zeros(1),torch.zeros(1))

class GeneralizedDistributedDataLoader(DataLoader):
    def __init__(self, stage_list, num_replicas, dataset, batch_size, shuffle, num_workers=0, pin_memory=False, drop_last=False, seed=0,**kwargs):
        '''
        E.g.: Supppose len(stage_list) = 3 and num_replicas = 2.Then:
        model 0 will be distributed across ranks [0,1,2] with first layer in rank 0 and second layer in rank 1 and so on.
        model 1 will be distributed across ranks [3,4,5] with first layer in rank 3 and second layer in rank 4 and so on.
        '''
        if batch_size > len(dataset):
            print(f"(WARNING) Batch size {batch_size} is greater than the dataset size {len(dataset)}. Setting batch size to dataset size.")
            batch_size = min(batch_size, len(dataset))
        first_layer_ranks = [0+len(stage_list)*i for i in range(num_replicas)]
        last_layer_ranks = [len(stage_list)-1+len(stage_list)*i for i in range(num_replicas)]

        rank = dist.get_rank()
        self.is_active_rank = True
        if len(dataset) % batch_size == 0:
            amount_of_batches = len(dataset) // (batch_size * num_replicas)
        else:
            if drop_last:
                amount_of_batches = len(dataset) // (batch_size * num_replicas)
            else:
                amount_of_batches = len(dataset) // (batch_size * num_replicas) + 1
        
        # amount_of_batches_in_rank = amount_of_batches // num_replicas
            
        if rank not in first_layer_ranks+last_layer_ranks: # rank in the middle does not require any real data
            # Make a mock dataset with the same amount of batches as the original dataset (this is needed to keep iterations consistent across all ranks)
            amount_of_batches = 1 if len(dataset) == batch_size else len(dataset) // batch_size
            dataset = MockDataset(dataset, amount_of_batches, first=None)    
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs)    
        elif rank in first_layer_ranks:
            dataset = MockDataset(dataset, len(dataset), first=True)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=first_layer_ranks, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//num_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs)
        else:
            dataset = MockDataset(dataset, len(dataset), first=False)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=last_layer_ranks, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//num_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs)
        temp = list(self)
        print(f"rank {rank} has {len(temp)} batches of size {[f'input {t[0].shape} output {t[1].shape}' for t in temp]}")
        
        # print('as')


class GeneralizedDistributedSampler(DistributedSampler):
    def __init__(self, layer_ranks, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, **kwargs): 
        '''
        Variables for the DistributedSampler: 
        dataset, num_replicas, rank, shuffle, seed, drop_last
        '''
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank() if rank is None else rank
        if num_replicas is not None and (len(layer_ranks) != num_replicas):
            raise ValueError("num_replicas should be equal to the number of first_layer_ranks.")
        rank = layer_ranks.index(rank)
        kwargs.update({'dataset': dataset, 'num_replicas': len(layer_ranks), 'rank': rank, 'shuffle': shuffle, 'seed': seed, 'drop_last': drop_last})
        super(GeneralizedDistributedSampler, self).__init__(**kwargs)
        # super(GeneralizedDistributedSampler, self).__init__(dataset=dataset, num_replicas=len(first_layer_ranks), rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last, **kwargs)

# class ParallelizedDataLoader(DataLoader):
#     '''
#     Assumptions: Dataset must have samples as rows (i.e. position 0 of the shape).
#     '''
#     def __init__(self, dataset,                 rank_list : list,
#                  device_list : list,            batch_size = 64,               
#                  overlap_ratio: float = 0.0,
#                  shuffle = True,                sampler = None,
#                  batch_sampler = None,          num_workers = 0, 
#                  collate_fn = None,             pin_memory = False, 
#                  drop_last = False,             timeout = 0, 
#                  worker_init_fn = None,         multiprocessing_context=None, 
#                  generator=None,                prefetch_factor = None,
#                  persistent_workers = False,    pin_memory_device = ""):
        
#         self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
#         self.rank_list = rank_list
#         self.device_list = device_list
#         all_devices = torch.cuda.device_count()
#         if any([gpu >= all_devices for gpu in device_list]):
#             raise ValueError("Some devices are not available.")
#         self.overlap_ratio = overlap_ratio # TODO: future implementation
#         self.iterator = None 
#         self.chunks_inputs = None
#         self.chunks_targets = None
    
#     def get_next_item(self):
#         num_chunks = len(self.device_list)
#         def chunkenizer(iterator):
#             try:
#                 inputs, targets = next(iterator)
#                 chunks_inputs = torch.chunk(inputs, num_chunks)
#                 chunks_targets = torch.chunk(targets, num_chunks)
#             except StopIteration:
#                 chunks_inputs = 'done'
#                 chunks_targets = 'done'
#             return chunks_inputs, chunks_targets
        
#         def to_cuda(inputs, targets, device):
#             return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    
#         with ProcessPoolExecutor(max_workers=1) as executor:
#             # Map `chunkenizer` across all the matrix chunks
#             results = list(executor.map(chunkenizer, self.iterator))
            
#         if self.chunks_inputs is None and self.chunks_targets is None:
#             self.chunks_inputs = [result[0] for result in results]
#             self.chunks_targets = [result[1] for result in results]
            
#         with ProcessPoolExecutor(max_workers=num_chunks) as executor:
#             # Map `to_cuda` across all the matrix chunks
#             results = list(executor.map(to_cuda, self.chunks_inputs, self.chunks_targets, self.device_list))
            
#         return results


#     def __iter__(self):
#         self.iterator = self.data_loader.__iter__()
#         return self

#     def __next__(self):
#         # Get the next item from the DataLoader iterator            
#         if self.chunks_inputs is not None and self.chunks_targets is not None:
#             if self.chunks_inputs == 'done' and self.chunks_targets == 'done':
#                 raise StopIteration()
#             chunks_inputs = self.chunks_inputs
#             chunks_targets = self.chunks_targets
#         else:
#             inputs, targets = next(self.iterator) # [ inputs, targets ]
#             chunks_inputs = torch.chunk(inputs, len(self.device_list))
#             chunks_inputs = [chunk.to(device, non_blocking=True) for chunk, device in zip(chunks_inputs, self.device_list)]
#             chunks_targets = torch.chunk(targets, len(self.device_list))
#             chunks_targets = [chunk.to(device, non_blocking=True) for chunk, device in zip(chunks_targets, self.device_list)]
            
#         tic = time.time()
#         result = self.get_next_item()
        
#         print(f"Time to get next item: {time.time() - tic}")

#         return chunks_inputs, chunks_targets
        
        


    

