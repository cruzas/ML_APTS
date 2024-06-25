from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from time import time
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
from torch.utils.data import Dataset

class MockDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # Return a dummy data sample, e.g., a zero tensor or None
        return torch.tensor([0])  # Adjust this depending on your actual data structure

class GeneralizedDistributedSampler(DistributedSampler):
    def __init__(self, rank_list, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, **kwargs): 
        '''
        Variables for the DistributedSampler: 
        dataset, num_replicas, rank, shuffle, seed, drop_last
        '''
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        current_rank = dist.get_rank() if rank is None else rank
        self.is_active_rank = current_rank in rank_list
        if num_replicas is None:
            kwargs.update({'num_replicas':len(rank_list)})
        else:
            if len(rank_list) != num_replicas:
                raise ValueError("World size must be equal to the number of ranks.")
        kwargs.update({'shuffle':shuffle})
        kwargs.update({'seed':seed})
        kwargs.update({'drop_last':drop_last})
        self.rank_list = rank_list
        if self.is_active_rank:
            kwargs.update({'dataset':dataset})
            kwargs.update({'rank':rank_list.index(current_rank)})
            self.sampler = DistributedSampler(**kwargs)
        else:
            ws = dist.get_world_size()
            unused_ranks = [r for r in range(ws) if r not in rank_list]
            kwargs.update({'dataset': MockDataset(), 'rank': unused_ranks.index(current_rank), 'num_replicas': len(unused_ranks)})
            self.sampler = DistributedSampler(**kwargs)

    def __getattr__(self, attr):
        if attr == 'epoch':
            self.sampler.set_epoch(self.sampler.epoch + 1)
        if not self.is_active_rank:
            return getattr(self.sampler, attr)
        return getattr(self.sampler, attr)



class ParallelizedDataLoader(DataLoader):
    '''
    Assumptions: Dataset must have samples as rows (i.e. position 0 of the shape).
    '''
    def __init__(self, dataset,                 rank_list : list,
                 device_list : list,            batch_size = 64,               
                 overlap_ratio: float = 0.0,
                 shuffle = True,                sampler = None,
                 batch_sampler = None,          num_workers = 0, 
                 collate_fn = None,             pin_memory = False, 
                 drop_last = False,             timeout = 0, 
                 worker_init_fn = None,         multiprocessing_context=None, 
                 generator=None,                prefetch_factor = None,
                 persistent_workers = False,    pin_memory_device = ""):
        
        self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
        self.rank_list = rank_list
        self.device_list = device_list
        all_devices = torch.cuda.device_count()
        if any([gpu >= all_devices for gpu in device_list]):
            raise ValueError("Some devices are not available.")
        self.overlap_ratio = overlap_ratio # TODO: future implementation
        self.iterator = None 
        self.chunks_inputs = None
        self.chunks_targets = None
    
    def get_next_item(self):
        num_chunks = len(self.device_list)
        def chunkenizer(iterator):
            try:
                inputs, targets = next(iterator)
                chunks_inputs = torch.chunk(inputs, num_chunks)
                chunks_targets = torch.chunk(targets, num_chunks)
            except StopIteration:
                chunks_inputs = 'done'
                chunks_targets = 'done'
            return chunks_inputs, chunks_targets
        
        def to_cuda(inputs, targets, device):
            return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Map `chunkenizer` across all the matrix chunks
            results = list(executor.map(chunkenizer, self.iterator))
            
        if self.chunks_inputs is None and self.chunks_targets is None:
            self.chunks_inputs = [result[0] for result in results]
            self.chunks_targets = [result[1] for result in results]
            
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            # Map `to_cuda` across all the matrix chunks
            results = list(executor.map(to_cuda, self.chunks_inputs, self.chunks_targets, self.device_list))
            
        return results


    def __iter__(self):
        self.iterator = self.data_loader.__iter__()
        return self

    def __next__(self):
        # Get the next item from the DataLoader iterator            
        if self.chunks_inputs is not None and self.chunks_targets is not None:
            if self.chunks_inputs == 'done' and self.chunks_targets == 'done':
                raise StopIteration()
            chunks_inputs = self.chunks_inputs
            chunks_targets = self.chunks_targets
        else:
            inputs, targets = next(self.iterator) # [ inputs, targets ]
            chunks_inputs = torch.chunk(inputs, len(self.device_list))
            chunks_inputs = [chunk.to(device, non_blocking=True) for chunk, device in zip(chunks_inputs, self.device_list)]
            chunks_targets = torch.chunk(targets, len(self.device_list))
            chunks_targets = [chunk.to(device, non_blocking=True) for chunk, device in zip(chunks_targets, self.device_list)]
            
        tic = time.time()
        result = self.get_next_item()
        
        print(f"Time to get next item: {time.time() - tic}")

        return chunks_inputs, chunks_targets
        
        


    

