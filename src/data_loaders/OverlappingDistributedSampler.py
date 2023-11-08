import math
import torch
from torch.utils.data import DistributedSampler

class OverlappingDistributedSampler(DistributedSampler):
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, overlapping_samples=0, random_overlap=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.overlapping_samples = overlapping_samples
        self.random_overlap = random_overlap


    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch * 100)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        my_indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(my_indices) == self.num_samples
        
        # TODO: take care of proper division in case drop last is true and in case of uneven division
        all_indices = []
        for rank in range(self.num_replicas):
            all_indices.append(indices[rank:self.total_size:self.num_replicas])
        if self.random_overlap:
            # Shuffle and add overlapping samples for each replica
            for rank in range(self.num_replicas):
                if rank != self.rank:
                    shuffled_indices = all_indices[rank]
                    torch.manual_seed(rank * 100 + self.epoch)  # Add rank as a factor to ensure different seeds
                    torch.randperm(len(shuffled_indices))
                    my_indices += shuffled_indices[:self.overlapping_samples]
        else:
            for rank in range(self.num_replicas):
                if rank != self.rank:
                    my_indices += all_indices[rank][:self.overlapping_samples]

        # print(f"Rank {self.rank} has indices of length {len(my_indices)}")
        return iter(my_indices)


    def __len__(self):
        return math.ceil(len(self.dataset) * self.overlapping_samples / self.num_replicas)
