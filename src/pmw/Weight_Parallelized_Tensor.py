import math
import torch
import torch.nn as nn
import torch.distributed as dist
from utils import *


# Global gradient class
class Weight_Parallelized_Tensor(nn.Module):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        
    def norm(self, p=2):
        if p == 2:
            return math.sqrt(self @ self)
        else:
            # Implement generic p
            raise NotImplementedError("Only L2 norm is implemented.")
    
    def __iter__(self):
        return iter(self.tensor)
    
    def __repr__(self):
        return f'Rank {self.rank}\nGradient: {self.model.subdomain.grad()}'
    
    def __matmul__(self, a): # self.grad @ a
        return self * a
    
    def __rmatmul__(self, a): # a @ self.grad
        return a * self

    def __rmul__(self, a):  # a * self.grad
        return self.__mul__(a)  # This handles the commutative property of multiplication
    
    def __mul__(self, a):  # self.grad * a
        if isinstance(a, Weight_Parallelized_Tensor):  # When both operands are Weight_Parallelized_Tensor instances
            g1 = torch.cat([p.flatten() for p in self.tensor], dim=0)  # Flatten the gradients
            g2 = torch.cat([p.flatten() for p in a.tensor], dim=0)  # Flatten the gradients
            g3 = g1 @ g2
            g3 = g3.to(f'cpu' if self.backend == 'gloo' else f'cuda:0')
            dist.all_reduce(tensor=g3, group=self.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3.item()
        else:
            return Weight_Parallelized_Tensor([p*a for p in self.tensor], backend=self.backend, master_group=self.master_group, rank=self.rank)   # Multiply model by a scalar or tensor

    def __add__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p+q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
    
    def __sub__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p-q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
