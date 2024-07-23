import torch
import torch.distributed as dist



def Parallelize(LayerList, rank_list, gpu_per_node):
    '''
    LayerList: contains the unloaded layers to avoid memory overflow.
    
    LayerList, e.g. LayerList = [ [ Layer_1_class,{dictionary of parameters} ] , [ Layer_2_class,{dictionary of parameters} ], ...]
    -> Therefore the global NN is NN = sequential( Layer_1_class({params1}), Layer_2_class({params2}), ... )
    
    rank_list, e.g. rank_list = [ [rank_0,rank_1,rank_2], [rank_3,rank_4], [rank_5], [rank_6,rank_7],... ]
    -> giving multiple ranks to each layer will parallelize the layer across the ranks through PyTorch's Dtensor library for horizontal slicing.
    
    Note that data parallelism is only available if you have multiple, say n, GPUs per node (in nccl). In that case you can divide the dataset into n chunks.
    '''
    
    # Here we define the forward function for the pipelined model with "send" and "recv" functions



class ParallelLoss():
    def __init__(self, rank_list, criterion, criterion_params={}):
        # check if criterion is a class or a function
        if isinstance(criterion, type):
            self.criterion = criterion(**criterion_params)
        else:
            if len(criterion_params) > 0:
                self.criterion = criterion.__class__(**criterion_params)
            else:
                self.criterion = criterion
        self.rank_list = rank_list
        
    def __call__(self, x, y):
        if dist.get_rank() == self.rank_list[-1][0]:
            print('ciao')
        return self.criterion(x,y) if dist.get_rank() == self.rank_list[-1][0] else None    

# Forward pass to defined the shapes of the tensors
def send_shape(shape: list, dst: int, device: torch.device):
    for s in shape:
        dist.send(tensor=torch.tensor(s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)
        
def receive_shape(src: int, device: torch.device):
    shape = []; temp = 0
    while True:
        temp = torch.tensor((0), dtype=torch.int32).to(device)
        dist.recv(tensor=temp, src=src)
        if temp == -1:
            break
        shape.append(temp.item())
    return shape

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False,data_chunks_amount=2):
    '''
    NOTE: Losses from different chunks are averaged.
    '''
    if model.__class__.__name__ != 'Weight_Parallelized_Model' and model.__class__.__name__ != 'Parallelized_Model' and model.__class__.__name__ != 'Parallel_Sequential_Model':
        raise ValueError('Model must be an instance of the "Weight_Parallelized_Model" class or Parallelized_Model or Parallel_Sequential_Model.')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.__class__.__name__ == 'Parallelized_Model':
        model = model.model
    if model.rank == model.rank_list[-1]:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount):
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
        # loss = torch.zeros(1).to(model.gpu_device)
        losses = []
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.rank == model.rank_list[-1]:
            for i,out in enumerate(outputs):
                losses.append(criterion(out, targets[i].to(out.device)))
            loss = sum(losses)/len(losses)
        dist.broadcast(tensor=loss.detach(), src=model.rank_list[-1], group=model.master_group)
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses, chunks_amount=data_chunks_amount)
        if return_output:
            if model.rank == model.rank_list[-1]:
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output for output in outputs]
            else:
                return loss.item(), None
        return loss.item()
    return closure2 