import torch
import torch.distributed as dist
import dill

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

def send_fun(tensor, dst):
    """Send a lambda function to a specific node."""
    serialized_func = dill.dumps(tensor)
    byte_tensor = torch.ByteTensor(list(serialized_func))
    send_shape(byte_tensor, dst)
    dist.send(byte_tensor, dst)

def recv_fun(src):
    """Receive a lambda function from a specific node."""
    shape = receive_shape(src)
    byte_tensor = torch.ByteTensor(shape)
    dist.recv(byte_tensor, src)
    serialized_func = byte_tensor.numpy().tobytes().rstrip(b'\x00')
    func = dill.loads(serialized_func)
    return func

# Forward pass to defined the shapes of the tensors
def send_shape(shape: list, dst: int, device = None):
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    for s in shape:
        dist.send(tensor=torch.tensor(s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)
        
def receive_shape(src: int, device = None):
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    shape = []; temp = 0
    while True:
        temp = torch.tensor((0), dtype=torch.int32).to(device)
        dist.recv(tensor=temp, src=src)
        if temp == -1:
            break
        shape.append(temp.item())
    return shape

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False, data_chunks_amount=1):
    '''
    NOTE: Losses from different chunks are averaged.
    '''
    if model.__class__.__name__ != 'Parallelized_Model':
        raise ValueError('Model must be an instance of the "Parallelized_Model".')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.rank == model.rank_list[-1]:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount):
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
        # loss = torch.zeros(1).to(model.gpu_device)
        losses = [0]*data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.rank == model.rank_list[-1]:
            for i,out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = sum(losses)/len(losses)
        # Average losses across replicas
        if model.rank == model.rank_list[-1]:
            dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.final_layer_group) # Summing the losses across final layers of each replicas
            loss = loss/model.num_replicas
        loss_broadcast = dist.broadcast(tensor=loss.detach(), src=model.model_ranks[0][-1], async_op=True) # Last layer of first model replica broadcasts the loss to all other ranks 
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
            # Synchronize model gradients
            model.sync_grads()
        loss_broadcast.wait()
        if return_output:
            if model.rank == model.rank_list[-1]:
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output for output in outputs]
            else:
                return loss.item(), None
        return loss.item()
    return closure2 