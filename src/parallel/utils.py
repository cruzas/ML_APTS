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
    
def layer_input_output_shape(layer):
    '''
    This function returns the input shape of a layer.
    '''
    if type(layer) is torch.nn.modules.linear.Linear:
        return layer.in_features, layer.out_features


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

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, output=False, data_chunks_amount=2, counter=True):
    '''
    NOTE: Losses from different chunks are averaged.
    '''
    if model.__class__.__name__ != 'Weight_Parallelized_Model':
        raise ValueError('Model must be an instance of the "Weight_Parallelized_Model" class.')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.rank == model.rank_list[-1][0]:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, output=output, data_chunks_amount=data_chunks_amount, counter=counter):
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, count_f=counter, chunks_amount=data_chunks_amount)
        # loss = torch.zeros(1).to(model.gpu_device)
        losses = []
        loss = torch.tensor(0.0).to(model.gpu_device)
        if model.rank == model.rank_list[-1][0]:
            for i,out in enumerate(outputs):
                losses.append(criterion(out, targets[i].to(out.device)))
            loss = sum(losses)/len(losses)
        dist.broadcast(tensor=loss.detach(), src=model.rank_list[-1][0], group=model.master_group)
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses, count_g=counter, chunks_amount=data_chunks_amount)
        if output:
            if model.rank == model.rank_list[-1][0]:
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output.detach() for output in outputs]
            else:
                return loss.item(), None
        else:
            return loss.item()
    return closure2 