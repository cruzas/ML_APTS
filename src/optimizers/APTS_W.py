import torch, copy
from torch.optim.optimizer import Optimizer
from optimizers.TR import *
import pickle
import itertools
import torch.distributed as dist


# TODO: We noticed that not using in-place operations creates highly unnecessary memory usage. 
# For example: a = torch.randn(100000); b = [a]; b[0] = b[0] + 1 will create a new vector for computing b[0] + 1. 
# This is not a problem for small vectors. But when we have large vectors, there is a huge memory overhead.
# OR we can simply use torch.cuda.empty_cache()

def list_subtract(list1, list2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [list1[i] - list2[i] for i in range(len(list1))] 
    return result


def list_linear_comb(a, list1, b=0, list2=[]): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [0] * len(list1)
        for i in range(len(list1)):
            a = torch.tensor(a, device=list1[i].device) 
            if b == 0:
                result[i] = a*list1[i]
            else:
                b = torch.tensor(b, device=list2[i].device) # Remember: list1[i] and list2[i] can NOT be on different devices
                result[i] = a*list1[i] + b*list2[i]
    return result

def list_sum(list1, list2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [list1[i] + list2[i] for i in range(len(list1))]
    return result
    

def list_norm(list1, p=2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    if p == torch.inf:
        with torch.no_grad():
            result = 0
            for i in range(len(list1)):
                result = max(result, torch.norm(list1[i].flatten(), p=p).item())
        return result
    else:
        with torch.no_grad():
            result = 0
            for i in range(len(list1)):
                result += torch.norm(list1[i].flatten(), p=p).item()**p
        return result**(1/p)


class APTS_W(Optimizer):
    def __name__(self):
        return 'APTS_W'
    
    def __init__(self,
                 params,
                 model=None,
                 loss_fn=None,
                 device=None,
                 max_iter=2,
                 nr_models=None,
                 global_opt=None,
                 global_opt_params=None,
                 local_opt=None,
                 local_opt_params=None,
                 global_pass=True,
                 forced_decreasing_loss=False,
                 shuffle_layers=False
                 ):

        super(APTS_W, self).__init__(params, {})

        self.rank = dist.get_rank()
        self.tot_layers = len(list(model.parameters()))
        self.model = model
        self.backend = dist.get_backend()

        self.max_iter = max_iter
        self.nr_models = nr_models if nr_models is not None else self.tot_layers
        self.global_pass = bool(global_pass) # in case global_pass is 0 or 1
        self.loss_fn = loss_fn
        self.iter = 0
        self.fdl = bool(forced_decreasing_loss) # in case forced_decreasing_loss is 0 or 1

        # Initialize the local models and optimizers
        self.rank2layer, self.layer2rank = self.define_trainable_layers(shuffle=shuffle_layers)
        self.set_trainable_layers()
        self.restricted_global_params = [torch.tensor(0)]*self.tot_layers
        params_list = list(self.model.parameters())
        idx_range = range(self.tot_layers) if self.rank == 0 else self.rank2layer[self.rank]
        for layer in idx_range:
            self.restricted_global_params[layer] = params_list[layer].clone().detach()

        if 'TR' in str(local_opt):
            local_opt_params['min_radius'] = 0 # avoids problems with "radius" being smallers than "min_radius"
            local_opt_params.update({'device': 'cuda'}) # TODO: make this more general
            
        self.local_optimizer = local_opt(self.model.parameters(), **local_opt_params)
        
        if self.rank == 0:
            self.global_model = copy.deepcopy(self.model)
            self.global_optimizer = global_opt(self.global_model.parameters(), **global_opt_params)
        self.radius = global_opt_params['radius']
        torch.cuda.empty_cache()

    def set_trainable_layers(self, is_global=False):
        if is_global and self.rank == 0:
            for index, param in enumerate(self.model.parameters()):
                param.requires_grad = True
        elif not is_global:
            for index, param in enumerate(self.model.parameters()):
                param.requires_grad = index in self.rank2layer[self.rank]

    def define_trainable_layers(self, shuffle=False):
        '''
        Creates a submodel with trainable layers distributed across ranks.
        Raises an error if the number of ranks exceeds the number of parameter groups.
        '''
        # TODO: For efficiency, change the following (E.g. choose a world size that is just enough, i.e. with every GPU needed filled)
        # TODO: If enough processes are availalb and each layer is not big enough, distribute even more layers per process...or something like that
        tot_layers = self.tot_layers
        if self.nr_models != dist.get_world_size():
            raise ValueError(f"Number of models ({self.nr_models}) should match world size ({dist.get_world_size()}).")
        
        index_shuffled = torch.randperm(tot_layers) if shuffle else list(range(tot_layers))

        trainable_layers = [None]*self.nr_models
        params_per_subset = [0]*self.nr_models
        for i in range(self.nr_models):
            params_per_subset[i] = int(tot_layers / (self.nr_models-i))
            if i == self.nr_models - 1:
                trainable_layers[i] = index_shuffled[sum(params_per_subset[:i]):]
            else:
                trainable_layers[i] = index_shuffled[sum(params_per_subset[:i]):sum(params_per_subset[:i+1])]
            tot_layers -= params_per_subset[i]

        rank2layer = {k: v for k, v in enumerate(trainable_layers)}
        layer2rank = {v: k for k, lst in rank2layer.items() for v in lst}
        return rank2layer, layer2rank
    

    def closure(self, inputs, targets, model):
        J = self.loss_fn(model(inputs), targets)
        if torch.is_grad_enabled():
            model.zero_grad()
            J.backward()
        return J.item()
        
    def restricted_global_params(self): # Remove in production
        sub = list(self.model.parameters())
        return torch.cat([p.flatten().detach() for i,p in enumerate(self.model.parameters()) if sub[i].requires_grad is True])
        
    def local_model_params(self):
        return torch.cat([p.flatten().detach() for p in self.model.parameters() if p.requires_grad is True])

    # TODO: With layer-pipelining, this will not work
    def local_net_sync(self):
        # Sync the nets
        if self.rank == 0:
            self.set_trainable_layers() # Set the global model back to the local model settings
        
        if self.rank == 0: # start sending
            global_opt_state_dict = {key: getattr(self.global_optimizer, key) for key in self.global_optimizer.collection_of_all_variables}
            dumped_dict = pickle.dumps(global_opt_state_dict)
            byte_dict = torch.ByteTensor(list(dumped_dict))
            object_list = [byte_dict]
            print(object_list)
        else:
            object_list = [0]

        dist.broadcast_object_list(object_list, src=0) # update the local optimizers
        global_opt_state_dict = pickle.loads(bytes(object_list[0].tolist()))

        # Sync the optimizers
        # TODO: for second order or momentum, make sure that collection_of_all_variables is not empty and update correctly the local optimizers
        for key in global_opt_state_dict:
            if key == 'hessian_approx':
                setattr(self.local_optimizer, key, copy.deepcopy(getattr(self.global_optimizer, key))) # TODO: fix this later
                if self.local_optimizer.hessian_approx.S is not None:
                    v1 = []; v2 = []; v3 = []
                    a = 0; a2=0
                    for j,p in enumerate(self.model.parameters()):
                        b2 = a2 + p.numel()
                        if j in self.trainable_layers:
                            b = a + p.numel()
                            v1.append( self.local_optimizer.hessian_approx.S[a2:b2] )
                            v2.append( self.local_optimizer.hessian_approx.Y[a2:b2] )
                            v3.append( self.local_optimizer.hessian_approx.Psi[a2:b2] )
                            a = b
                        a2 = b2
                        
                    self.local_optimizer.hessian_approx.S = torch.concatenate(v1)
                    self.local_optimizer.hessian_approx.Y = torch.concatenate(v2)
                    self.local_optimizer.hessian_approx.Psi = torch.concatenate(v3)

            elif key in ['var', 'mom']:
                v = []
                v2 = getattr(self.global_optimizer, key)
                a = 0; a2=0
                for j,p in enumerate(self.model.parameters()): 
                    b2 = a2 + p.numel()
                    if j in self.trainable_layers:
                        b = a + p.numel()
                        v.append( v2[a2:b2] )
                        a = b
                    a2 = b2
                setattr(self.local_optimizer, key, torch.cat(v))
            else:
                setattr(self.local_optimizer, key, getattr(self.global_optimizer, key))
            

    def local_steps(self, inputs, targets):
        # Update the local model 
        TEMP = 1 # TODO: test whether having TEMP=2 here is beneficial
        local_loss = 0
        lr = self.radius/TEMP
        # g = torch.zeros(self.tot_params, device=self.device); grad_norm2 = 0
        if self.rank == 0:
            receive_device = 'cpu' if self.backend == 'gloo' else 'cuda'
            g = [torch.zeros_like(p).to(receive_device) for p in self.model.parameters()]
        else:
            g = [torch.tensor(0)] * self.tot_layers
            
        grad_norm2 = 0

        global_params_list = list(self.model.parameters())
        for j in range(self.max_iter): # Iterations of the preconditioning step            
            self.local_optimizer.radius = lr
            self.local_optimizer.max_radius = lr            
            local_loss, grad, grad_norm = self.local_optimizer.step(lambda dummy=1: self.closure(inputs, targets, self.model)) 
            
            if j == 0:
                grad_norm2 = max(grad_norm2, grad_norm)
                for i in self.rank2layer[self.rank]:
                    send_device = 'cpu' if self.backend == 'gloo' else grad[i].device
                    g[i] = grad[i].clone().detach().to(send_device, non_blocking=True)
                        
            # Update radius to make sure the update is inside the global trust region radius
            lr = self.radius - list_norm(list_linear_comb(1, self.restricted_global_params, -1, global_params_list), p=torch.inf)
            if lr < torch.finfo(torch.get_default_dtype()).eps:
                break
                
        # Collect the gradients from local models
        if self.rank != 0:
            for layer in self.rank2layer[self.rank]:
                send_device = 'cpu' if self.backend == 'gloo' else g[layer].device
                dist.send(g[layer].to(send_device), dst=0, tag=layer)
        else:
            for i in range(1, self.nr_models):
                for layer in self.rank2layer[i]:
                    receive_device = 'cpu' if self.backend == 'gloo' else g[layer].device
                    # print(f'{g[layer]} layer {layer} device {receive_device}')
                    g2 = torch.zeros_like(g[layer]).to(receive_device)
                    dist.recv(g2, src=i, tag=layer)
                    g[layer].copy_(g2)

        return local_loss, g, grad_norm2
    
    def synchronize_global_to_local(self):
        if self.rank == 0:
            dist.broadcast_object_list(list(self.global_model.parameters()), src=0)
        else:
            params_list = list(self.model.parameters())
            dist.broadcast_object_list(params_list, src=0)
            with torch.no_grad():
                for layer,p in enumerate(self.model.parameters()):
                    p.copy_(params_list[layer])
                    if layer not in self.rank2layer[self.rank]:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            
        # if self.rank == 0:
        #     model_params_list = list(self.global_model.parameters())
        #     for model_id, layer in itertools.product(range(self.nr_models), range(self.tot_layers)):
        #         if model_id != 0 and layer in self.rank2layer[model_id]:
        #             send_device = 'cpu' if self.backend == 'gloo' else self.restricted_global_params[layer].device
        #             dist.send(model_params_list[layer].to(send_device), dst=model_id, tag=layer)
        # else:
        #     for layer in range(self.tot_layers):
        #         if layer in self.rank2layer[self.rank]:
        #             device = self.restricted_global_params[layer].device
        #             receive_device = 'cpu' if self.backend == 'gloo' else device
        #             layers_ = torch.zeros_like(self.restricted_global_params[layer]).to(receive_device)
        #             dist.recv(layers_, src=0, tag=layer)
        #             self.restricted_global_params[layer] = layers_.to(device)

    def synchronize_new_params_to_rank0_local(self):
        if self.rank != 0:
            for layer, p in enumerate(self.model.parameters()):
                if layer in self.rank2layer[self.rank]:
                    send_device = 'cpu' if self.backend == 'gloo' else p.device
                    dist.send(p.to(send_device), dst=0, tag=layer)
        else:
            for i in range(1, self.nr_models):
                for layer, p in enumerate(self.model.parameters()):
                    if layer in self.rank2layer[i]:
                        receive_device = 'cpu' if self.backend == 'gloo' else p.device
                        layer_from_i = torch.zeros_like(p).to(receive_device)
                        dist.recv(layer_from_i, src=i, tag=layer)
                        p.copy_(layer_from_i) # TODO: test if this modifies things
                        p.requires_grad = True

    def synchronize_local_to_global(self):
        if self.rank != 0:
            for layer, p in enumerate(self.model.parameters()):
                if layer in self.rank2layer[self.rank]:
                    send_device = 'cpu' if self.backend == 'gloo' else p.device
                    dist.send(p.to(send_device), dst=0, tag=layer)
        else:
            with torch.no_grad():
                for i in range(1, self.nr_models):
                    for layer,p in enumerate(self.global_model.parameters()):
                        if layer in self.rank2layer[i]:
                            receive_device = 'cpu' if self.backend == 'gloo' else p.device
                            layer_from_i = torch.zeros_like(p).to(receive_device)
                            dist.recv(layer_from_i, src=i, tag=layer)
                            p.copy_(layer_from_i) # TODO: test if this modifies things
                            p.requires_grad = True
    

    def step(self, inputs, targets):  
        '''
        One step of APTS
        '''
        self.iter += 1
        skip_preconditioning = False
        if skip_preconditioning is False:
            self.local_net_sync() # Sharing weights of the global NN with the local ones
            a = time.time()

            self.synchronize_global_to_local() # Synchronize the local models with global model parameters
            torch.cuda.empty_cache()

            _, g, _ = self.local_steps(inputs, targets)
            # sync the local models with the global model
            print(f'Rank {self.rank}, Local steps took {time.time()-a} seconds')
            a = time.time()
            self.synchronize_new_params_to_rank0_local()
            print(f'Rank {self.rank}, sync took {time.time()-a} seconds')
        
            # check that the loss is decreasing
            if self.fdl and self.rank == 0: # TODO: make this parallel, do some dogleg iterations  in parallel and only keep the longest step which satisfies the decreasing loss condition
                with torch.no_grad():
                    step = list_linear_comb(1, list(self.model.parameters()), -1, list(self.global_model.parameters())) # new_params - old_params
                    step_norm = list_norm(step, p=torch.inf)
                    g = list_linear_comb(step_norm/list_norm(g, p=torch.inf), g) # g/torch.norm(g, p=torch.inf)
                    old_loss = self.closure(inputs, targets, self.global_model)
                    self.synchronize_local_to_global() # update global model
                    
                    loss = self.closure(inputs, targets, self.global_model) # current loss
                    radius = self.radius
                    w = 0; c = 0
                    while loss > old_loss: 
                        c += 1
                        # print(f'----------> Loss is increasing ({loss} > {self.old_loss}), reducing step size. W = '+ str(w))
                        radius = radius/2
                        w = min(w + 0.2, 1)
                        step2 = step*(1-w) + w*g
                        x_loc = list_linear_comb(1, list(self.model.parameters()),) #x_glob + step2/torch.norm(step2, p=torch.inf)*radius
                        self.synchronize_local_to_global()
                        loss = self.closure(inputs, targets, self.global_model)
                        if c > 5:
                            print(f'Warning: APTS is not decreasing the loss {loss} > {old_loss}')
                            break
            elif not self.fdl:
                self.synchronize_local_to_global() # update global model

        torch.cuda.empty_cache()
        dist.barrier()
        # Global smoothing step 
        new_loss = 0
        if self.global_pass and self.rank == 0:
            new_loss = self.global_optimizer.step(lambda dummy=1: self.closure(inputs, targets, self.global_model))[0]
            self.radius = self.global_optimizer.radius

        # print(f'Iteration {self.iter} | initial loss: {initial_loss}, after local steps: {local_loss}, loss before global step {loss}, final loss {new_loss}') ##
        if self.iter == 1:
            if skip_preconditioning:
                print('Skipping preconditioning')
            else:
                if self.fdl:
                    print('Modified APTS with forced decreasing loss')
        
        torch.cuda.empty_cache()
        return new_loss
















