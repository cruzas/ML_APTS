import torch, copy
from torch.optim.optimizer import Optimizer
from optimizers.TR import *
import pickle
import itertools
import torch.distributed as dist
import inspect

# TODO: We noticed that not using in-place operations creates highly unnecessary memory usage. 
# For example: a = torch.randn(100000); b = [a]; b[0] = b[0] + 1 will create a new vector for computing b[0] + 1. 
# This is not a problem for small vectors. But when we have large vectors, there is a huge memory overhead.
# OR we can simply use torch.cuda.empty_cache()


# TODO: IMPORTANT TODO: In case of model approximation (gradient approx to have independent models), First Order Consistency has to be considered!


def list_subtract(list1, list2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [list1[i] - list2[i] for i in range(len(list1))] 
    return result


def list_linear_comb(a, list1, b=0, list2=[]): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [0] * len(list1)
        for i in range(len(list1)):
            if type(a) is not torch.Tensor:
                a = torch.tensor(a, device=list1[i].device) 
            else:
                a = a.to(list1[i].device)

            if b == 0:
                result[i] = a*list1[i]
            else:
                if type(b) is not torch.Tensor:
                    b = torch.tensor(b, device=list2[i].device) # Remember: list1[i] and list2[i] can NOT be on different devices
                else:
                    b = b.to(list2[i].device)
                
                result[i] = a*list1[i] + b*list2[i]
    return result

def list_sum(list1, list2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = [list1[i] + list2[i] for i in range(len(list1))]
    return result


def list_scalar_product(list1, list2): #TODO: parallelize maybe through streams (when model pipelining is implemented or even with multiple layers on one GPU)
    with torch.no_grad():
        result = 0
        for i in range(len(list1)):
            result += torch.sum(list1[i]*list2[i]).item()
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
                 max_iter=2,
                 nr_models=None,
                 global_opt=None,
                 global_opt_params=None,
                 local_opt=None,
                 local_opt_params=None,
                 global_pass=True,
                 forced_decreasing_loss=False,
                 shuffle_layers=False,
                 sequential_derivative=None,
                 loss_fn=torch.nn.CrossEntropyLoss
                 ):

        super(APTS_W, self).__init__(params, {})
        if model is None or global_opt is None or global_opt_params is None or local_opt is None or local_opt_params is None:
            raise ValueError('model, global_opt, global_opt_params, local_opt, local_opt_params must be defined.')
        self.rank = dist.get_rank()
        with torch.no_grad():
            self.tot_layers = len(list(model.parameters()))
        self.loss_fn = loss_fn()
        self.backend = dist.get_backend()
        self.sequential_derivative = sequential_derivative
        self.max_iter = max_iter
        self.nr_models = nr_models if nr_models is not None else self.tot_layers
        self.global_pass = bool(global_pass) # in case global_pass is 0 or 1
        self.iter = 0
        self.fdl = bool(forced_decreasing_loss) # in case forced_decreasing_loss is 0 or 1
        self.total_args = len(inspect.getfullargspec(model.forward).args)
        
        if self.rank == 0:
            self.model = model
            self.global_model_params = copy.deepcopy(list(model.parameters()))
            self.global_optimizer = global_opt(self.global_model_params, **global_opt_params)
        else:
            self.model = model

        self.radius = torch.tensor(global_opt_params['radius'], device='cpu')

        # Initialize the local models and optimizers
        self.rank2layer, self.layer2rank = self.define_trainable_layers(shuffle=shuffle_layers)
        self.set_trainable_layers()
        self.restricted_global_params = [torch.tensor(0)]*self.tot_layers

        if 'TR' in str(local_opt):
            local_opt_params['min_radius'] = 0 # avoids problems with "radius" being smallers than "min_radius"
            local_opt_params.update({'device': 'cuda'}) # TODO: make this more general
            
        self.local_optimizer = local_opt(self.model.parameters(), **local_opt_params)
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
    

    def closure(self, inputs, targets, model, sequential_derivative=None):
        # model.train()
        # model.eval()
        if sequential_derivative is not None:
            # here we compute the derivative using microbatches to reduce memory usage
            model.zero_grad()
            micro_batch_size = inputs.shape[0] // sequential_derivative
            J_total = 0
            for i in range(sequential_derivative):
                if i == sequential_derivative - 1:
                    inp = inputs[i*micro_batch_size:]
                    tar = targets[i*micro_batch_size:]
                else:
                    inp = inputs[i*micro_batch_size:(i+1)*micro_batch_size]        
                    tar = targets[i*micro_batch_size:(i+1)*micro_batch_size]
                    
                type_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)
                with type_ctx:
                    output = self.model(inp, tar)
                    if isinstance(output, dict): # This is needed for LLMs 
                        J = output['loss']
                    else:
                        J = self.loss_fn(output, tar)

                J = J/sequential_derivative
                J_total += J.item()
                if torch.is_grad_enabled():
                    # print(f"Rank {self.rank} J is a leaf node: {J.is_leaf}")
                    J.backward()
                    torch.cuda.empty_cache()    
        else:
            type_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)
            with type_ctx:
                if self.total_args == 3:
                    output = model(inputs,targets) # FOR LLMs 
                else:
                    output = model(inputs)
                if isinstance(output, dict): # This is needed for LLMs 
                    J = output['loss']
                else:
                    J = self.loss_fn(output, targets)

            if torch.is_grad_enabled():
                model.zero_grad()
                # print(f"Rank {self.rank} J is a leaf node: {J.is_leaf}")
                J.backward()
                torch.cuda.empty_cache()
            J_total = J.item()

        # print gradient norm:
        # if torch.is_grad_enabled():
        #     print(f'Rank {self.rank} seq_der {sequential_derivative} - J_tot {J_total} - grad norm {list_norm([p.grad for p in model.parameters() if p.requires_grad is True], p=torch.inf)}')
        # else:
        #     print(f'Rank {self.rank} seq_der {sequential_derivative} - J_tot {J_total}')
        # if sequential_derivative is not None:
        #     self.closure( inputs, targets, model, sequential_derivative=None)
        return J_total
        
    # def restricted_global_params(self): # Remove in production
    #     sub = list(self.model.parameters())
    #     return torch.cat([p.flatten().detach() for i,p in enumerate(self.model.parameters()) if sub[i].requires_grad is True])
        
    def local_model_params(self):
        return torch.cat([p.flatten().detach() for p in self.model.parameters() if p.requires_grad is True])

    # TODO: With layer-pipelining, this will not work
    def local_optimizers_sync(self):
        # Sync the nets
        if self.rank == 0:
            self.set_trainable_layers() # Set the global model back to the local model settings
        
        if self.rank == 0: # start sending
            global_opt_state_dict = {key: getattr(self.global_optimizer, key) for key in self.global_optimizer.collection_of_all_variables}
            dumped_dict = pickle.dumps(global_opt_state_dict)
            byte_dict = torch.ByteTensor(list(dumped_dict))
            object_list = [byte_dict]
            # print(object_list)
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
    
            
    def local_steps(inputs, targets):
        # Initialize constants and variables
        TEMP, local_loss, grad_norm2 = 2, 0, 0
        learning_rate = radius / TEMP
        gradients = initialize_gradients()
        # Compute initial loss
        initial_loss = closure(inputs, targets, model, derivative)
        # Preconditioning iterations
        for iteration in range(max_iterations):
            # Update optimizer settings
            update_optimizer(learning_rate)
            # Perform optimization step
            local_loss, gradient, grad_norm = optimizer_step(closure)
            # First iteration operations
            if is_first_iteration(iteration):
                grad_norm2 = max(grad_norm2, grad_norm)
                update_gradients(gradients, gradient)
            # Adjust learning rate
            learning_rate = adjust_learning_rate()
            # Break if learning rate is too small
            if learning_rate_too_small(learning_rate):
                break
        
    def local_steps(self, inputs, targets):
        TEMP = 2 # TODO: test whether having TEMP=2 here is beneficial
        local_loss = 0
        lr = self.radius/TEMP
        if self.rank == 0:
            receive_device = 'cpu' if self.backend == 'gloo' else 'cuda'
            g = [torch.zeros_like(p).to(receive_device) for p in self.model.parameters()]
        else:
            g = [torch.tensor(0)] * self.tot_layers   
        grad_norm2 = 0
        global_params_list = [p.clone().detach() for p in self.model.parameters() if p.requires_grad is True]
        OLD_LOSS= self.closure(inputs, targets, self.model, self.sequential_derivative)
        for j in range(self.max_iter): # Iterations of the preconditioning step            
            self.local_optimizer.radius = lr
            self.local_optimizer.max_radius = lr        
            local_loss, grad, grad_norm = self.local_optimizer.step(
                lambda dummy=1: self.closure(inputs, targets, self.model, self.sequential_derivative)) 
            if j == 0:
                grad_norm2 = max(grad_norm2, grad_norm)
                for i in range(self.tot_layers):
                    send_device = 'cpu' if self.backend == 'gloo' else grad[i].device
                    if i in self.rank2layer[self.rank]:
                        if self.rank != 0:
                            g[i] = grad[i].clone().detach().to(send_device, non_blocking=True)
                        else:
                            g[i] = grad[i].clone().detach().to('cuda', non_blocking=True)
                    else:
                        if self.rank != 0:
                            g[i] = torch.tensor(0.0, device=send_device)
            lr = self.radius/TEMP - list_norm(list_linear_comb(1, [p for p in self.model.parameters() if p.requires_grad is True],
                                                               -1, global_params_list), p=torch.inf)
            if lr < torch.finfo(torch.get_default_dtype()).eps:
                break
        # THEN SHARE THE LAYERS WITH THE MAIN MODEL
        if self.rank != 0:
            for layer in self.rank2layer[self.rank]:
                send_device = 'cpu' if self.backend == 'gloo' else g[layer].device
                dist.send(g[layer].to(send_device), dst=0, tag=layer)
        else:
            for i in range(1, self.nr_models):
                for layer in self.rank2layer[i]:
                    receive_device = 'cpu' if self.backend == 'gloo' else g[layer].device
                    g2 = torch.zeros_like(g[layer]).to(receive_device)
                    dist.recv(g2, src=i, tag=layer)
                    if self.backend == 'gloo':
                        g[layer] = g[layer].copy_(g2).to('cuda:0') # because only main process should receive (i.e. process 0)
                    else: # Daint
                        g[layer] = g[layer].copy_(g2).to('cuda')
        return local_loss, g, grad_norm2
    
    
    def synchronize_global_to_local(self):
        '''
        Synchronize the local models with global model parameters
        '''
        with torch.no_grad():
            if self.rank == 0:
                # Save the global model parameters
                self.global_model_params = copy.deepcopy(list(self.model.parameters()))
                # Send to other ranks
                dist.broadcast_object_list(self.global_model_params, src=0)

                # Update the model on rank 0 to be the local model
                for layer, p in enumerate(self.model.parameters()):  
                    if layer in self.rank2layer[self.rank]:
                        p.requires_grad = True  
                    else:
                        p.requires_grad = False
            else:
                # Placeholder to receive global parameters
                params_list = list(self.model.parameters())
                # Receive update global parameters from rank 0
                dist.broadcast_object_list(params_list, src=0)
                for layer, p in enumerate(self.model.parameters()):
                    p.copy_(params_list[layer])
                    if layer in self.rank2layer[self.rank]: # Trainable layer locally
                        p.requires_grad = True 
                    else: # Non-trainable layer locally
                        p.requires_grad = False
        # Ensure all processes have reached this point
        dist.barrier()
            
    def synchronize_local_to_global(self):
        with torch.no_grad():
            if self.rank == 0:
                for i in range(1, self.nr_models):
                    for layer, p in enumerate(self.model.parameters()):
                        if layer in self.rank2layer[i]:
                            orig_device = p.device
                            receive_device = 'cpu' if self.backend == 'gloo' else p.device
                            # layer_from_i = torch.zeros_like(p).to(receive_device)
                            dist.recv(p.to(receive_device), src=i, tag=layer)
                            p.copy_(p.to(orig_device))
                            # dist.recv(layer_from_i, src=i, tag=layer)
                            # assert p.shape == layer_from_i.shape
                            # p.copy_(layer_from_i) # TODO: test if this modifies things
                            p.requires_grad = True
                            # p = p.to(orig_device) <- this needs further testing TODO
            else:
                for layer, p in enumerate(self.model.parameters()):
                    if layer in self.rank2layer[self.rank]:
                        send_device = 'cpu' if self.backend == 'gloo' else p.device
                        dist.send(p.to(send_device), dst=0, tag=layer)

        # Ensure all processes have reached this point
        dist.barrier()
    
    def step(self, inputs, targets):  
        '''
        One step of APTS
        '''
        self.iter += 1
        skip_preconditioning = False
        
        if skip_preconditioning is False: 
            self.synchronize_global_to_local() # Synchronize the local models with global model parameters
            if self.rank == 0 and self.fdl: # Here self.model is the global model
                old_loss = self.closure(inputs, targets, self.model, self.sequential_derivative)
                
            # Compute local losses, global gradient at the previous iteration and new local params
            loss, g, _ = self.local_steps(inputs, targets) 
            
            print(f'Rank {self.rank} - loss {loss}')
            
            # Check that the loss is decreasing
            with torch.no_grad():
                self.synchronize_local_to_global() # Update self.model with new params from "local trained models" -> global_model_params = old params
                if self.fdl: # TODO: make this parallel, do some dogleg iterations in parallel and only keep the longest step which satisfies the decreasing loss condition
                    if self.rank == 0:        
                        # Compute step and step norm
                        step = list_linear_comb(1, list(self.model.parameters()), -1, self.global_model_params) # new global params - old global params                
                        step_norm = list_norm(step, p=torch.inf)

                        # Compute gradient scaled by step norm, so that in Dogleg, both have a similar contribution
                        g = list_linear_comb(step_norm/list_norm(g, p=torch.inf), g) 
                        
                        # Compute new loss
                        new_loss = self.closure(inputs, targets, self.model, self.sequential_derivative)

                        # TODO: this seems wrong. We should check if the new loss leads to a reduction. If not, we should go back to the original parameters.

                        # Compute the dogleg step with the hope that new_loss <= old_loss
                        radius = self.radius
                        w = 0; c = 0
                        while new_loss > old_loss: 
                            c += 1
                            # Decrease radius to decrease size of step...
                            radius = radius/2
                            # ... while moving towards the steepest descent direction (-g)
                            w = min(w + 0.2, 1)
                            step2 = list_linear_comb(1-w, step, -w, g) 
                            # The step length is "radius", with   radius <= self.radius   (global TR radius)
                            step2 = list_linear_comb(radius/list_norm(step2, p=torch.inf), step2)
                            
                            # Compute new global params, summing the old params with the new step
                            new_params = list_linear_comb(1, self.global_model_params, 1, step2)
                            # Update the model with the new params
                            for i,p in enumerate(self.model.parameters()):
                                p.copy_(new_params[i])
                                p.requires_grad = True # Just making sure not to lose the differentiability of the model
                                
                            # Compute new global loss
                            new_loss = self.closure(inputs, targets, self.model, self.sequential_derivative)
                            # Empty cache to avoid memory problems
                            torch.cuda.empty_cache()
                            if c > 5:
                                # In this case, we did not manage to decrease the global loss compared to the old global loss after 5 iterations (steep descent direction)
                                print(f'Warning: APTS is not decreasing the loss {new_loss} > {old_loss}')
                                break
                        # print(f' FDL : loss {new_loss} - old loss {old_loss} - c {c}')
        else:
            print('Skipping preconditioning')
            
        # Empty cache to avoid memory problems
        torch.cuda.empty_cache()
        # Ensure all processes have reached this point
        dist.barrier()
        # ------------------------------- Global smoothing step -------------------------------
        new_loss = 0
        if self.global_pass and self.rank == 0:
            self.global_optimizer.param_groups[0]['params'] = list(self.model.parameters())
            new_loss = self.global_optimizer.step(lambda dummy=1: self.closure(inputs, targets, self.model, self.sequential_derivative))[0]
            self.radius = torch.tensor(self.global_optimizer.radius)

        self.radius = self.radius.to('cpu' if self.backend=='gloo' else 'cuda:0')
        dist.barrier()
        dist.broadcast(self.radius, src=0) # Broadcast the radius
        if self.iter == 1:
            if skip_preconditioning:
                print('Skipping preconditioning')
            else:
                if self.fdl:
                    print('Modified APTS with forced decreasing loss')
        
        torch.cuda.empty_cache()
        return new_loss
















