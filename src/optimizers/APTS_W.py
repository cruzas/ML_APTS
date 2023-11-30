import torch, copy, random, inspect
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.optim as optim  # Optimizers
import numpy as np
from optimizers.TR import *
from torch.multiprocessing import Process
import pickle


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
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.max_iter = max_iter
        self.nr_models = nr_models if nr_models is not None else len(list(self.model.parameters()))
        self.global_pass = bool(global_pass) # in case global_pass is 0 or 1
        self.loss_fn = loss_fn
        self.iter = 0
        self.fdl = bool(forced_decreasing_loss) # in case forced_decreasing_loss is 0 or 1

        # Initialize the local models and optimizers
        self.rank2layer, self.layer2rank = self.define_trainable_layers(shuffle=shuffle_layers)
        self.set_trainable_layers(False)

        if 'TR' in str(local_opt):
            local_opt_params['min_radius'] = 0 # avoids problems with "radius" being smallers than "min_radius"
            local_opt_params.update({'device': self.device})
        self.local_optimizer = local_opt(self.model.parameters(), **local_opt_params)
        
        if dist.get_rank() == 0:
            self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
        self.radius = global_opt_params['radius']

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
        tot_layers = len(list(self.model.parameters()))
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
    
    def closure(self, inputs, targets):
        J = self.loss_fn(self.model(inputs), targets)
        if torch.is_grad_enabled():
            self.model.zero_grad()
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
            self.set_trainable_layers(False) # Set the global model back to the local model settings
        
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
        start_time = time.time()

        # Update the local model 
        TEMP = 2
        local_loss = [0 for _ in range(self.nr_models)]
        lr = [self.radius/TEMP] * self.nr_models
        # g = torch.zeros(self.tot_params, device=self.device); grad_norm2 = 0
        g = [0] * len(list(self.model.parameters())); grad_norm2 = 0

        for j in range(self.max_iter): # Iterations of the preconditioning step            
            self.local_optimizer.radius = lr
            self.local_optimizer.max_radius = lr            
            print(self.closure(inputs, targets))
            local_loss, grad, grad_norm = self.local_optimizer.step(lambda dummy=1: self.closure(inputs, targets)) 
            
            if j == 0:
                grad_norm2 = max(grad_norm2, grad_norm)
                for i, p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        g[i] = grad[i].clone().detach().to('cpu', non_blocking=True)
                        
            # Update radius to make sure the update is inside the global trust region radius
            lr = self.radius/TEMP - torch.norm(self.restricted_global_params())

            # print(f'Iteration {j}: (POST) radius {lr}, TR radius {self.local_optimizer.radius}')
            if lr < self.global_optimizer.min_radius/10:
                break
            
        # Final check
        if torch.norm(self.restricted_global_params()) > self.radius/TEMP+10**-3 or lr < -10**-3:
            print(f'Step too large or negative radius: {torch.norm(self.restricted_global_params())}>{self.radius/TEMP},   {lr}')

        # Sync the local models
        if self.rank != 0:
            dist.send(g, dst=0)
        else:
            for i in range(1, self.nr_models):
                g2 = dist.recv(src=i)
                for j,p in enumerate(self.model.parameters()):
                    if not p.requires_grad:
                        g[j].copy_(g2[j])

        print(f'Local steps took {time.time()-start_time} seconds')
        return local_loss, g, grad_norm2
    
    def get_local_params(self):
        '''
        Get all local trainable parameters
        '''
        with torch.no_grad():
            local_params = [0]*len(list(self.model.parameters()))
            for i in range(self.nr_models):
                for j,p in enumerate(self.local_optimizer[i].params):
                    if j in self.trainable_layers[i]:
                        local_params[j] = p.data
        return local_params #torch.cat(local_params)


    def global_params_update(self,local_params):
        '''
        Update the global model with the average of the local models
        '''
        with torch.no_grad():
            #if local_params is a list
            if isinstance(local_params, list):
                for j,p in enumerate(self.model.parameters()):
                    p.data.copy_(local_params[j])
            else:
                a=0
                for j,p in enumerate(self.model.parameters()):
                    # take the correct amount of parameters from local_params and reshape them into the shape of p
                    p.data.copy_(local_params[a:a+p.numel()].reshape(p.shape))       
                    a+=p.numel()
    
    def step(self, inputs, targets):  
        '''
        One step of APTS
        '''
        self.iter += 1
        skip_preconditioning = False
        if skip_preconditioning is False:
            # check current loss of local models:
            # with torch.no_grad():
                # initial_loss = [0 for _ in range(self.nr_models)]
                # for i in range(self.nr_models):
                #     initial_loss[i] = self.closure(inputs,targets,i)
                #     initial_loss[i] = round(initial_loss[i].item(),3)

            self.local_net_sync() # Sharing weights of the global NN with the local ones
            a = time.time()
            local_loss, g, g_norm = self.local_steps(inputs, targets)
            print(f'Local steps took {time.time()-a} seconds')

            for i in range(len(local_loss)): ### erease this
                local_loss[i] = round(local_loss[i],3)###
        
            # check that the loss is decreasing
            if self.fdl:
                x_loc = torch.cat([p.flatten() for p in self.get_local_params()])
                x_glob = torch.cat([p.flatten() for p in self.model.parameters()]) # TODO: this should be initialized before the preconditioning step, and only from rank 0
                step = x_loc - x_glob
                step = step/torch.norm(step, p=torch.inf)
                g = g/torch.norm(g, p=torch.inf)
                with torch.no_grad():
                    old_loss = self.closure(inputs, targets)
                self.global_params_update(self.get_local_params()) # update global model
                with torch.no_grad():
                    loss = self.closure(inputs, targets) # current loss
                radius = self.radius
                w=0; c=0
                while loss > old_loss: 
                    c += 1
                    # print(f'----------> Loss is increasing ({loss} > {self.old_loss}), reducing step size. W = '+ str(w))
                    radius = radius/2
                    w = min(w+0.2,1)
                    step2 = step*(1-w)+w*g
                    x_loc = x_glob + step2/torch.norm(step2, p=torch.inf)*radius
                    self.global_params_update(x_loc)
                    with torch.no_grad():
                        loss = self.closure(inputs, targets)
                    if c > 5:
                        print(f'Warning: APTS is not decreasing the loss {loss} > {old_loss}')
                        break
            else:
                self.global_params_update(self.get_local_params())

        # Global smoothing step 
        if self.global_pass:
            new_loss = self.global_optimizer.step(lambda x=1: self.closure(inputs, targets))[0]
            self.radius = self.global_optimizer.radius

        # print(f'Iteration {self.iter} | initial loss: {initial_loss}, after local steps: {local_loss}, loss before global step {loss}, final loss {new_loss}') ##
        if self.iter == 1:
            if skip_preconditioning:
                print('Skipping preconditioning')
            else:
                if self.fdl:
                    print('Modified APTS with forced decreasing loss')

        return new_loss, 2 # dummy output
















