import torch
import torch.distributed as dist

class APTS(torch.optim.Optimizer):
    def __init__(self, model, criterion, subdomain_optimizer, subdomain_optimizer_defaults, global_optimizer, global_optimizer_defaults, lr=0.01, max_subdomain_iter=5, dogleg=False):
        '''
        We use infinity norm for the gradient norm.
        '''
        super(APTS, self).__init__(model.parameters(), {'lr': lr, 'max_subdomain_iter': max_subdomain_iter, 'dogleg': dogleg})
        for key in self.param_groups[0].keys():  
            if key not in ['params']:
                setattr(self, key, self.param_groups[0][key])
        self.model = model # subdomain model
        self.criterion = criterion # loss function
        # Throw an error if 'lr' not in subdomain_optimizer_defaults.keys()
        if lr <= 0:
            raise ValueError('The learning rate "lr" must be bigger than 0.')
        subdomain_optimizer_defaults.update({'lr': lr})
        self.subdomain_optimizer = subdomain_optimizer(params=model.subdomain.parameters(), **subdomain_optimizer_defaults) # subdomain optimizer
        if 'TR' in str(global_optimizer):
            self.global_optimizer = global_optimizer(model=model, criterion=criterion, **global_optimizer_defaults) # TR optimizer
        else:
            global_optimizer_defaults.update({'lr': lr})
            self.global_optimizer = global_optimizer(params=model.subdomain.parameters(), **global_optimizer_defaults) # standard PyTorch optimizers
    
    # override of param_group (update)
    def update_param_group(self):
        for key in self.param_groups[0].keys():
            if key not in ['params']:
                self.param_groups[0][key] = getattr(self, key)
            
    def subdomain_steps(self):
        # Set up the learning rate
        self.subdomain_optimizer.param_groups[0]['lr'] = self.lr/self.max_subdomain_iter
        # Do subdomain steps
        for _ in range(self.max_subdomain_iter):
            self.subdomain_optimizer.zero_grad()
            self.model.subdomain.forward()
            self.model.subdomain.backward()
            #normalize gradient to 1 to avoid going out of the trust region
            grad_norm = self.model.subdomain.grad_norm()
            if grad_norm > 1: 
                for param in self.model.subdomain.parameters():
                    param.grad /= grad_norm
            self.subdomain_optimizer.step()
        # TODO: we have the gradient, we can compute its norm, we could use the norm to have an idea of the convergence of the subdomain optimization
        self.update_param_group()

    def step(self, closure):
        # Compute loss
        initial_loss = closure(compute_grad=True, zero_grad=True)
        # Store the initial parameters and gradients
        initial_parameters = self.model.parameters(clone=True)
        initial_grads = self.model.grad(clone=True)
        # Do subdomain steps
        self.subdomain_steps()
        with torch.no_grad():
            new_loss = closure(compute_grad=False, zero_grad=True)
            step = self.model.parameters(clone=False) - initial_parameters
            # Compute the dogleg step with the hope that new_loss <= old_loss
            lr = self.lr
            w = 0; c = 0
            while new_loss > initial_loss and self.dogleg and c>=5: 
                c += 1
                # Decrease lr to decrease size of step...
                lr = lr/2
                # ... while moving towards the steepest descent direction (-g)
                w = min(w + 0.2, 1)
                step2 = ((1-w)*step) - (w*initial_grads)
                # The step length is "lr", with   lr <= self.lr (global TR lr)
                step2 = (lr/step2.norm())*step2
                # Update the model with the new params
                for i,p in enumerate(self.model.parameters()):
                    p.copy_(initial_parameters.tensor[i] + step2.tensor[i])
                # Compute new global loss
                new_loss = closure(compute_grad=False, zero_grad=True)
                # Empty cache to avoid memory problems
                torch.cuda.empty_cache()
        # Do global TR step
        self.global_optimizer.step(closure)    
        # Update the learning rate
        self.lr = self.global_optimizer.lr
        self.update_param_group()

class TR(torch.optim.Optimizer):
    def __init__(self, model, criterion, lr=0.01, max_lr=1.0, min_lr=0.0001, nu=0.5, inc_factor=2.0, dec_factor=0.5, nu_1=0.25, nu_2=0.75, max_iter=5, norm_type=2):
        '''
        We use infinity norm for the gradient norm.
        '''
        super().__init__(model.parameters(), {'lr': lr, 'max_lr': max_lr, 'min_lr': min_lr, 'max_iter': max_iter})
        self.model = model
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.criterion = criterion
        self.inc_factor = inc_factor # increase factor for the learning rate
        self.dec_factor = dec_factor # decrease factor for the learning rate
        self.nu_1 = nu_1 # lower bound for the reduction ratio
        self.nu_2 = nu_2 # upper bound for the reduction ratio
        self.nu = min(nu, nu_1) # acceptable reduction ratio (cannot be bigger than nu_1)
        self.max_iter = max_iter # max iterations for the TR optimization
        self.norm_type = norm_type 
    
    def step(self, closure):
        # Compute the loss of the model
        old_loss = closure(compute_grad=True)
        # Retrieve the gradient of the model
        grad = self.model.grad()
        # Compute the norm of the gradient
        grad_norm2 = grad.norm(p=2)
        grad_norm = grad.norm(p=self.norm_type) if self.norm_type != 2 else grad_norm2 
        # Rescale the gradient to e at the edges of the trust region
        grad = grad * (self.lr/grad_norm)
        # Make sure the loss decreases
        new_loss = torch.inf; c = 0

        while old_loss - new_loss < 0 and c < self.max_iter:
            stop = True if abs(self.lr - self.min_lr)/self.min_lr < 1e-6 else False
            # TODO: Adaptive reduction factor -> if large difference in losses maybe divide by 4
            for i,param in enumerate(self.model.parameters()):
                param.data -= grad.tensor[i].data
            new_loss = closure(compute_grad=False)
            old_lr = self.lr
            
            # Compute the ratio of the loss reduction       
            act_red = old_loss - new_loss # actual reduction
            pred_red = self.lr*grad_norm2 # predicted reduction (first order approximation of the loss function)
            red_ratio = act_red / pred_red # reduction ratio
            
            if dist.get_rank() == 0:
                print(f'old loss: {old_loss}, new loss: {new_loss}, act_red: {act_red}, pred_red: {pred_red}, red_ratio: {red_ratio}')
            if red_ratio < self.nu_1: # the reduction ratio is too small -> decrease the learning rate
                self.lr = max(self.min_lr, self.dec_factor*self.lr)
            elif red_ratio > self.nu_2: # the reduction ratio is good enough -> increase the learning rate
                self.lr = min(self.max_lr, self.inc_factor*self.lr)
                break
            
            # Else learning rate remains unchanged
            if stop:
                break
            
            if red_ratio < self.nu:
                # In place of storing initial weights, we go backward from the current position whenever the step gets rejected (This only works with first order approximation of the loss)
                if self.lr != self.min_lr:
                    if c == 0: 
                        grad = grad * (-self.lr/old_lr)
                    else:
                        grad = grad * (self.lr/old_lr)
                else: # self.lr == self.min_lr
                    if c == 0:
                        grad = grad * (-(old_lr-self.lr)/old_lr)
                    else:
                        grad = grad * ((old_lr-self.lr)/old_lr)
                if dist.get_rank() == 0:
                    print(f'old loss: {old_loss}, new loss: {new_loss}, lr: {self.lr}')
            else:
                break
            c += 1