import torch
import time
import torch.distributed as dist
# Paper [1]: OFFO minimization algorithms for second-order optimality and their complexity, S. Gratton and Ph. L. Toint, https://arxiv.org/pdf/2203.03351
# import adagrad

class TRAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, norm_type=torch.inf): #torch.inf

        super(TRAdam, self).__init__(params, {'lr': lr, 'betas': betas, 'eps': eps, 'norm_type': norm_type})
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.norm_type = norm_type
        if norm_type != 2 and norm_type != torch.inf:
            raise ValueError('The norm type must be 2 or torch.inf')
        self.m = [0 for _ in self.param_groups[0]['params']]
        self.v = [0 for _ in self.param_groups[0]['params']]
        self.t = 0
        self.timings = {}
    
    def reset_momentum(self):
        self.m = [0 for _ in self.param_groups[0]['params']]
        self.v = [0 for _ in self.param_groups[0]['params']]

    def get_timings(self):
        timings = {key: self.timings[key] for key in self.timings.keys()}
        return timings
    
    def zero_timers(self):
        self.timings = {key: 0 for key in self.timings.keys()}
        
    def display_avg_timers(self):
        '''Display the timings of the optimizer in a table format with ordered columns'''
        timings = self.get_timings()
        total_time = sum(timings.values())
        
        # Create headers
        headers = ["Timer", "Time (s)", "Percentage (%)"]
        
        # Create rows for each timer
        rows = [
            [key, f'{timings[key]:.4f}', f'{(timings[key]/total_time)*100:.2f}%']
            for key in sorted(timings.keys())
        ]
        
        # Find the maximum width of each column
        col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(3)]
        
        # Create a format string for each row
        row_format = '  '.join(f'{{:<{col_width}}}' for col_width in col_widths)
        
        # Create the table
        table = []
        table.append(row_format.format(*headers))
        table.append('-' * sum(col_widths))
        for row in rows:
            table.append(row_format.format(*row))
        
        print('\n'.join(table))
        return '\n'.join(table)
            
    def step(self ,closure=None):
        self.t += 1
        tic = time.time()
        loss = closure() if closure is not None else None 
        self.timings['TRAdam_closure_1'] = self.timings.get('TRAdam_closure_1',0) + time.time() - tic
        with torch.no_grad():
            s = [0 for _ in self.param_groups[0]['params']]
            step_length = 0
            for i, p in enumerate(self.param_groups[0]['params']):  # TODO: Run this in parallel through cuda streams ?
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                tic = time.time()
                # Initialize m
                self.m[i] = self.betas[0]*self.m[i] + (1-self.betas[0])*grad
                self.timings['TRAdam_init_m'] = self.timings.get('TRAdam_init_m',0) + time.time() - tic
                # Initialize v
                tic = time.time()
                self.v[i] = self.betas[1]*self.v[i] + (1-self.betas[1])*grad**2
                self.timings['TRAdam_init_v'] = self.timings.get('TRAdam_init_v',0) + time.time() - tic

                tic = time.time()
                m_hat = self.m[i]/(1 - self.betas[0]**self.t) # m_hat
                self.timings['TRAdam_init_mhat'] = self.timings.get('TRAdam_init_mhat',0) + time.time() - tic
                tic = time.time()
                v_hat = self.v[i]/(1 - self.betas[1]**self.t) # v_hat
                self.timings['TRAdam_init_vhat'] = self.timings.get('TRAdam_init_vhat',0) + time.time() - tic
                
                tic = time.time()
                s[i] = m_hat/(v_hat.sqrt() + self.eps)
                self.timings['TRAdam_comp_step'] = self.timings.get('TRAdam_comp_step',0) + time.time() - tic
                
                tic = time.time()
                if self.norm_type == torch.inf:
                    #step_length = max(step_length, torch.norm(s[i], p=self.norm_type).item())
                    step_length = 1 # upper bound for the step length to spare computation time (TODO: Check if this is actually true)
                else:
                    step_length += torch.norm(s[i]).item()**2
                self.timings['TRAdam_step_length'] = self.timings.get('TRAdam_step_length',0) + time.time() - tic
                
            tic = time.time()
            for i,p in enumerate(self.param_groups[0]['params']):
                # p.data -= s[i] * self.lr/step_length if step_length > self.lr and 2==1 else self.lr*s[i]
                p.data -= s[i] * self.lr/step_length if step_length > self.lr else s[i]
            self.timings['TRAdam_take_step'] = self.timings.get('TRAdam_take_step',0) + time.time() - tic
        return loss

        
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
        self.timings = {'smoother':0, 'precond':0, 'copy_params': 0, 'step_comp':0, 'dogleg':0, 'closure_1':0, 'closure_2':0}
   
    def get_timings(self):
        timings = {key: self.timings[key] for key in self.timings.keys()}
        if 'tradam' in str(self.subdomain_optimizer).lower():
            timings.update(self.subdomain_optimizer.get_timings())
        return timings

    def display_avg_timers(self):
        '''Display the timings of the optimizer in a table format with ordered columns'''
        timings = self.get_timings()
        timings.pop('precond')
        total_time = sum(timings.values())
        headers = ["Timer", "Time (s)", "Percentage (%)"]
        
        # Create rows for each timer
        rows = [
            [key, f'{timings[key]:.4f}', f'{(timings[key]/total_time)*100:.2f}%']
            for key in sorted(timings.keys())
        ]
        
        # Find the maximum width of each column
        col_widths = [max(len(row[i]) for row in rows + [headers]) for i in range(3)]
        
        # Create a format string for each row
        row_format = '  '.join(f'{{:<{col_width}}}' for col_width in col_widths)
        
        # Create the table
        table = []
        table.append(row_format.format(*headers))
        table.append('-' * sum(col_widths))
        for row in rows:
            table.append(row_format.format(*row))
        
        print('\n'.join(table))
        return '\n'.join(table)
        
        
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
            self.subdomain_optimizer.step()
        # Check if TRAdam is used as the local optimizer
        # if 'tradam' in str(self.subdomain_optimizer).lower():
        #     self.subdomain_optimizer.reset_momentum()
        self.update_param_group()

    def zero_timers(self):
        self.timings = {key: 0 for key in self.timings.keys()}
        self.subdomain_optimizer.zero_timers()

    def step(self, closure):
        # Compute loss
        tic = time.time()
        initial_loss = closure(compute_grad=True, zero_grad=True)
        self.timings['closure_1'] += time.time() - tic
        
        # Store the initial parameters and gradients
        tic = time.time()
        initial_parameters = self.model.parameters(clone=True)
        initial_grads = self.model.grad(clone=True)
        self.timings['copy_params'] += time.time() - tic
        # Do subdomain steps
        tic = time.time()
        self.subdomain_steps()
        self.timings['precond'] += time.time() - tic
        with torch.no_grad():
            tic = time.time()
            new_loss = closure(compute_grad=False, zero_grad=True)
            self.timings['closure_2'] += time.time() - tic
            tic = time.time()
            step = self.model.parameters(clone=False) - initial_parameters
            # Compute the dogleg step with the hope that new_loss <= old_loss
            lr = self.lr
            w = 0; c = 0
            self.timings['step_comp'] += time.time() - tic
            tic = time.time()
            if self.dogleg:
                while new_loss > initial_loss and c>=5: 
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
            else:
                # Update the model with the new params
                for i,p in enumerate(self.model.parameters()):
                    p.copy_(initial_parameters.tensor[i] + step.tensor[i])
            self.timings['dogleg'] += time.time() - tic

        # Do global TR step
        tic = time.time()
        self.global_optimizer.step(closure)   
        self.timings['smoother'] += time.time() - tic
         
        # Update the learning rate
        self.lr = self.global_optimizer.lr
        self.update_param_group()
        return new_loss

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
        # Retrieve the gradient of the model  TODO: check if this is a copy by reference or not (if not we could use param.data -= param.grad ..... below)
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
            
            # if dist.get_rank() == 0:
            #     print(f'old loss: {old_loss}, new loss: {new_loss}, act_red: {act_red}, pred_red: {pred_red}, red_ratio: {red_ratio}')
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
                # if dist.get_rank() == 0:
                #     print(f'old loss: {old_loss}, new loss: {new_loss}, lr: {self.lr}')
            else:
                break
            c += 1
