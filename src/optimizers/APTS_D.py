import torch
import copy
import torch.optim as optim  # Optimizers
import torch.distributed as dist  # For parallel training
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from optimizers.TR import *

class APTS_D(Optimizer):
    def __init__(self,
                 params,
                 model=None,
                 loss_fn=None,
                 device=None,
                 max_iter=5, 
                 nr_models=None,
                 global_opt=None,
                 global_opt_params=None,
                 local_opt=None,
                 local_opt_params=None,
                 global_pass=True,
                 foc=True):

        super(APTS_D, self).__init__(params, {})

        self.model = model
        self.local_model = copy.deepcopy(model)
        self.max_iter = max_iter
        self.nr_models = nr_models
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.loss_fn = loss_fn # loss function
        self.global_pass = bool(global_pass) # in case global_pass is 0 or 1
        self.foc = bool(foc)  # first-order-consistency

        # Needed for closures
        self.inputs = None # Needs to be passed to step()
        self.labels = None # Needs to be passed to step()
        self.residual = None # Computed inside step() 
        # NOTE: no need for a counter inside local or global optimizers as TR methods only evaluate the gradient once, even if multiple iterations are performed inside
        self.grad_evals_counter = torch.tensor(0.0, device=self.device) # Count number of gradient evaluations assuming one full batch (user should take care of calculations for multiple mini-batches in their main script)

        self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
        self.local_optimizer = local_opt(self.local_model.parameters(), **local_opt_params)

        self.radius = self.global_optimizer.radius
        

    # Average gradients across NCCL processes
    def average_gradients(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.nr_models


    # Normal closure for local (TR) optimizer without first-order consistency term
    def non_foc_local_closure(self):
        self.local_optimizer.zero_grad()  # Zero the gradients
        outputs = self.local_model(self.inputs)  # Forward pass
        loss = self.loss_fn(outputs, self.labels) # Compute the loss
        if torch.is_grad_enabled():
            loss.backward()  # Backpropagation
        return loss


    # First-order consistent closure for local (TR) optimizer
    def foc_local_closure(self):
        local_loss = self.non_foc_local_closure()
        # Flatten global model parameters
        global_params = torch.cat([p.view(-1) for p in self.model.parameters()])
        local_params = torch.cat([p.view(-1) for p in self.local_model.parameters()])
        s = local_params - global_params

        local_loss = local_loss + (self.residual @ s) if not torch.all(torch.abs(s) < 1e-8) else local_loss
        return local_loss


    # Normal closure for global (TR) optimizer
    def global_closure(self):
        self.zero_grad()
        outputs = self.model(self.inputs)
        global_loss = self.loss_fn(outputs, self.labels)
        if torch.is_grad_enabled():
            global_loss.backward()
        if self.nr_models > 1:
            # Global loss
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            global_loss /= self.nr_models
            self.average_gradients()

        return global_loss


    # One step of APTS
    def step(self, inputs, labels):
        self.grad_evals_counter = torch.tensor(0.0, device=self.device)
        self.inputs = inputs
        self.labels = labels
        self.residual = torch.tensor(0.0, device=self.device)
        local_closure = self.foc_local_closure if self.foc else self.non_foc_local_closure
        with torch.no_grad():
            initial_params = torch.cat([p.view(-1) for p in self.model.parameters()])
        
        # Compute required initial values
        initial_global_loss = self.global_closure()
        self.grad_evals_counter += 1 # 1 full gradient evaluation 
        initial_local_loss = local_closure()
        # self.grad_evals_counter += 1.0 # after all subdomain gradient evaluations are added together, they add up to 1, since local_closure() computes the subdomain gradient only once
        # NOTE/TODO: currently, not counting in line 162 since after an optimization (saving the initial local gradient in TR), we don't need to compute it in the first iteration of the for loop starting on line 170

        global_gradient = torch.cat([p.grad.flatten() for p in self.model.parameters()]).detach()
        initial_local_gradient = (torch.cat([p.grad.flatten() for p in self.local_model.parameters()]).clone().detach())
        self.residual = global_gradient - initial_local_gradient

        # Compute local correction
        for _ in range(self.max_iter):
            local_loss, _, _ = self.local_optimizer.step(local_closure)  # Local optimization step
            local_grad_evals_counter = torch.tensor(1.0, device=self.device) / self.nr_models 
            if self.nr_models > 1:
                # Add all local gradient evaluation counters
                dist.all_reduce(local_grad_evals_counter, op=dist.ReduceOp.SUM)
            self.grad_evals_counter += local_grad_evals_counter
            with torch.no_grad():
                # Local step = new parameters - initial parameters
                step = torch.cat([p.view(-1) for p in self.local_model.parameters()]) - initial_params
                # If local step has reached local trust-region radius, break the local trust-region preconditioning
                if torch.norm(step, p=2) >= self.local_optimizer.max_radius:
                    break

        # Local reduction
        with torch.no_grad():
            local_reduction = initial_local_loss - torch.tensor(local_loss, device=self.device)

            # Sum local steps
            if self.nr_models > 1:
                dist.all_reduce(step, op=dist.ReduceOp.SUM)

            # Apply step to global model
            a = 0
            for param in self.model.parameters():
                b = param.numel()  # Number of parameters in the given layer
                # This command is probably slow. Maybe use .add_() in some way.
                param.data.copy_(torch.reshape(initial_params[a:a+b] + step[a:a+b], param.shape))
                a += b

            # Compute global trial loss
            trial_loss = self.global_closure() # NOTE: no gradient evaluated here (see 'with torch.no_grad()' in line 179), so not counted towards grad_evals_counter

            # Sum all local reductions
            if self.nr_models > 1:
                # Compute global acceptance ratio, i.e. actual reduction over sum of predicted local reductions
                dist.all_reduce(local_reduction, op=dist.ReduceOp.SUM)

            # Acceptance ratio = global reduction / local reduction measures the quality of the local models
            acceptance_ratio = (initial_global_loss - trial_loss) / local_reduction

            # Update the trust region radius
            new_loss = trial_loss
            if acceptance_ratio < self.global_optimizer.reduction_ratio:
                # Failed step: not enough reduction, so shrink trust-region radius
                self.radius = max(self.radius*self.global_optimizer.decrease_factor, self.global_optimizer.min_radius)
                # Restore initial model parameters
                a = 0
                for param in self.model.parameters():
                    b = param.numel()  # Number of parameters in the given layer
                    # This command is probably slow. Maybe use .add_() in some way.
                    param.data.copy_(torch.reshape(initial_params[a:a+b], param.shape))
                    a += b

                new_loss = initial_global_loss

            elif acceptance_ratio > self.global_optimizer.acceptance_ratio:
                # Expand trust-region radius
                self.radius = min(self.radius*self.global_optimizer.increase_factor, self.global_optimizer.max_radius)

            # Update global radius
            self.global_optimizer.radius = self.radius

        # Global smoothing
        if self.global_pass:
            new_loss, _, _ = self.global_optimizer.step(self.global_closure)
            self.grad_evals_counter += 1 # Global closure performs a full gradient evaluation

        with torch.no_grad():
            # Update global and local optimizer learning rates (radius)
            self.radius = self.global_optimizer.radius
            self.local_optimizer.radius = self.radius / self.nr_models
            # Copy global model parameters to each local model
            self.local_model.load_state_dict(self.model.state_dict())

        return new_loss
