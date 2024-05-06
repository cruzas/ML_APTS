import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist # For parallel training
import numpy as np
# from scipy.optimize._hessian_update_strategy import BFGS, SR1
from hess_approx.OBS import *
from hess_approx.LSR1 import *
from hess_approx.LBGS import *

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

class TR(Optimizer): # TR optimizer
    def __init__(self,params,       # NN params
    radius=1,                       # TR radius
    max_radius=None,                # TR maximum radius
    min_radius=None,                # TR maximum radius
    decrease_factor=0.5,            # Factor by which to decrease the TR radius in case a step is not accepted
    increase_factor=None,           # Factor by which to increase the TR radius in case a step is accepted
    second_order=True,              # Enable approximation of Hessian (or inverse Hessian)
    second_order_method='SR1',      # Method to approximate Hessian (or inverse Hessian)
    delayed_second_order=0,         # Second order information will kick in after "delayed_second_order" calls to the "step" method
    device=None,                    # CPU or GPU
    accept_all=False,               # If accept all, accepts all steps, even if they lead to an increase in the loss
    acceptance_ratio=0.75,          # Increase factor of the radius when the region is flat enough
    reduction_ratio=0.25,           # Reduction factor of the radius when the region is not flat enough
    history_size=5,                 # Amount of stored vectors which forms the approximated Hessian
    momentum=True,                  # Momentum
    beta1=0.9,                      # Parameter for Adam momentum
    beta2=0.999,                    # Parameter for Adam momentum
    norm_type=2):                   # Norm type for the trust region radius

        super().__init__(params, {})
        self.steps = 0       # Number of times TR has been called
        self.iter_count = [] # Number of TR iterations per step in while-loop

        # Standard TR parameters
        self.radius = radius
        self.params = self.param_groups[0]['params']
        self.max_radius = max_radius if max_radius is not None else max(1.0, radius)
        self.min_radius = min_radius if min_radius is not None else min(0.001, radius)
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor if increase_factor is not None else 1.0 / decrease_factor
        self.acceptance_ratio = acceptance_ratio
        self.reduction_ratio = reduction_ratio
        # ACCEPT_ALL == True means we forgo monotonicity
        self.ACCEPT_ALL = bool(accept_all) # In case accept_all is 0 or 1
        # Device in which tensors reside     
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # TR variants (momentum, second-order)
        self.MOMENTUM = bool(momentum) # in case momentum is 0 or 1    
        self.SECOND_ORDER = bool(second_order) # in case second_order is 0 or 1
        self.delayed_second_order = delayed_second_order 
        
        self.norm_type = norm_type  
        self.tot_params = sum(p.numel() for p in self.params)

        if self.SECOND_ORDER:
            # Needed for second order information
            self.history_size = history_size #TODO: current H_prod function is working only for infinite memory.. fix it
            # print(f'Forcing infinite history of hessian : len = {history_size}')
            self.second_order_method = second_order_method if second_order else 'LBFGS'
            
            self.list_y = []  # List of gradient differences (y_k = g_k - g_{k-1})
            self.list_s = []  # List of iterate differences (s_k = x_k - x_{k-1})
            self.prev_x = None # parameters x_{k-1}
            self.prev_g = None # gradient g_{k-1}
            self.hessian_len = self.history_size  # How many iterate/gradient differences to keep in memory
            self.gamma = None # Initial approximation of the Hessian B_{0}.. "None" means adaptive according to book Nocedal
            # BFGS
            self.B = torch.tensor(1.0, device=self.device)
            self.list_r = []
            self.list_rho = []  # List of (1 / y_k^T s_k)
            self.list_rho2 = [] # List of (1 / s_k^T B_k s_k)
            self.list_Bs = []
            self.list_Binvr = []
            self.dampingLBFGS = torch.tensor(1.0, device=self.device) # TODO: positive values <1 seem to speed up convergence (no theoretical evidences)
            # SR1
            self.list_sigma = []
            self.list_sigma_inv = []
            self.list_v = []
            self.list_v_inv = []
            self.list_Sigma = []
            self.list_Sigma_inv = []
            self.list_V = [] # torch.empty((self.tot_params,1), device=self.device)
            self.list_V_inv = []
            self.r = torch.tensor(10**-8, device=self.device)
        # ADAM Momentum
        if self.MOMENTUM:
            self.var = torch.zeros(self.tot_params, device=self.device)
            self.mom = torch.zeros(self.tot_params, device=self.device)
            self.beta1 = torch.tensor(beta1, device=device)
            self.beta2 = torch.tensor(beta2, device=device)
            self.eps = torch.sqrt(torch.finfo(list(self.params)[0].dtype).eps).to(device) # sqrt of the machine precision

        self.collection_of_all_variables = []
        if self.SECOND_ORDER:
            if "SR1" in second_order_method:
                self.hessian_approx = LSR1(ha_memory=self.history_size)
            else:
                raise NotImplementedError
                # self.hessian_approx = LBFGS() # TODO: implement LBFGS
        
            self.subproblem = OBS()

            temp = [
                'B', 'list_r', 'list_rho', 'list_rho2', 'list_Bs', 'list_Binvr', 
                'dampingLBFGS', 'list_sigma', 'list_sigma_inv', 'list_v', 
                'list_v_inv', 'list_Sigma', 'list_Sigma_inv', 'list_V', 'list_V_inv', 'r', 'hessian_approx', 'subproblem'
            ]
            self.collection_of_all_variables.extend(temp)
        if self.MOMENTUM:
            temp = [
                'var', 'mom', 'beta1', 'beta2', 'eps'
            ]
            self.collection_of_all_variables.extend(temp)

    def update_hessian_history(self, x, g):
        """
        Updates the list of parameters needed to compute the inverse of the Hessian.

        Args:
        x: current parameters (x_k)
        g: current gradient (g(x_k))
        """
        with torch.no_grad():
            x = x.detach()
            g = g.detach()

            if self.prev_x is not None:
                x_diff = x - self.prev_x
                g_diff = g - self.prev_g

                if torch.norm(x_diff) > 10**-6 and torch.norm(g_diff) > 10**-6: #update only if needed
                    self.hessian_approx.update_memory(x_diff, g_diff)
                    # self.hessian_approx2.update_memory(x_diff.cpu().numpy(), g_diff.cpu().numpy())

            self.prev_x = x
            self.prev_g = g


    def step(self, closure, seed=[]):
        """
        Compute a step of the trust-region method.

        Args:
        loss: objective function f evaluated on current inputs, i.e. f(x)
        inputs: the inputs to the model/net
        labels: the ground truth/labels for the inputs

        Returns:
        If the trial point (x + step) is accepted, this returns the loss at the trial point f(x + s)
        """
        # TODO: Stop iterating when reaching the minimum radius and the direction is the gradient direction (smaller step is needed but this is not possible)
        # TODO: check on the way we computethe update of SR1 because the difference in gradient and weights are unrelated

        self.zero_grad()
        self.steps += 1
        self.iter_count.append(0)
        loss = closure() # NOTE: this is the only time we compute the gradient inside step()

        weights = [w for w in self.param_groups[0]['params'] if w.requires_grad] # Store the model's parameters
        x = [p.clone().detach() for p in weights]      # Store the model's parameters (flat)
        try:
            g = [p.grad.clone().detach() for p in weights] # Store the model's gradients (flat)
        except Exception as e:
            print('gradient is None: ' + e)
        
        # TODO: do this in APTS
        g_list = []; j = 0 
        for i,p in enumerate(self.param_groups[0]['params']):
            if p.requires_grad:
                g_list.append(g[j])
                j += 1
            else:
                g_list.append(torch.tensor(0))
        g_norm = list_norm(g, p=self.norm_type)                    # Gradient norm

        if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
            raise NotImplementedError
            self.update_hessian_history(x, g) # current weights and gradient
            if self.steps > self.delayed_second_order+1:
                self.hessian_approx.precompute()
                r = self.radius
                g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                s = g2.clone()
            else:
                s = g.clone()
        elif self.MOMENTUM:
            raise NotImplementedError
            self.mom = self.beta1 * self.mom + (1 - self.beta1) * g
            self.var = self.beta2 * self.var + (1 - self.beta2) * g**2
            g_mom = ((self.mom / (1 - self.beta1**self.steps)) / (torch.sqrt(self.var / (1 - self.beta2**self.steps)) + self.eps))
            s = g_mom.clone()
        else:
            s = g  
            
        # Update the model parameters
        END = 0
        actual_improvement = -1
        predicted_improvement = 1
        self.iter_count[self.steps-1] = 0 # NOTE: we can replace counter with self.iter_count
        while actual_improvement / predicted_improvement < 0 or actual_improvement < 0:
            lr = self.radius
            if seed:
                torch.manual_seed(seed)

            with torch.no_grad():
                if self.SECOND_ORDER and self.steps > self.delayed_second_order and self.steps > 1:
                    raise NotImplementedError
                    if self.iter_count[self.steps-1] > 1:
                        r = self.radius
                        try:
                            g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                        except:
                            g2 = g.clone()
                            print('(row 238) subproblem failed')
                        assert torch.isnan(g2).any() == False, "Nan in the update"
                        s = g2.clone()

                
                s_norm = list_norm(s, p=self.norm_type)
                if s_norm < abs(lr) and (self.SECOND_ORDER and self.delayed_second_order<=self.steps):
                    raise NotImplementedError
                    s_incr = 1
                else:
                    s_incr = lr/s_norm

                # Update model parameters
                if type(s_incr) is not torch.Tensor:
                    s_incr = torch.tensor(s_incr, device=self.device)#, dtype=torch.float32)
                    
                # Update the model parameters
                for i,p in enumerate(weights):
                    p.copy_(x[i] - s[i]*s_incr)
                    p.requires_grad = True

                # Compute the new loss
                new_loss = closure()

            # Compute the accuracy factor between predicted and actual change (rho)
            actual_improvement = loss - new_loss
            if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
                raise NotImplementedError
                predicted_improvement = (g @ s) * s_incr - 1/2 * (s @ self.hessian_approx.apply(s)) * s_incr**2
            elif self.MOMENTUM: # First order information only
                raise NotImplementedError
                predicted_improvement = g@(s*s_incr)# g_norm**2 * abs(s_incr) #- 1/2 * g_norm * s**2  # is like considering an identity approx of the Hessian
            else: # First order information only
                if self.norm_type < 1000: # infinity norm ;)
                    predicted_improvement = g_norm**self.norm_type * abs(s_incr) 
                else:
                    predicted_improvement = g@s * abs(s_incr) 
                    
            if actual_improvement > 0:
                if actual_improvement / predicted_improvement < self.reduction_ratio and predicted_improvement > 0: # Update the trust region radius
                    END = 1 if self.radius == self.min_radius else 0
                    self.radius = max(self.radius*self.decrease_factor, self.min_radius)
                elif actual_improvement / predicted_improvement > self.acceptance_ratio:
                    self.radius = min(self.radius*self.increase_factor, self.max_radius)
                    END = 1
                # TODO: add another criterion when the ratio is close to 1
                else:
                    END = 1
            else:
                END = 1 if self.radius == self.min_radius else 0
                self.radius = max(self.radius*self.decrease_factor, self.min_radius)

            lr = self.radius
            if self.ACCEPT_ALL:
                END = 1

            self.iter_count[self.steps-1] += 1       
            if self.radius <= self.min_radius+self.min_radius/100 or self.iter_count[self.steps-1]>10: #safeguard to avoid infinite loop
                END = 1

            if END == 1:
                break
                   
        return new_loss, g_list, g_norm
    


