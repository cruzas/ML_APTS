from models.simple_nn import * #import all NN models
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist # For parallel training
from scipy.optimize import minimize_scalar
import numpy as np
import time
from scipy.optimize._hessian_update_strategy import BFGS, SR1
from optimizers.Hessian_approx import *
from hess_approx.LSR1 import *
from hess_approx.LBGS import *



# class TR(Optimizer): # TR optimizer
#     def __init__(self,params,       # NN params
#     radius=1,                       # TR radius
#     max_radius=None,                # TR maximum radius
#     min_radius=None,                # TR maximum radius
#     decrease_factor=0.5,            # Factor by which to decrease the TR radius in case a step is not accepted
#     increase_factor=None,           # Factor by which to increase the TR radius in case a step is accepted
#     is_adaptive=False,              # If adaptive, performs polynomial interpolation to find TR radius
#     second_order=True,              # Enable approximation of Hessian (or inverse Hessian)
#     second_order_method='SR1',      # Method to approximate Hessian (or inverse Hessian)
#     delayed_second_order=0,         # Second order information will kick in after "delayed_second_order" calls to the "step" method
#     device=None,                    # CPU or GPU
#     accept_all=False,               # If accept all, accepts all steps, even if they lead to an increase in the loss
#     acceptance_ratio=0.75,          # Increase factor of the radius when the region is flat enough
#     reduction_ratio=0.25,           # Reduction factor of the radius when the region is not flat enough
#     history_size=5,                 # Amount of stored vectors which forms the approximated Hessian
#     momentum=True,                  # Momentum
#     beta1=0.9,                      # Parameter for Adam momentum
#     beta2=0.999,                    # Parameter for Adam momentum
#     norm_type=2):                   # Norm type for the trust region radius

#         super().__init__(params, {})
        
#         self.steps = 0
#         self.radius = radius
#         self.params = self.param_groups[0]['params']
#         self.max_radius = max_radius if max_radius is not None else max(1.0, radius)
#         self.min_radius = min_radius if min_radius is not None else min(0.001, radius)
#         self.decrease_factor = decrease_factor
#         self.increase_factor = increase_factor if increase_factor is not None else 1.0 / decrease_factor
#         self.ADAPTIVE = bool(is_adaptive) # in case is_adaptive is 0 or 1
#         self.SECOND_ORDER = bool(second_order) # in case second_order is 0 or 1
#         self.delayed_second_order = delayed_second_order
#         self.second_order_method = second_order_method if second_order else 'LBFGS'
#         self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.ACCEPT_ALL = bool(accept_all) # in case accept_all is 0 or 1
#         self.acceptance_ratio = acceptance_ratio
#         self.reduction_ratio = reduction_ratio
#         # print(f'Forcing infinite history of hessian : len = {history_size}')
#         self.history_size = history_size #TODO: current H_prod function is working only for infinite memory.. fix it
#         self.MOMENTUM = bool(momentum) # in case momentum is 0 or 1    
#         self.norm_type = norm_type  

#         # self.tot_params = sum(p.numel() for p in self.params if p.requires_grad)
#         self.tot_params = sum(p.numel() for p in self.params)

#         # Needed for second order information
#         self.list_y = []  # List of gradient differences (y_k = g_k - g_{k-1})
#         self.list_s = []  # List of iterate differences (s_k = x_k - x_{k-1})
#         self.prev_x = None # parameters x_{k-1}
#         self.prev_g = None # gradient g_{k-1}
#         self.hessian_len = self.history_size  # How many iterate/gradient differences to keep in memory
#         self.gamma = None # Initial approximation of the Hessian B_{0}.. "None" means adaptive according to book Nocedal
        
#         if self.SECOND_ORDER:
#             # BFGS
#             self.B = torch.tensor(1.0, device=self.device)
#             self.list_r = []
#             self.list_rho = []  # List of (1 / y_k^T s_k)
#             self.list_rho2 = [] # List of (1 / s_k^T B_k s_k)
#             self.list_Bs = []
#             self.list_Binvr = []
#             self.dampingLBFGS = torch.tensor(1.0, device=self.device) # TODO: positive values <1 seem to speed up convergence (no theoretical evidences)
#             # SR1
#             self.list_sigma = []
#             self.list_sigma_inv = []
#             self.list_v = []
#             self.list_v_inv = []
#             self.list_Sigma = []
#             self.list_Sigma_inv = []
#             self.list_V = [] # torch.empty((self.tot_params,1), device=self.device)
#             self.list_V_inv = []
#             self.r = torch.tensor(10**-8, device=self.device)
#         # ADAM Momentum
#         if self.MOMENTUM:
#             self.var = torch.zeros(self.tot_params, device=self.device)
#             self.mom = torch.zeros(self.tot_params, device=self.device)
#             self.beta1 = torch.tensor(beta1, device=device)
#             self.beta2 = torch.tensor(beta2, device=device)
#             self.eps = torch.sqrt(torch.tensor(torch.finfo(list(self.params)[0].dtype).eps, device=device)) # sqrt of the machine precision



#         # a =       16.84
#         # b =      -17.04
#         # c =         0.5
#         # self.r_adaptive = lambda x: a*x+b*np.sin(x)+c#a*np.exp(-b*x)+c#(2.66*x**2+0.01*x+0.33)/1.5
#         self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.5,0.5,0.7,1,1.4,2,3])#slightly better
#         # self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.6,0.7,0.8,1,1.2,1.5,2])#

#         self.neg = 0
#         # Dogleg
#         self.tau = 1.5
#         self.lr_sup = None
#         self.lr_inf = None
#         self.list_lr= []

#         # self.hessian_approx = Hessian_approx(gamma=1, device=self.device, tol=10**-6, method=self.second_order_method)
#         # self.hessian_approx = OBS(gamma=1, device=self.device, tol=10**-6, method=self.second_order_method)
#         if self.SECOND_ORDER:
#             if "SR1" in second_order_method:
#                 self.hessian_approx = LSR1_TORCH(ha_memory=self.history_size)
#                 # self.hessian_approx2 = LSR1(ha_memory=self.history_size)
#             else:
#                 raise NotImplementedError
#                 # self.hessian_approx = LBFGS()
        
#         self.subproblem = OBS_TORCH()
#         # self.subproblem2 = OBS()
#         self.iter_count = 0 # could we use self.steps instead?

#         self.collection_of_all_variables = []
#         if self.SECOND_ORDER:
#             temp = [
#                 'B', 'list_r', 'list_rho', 'list_rho2', 'list_Bs', 'list_Binvr', 
#                 'dampingLBFGS', 'list_sigma', 'list_sigma_inv', 'list_v', 
#                 'list_v_inv', 'list_Sigma', 'list_Sigma_inv', 'list_V', 'list_V_inv', 'r', 'hessian_approx', 'subproblem'
#             ]
#             self.collection_of_all_variables.extend(temp)
#         if self.MOMENTUM:
#             temp = [
#                 'var', 'mom', 'beta1', 'beta2', 'eps'
#             ]
#             self.collection_of_all_variables.extend(temp)

#     def update_hessian_history(self, x, g):
#         """
#         Updates the list of parameters needed to compute the inverse of the Hessian.

#         Args:
#         x: current parameters (x_k)
#         g: current gradient (g(x_k))
#         """
#         with torch.no_grad():
#             x = x.detach()
#             g = g.detach()

#             if self.prev_x is not None:
#                 x_diff = x - self.prev_x
#                 g_diff = g - self.prev_g

#                 if torch.norm(x_diff) > 10**-6 and torch.norm(g_diff) > 10**-6: #update only if needed
#                     self.hessian_approx.update_memory(x_diff, g_diff)
#                     # self.hessian_approx2.update_memory(x_diff.cpu().numpy(), g_diff.cpu().numpy())

#             self.prev_x = x
#             self.prev_g = g


#     def step(self, closure, seed=[]):
#         """
#         Compute a step of the trust-region method.

#         Args:
#         loss: objective function f evaluated on current inputs, i.e. f(x)
#         inputs: the inputs to the model/net
#         labels: the ground truth/labels for the inputs

#         Returns:
#         If the trial point (x + step) is accepted, this returns the loss at the trial point f(x + s)
#         """
#         # TODO: Stop iterating when reaching the minimum radius and the direction is the gradient direction (smaller step is needed but this is not possible)
#         # TODO: check on the way we computethe update of SR1 because the difference in gradient and weights are unrelated

#         self.zero_grad()
#         self.steps += 1
#         loss = closure() # NOTE: this is the only time we compute the gradient inside step()

#         # weights = self.param_groups[0]['params']           
#         weights = [w for w in self.param_groups[0]['params'] if w.requires_grad] # Store the model's parameters
#         x = torch.cat([p.flatten() for p in weights]).detach()      # Store the model's parameters (flat)
#         g = torch.cat([p.grad.flatten() for p in weights]).detach() # Store the model's gradients (flat)
#         g_norm = torch.norm(g, p=self.norm_type)                    # Gradient norm
#         assert g.isnan().sum() == 0, "Gradient is NaN"
#         assert g_norm != 0, "Gradient is zero. Cannot update the model."

#         if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
#             self.update_hessian_history(x, g) # current weights and gradient
#             if self.steps > self.delayed_second_order+1:
#                 self.hessian_approx.precompute()
#                 # self.hessian_approx2.precompute()
#                 r=self.radius
#                 try:
#                     g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
#                 except:
#                     g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
#                     # g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                    
#                     # g2 = -self.subproblem2.solve_tr_subproblem(g.cpu().numpy(), r, self.hessian_approx2.gamma, self.hessian_approx2.Psi, self.hessian_approx2.M_inv)
#                     # print(torch.norm(g2.cpu()-torch.tensor(g3)))
#                     # if torch.norm(g2.cpu()-torch.tensor(g3))>1:
#                     #     print(torch.norm(torch.tensor(self.hessian_approx2.M_inv)-self.hessian_approx.M_inv.cpu()))
#                     #     print(torch.norm(torch.tensor(self.hessian_approx2.Psi)-self.hessian_approx.Psi.cpu()))
#                     #     print(torch.norm(torch.tensor(self.hessian_approx2.gamma)-self.hessian_approx.gamma.cpu()))
#                     #     g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
#                     #     g3 = -self.subproblem2.solve_tr_subproblem(g.cpu().numpy(), r, self.hessian_approx2.gamma, self.hessian_approx2.Psi, self.hessian_approx2.M_inv)

                    
#                     g2 = g.clone()
#                     print('(row 203) subproblem failed')
#                 g2 = torch.tensor(g2, device=self.device, dtype=torch.float32)
#                 s = g2.clone()
#             else:
#                 s = g.clone()

#         else:
#             if self.MOMENTUM:
#                 self.mom = self.beta1 * self.mom + (1 - self.beta1) * g
#                 self.var = self.beta2 * self.var + (1 - self.beta2) * g**2
#                 g_mom = ((self.mom / (1 - self.beta1**self.steps)) / (torch.sqrt(self.var / (1 - self.beta2**self.steps)) + self.eps))
#                 s = g_mom.clone()
#             else:
#                 s = g.clone()    
#         # Update the model parameters
#         END = 0
#         actual_improvement = -1
#         predicted_improvement = 1
#         c = 0 # counter
#         self.iter_count = 0 # NOTE: we can replace counter with self.iter_count
#         while actual_improvement / predicted_improvement < 0 or actual_improvement < 0:
#             lr = self.radius
#             if seed:
#                 torch.manual_seed(seed)

#             with torch.no_grad():
#                 if self.SECOND_ORDER and self.steps > self.delayed_second_order and self.steps > 1:
#                     if c > 1:
#                         r=self.radius
#                         try:
#                             g2 = -self.subproblem.solve_tr_subproblem(g, r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
#                         except:
#                             g2 = g.clone()
#                             print('(row 238) subproblem failed')
#                         g2 = torch.tensor(g2, device=self.device, dtype=torch.float32)
#                         assert torch.isnan(g2).any() == False, "Nan in the update"
#                         g2_norm = torch.norm(g2, p=self.norm_type)
#                         s = g2.clone()

                
#                 s_norm = torch.norm(s, p=self.norm_type)
#                 if s_norm < abs(lr) and (self.SECOND_ORDER and self.delayed_second_order<=self.steps):
#                     s_incr = 1
#                 else:
#                     s_incr = lr/s_norm

#                 # Update model parameters
#                 a=0; 
#                 if type(s_incr) is not torch.Tensor:
#                     s_incr = torch.tensor(s_incr, device=self.device, dtype=torch.float32)
#                 for param in weights:
#                     b = param.numel()
#                     assert torch.isnan(x[a:a+b] - s[a:a+b]*s_incr).any() == False, "Nan in the update"
#                     param.data.copy_(torch.reshape(x[a:a+b] - s[a:a+b]*s_incr, param.shape))
#                     a += b

#                 # Compute the new loss
#                 new_loss = closure()

#             # print(f"norm of new weights: {torch.norm(torch.cat([p.flatten() for p in weights]), p=2)}")
#             # Compute the accuracy factor between predicted and actual change (rho)
#             actual_improvement = loss - new_loss.item()
#             if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
#                 predicted_improvement = (g @ s) * s_incr - 1/2 * (s @ self.hessian_approx.apply(s)) * s_incr**2
#             elif self.MOMENTUM: # First order information only
#                 predicted_improvement = g@(s*s_incr)# g_norm**2 * abs(s_incr) #- 1/2 * g_norm * s**2  # is like considering an identity approx of the Hessian
#             else: # First order information only
#                 predicted_improvement = g_norm**2 * abs(s_incr) 
                    
#             if actual_improvement>0:
#                 if self.ADAPTIVE: # Uses a function to adaptively change the radius
#                     ratio = max(0,min(1,actual_improvement/predicted_improvement.item()))
#                     LR = self.radius * self.r_adaptive(ratio)
#                     self.radius = min(self.max_radius,max(self.min_radius,LR))
#                     if ratio >= 1-self.acceptance_ratio:
#                         END = 1
#                 else: # Standard TR update
#                     if actual_improvement / predicted_improvement < self.reduction_ratio and predicted_improvement > 0: # Update the trust region radius
#                         END = 1 if self.radius == self.min_radius else 0
#                         self.radius = max(self.radius*self.decrease_factor, self.min_radius)
#                     elif actual_improvement / predicted_improvement > self.acceptance_ratio:
#                         self.radius = min(self.radius*self.increase_factor, self.max_radius)
#                         END = 1
#                     # TODO: add another criterion when the ratio is close to 1
#                     else:
#                         END = 1
#             else:
#                 END = 1 if self.radius == self.min_radius else 0
#                 self.radius = max(self.radius*self.decrease_factor, self.min_radius)

#             lr = self.radius
#             if self.ACCEPT_ALL:
#                 END = 1

#             c += 1     
#             self.iter_count += 1       
#             if self.radius <= self.min_radius+self.min_radius/100 or c>10: #safeguard to avoid infinite loop
#                 END = 1

#             if END==1:
#                 break
                   
#         return new_loss.item(), g, g_norm
    
    


class TR(Optimizer): # TR optimizer
    def __init__(self,params,       # NN params
    radius=1,                       # TR radius
    max_radius=None,                # TR maximum radius
    min_radius=None,                # TR maximum radius
    decrease_factor=0.5,            # Factor by which to decrease the TR radius in case a step is not accepted
    increase_factor=None,           # Factor by which to increase the TR radius in case a step is accepted
    is_adaptive=False,              # If adaptive, performs polynomial interpolation to find TR radius
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
        
        self.steps = 0
        self.radius = radius
        self.params = self.param_groups[0]['params']
        self.max_radius = max_radius if max_radius is not None else max(1.0, radius)
        self.min_radius = min_radius if min_radius is not None else min(0.001, radius)
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor if increase_factor is not None else 1.0 / decrease_factor
        self.ADAPTIVE = bool(is_adaptive) # in case is_adaptive is 0 or 1
        self.SECOND_ORDER = bool(second_order) # in case second_order is 0 or 1
        self.delayed_second_order = delayed_second_order
        self.second_order_method = second_order_method if second_order else 'LBFGS'
        self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ACCEPT_ALL = bool(accept_all) # in case accept_all is 0 or 1
        self.acceptance_ratio = acceptance_ratio
        self.reduction_ratio = reduction_ratio
        # print(f'Forcing infinite history of hessian : len = {history_size}')
        self.history_size = history_size #TODO: current H_prod function is working only for infinite memory.. fix it
        self.MOMENTUM = bool(momentum) # in case momentum is 0 or 1    
        self.norm_type = norm_type  

        # self.tot_params = sum(p.numel() for p in self.params if p.requires_grad)
        self.tot_params = sum(p.numel() for p in self.params)

        # Needed for second order information
        self.list_y = []  # List of gradient differences (y_k = g_k - g_{k-1})
        self.list_s = []  # List of iterate differences (s_k = x_k - x_{k-1})
        self.prev_x = None # parameters x_{k-1}
        self.prev_g = None # gradient g_{k-1}
        self.hessian_len = self.history_size  # How many iterate/gradient differences to keep in memory
        self.gamma = None # Initial approximation of the Hessian B_{0}.. "None" means adaptive according to book Nocedal
        self.model_accuracy_lvl = -1 # 2**model_accuracy Needed to average 2nd order direction with 1st order direction
        self.model_accuracy_range = [0, 0.1, 0.3, 0.5, 0.75, 1] # 2**model_accuracy Needed to average 2nd order direction with 1st order direction
        
        if self.SECOND_ORDER:
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
            self.eps = torch.sqrt(torch.tensor(torch.finfo(list(self.params)[0].dtype).eps, device=device)) # sqrt of the machine precision



        # a =       16.84
        # b =      -17.04
        # c =         0.5
        # self.r_adaptive = lambda x: a*x+b*np.sin(x)+c#a*np.exp(-b*x)+c#(2.66*x**2+0.01*x+0.33)/1.5
        self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.5,0.5,0.7,1,1.4,2,3])#slightly better
        # self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.6,0.7,0.8,1,1.2,1.5,2])#

        self.neg = 0
        # Dogleg
        self.tau = 1.5
        self.lr_sup = None
        self.lr_inf = None
        self.list_lr= []

        # self.hessian_approx = Hessian_approx(gamma=1, device=self.device, tol=10**-6, method=self.second_order_method)
        # self.hessian_approx = OBS(gamma=1, device=self.device, tol=10**-6, method=self.second_order_method)
        if self.SECOND_ORDER:
            if "SR1" in second_order_method:
                self.hessian_approx = LSR1(ha_memory=self.history_size)
            else:
                self.hessian_approx = LBFGS()
        
        self.subproblem = OBS()
        self.iter_count = 0 # could we use self.steps instead?

        self.collection_of_all_variables = []
        if self.SECOND_ORDER:
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
                
                x_diff = x_diff.cpu().numpy()
                g_diff = g_diff.cpu().numpy()

                if np.linalg.norm(x_diff) > 10**-6 and np.linalg.norm(g_diff) > 10**-6: #update only if needed
                    self.hessian_approx.update_memory(x_diff, g_diff)

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
        loss = closure() # NOTE: this is the only time we compute the gradient inside step()

        # weights = self.param_groups[0]['params']           
        weights = [w for w in self.param_groups[0]['params'] if w.requires_grad] # Store the model's parameters
        x = torch.cat([p.flatten() for p in weights]).detach()      # Store the model's parameters (flat)
        g = torch.cat([p.grad.flatten() for p in weights]).detach() # Store the model's gradients (flat)
        g_norm = torch.norm(g, p=self.norm_type)                    # Gradient norm
        assert g.isnan().sum() == 0, "Gradient is NaN"
        assert g_norm != 0, "Gradient is zero. Cannot update the model."

        if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
            self.update_hessian_history(x, g) # current weights and gradient
            if self.steps > self.delayed_second_order+1:
                self.hessian_approx.precompute()
                try:
                    r=self.radius.cpu().numpy()
                except:
                    r=self.radius
                try:
                    g2 = -self.subproblem.solve_tr_subproblem(g.cpu().numpy(), r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                except:
                    g2 = g.clone()
                    print('(row 203) subproblem failed')
                g2 = torch.tensor(g2, device=self.device, dtype=torch.float32)
                s = g2.clone()
            else:
                s = g.clone()

        else:
            if self.MOMENTUM:
                self.mom = self.beta1 * self.mom + (1 - self.beta1) * g
                self.var = self.beta2 * self.var + (1 - self.beta2) * g**2
                g_mom = ((self.mom / (1 - self.beta1**self.steps)) / (torch.sqrt(self.var / (1 - self.beta2**self.steps)) + self.eps))
                s = g_mom.clone()
            else:
                s = g.clone()    
        # Update the model parameters
        END = 0
        actual_improvement = -1
        predicted_improvement = 1
        c = 0 # counter
        self.iter_count = 0 # NOTE: we can replace counter with self.iter_count
        while actual_improvement / predicted_improvement < 0 or actual_improvement < 0:
            lr = self.radius
            if seed:
                torch.manual_seed(seed)

            with torch.no_grad():
                if self.SECOND_ORDER and self.steps > self.delayed_second_order and self.steps > 1:
                    if c > 1:
                        try:
                            r=self.radius.cpu().numpy()
                        except:
                            r=self.radius
                        try:
                            g2 = -self.subproblem.solve_tr_subproblem(g.cpu().numpy(), r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                        except:
                            g2 = -self.subproblem.solve_tr_subproblem(g.cpu().numpy(), r, self.hessian_approx.gamma, self.hessian_approx.Psi, self.hessian_approx.M_inv)
                            
                            g2 = g.clone()
                            print('(row 238) subproblem failed')
                        g2 = torch.tensor(g2, device=self.device, dtype=torch.float32)
                        assert torch.isnan(g2).any() == False, "Nan in the update"
                        g2_norm = torch.norm(g2, p=self.norm_type)
                        s = g2.clone()

                
                s_norm = torch.norm(s, p=self.norm_type)
                if s_norm < abs(lr) and (self.SECOND_ORDER and self.delayed_second_order<=self.steps):
                    s_incr = 1
                else:
                    s_incr = lr/s_norm

                # Update model parameters
                a=0; 
                if type(s_incr) is not torch.Tensor:
                    s_incr = torch.tensor(s_incr, device=self.device, dtype=torch.float32)
                for param in weights:
                    b = param.numel()
                    assert torch.isnan(x[a:a+b] - s[a:a+b]*s_incr).any() == False, "Nan in the update"
                    param.data.copy_(torch.reshape(x[a:a+b] - s[a:a+b]*s_incr, param.shape))
                    a += b

                # Compute the new loss
                new_loss = closure()

            # print(f"norm of new weights: {torch.norm(torch.cat([p.flatten() for p in weights]), p=2)}")
            # Compute the accuracy factor between predicted and actual change (rho)
            actual_improvement = loss - new_loss.item()
            if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
                predicted_improvement = (g @ s) * s_incr - 1/2 * (s @ torch.tensor(self.hessian_approx.apply(s.cpu().numpy()), device=self.device, dtype=torch.float32)) * s_incr**2
            elif self.MOMENTUM: # First order information only
                predicted_improvement = g@(s*s_incr)# g_norm**2 * abs(s_incr) #- 1/2 * g_norm * s**2  # is like considering an identity approx of the Hessian
            else: # First order information only
                predicted_improvement = g_norm**2 * abs(s_incr) 
                    
            if actual_improvement>0:
                if self.ADAPTIVE: # Uses a function to adaptively change the radius
                    ratio = max(0,min(1,actual_improvement/predicted_improvement.item()))
                    LR = self.radius * self.r_adaptive(ratio)
                    self.radius = min(self.max_radius,max(self.min_radius,LR))
                    if ratio >= 1-self.acceptance_ratio:
                        END = 1
                else: # Standard TR update
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

            c += 1     
            self.iter_count += 1       
            if c > 10:
                # print(f"too many iterations. tr radius: {self.radius}")
                # if self.radius <= self.min_radius:
                END = 1

            # This is needed to fake dogleg method
            if END==1:
                # self.model_accuracy_lvl = max(self.model_accuracy_lvl-1,0) # TODO: I think this line is not necessary since it's the same as what's on line 392 after the loop has been broken
                break
            else:
                self.model_accuracy_lvl = min(self.model_accuracy_lvl+1,len(self.model_accuracy_range)-1)
            
        self.model_accuracy_lvl = max(self.model_accuracy_lvl-1,0)
        # if new_loss.item()>loss.item() and not self.ACCEPT_ALL:
        #     print('Something went wrong... loss is increasing')          
        return new_loss.item(), g, g_norm
    


# class TR(Optimizer): # TR optimizer
#     def __init__(self,params,       # NN params
#     radius=1,                       # TR radius
#     max_radius=None,                # TR maximum radius
#     min_radius=None,                # TR maximum radius
#     decrease_factor=0.5,            # Factor by which to decrease the TR radius in case a step is not accepted
#     increase_factor=None,           # Factor by which to increase the TR radius in case a step is accepted
#     is_adaptive=False,              # If adaptive, performs polynomial interpolation to find TR radius
#     second_order=True,              # Enable approximation of Hessian (or inverse Hessian)
#     second_order_method='SR1',      # Method to approximate Hessian (or inverse Hessian)
#     delayed_second_order=20,        # Second order information will kick in after "delayed_second_order" calls to the "step" method
#     device=None,                    # CPU or GPU
#     accept_all=False,               # If accept all, accepts all steps, even if they lead to an increase in the loss
#     acceptance_ratio=0.75,          # Increase factor of the radius when the region is flat enough
#     reduction_ratio=0.25,           # Reduction factor of the radius when the region is not flat enough
#     history_size=1000000000,        # Amount of stored vectors which forms the approximated Hessian
#     momentum=True,                  # Momentum
#     beta1=0.9,                      # Parameter for Adam momentum
#     beta2=0.999,                    # Parameter for Adam momentum
#     DOG=False,                      # If true, search direction is convex combination of adam direction and gradient direction
#     norm_type=2):                   # Norm type for the trust region radius

#         super().__init__(params, {})
        
#         self.steps = 0
#         self.radius = radius
#         self.params = self.param_groups[0]['params']
#         self.max_radius = max_radius if max_radius is not None else max(1.0, radius)
#         self.min_radius = min_radius if min_radius is not None else min(0.001, radius)
#         self.decrease_factor = decrease_factor
#         self.increase_factor = increase_factor if increase_factor is not None else 1.0 / decrease_factor
#         self.ADAPTIVE = is_adaptive
#         self.SECOND_ORDER = second_order
#         self.delayed_second_order = delayed_second_order
#         self.second_order_method = second_order_method if second_order else 'LBFGS'
#         self.device = device if device is not None else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.ACCEPT_ALL = accept_all
#         self.acceptance_ratio = acceptance_ratio
#         self.reduction_ratio = reduction_ratio
#         # print(f'Forcing infinite history of hessian : len = {history_size}')
#         self.history_size = history_size #TODO: current H_prod function is working only for infinite memory.. fix it
#         self.MOMENTUM = momentum    
#         self.norm_type = norm_type  

#         # self.tot_params = sum(p.numel() for p in self.params if p.requires_grad)
#         self.tot_params = sum(p.numel() for p in self.params)

#         # Needed for second order information
#         self.list_y = []  # List of gradient differences (y_k = g_k - g_{k-1})
#         self.list_s = []  # List of iterate differences (s_k = x_k - x_{k-1})
#         self.prev_x = None # parameters x_{k-1}
#         self.prev_g = None # gradient g_{k-1}
#         self.hessian_len = self.history_size  # How many iterate/gradient differences to keep in memory
#         self.gamma = None # Initial approximation of the Hessian B_{0}.. "None" means adaptive according to book Nocedal
#         self.model_accuracy_lvl = -1 # 2**model_accuracy Needed to average 2nd order direction with 1st order direction
#         self.model_accuracy_range = [0, 0.1, 0.3, 0.5, 0.75, 1] # 2**model_accuracy Needed to average 2nd order direction with 1st order direction
        
#         # BFGS
#         self.B = torch.tensor(1.0, device=self.device)
#         self.list_r = []
#         self.list_rho = []  # List of (1 / y_k^T s_k)
#         self.list_rho2 = [] # List of (1 / s_k^T B_k s_k)
#         self.list_Bs = []
#         self.list_Binvr = []
#         self.dampingLBFGS = torch.tensor(1.0, device=self.device) # TODO: positive values <1 seem to speed up convergence (no theoretical evidences)
#         # SR1
#         self.list_sigma = []
#         self.list_sigma_inv = []
#         self.list_v = []
#         self.list_v_inv = []
#         self.list_Sigma = []
#         self.list_Sigma_inv = []
#         self.list_V = [] # torch.empty((self.tot_params,1), device=self.device)
#         self.list_V_inv = []
#         self.r = torch.tensor(10**-8, device=self.device)
#         # ADAM Momentum
#         self.var = torch.zeros(self.tot_params, device=self.device)
#         self.mom = torch.zeros(self.tot_params, device=self.device)
#         self.beta1 = torch.tensor(beta1, device=device)
#         self.beta2 = torch.tensor(beta2, device=device)
#         self.eps = torch.sqrt(torch.tensor(torch.finfo(list(self.params)[0].dtype).eps, device=device)) # sqrt of the machine precision
#         # DOGLEG
#         self.DOG = DOG


#         # a =       16.84
#         # b =      -17.04
#         # c =         0.5
#         # self.r_adaptive = lambda x: a*x+b*np.sin(x)+c#a*np.exp(-b*x)+c#(2.66*x**2+0.01*x+0.33)/1.5
#         self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.5,0.5,0.7,1,1.4,2,3])#slightly better
#         # self.r_adaptive = lambda x: np.interp(x, [0,0.125,0.25,0.5,0.75,0.875,1], [0.6,0.7,0.8,1,1.2,1.5,2])#

#         self.neg = 0
#         # Dogleg
#         self.tau = 1.5
#         self.lr_sup = None
#         self.lr_inf = None
#         self.list_lr= []

#         self.hessian_approx = Hessian_approx(gamma=1, device=self.device, tol=10**-6)

#     def reset_lists(self):
#         self.list_y = []  # List of gradient differences (y_k = g_k - g_{k-1})
#         self.list_s = []  # List of iterate differences (s_k = x_k - x_{k-1})
#         self.list_Binvr = []
#         self.list_Bs = []
#         self.list_r = []
#         self.list_rho2 = []
#         self.list_rho = [] # List of inverse scalar product of gradient and iterate differences (1 / y_k^T s_k)
#         self.prev_x = None # parameters x_{k-1}
#         self.prev_g = None # gradient g_{k-1}

#     def compute_cauchy_point(self, g, Bg):
#         g_norm = torch.norm(g, p=2)
#         s = (-self.radius / g_norm) * g

#         if self.SECOND_ORDER:
#             gBg = g @ (Bg)
#             if gBg > 0:
#                 tau = torch.min((g_norm**3 / (self.radius * gBg)), torch.tensor(1.0, device=self.device))
#                 s *= tau
#         return s
    
#     def cg_steihaug(self, g, Bg):
#         s = torch.zeros_like(g, device=self.device)
#         r = g
#         d = -r
#         eps = torch.finfo(torch.float32).eps

#         if torch.norm(r, p=2) <= eps:
#             return s
        
#         for _ in range(50):
#             # print(f"CG Steihaug iteration {j}")
#             Bd = self.sr1_two_loop_recursion(d)
#             if d@Bd <= 0:
#                 return self.compute_cauchy_point(g, Bg)
            
#             alpha = (r@r)/(d@Bd)
#             s += alpha*d

#             if torch.norm(s, p=2) >= self.radius:
#                 return self.compute_cauchy_point(g, Bg)

#             rnew = r + alpha*Bd

#             if torch.norm(rnew, p=2) <= eps * torch.norm(r, p=2):
#                 return s
            
#             beta = (rnew@rnew)/(r@r)
#             d = rnew + beta*d
#             r = rnew
        
#         return s
    
#     def dogleg(self, g):
#         # pb = -H@g # full newton step
#         pb = -self.lbfgs(g)
#         Bg = self.Hessian_times_g(g)
#         norm_pb = torch.norm(pb, p=2)
        
#         # full newton step lies inside the trust region
#         if torch.norm(pb, p=2) <= self.radius:
#             return pb
#         # step along the steepest descent direction
#         pu = - (g @ g) / (g @ Bg) * g
#         dot_pu = pu @ pu
#         norm_pu = np.sqrt(dot_pu)
#         if norm_pu >= self.radius:
#             return self.radius * pu / norm_pu
        
#         # solve ||pu**2 +(tau-1)*(pb-pu)**2|| = trusr_radius**2
#         pb_pu = pb - pu
#         pb_pu_sq = np.dot(pb_pu, pb_pu)
#         pu_pb_pu_sq = np.dot(pu, pb_pu)
#         d = pu_pb_pu_sq ** 2 - pb_pu_sq * (dot_pu - self.radius ** 2)
#         tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq+1

#         # 0<tau<1
#         if tau < 1:
#             return pu*tau
#         #1<tau<2
#         return pu + (tau-1) * pb_pu    

        
#     def lbfgs(self, g):
#         """
#         Compute the search direction z=H^{-1}g, i.e. to be used in x_{k+1} = x_k - z
#         H is the LBFGS approximation of the Hessian at x_k.

#         Args:
#         g: current gradient (g(x_k))

#         Returns:
#         The search direction.
#         """
#         if len(self.list_s) > 0:
#             q = g.clone()
#             alpha = [0] * len(self.list_s)
#             for i in range(len(self.list_s) - 1, -1, -1):
#                 alpha[i] = self.list_rho[i] * (self.list_s[i] @ q)
#                 q -= alpha[i]*self.list_r[i] #list_y
            
#             r = (1/self.gamma)*q
#             for i in range(len(self.list_s)):
#                 beta = self.list_rho[i] * (self.list_r[i] @ r) #list_y
#                 r += self.list_s[i]*(alpha[i] - beta)

#             return r
#         else:
#             return g


#     def lbfgs_Bprod(self, g):
#         """
#         Compute the product z=Hg, where H is the Damped LBFGS approximation of the Hessian at x_k.

#         Args:
#         g: current gradient (g(x_k))

#         Returns:
#         Approximatd Hessian - vector product.
#         """
#         if len(self.list_s) > 0:
#             q = g.clone()
#             alpha = [0] * len(self.list_s)
#             for i in range(len(self.list_s) - 1, -1, -1):
#                 alpha[i] = self.list_rho[i] * (self.list_r[i] @ q)
#                 q -= alpha[i]*self.list_s[i]
            
#             r = (1/self.gamma)*q
#             for i in range(len(self.list_s)):
#                 beta = self.list_rho[i] * (self.list_s[i] @ r)
#                 r += self.list_r[i]*(alpha[i] - beta)

#             return r
#         else:
#             return g
        
   
#     def dogleg(self, g):
#         pU = (-(g@g) / (g@self.H_prod(g))) * g
#         pB = -self.Hinv_prod(g) # - B_inv @ g

#         if self.tau <= 1 and self.tau >= 0:
#             s = self.tau * pU
#         elif self.tau <= 2 and self.tau > 1:
#             s = pU + (self.tau - 1)*(pB - pU)

#         return s
    
#     def H_prod(self,x):
#         if torch.numel(self.gamma) == 1: #gamma is a scalar (i.e. the initial approx of the Hessian is a multiple of an identity matrix: B_0 = gamma * I)
#             Bx = self.gamma * x
#         else: #otherwise gamma might be a matrix
#             Bx = self.gamma @ x

#         if self.second_order_method == 'SR1':
#             Bx2 = Bx.clone()
#             for i in range(len(self.list_v)):
#                 Bx = Bx+ self.list_sigma[i] * ( self.list_v[i] * ( self.list_v[i] @ x ) ) 
#                 # print(torch.norm(self.list_sigma[i] * ( self.list_v[i] * ( self.list_v[i] @ torch.ones(len(self.list_s[-1]),device='cuda') ) ),p=2) )

#             if len(self.list_V) > 0:
#                 x = x.unsqueeze(1)
#                 Bx2 += (self.list_Sigma @ (self.list_V * (self.list_V @ x)))

#         elif 'BFGS' in self.second_order_method:
#             #THIS ONLY WORKS FOR INFINITE MEMORY:
#             for i in range(len(self.list_Bs)):
#                 Bx = Bx - self.list_Bs[i] * (torch.inner(self.list_Bs[i],x)*self.list_rho2[i]) + self.list_r[i] * torch.inner(self.list_r[i],x)*self.list_rho[i]
#             # Bx = self.lbfgs_Bprod(x)
#         return Bx 
    
#     def Hinv_prod(self, x):
#         if self.gamma.numel()==1: #gamma is a scalar (i.e. the initial approx of the Hessian is a multiple of an identity matrix: B_0 = gamma * I)
#             Hx=(1/self.gamma)*x
#         else: #otherwise gamma might be a matrix
#             Hx=torch.inverse(self.gamma)@x
#         if self.second_order_method == 'SR1':
#             # Hx2 = Hx.clone()
#             for i in range(len(self.list_v_inv)):
#                 Hx += self.list_sigma_inv[i] * ( self.list_v_inv[i] * ( self.list_v_inv[i] @ x ) ) 
#                 # NOTE: a different handling of the list (into matrices) will allow to remove the for loop and parallelize everything

#             # if len(self.list_V_inv) > 0:
#             #     x = x.unsqueeze(1)
#             #     Hx2 = Hx2 + (self.list_Sigma_inv @ (self.list_V_inv * (self.list_V_inv @ x))) #

#                 # if torch.norm(Hx-Hx2,2)/torch.norm(Hx)>10**-4:
#                 #     print('asddddddddddddddddd '+str(torch.norm(Hx-Hx2,2)/torch.norm(Hx)))
#         elif 'BFGS' in self.second_order_method:
#             # THIS ONLY WORKS FOR INFINITE MEMORY:
#             for i in range(len(self.list_Binvr)):
#                 Hx = Hx +self.list_s[i]*((1/self.list_rho[i]+torch.inner(self.list_r[i],self.list_Binvr[i]))*(self.list_rho[i]**2))*torch.inner(self.list_s[i],x)\
#                 -(self.list_Binvr[i]*torch.inner(self.list_s[i],x)+self.list_s[i]*torch.inner(self.list_Binvr[i],x))*self.list_rho[i]
#             # Hx = self.lbfgs(x)
#         return Hx

#     def full_bfgs(self):# Debug only
#         """
#         Updates the Hessian (full matrix). This function is disabled by default, 
#         enable it inside "update_hessian_history(self, x, g)" if you need it.
#         """
#         if self.B.numel()==1:
#             self.B = self.gamma*torch.eye(self.tot_params, device = self.device)
#         if len(self.list_s)>0:
#             Bs=self.B@self.list_s[-1] #TODO: there is a slight difference between this and 
#             c=torch.inner(self.list_s[-1],self.list_y[-1])
#             rho2=1/torch.inner(self.list_s[-1],Bs)
#             if c>=0.2/self.list_rho2[-1]:
#                 t=1
#             else:
#                 t=(0.8/rho2)/(1/rho2-c)
#             t=t*self.dampingLBFGS
#             # t=t/2
#             r=t*self.list_y[-1]+(1-t)*Bs
#             self.B = self.B - torch.outer(Bs,Bs)*rho2 + torch.outer(r,r)/torch.inner(self.list_s[-1],r)

#     def sr1(self): #full matrix... expensive (testing only)
#         if len(self.list_s) == 0:
#             self.B = torch.eye(torch.numel(torch.cat([p.flatten() for p in self.param_groups[0]['params']])), device=self.device) 
#         else:
#             s = self.list_s[-1]
#             y = self.list_y[-1]
#             ymBs = y - self.B@s

#             norm_s = torch.norm(s, p=2)
#             norm_ymBs = torch.norm(ymBs, p=2)
#             eps = torch.finfo(torch.float32).eps

#             print(torch.abs(s @ ymBs) >= eps * norm_s * norm_ymBs)
#             if torch.abs(s @ ymBs) >= eps * norm_s * norm_ymBs:
#                 self.B.add_(torch.outer(ymBs, ymBs) / (ymBs @ s))

#         return self.B
    
#     def update_hessian_history(self, x, g):
#         """
#         Updates the list of parameters needed to compute the inverse of the Hessian.

#         Args:
#         x: current parameters (x_k)
#         g: current gradient (g(x_k))
#         """
#         with torch.no_grad():
#             x = x.detach()
#             g = g.detach()
#             if self.gamma is None:
#                 self.gamma=torch.tensor(0.01,device=self.device)

#             if self.prev_x is not None:
#                 x_diff = x - self.prev_x
#                 g_diff = g - self.prev_g
#                 self.hessian_approx.update_memory(x_diff, g_diff)
#                 # self.list_s.append(x_diff)
#                 # self.list_y.append(g_diff)
#                 # if self.gamma is None: #gamma is the approx of H (the hessian.. not the inverse)
#                 # gamma_old=self.gamma
#                 # gamma = abs(1/((g_diff @ x_diff) /(g_diff @ g_diff)))
#                 # gamma=1
#                 # self.gamma = torch.tensor(gamma,device=self.device) #Numerical Optimization, Nocedal and Wright, 2006, equation (6.20)
#                 # if self.B.numel()==1:
#                 #     self.B = self.gamma
#                 # else:
#                 #     self.B = self.B + (-gamma_old+self.gamma)*torch.eye(self.B.shape[0],device=self.device)
#                 if self.second_order_method == 'SR1':
#                     # Numerical Optimization, Nocedal and Wright, 2006
#                     v = g_diff - self.H_prod(x_diff)  # y_k-B_ks_k
#                     sv = x_diff @ v
#                     if torch.abs(sv) >= self.r * torch.norm(x_diff,p=2) * torch.norm(v,p=2): # Condition to avoid singular update
#                         v_inv = (x_diff - self.Hinv_prod(g_diff)).unsqueeze(0)
#                         self.list_sigma.append(1 / (v @ x_diff))
#                         self.list_v.append(v)
#                         # if len(self.list_V) > 0:  # TODO: NEEDED WHEN OPTIMIZING SR1 
#                         #     self.list_Sigma = torch.cat((self.list_Sigma, (1 / (v @ x_diff)).unsqueeze(0)), 0) 
#                         #     self.list_V = torch.cat((self.list_V, v.unsqueeze(0)), 0)
#                         #     self.list_Sigma_inv = torch.cat( (self.list_Sigma_inv, (1 / (v_inv @ g_diff))), 0) 
#                         #     self.list_V_inv = torch.cat((self.list_V_inv, v_inv), 0)
#                         # else:
#                         #     self.list_Sigma = (1 / (v @ x_diff)).unsqueeze(0) # detach is needed to remove the "grad_fn"
#                         #     self.list_V = v.unsqueeze(0)
#                         #     self.list_Sigma_inv = (1 / (v_inv @ g_diff))
#                         #     self.list_V_inv = v_inv
#                         self.list_v_inv.append(x_diff - self.Hinv_prod(g_diff)) # s_k-H_ky_k
#                         self.list_sigma_inv.append(1 / (self.list_v_inv[-1] @ g_diff))
#                 elif 'BFGS' in self.second_order_method: #  DAMPED BFGS -> avoids skipping too many updates (book nocedal chap. 18.2)
#                     self.list_Bs.append(self.H_prod(x_diff)) #B@s
#                     self.list_rho2.append(1 / torch.inner(x_diff , self.list_Bs[-1]))
#                     c=torch.inner(x_diff,g_diff)
#                     if c>=0.2/self.list_rho2[-1]:
#                         t=torch.tensor(1, device = self.device)
#                     else:
#                         t=(0.8/self.list_rho2[-1])/(1/self.list_rho2[-1]-c)
#                     t = t*self.dampingLBFGS
#                     # print(t)
#                     self.list_r.append(t*g_diff+(1-t)*self.list_Bs[-1])
#                     self.list_Binvr.append(self.Hinv_prod(self.list_r[-1])) # Binv @ r
#                     self.list_rho.append(1 / (x_diff @ self.list_r[-1]))

#                     self.list_s.append(x_diff)
#                     self.list_y.append(g_diff)

#                     # self.full_bfgs() # Debug only
#                 else:
#                     raise ValueError('Unknown second order method')

#                 if len(self.list_s) > self.hessian_len:
#                     self.list_s.pop(0)
#                     self.list_y.pop(0)
#                     self.list_Bs.pop(0)
#                     self.list_Binvr.pop(0)
#                     self.list_r.pop(0)

#                 if len(self.list_rho) > self.hessian_len:
#                     self.list_rho.pop(0)
#                     self.list_rho2.pop(0)

#                 if len(self.list_sigma) > self.hessian_len:
#                     self.list_sigma.pop(0)
#                     self.list_sigma_inv.pop(0)
#                     self.list_v.pop(0)
#                     self.list_v_inv.pop(0)
                    
#             self.prev_x = x
#             self.prev_g = g


    


#     def step(self, closure, seed=[]):
#         """
#         Compute a step of the trust-region method.

#         Args:
#         loss: objective function f evaluated on current inputs, i.e. f(x)
#         inputs: the inputs to the model/net
#         labels: the ground truth/labels for the inputs

#         Returns:
#         If the trial point (x + step) is accepted, this returns the loss at the trial point f(x + s)
#         """
#         # TODO: Stop iterating when reaching the minimum radius and the direction is the gradient direction (smaller step is needed but this is not possible)

#         self.steps += 1
#         loss = closure()
#         self.list_lr.append(self.radius)
#         if len(self.list_lr)>10 and self.DOG:
#             self.lr_inf,self.lr_sup = biased_average(self.list_lr,70)
#             fun = lambda x: 1 - x / (self.lr_sup - self.lr_inf) + self.lr_inf / (self.lr_sup - self.lr_inf)
#             dogleg_mod = lambda lr: max(0,min(1,fun(lr))) # to make sure to be inside [0,1]

#         weights = self.param_groups[0]['params']                    # Store the model's parameters
#         x = torch.cat([p.flatten() for p in weights]).detach()      # Store the model's parameters (flat)
#         g = [] #this works even if there are layers which are not trainable

#         for p in weights:
#             if p.grad is None:
#                 g.append(torch.zeros_like(p).flatten())
#                 continue
#             g.append(p.grad.flatten().detach())
#         g = torch.cat(g)


#         params = [p for p in weights if p.requires_grad]
#         hessian = torch.zeros(len(params), len(params))

#         for i in range(len(params)):
#             grad_i = torch.autograd.grad(loss, params[i], create_graph=True)[0]
#             for j in range(len(params)):
#                 if grad_i.requires_grad and params[j].requires_grad:
#                     # Note the retain_graph=True
#                     if i == len(params)-1 and j == len(params)-1:
#                         hessian[i, j] = torch.autograd.grad(grad_i, params[j], retain_graph=False)[0]
#                     else:
#                         hessian[i, j] = torch.autograd.grad(grad_i, params[j], retain_graph=True)[0]



#         self.update_hessian_history(x, g) # current weights and gradient
#         g = -self.hessian_approx.solve_tr_subproblem(g, self.radius).real
#         dir = -self.hessian_approx.solve_tr_subproblem(g, self.radius).real
#         dir2 = -self.hessian_approx.solve_tr_subproblem(g, self.radius/4).real

#         print(torch.norm(dir-dir2)/torch.norm(dir))

#         #angle between gradient and direction
#         print(f"dir @ g = {torch.dot(g,dir)/(torch.norm(g)*torch.norm(dir))},     dir2 @ g {torch.dot(g,dir2)/(torch.norm(g)*torch.norm(dir2))}" )

#         # g = torch.cat([p.grad.flatten() for p in weights]).detach() # Store the model's gradient (flat)
#         g_norm = torch.norm(g, p=self.norm_type)                                 # Gradient norm
#         # TODO: AVOID FLATTENING ?
        
#         if self.SECOND_ORDER:
#             self.update_hessian_history(x, g) # current weights and gradient

#             if self.delayed_second_order<=self.steps:
#                 g2 = self.Hinv_prod(g)
#                 g2_norm = torch.norm(g2, p=self.norm_type)
#             else:
#                 s = g.clone()
            
#             # problem=False
#             # while problem==True:
#             #     g2 = self.Hinv_prod(g)
#             #     try:
#             #         g3 = torch.inverse(self.B)@g
#             #     except:
#             #         g3 = g
#             #     g2=g3
                
#                 # print('difference B^-1 g and Hinv(g) : '+str(torch.norm(g3-g2)/torch.norm(g3)))
#                 # print('difference B^-1 g and lbfgs(g) : '+str(torch.norm(g3-self.lbfgs(g))/torch.norm(g3)))
#                 # try:
#                 #     print('difference B g and H(g) : '+str(torch.norm(self.B@g-self.H_prod(g))/torch.norm(self.B@g)))
#                 #     print('difference B g and lbfgs_Bprod(g) : '+str(torch.norm(self.B@g-self.lbfgs_Bprod(g))/torch.norm(self.B@g)))
#                 # except:pass

#                 # g2_norm = torch.norm(g2, p=2)
#                 # if self.SECOND_ORDER and g@g2<0:
#                 #     ll=self.list_rho
#                 #     self.reset_lists()
#                 #     g_norm = torch.norm(g, p=2)
#                 #     self.neg+=1
#                 #     print(f'PROBLEM!! {self.neg} times, hessian len = {len(ll)} ,  < g , g2 > = {((g@g2)/(g2_norm*g_norm)):2f}') 
                    
#                 # else:
#                 #     problem=False

#                 # problem=False
            
#             # print('as')
#             # gg=g
#             # try:
#             #     print(f'Consistency in inversion : {torch.norm(gg-self.H_prod(self.Hinv_prod(gg)))}, inv_prod = {torch.norm(torch.inverse(self.B)@gg-self.Hinv_prod(gg))/torch.norm(torch.inverse(self.B)@gg)}, prod = {torch.norm(self.B@gg-self.H_prod(gg))/torch.norm(self.B@gg)}')
#             #     gg=torch.rand_like(g)
#             #     print(f'Consistency in inversion : {torch.norm(gg-self.H_prod(self.Hinv_prod(gg)))}, inv_prod = {torch.norm(torch.inverse(self.B)@gg-self.Hinv_prod(gg))/torch.norm(torch.inverse(self.B)@gg)}, prod = {torch.norm(self.B@gg-self.H_prod(gg))/torch.norm(self.B@gg)}')
#             #     print(f' {torch.norm(g-self.B@self.Hinv_prod(g))}')
#             # except:pass
#             # print('---')

#         else:
#             if self.MOMENTUM:
#                 self.mom = self.beta1 * self.mom + (1 - self.beta1) * g
#                 self.var = self.beta2 * self.var + (1 - self.beta2) * g**2
#                 g_mom = ((self.mom / (1 - self.beta1**self.steps)) / (torch.sqrt(self.var / (1 - self.beta2**self.steps)) + self.eps))
#                 s = g_mom.clone()
#             else:
#                 s = g.clone()

                
        
#         # Update the model parameters
#         END = 0
#         x_list = [0] # list of positions
#         actual_improvement = -1
#         predicted_improvement = 1
#         list_loss = [loss.item()]
#         c = 0 # counter
#         while actual_improvement / predicted_improvement < 0 or actual_improvement < 0:
#             lr = self.radius
#             # print(f"norm of x: {torch.norm(x, p=2)}")
#             # print(f"norm of initial weights: {torch.norm(torch.cat([p.flatten() for p in weights]), p=2)}")

#             if seed:
#                 torch.manual_seed(seed)

#             with torch.no_grad():
#                 # if len(x_list) > 2:
#                 #     if self.ADAPTIVE and False:
#                 #         lr = find_minima(x_list, list_loss, self.radius, max_ord=3)
#                 #     else:
#                 #         lr = self.radius
#                 x_list.append(lr)

#                 if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
#                     if c==0:
#                         w = 1
#                         s = g2.clone()
#                         # print(f' model accu {self.model_accuracy_lvl:.2f}, c={c}')
#                     else:
#                         w = self.model_accuracy_range[self.model_accuracy_lvl]
#                         s = ((1-w)/g2_norm)*g2+((w)/g_norm)*g
#                         # print(f' model accu {self.model_accuracy_lvl:.2f}, c={c}, w={w}, lr={lr}')
#                 else:
#                     if self.DOG and ('dogleg_mod' in locals()):
#                         w = dogleg_mod(lr)
#                         print(f'inf lr : {self.lr_inf:.1e},   sup lr : {self.lr_sup:.1e},   lr : {lr},   w : {w}')
#                         w=torch.tensor(w,device=self.device)
#                         s = (w)*g_mom/torch.norm(g_mom, p=self.norm_type)+(1-w)*g/torch.norm(g, p=self.norm_type)
#                         s=s/torch.norm(s, p=self.norm_type)
#                         s_incr=1#lr/g_norm
#                         # print(f'{g_norm}, {torch.norm(s2)}')

                
#                 s_norm = torch.norm(s, p=self.norm_type)
#                 if s_norm < abs(lr) and (self.SECOND_ORDER and self.delayed_second_order<=self.steps):
#                     s_incr = 1
#                 else:
#                     s_incr = lr/s_norm

#                 # Update model parameters
#                 a=0; 
#                 if type(s_incr) is not torch.Tensor:
#                     s_incr = torch.tensor(s_incr, device=self.device)
#                 for param in weights:
#                     b = param.numel()
#                     param.data.copy_(torch.reshape(x[a:a+b] - s[a:a+b]*s_incr, param.shape))
#                     a += b

#                 # Compute the new loss
#                 # TODO: should we maybe un-indent this by one level? 
#                 # ANSWER: No it's fine this way. So the "closure()" function does not compute "backward()" since it is expensive and we don't need it
#                 new_loss = closure()
#                 list_loss.append(new_loss.item())

#             # print(f"norm of new weights: {torch.norm(torch.cat([p.flatten() for p in weights]), p=2)}")
#             # Compute the accuracy factor between predicted and actual change (rho)
#             actual_improvement = list_loss[0] - new_loss.item()
#             if self.SECOND_ORDER and self.delayed_second_order<=self.steps:
#                 predicted_improvement = (g @ s) * s_incr - 1/2 * (s @ self.H_prod(s)) * s_incr**2
#             else:
#                 if self.MOMENTUM or self.SECOND_ORDER:
#                     predicted_improvement = g@(s*s_incr)# g_norm**2 * abs(s_incr) #- 1/2 * g_norm * s**2  # is like considering an identity approx of the Hessian
#                 else:
#                     predicted_improvement = g_norm**2 * abs(s_incr) #- 1/2 * g_norm * s**2  # is like considering an identity approx of the Hessian
            
            

#             # print(g@s)
#             # print(f'predicted improvement {predicted_improvement}')
#             # if predicted_improvement < 0:
#             #     raise ValueError('Predicted improvement cannot be <0')
#             # print(actual_improvement / predicted_improvement)

#             # TODO: come with a continuous function that adapts the trust region radius 
#             # TODO: add another criterion when the ratio is close to 0
#             # TODO: check if predicted improvement < 0 and actual improvement > 0: accept step (potentially)
#             # TODO: if both < 0, we might need to change direction: dogleg?
#             # print(f"radius: {self.radius},  ratio: {actual_improvement / predicted_improvement}")

#             if actual_improvement>0:
#                 if not self.ADAPTIVE:
#                     if actual_improvement / predicted_improvement < self.reduction_ratio: # Update the trust region radius
#                         self.radius = max(self.radius*self.decrease_factor, self.min_radius)
#                     elif actual_improvement / predicted_improvement > self.acceptance_ratio:
#                         self.radius = min(self.radius*self.increase_factor, self.max_radius)
#                         END = 1
#                     # TODO: add another criterion when the ratio is close to 1
#                     else:
#                         END = 1
#                 else:
#                     ratio = max(0,min(1,actual_improvement/predicted_improvement.item()))
#                     LR = self.radius * self.r_adaptive(ratio)
#                     self.radius = min(self.max_radius,max(self.min_radius,LR))
#                     if ratio >= 1-self.acceptance_ratio:
#                         END = 1
#             else:
#                 self.radius = max(self.radius*self.decrease_factor, self.min_radius)


#             lr = self.radius
#             if self.ACCEPT_ALL:
#                 END = 1

#             c += 1            
#             if c > 10:
#                 print(f"too many iterations. tr radius: {self.radius}")
#                 # if self.radius <= self.min_radius:
#                 END = 1

#             if END==1:
#                 self.model_accuracy_lvl = max(self.model_accuracy_lvl-1,0)
#                 break
#             else:
#                 self.model_accuracy_lvl = min(self.model_accuracy_lvl+1,len(self.model_accuracy_range)-1)
            

#             # self.update_hessian_history(torch.cat([p.flatten() for p in weights]), g) # current weights and gradient
#             # return new_loss  
#         self.model_accuracy_lvl = max(self.model_accuracy_lvl-1,0)
#         if new_loss.item()>loss.item() and not self.ACCEPT_ALL:
#             print('Something went wrong... loss is increasing')          
#         return actual_improvement,new_loss



def biased_average(numbers,perc):
    sorted_numbers = np.sort(numbers)
    avg_sup = np.mean(sorted_numbers[-int(len(numbers) * perc/100):])
    avg_inf = np.mean(sorted_numbers[:int(len(numbers) * perc/100)])
    return avg_inf,avg_sup

def find_minima(x, y, r, max_ord=3):
    """
    Find the position of the minima in a ball of radius "r" around x[0] using polynomial interpolation of the data.

    Args:
    x: a list of scalar positions
    y: a list of the scalar values that an unknown function "f" assumes at each previous "x" position
    r: the radius where to look for the minima.

    Returns:
    pos_minima: the position of the minima found in the ball of radius "r"
    """
    order = min(len(x)-1, max_ord)    # Compute the order of the polynomial based on the number of data points
    poly = np.poly1d(np.polyfit(x, y, order))# Interpolate the data using a polynomial of order "order"
    def poly_func(x):
        return poly(x)
    # Find the minimum of the polynomial within the radius
    a = x[0] - r;    b = x[0] + r
    result = minimize_scalar(poly_func, bounds=(a, b), method='bounded')
    if not result.success:
        raise ValueError("Failed to find minimum of polynomial within radius")
    return result.x



