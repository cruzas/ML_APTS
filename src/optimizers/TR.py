import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist # For parallel training
import time
from optimizers.Hessian_approx import *
from hess_approx.LSR1 import *
# from hess_approx.LBGS import *



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



