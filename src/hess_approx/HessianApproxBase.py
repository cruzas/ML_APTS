from abc import abstractmethod
import numpy as np
from hess_approx.JFNK import * 


class ApplyHessDirect():
  def __init__( self,  fun, x,  x_batch, y_batch):
        self.x = x
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.fun = fun

  def apply_hessian(self, vec):
    return self.fun.apply_hessian(self.x, vec, self.x_batch, self.y_batch)
  

class HessianApproxBase():
  def __init__( self, config):
        self.memory_init = False
        self.Y          = None
        self.S          = None
        self.M          = None
        self.Psi        = None 
        self.gamma      = 1.0
                
        self.memory         = config.get("ha_memory")        
        self.tol            = config.get("ha_tol")
        self.r              = config.get("ha_r")                
        self.eig_type       = config.get("ha_eig_type")
        self.print_errors   = config.get("ha_print_errors")  
        self.sampling_radius = config.get("ha_sampling_radius")      

  @abstractmethod
  def update_memory_inv(self, s, y): 
    raise NotImplementedError  
    

  @abstractmethod
  def update_memory(self, s, y): 
    raise NotImplementedError  
         

  @abstractmethod
  def reset_memory(self):
    self.memory_init    = False
    self.S              = []
    self.Y              = []
    self.M_inv          = []
    self.M              = []
    self.gamma          = 1
    self.Psi            = []


  @abstractmethod
  def apply(self, v): 
    raise NotImplementedError  


  @abstractmethod
  def apply_inv(self, v): 
    raise NotImplementedError  


  def sample_dir_update_memory_inv(self, fun, x, grad_batch_old, x_batch, y_batch):
    size = len(x)

    for m in range(self.memory):
      p_k = self.sampling_radius*np.random.randn(size)

      if(np.dot(p_k, grad_batch_old) > 0):
        p_k = -1.0*p_k   # new test for DD


      x_new = x + p_k
      [__, grad_batch_new] = fun.loss_grad(x_new, x_batch, y_batch)

      y = grad_batch_new - grad_batch_old
      perform_up = self.update_memory_inv(p_k, y)    



  def sample_dir_update_memory(self, fun, x, grad_batch_old, x_batch, y_batch):
    size = len(x)

    for m in range(self.memory):
      p_k = self.sampling_radius*np.random.randn(size)

      if(np.dot(p_k, grad_batch_old) > 0):
        # p_k = -1.0*p_k   # new test for DD
        m -= 1
        break

      x_new = x + p_k


      # option one - paper
      # [__, grad_batch_new] = fun.loss_grad(x_new, x_batch, y_batch)
      # y = grad_batch_new - grad_batch_old
      
      # option two - paper
      # TODO:: this can be speeded up quite a lot... by evaluating all together 
      action_class = ApplyHessDirect(fun, x, x_batch, y_batch)
      y = action_class.apply_hessian(p_k)


      perform_up = self.update_memory(p_k, y)    












