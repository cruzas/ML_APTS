from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import scipy
from scipy import sparse
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import eigh
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh



class ApplyHessDirect():
  def __init__( self,  fun, x,  x_batch, y_batch):
        self.x = x
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.fun = fun

  def apply_hessian(self, vec):
    return self.fun.apply_hessian(self.x, vec, self.x_batch, self.y_batch)



class ApplyHess():
  def __init__( self, H):
        self.H = H

  def apply_hessian(self, vec):
    return np.matmul(self.H, vec)


class ApplyIdentity():

  def apply_hessian(self, vec):
    return vec



class ApplyHessJFNK():
  def __init__(self,  fun, x, x_batch, y_batch, g):
        self.x = x
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.fun = fun
        
        self.grad_x_k = g
        self.eps = 1e-12

        self.n_glob = len(x)

  # eval pertubation
  def apply_hessian(self, vec):

    norm_v = np.linalg.norm(vec) 
    if(norm_v < self.eps):
        return vec

    x_p = np.sqrt(self.eps) * (1.0+self.x)
    sum_a = np.sum(x_p)
    
    
    if(norm_v > self.eps): 
        per = sum_a *(1./(self.n_glob * norm_v))
    else:
        per = sum_a/self.n_glob

    if(np.isnan(per)):
        return vec


    x_new = self.x + per*vec
    __, g_new = self.fun.loss_grad(x_new, self.x_batch, self.y_batch)


    return (g_new-self.grad_x_k)/per
