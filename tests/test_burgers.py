import numpy as np
import scipy.io as io
import os
import matplotlib.pyplot as plt
import time
import scipy.io as io
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import copy
import torch.optim as optim
import argparse
import pandas as pd
from pathlib import Path

from models.FFNConstantWidth import *
from datasets.Domain2D import *
from models.ResNetDenseConstantWidth import *
from PDEs.Burgers import *

from trainers.UnconstrainedRegressionTrainer import *
from trainers.PrecLBFGS import *
from trainers.GDLinesearch import *

torch.set_default_dtype(torch.float64)
args = argparse.Namespace()
args.seed = 0; args.num_points_x = 25; args.num_points_y = 50; args.num_ref = 4; args.num_levels = 1; args.width = 20; args.hiddenlayers_coarse = 6; args.ada_net = False; args.T = 1; args.history = 1; args.use_adaptive_activation = False; args.opt_type = 0; args.lr_global = 1e-3; args.max_epochs = 1000
print(" \n \n args ", args, " \n \n ")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

pi = torch.acos(torch.zeros(1)).item() * 2

train_samples = np.array([args.num_ref*args.num_points_x, args.num_ref*args.num_points_y])
test_samples = np.array([200, 400])

start_point = np.array([0., -1.0]) # (t_0, x_0)
end_point = np.array([1.0, 1.]) # (t_end, x_end)


domain = RectangleDomain2D( start_point,
                            end_point,
                            train_samples,
                            test_samples,
                            use_BC=False,
                            sampling_strategy="hammersly", # uniform_random, hammersly
                            ref_strategy="None")


bc  = BC_Burgers(start_point=start_point, end_point=end_point)
pde = Burgers(bc_exact=bc)


ml_hierarchy = MLHierarchyResNets(num_levels=args.num_levels)
nets = ml_hierarchy.build(inputs=2, 
                          outputs=1,
                          width=args.width,
                          hiddenlayers_coarse=args.hiddenlayers_coarse,
                          T=args.T,
                          use_adaptive_activation=args.use_adaptive_activation)


domain.append_analytical_sol_test(pde)


elapsed_time = 0
epoch = 0
train_loss = 0
converged = False

args.opt_type = 2
if(args.opt_type == 0):
    algo = GDLinesearch
elif(args.opt_type == 1):
    algo = PrecLBFGS
elif(args.opt_type == 2):
    algo = optim.Adam
else:
    algo = optim.SGD

print(f"Optimizer type: {algo}")

dir_name = "Algo_"+str(args.opt_type)
Path(dir_name).mkdir(parents=True, exist_ok=True)

trainer = UnconstrainedRegressionTrainer(training_pinn_flg=True, num_subdomains=1, type="regression", freq_print=100, use_wandb=False)

trainer.config["max_epochs"] = args.max_epochs
trainer.config["conv_history_csv_name"] = dir_name + "/" + \
                                          'Summary_seed_' + str(args.seed)+\
                                          '_lrg'+str(args.lr_global) +\
                                          "_opt_type_" + str(args.opt_type) +\
                                          "max_epochs" + str(args.max_epochs) +\
                                          "_num_points_x_"+str(args.num_points_x)+\
                                          "_num_points_y_"+str(args.num_points_y)+\
                                          "_num_ref_"+str(args.num_ref)+\
                                          "_num_levels_"+str(args.num_levels)+\
                                          "_width_"+str(args.width)+\
                                          "_hiddenlayers_coarse_"+str(args.hiddenlayers_coarse)+\
                                          "_ada_net_"+str(args.ada_net)+\
                                          "_T_"+str(args.T)+\
                                          "_history_"+str(args.history)+\
                                          "_use_adaptive_activation_"+str(args.use_adaptive_activation)
    

start_time = time.time()
if(args.ada_net==False):
  convergence_status = trainer.train(dataset=domain,
                                      net=nets[-1],
                                      criterion=pde.criterion,
                                      optimizer_type=algo,
                                      args=args)
  elapsed_time = time.time() - start_time

  # if(rank==0):
  out_name = dir_name + "/" + 'Summary.csv'

  df = pd.DataFrame({'seed': [args.seed],
                     'num_points_x':[args.num_points_x],
                     'num_points_y':[args.num_points_y],
                     'num_ref':[args.num_ref],
                     'num_levels': [args.num_levels],
                     'width': [args.width],
                     'hiddenlayers_coarse': [args.hiddenlayers_coarse],                     
                     'ada_net': [args.ada_net], 
                     'use_adaptive_activation':[args.use_adaptive_activation],
                     'T': [args.T], 
                     'opt_type': [args.opt_type],
                     'lr': [args.lr_global],
                     'history': [args.history],
                     'time': [elapsed_time],
                     'epochs': [convergence_status["epochs"]],
                     'loss': [convergence_status["loss_train"]],
                     'loss_test': [convergence_status["loss_val"]],
                     'L2_error': [convergence_status["L2_error"]],
                     'L2_error_rel': [convergence_status["L2_error_rel"]], 
                     'num_loss_evals': [convergence_status["num_loss_evals"]]})


else:

  convergence_statuses = []

  

  for l in range(0, ml_hierarchy.num_levels): 
    print("------------  level l ", l, " ------------")

    if(l > 0):
      print("l-1: ", l-1, "l:  ", l)
      ml_hierarchy.prolong_params(nets[l-1], nets[l])


    ml_scaling_factor =  (2**((l+1)- args.num_levels))
    trainer.config["max_epochs"] = int((1./ml_scaling_factor)*(args.max_epochs/ml_hierarchy.num_levels))


    convergence_status = trainer.train( dataset=domain,
                                        net=nets[l],
                                        criterion=pde.criterion,
                                        optimizer_type=algo,
                                        args=args)

    convergence_statuses.append(convergence_status)


  elapsed_time = time.time() - start_time
  epochs=0
  num_loss_evals=0

  for l in range(0, len(convergence_statuses)): 
    
    ml_scaling_factor =  (2**((l+1)- args.num_levels))
    print("l ", l, "   ml_scaling_factor ", ml_scaling_factor)

    epochs          += ml_scaling_factor * convergence_statuses[l]["epochs"]
    num_loss_evals  += ml_scaling_factor * convergence_statuses[l]["num_loss_evals"]


  out_name = dir_name + "/" + 'Summary.csv'


  df = pd.DataFrame({'seed': [args.seed],
                     'num_points_x':[args.num_points_x],
                     'num_points_y':[args.num_points_y],
                     'num_ref':[args.num_ref],
                     'num_levels': [args.num_levels],
                     'width': [args.width],
                     'hiddenlayers_coarse': [args.hiddenlayers_coarse],
                     'ada_net': [args.ada_net], 
                     'use_adaptive_activation':[args.use_adaptive_activation],
                     'T': [args.T],                      
                     'opt_type': [args.opt_type],
                     'lr': [args.lr_global],
                     'history': [args.history],
                     'time': [elapsed_time],
                     'epochs': [epochs],
                     'loss': [convergence_status["loss_train"]],
                     'loss_test': [convergence_status["loss_val"]],
                     'L2_error': [convergence_status["L2_error"]],
                     'L2_error_rel': [convergence_status["L2_error_rel"]], 
                     'num_loss_evals': [num_loss_evals]})



if Path(out_name).exists():
    df.to_csv(out_name, index=False, header=False, mode='a')
else:
    df.to_csv(out_name, index=False, header=True, mode='a')


pde.plot_results(domain, nets[-1], dpi=1000)


