import utils
from pmw.parallelized_model import ParallelizedModel
from optimizers import APTS, TR
from dataloaders import GeneralizedDistributedDataLoader
import torch
from trainers.GDLinesearch import *
from trainers.PrecLBFGS import *
from trainers.UnconstrainedRegressionTrainer import *
from PDEs.Burgers import *
from models.ResNetDenseConstantWidth import *
from datasets.Domain2D import *
from models.FFNConstantWidth import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
import torch.multiprocessing as mp
import torch.optim as optim
import argparse
import pandas as pd
from pathlib import Path

# Make ../src visible for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import networks
import utils
from pmw.parallelized_model import ParallelizedModel



def parse_cmd_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="APTS", required=False)
    parser.add_argument("--lr", type=float, default=1.0, required=False)
    parser.add_argument("--dataset", type=str, default="mnist", required=False)
    parser.add_argument("--batch_size", type=int, default=28000, required=False)
    parser.add_argument("--model", type=str,
                        default="feedforward", required=False)
    parser.add_argument("--num_subdomains", type=int, default=1, required=False)
    parser.add_argument("--num_replicas_per_subdomain",
                        type=int, default=1, required=False)
    parser.add_argument("--num_stages_per_replica",
                        type=int, default=1, required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--trial", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--is_sharded", type=bool, default=False, required=False)
    parser.add_argument("--data_chunks_amount", type=int, default=10, required=False)
    return parser.parse_args(args)




def main(rank=None, cmd_args=None, master_addr=None, master_port=None, world_size=None):
    utils.prepare_distributed_environment(
        rank, master_addr, master_port, world_size, is_cuda_enabled=True)
    
    torch.set_default_dtype(torch.float64)
    parsed_cmd_args = parse_cmd_args(sys.argv[1:])

    args = argparse.Namespace()
    args.seed = 0
    args.num_points_x = 25
    args.num_points_y = 50
    args.num_ref = 4
    args.num_levels = 1
    args.width = 20
    args.hiddenlayers_coarse = 6
    args.ada_net = False
    args.T = 1
    args.history = 1
    args.use_adaptive_activation = False
    args.opt_type = 0
    args.lr_global = 1e-1
    args.max_epochs = 1000
    print(" \n \n args ", args, " \n \n ")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    pi = torch.acos(torch.zeros(1)).item() * 2

    train_samples = np.array(
        [args.num_ref*args.num_points_x, args.num_ref*args.num_points_y])
    test_samples = np.array([200, 400])

    start_point = np.array([0., -1.0])  # (t_0, x_0)
    end_point = np.array([1.0, 1.])  # (t_end, x_end)

    # Create stage list
    stage_list = networks.construct_stage_list(
        parsed_cmd_args.model, parsed_cmd_args.num_stages_per_replica)

    random_input = torch.tensor(start_point).unsqueeze(0).to(device)
    par_model = ParallelizedModel(stage_list=stage_list, 
                                sample=random_input,
                                num_replicas_per_subdomain=parsed_cmd_args.num_replicas_per_subdomain, 
                                num_subdomains=parsed_cmd_args.num_subdomains,
                                is_sharded=parsed_cmd_args.is_sharded)

    domain = RectangleDomain2D(start_point,
                               end_point,
                               train_samples,
                               test_samples,
                               use_BC=False,
                               sampling_strategy="hammersly",  # uniform_random, hammersly
                               ref_strategy="None",
                               model_structure=par_model.all_model_ranks)

    bc = BC_Burgers(start_point=start_point, end_point=end_point)
    pde = Burgers(bc_exact=bc)

    if dist.is_initialized():
        nets = [par_model]
    else:
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

    args.opt_type = 4
    if (args.opt_type == 0):
        algo = GDLinesearch
    elif (args.opt_type == 1):
        algo = PrecLBFGS
    elif (args.opt_type == 2):
        algo = optim.Adam
    elif (args.opt_type == 3):
        algo = optim.SGD
    else:
        algo = APTS

    print(f"Optimizer type: {algo}")

    dir_name = "Algo_"+str(args.opt_type)
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    trainer = UnconstrainedRegressionTrainer(
        training_pinn_flg=True, num_subdomains=1, type="regression", freq_print=100, use_wandb=False)

    trainer.config["max_epochs"] = args.max_epochs
    trainer.config["conv_history_csv_name"] = dir_name + "/" + \
        'Summary_seed_' + str(args.seed) +\
        '_lrg'+str(args.lr_global) +\
        "_opt_type_" + str(args.opt_type) +\
        "max_epochs" + str(args.max_epochs) +\
        "_num_points_x_"+str(args.num_points_x) +\
        "_num_points_y_"+str(args.num_points_y) +\
        "_num_ref_"+str(args.num_ref) +\
        "_num_levels_"+str(args.num_levels) +\
        "_width_"+str(args.width) +\
        "_hiddenlayers_coarse_"+str(args.hiddenlayers_coarse) +\
        "_ada_net_"+str(args.ada_net) +\
        "_T_"+str(args.T) +\
        "_history_"+str(args.history) +\
        "_use_adaptive_activation_" + \
        str(args.use_adaptive_activation)

    start_time = time.time()
    if (args.ada_net == False):
        convergence_status = trainer.train(dataset=domain,
                                           net=nets[-1],
                                           criterion=pde.criterion,
                                           optimizer_type=algo,
                                           args=args)
        elapsed_time = time.time() - start_time

        # if(rank==0):
        out_name = dir_name + "/" + 'Summary.csv'

        df = pd.DataFrame({'seed': [args.seed],
                          'num_points_x': [args.num_points_x],
                           'num_points_y': [args.num_points_y],
                           'num_ref': [args.num_ref],
                           'num_levels': [args.num_levels],
                           'width': [args.width],
                           'hiddenlayers_coarse': [args.hiddenlayers_coarse],
                           'ada_net': [args.ada_net],
                           'use_adaptive_activation': [args.use_adaptive_activation],
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

            if (l > 0):
                print("l-1: ", l-1, "l:  ", l)
                ml_hierarchy.prolong_params(nets[l-1], nets[l])

            ml_scaling_factor = (2**((l+1) - args.num_levels))
            trainer.config["max_epochs"] = int(
                (1./ml_scaling_factor)*(args.max_epochs/ml_hierarchy.num_levels))

            convergence_status = trainer.train(dataset=domain,
                                               net=nets[l],
                                               criterion=pde.criterion,
                                               optimizer_type=algo,
                                               args=args)

            convergence_statuses.append(convergence_status)

        elapsed_time = time.time() - start_time
        epochs = 0
        num_loss_evals = 0

        for l in range(0, len(convergence_statuses)):

            ml_scaling_factor = (2**((l+1) - args.num_levels))
            print("l ", l, "   ml_scaling_factor ", ml_scaling_factor)

            epochs += ml_scaling_factor * convergence_statuses[l]["epochs"]
            num_loss_evals += ml_scaling_factor * \
                convergence_statuses[l]["num_loss_evals"]

        out_name = dir_name + "/" + 'Summary.csv'

        df = pd.DataFrame({'seed': [args.seed],
                          'num_points_x': [args.num_points_x],
                           'num_points_y': [args.num_points_y],
                           'num_ref': [args.num_ref],
                           'num_levels': [args.num_levels],
                           'width': [args.width],
                           'hiddenlayers_coarse': [args.hiddenlayers_coarse],
                           'ada_net': [args.ada_net],
                           'use_adaptive_activation': [args.use_adaptive_activation],
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


if __name__ == '__main__':
    if 1 == 2:
        main(cmd_args=sys.argv[1:])
    else:
        WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if WORLD_SIZE == 0:
            print("No CUDA device(s) detected.")
            exit(0)

        MASTER_ADDR = 'localhost'
        MASTER_PORT = '12345'
        WORLD_SIZE = 1
        # Also add command-line arguments to main
        # --optimizer "APTS" --dataset "MNIST" --batch_size "60000" --model "feedforward" --num_subdomains "2" --num_replicas_per_subdomain "1" --num_stages_per_replica "2" --trial "0" --epochs "2"

        mp.spawn(main, args=(None, MASTER_ADDR, MASTER_PORT, WORLD_SIZE),
                 nprocs=WORLD_SIZE, join=True)
