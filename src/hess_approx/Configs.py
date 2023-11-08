from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import argparse

# def init():
global use_keras_optimizer
use_keras_optimizer=False


base_optimizer_config = dict(
    max_epochs=300,
    seed=42,
    acc_train_tol = 0.98,
    acc_val_tol = 0.98,
    loss_train_tol=1e-5,
    loss_val_tol=1e-5,
    loss_diff_tol = 1e-9,
    use_line_search=True,
    full_batch_stats=True,
    use_full_train_stats = False,
    g_norm_tol = 1e-9 ,
    g_norm_diff_tol = 1e-12,
    output_csv_name=None,
    name=None,
    max_loss_evals=1000,
    max_grad_evals=1000,
    max_Hessian_evals=1000,
    use_dropout=False,
    weight_decay=False,
    weight_decay_param=0.0005
)


config_SGD = dict(
    learning_rate=0.1,
    lr_decay_type=None,
    patience=25,
    learning_rate_decay=0.001,
    step_decay_freq=10,
    k=0.1,
    eta=0.9,
    momentum_type=None,
    # stuff for stopping criterium
    patience_loss=15,
    patience_train_acc=15,
    patience_val_acc=15,
    diff_loss=1e-3,
    diff_acc_train=0.01,
    diff_acc_val=0.01,
)

config_WRSA = dict(
    use_warm_restart = True, 
    eta_min     = 0.001, 
    eta_max     = 0.5,
    T_cycle     = 60,
    num_cycles  = 3,
    T_mult      = 2,
)



config_Adam = dict(
    beta1=0.9,
    beta2=0.999, 
    epsilon=1e-8,    
)


config_HB = dict(
    use_HB_precond=True
)


config_HB_SGD=dict(config_SGD, **config_HB)
config_HB_Adam=dict(config_HB_SGD, **config_Adam)


config_SSN = dict(
    learning_rate=0.1,
    use_direct_Hess_eval=True,
    linear_solve_max_it = 5
)


config_line_search = dict(
    line_search_max_it=15,
    line_search_rho=0.5,
    line_search_c=1e-4
)


config_lin_solver = dict(
    ls_max_it=10,
    ls_atol = 1e-9,
    ls_verbose = False
)


config_hess_approx = dict(
    ha_type="LSR1",
    ha_memory = 10,
    ha_eig_type="standard",
    ha_tol=1e-15,
    ha_r = 1e-8,
    ha_print_errors = False,
    ha_update_type="overlap", # "sampling"
    ha_sampling_radius=0.05
)


config_tr_base = dict(
    delta0=10,
    delta_max=1e16,
    delta_min=1e-16,
    gamma1_ns=0.5,
    gamma2_ns=2.0,
    gamma1_s=0.25,
    gamma2_s=2.5,
    eta1=0.25,
    eta2=0.75,
    rho_tol=1e-7,
    subproblem_type="OBS",
    max_succ_its=300
)


config_var_proj_help = dict(
    smoothing=False,
)

config_var_proj = dict(config_var_proj_help, **config_SGD)



config_ASTR_help=dict(
    use_line_search_global=False,
    print_local = False,
    max_fails = 5,
    delta_max0=1.0,
    use_BN_inner = False,
    use_BN_outer = False,
    f_cycle = False,
    mb_increase_levels=False,
    mb_decrease_levels=False,
    patience_decrease_levels=False,
    patience_decrease_factor=1.5,
    ha_adaptivity=False,
    ha_adaptivity_start_size=1,
    monotonicity_check_frequency=5,
    max_epochs_coarsest=100,
    check_acc = False)

config_ASTR=dict(config_ASTR_help, **config_SGD)


config_VR=dict(
    VR_eta=1./8.,
    VR_inner_its=100)


config_multilevel=dict(
    num_levels=3,
    coarse_steps=10,
    smoothing_steps=2,
    succ_coarse_steps=5,
    succ_smoothing_steps=1,
    coarse_QP_steps=10,
    fine_QP_steps=5,
    print_local=False,
    )


config_MLVR_help=dict(
    consistency_type="additive_svrg", # additive, multiplicative, additive_svrg
    coarsening_factor=2,
    coarse_solver="gd", # newton
    samples_sizes=None,
    nested_sampling=True,
    use_line_search_global=True,
    )


config_MLVR=dict(config_MLVR_help, **config_multilevel)

config_DD_SGD = dict(config_SGD, **config_MLVR)

config_MGOPT_help=dict(
    post_smooth_fine=False,
    gradient_lagg_consistency_term=False)


config_MGOPT=dict(config_MGOPT_help, **config_multilevel)



config_RMTR_help=dict(
    post_smooth_fine=False,
    gradient_lagg_consistency_term=False,
    use_s_norm_update=False,
    use_post_smoothing=False,
    use_BN=False,
    scaling_rho=1.0,
    subproblem_type_fine="OBS",
    subproblem_type_coarse="OBS",
    ha_only_coarse=True,
    use_line_search_global=True,
    use_line_search_local=False,
    f_cycle=True,
    f_cg_steps=100,
    )


config_RMTR=dict(config_RMTR_help, **config_multilevel)





def collect_command_line_args(  reg_def = 0.001, \
                                T_def = 1,\
                                blks_def = 5, \
                                node_layer_def = 32,\
                                mb_size_def = 1000, \
                                lr_def = 0.1,\
                                llr_def = 0.001,\
                                epochs_def = 200,\
                                acc_def = 0.85, \
                                lev_def = 4, \
                                smoothing_its_def = 1, \
                                coarse_its_def = 5, \
                                out_def = '',\
                                lr_decay_def = 'none',\
                                seed=1,\
                                gamma_corr=0,
                                dsn_def='mushrooms',\
                                sas_def=None,\
                                uls_def=False,\
                                pl_def=False,\
                                lgds_def=1,\
                                sgd_on_def=0,\
                                dd_sgd_on_def=0,\
                                sarah_on_def=0,\
                                svrg_on_def=0,\
                                vr_inner_its_def=100,\
                                obj_def='additive',\
                                dd_type_def="additive"
                                ):

    ap = argparse.ArgumentParser()
    ap.add_argument("-dsn", "--dataset_name", default=dsn_def, required=False, type=str, help="Name of dataset")
    ap.add_argument("-sas", "--sample_sizes", default=sas_def, required=False, type=None, help="Sample sizes")
    ap.add_argument("-uls", "--use_line_search", default=uls_def, required=False, type=int, help="Use line search")
    ap.add_argument("-pl", "--print_local", default=pl_def, required=False, type=bool, help="Print local")
    ap.add_argument("-lgds", "--local_grad_desc_steps", default=lgds_def, required=False, type=int, help="Total number of local gradient descent steps")
    ap.add_argument("-sgd_on", "--run_sgd_test", default=sgd_on_def, required=False, type=int, help="Activate SGD test")
    ap.add_argument("-dd_sgd_on", "--run_dd_sgd_test", default=dd_sgd_on_def, required=False, type=int, help="Activate DD-SGD test")
    ap.add_argument("-sarah_on", "--run_sarah_test", default=sarah_on_def, required=False, type=int, help="Activate SARAH test")
    ap.add_argument("-svrg_on", "--run_svrg_test", default=svrg_on_def, required=False, type=int, help="Activate SVRG test")
    ap.add_argument("-vr_inits", "--vr_inner_its", default=vr_inner_its_def, required=False, type=int, help="Number of VR inner iterations")

    ap.add_argument("-reg", "--reg_parameter_coarse", default=reg_def, required=False, type=float, help="Regularization param on the coarsest level")
    ap.add_argument("-T", "--final_time", default=T_def,  required=False, type=float, help="Final time on all levels")
    ap.add_argument("-blks", "--residual_blocks_coarse", default=blks_def,  required=False, type=int, help="Number of residual blocks on the coarsest level")
    ap.add_argument("-node_layer", "--no_nodes_per_layer", default=node_layer_def,  required=False, type=int, help="Number of nodes per layer/number of filter")

    ap.add_argument("-mb_size", "--mini_batch_size", default=mb_size_def, required=False, type=int, help="Mini-batch size used for SGD")
    ap.add_argument("-lr", "--lr_rate", default=lr_def, required=False, type=float, help="Learning rate used on all levels")
    ap.add_argument("-llr", "--local_lr_rate", default=llr_def, required=False, type=float, help="Local learning rate used on all levels")
    ap.add_argument("-glr", "--global_lr_rate", default=lr_def, required=False, type=float, help="Global learning rate used on all levels")
    ap.add_argument("-pglr", "--prolongation_global_lr_rate", default=lr_def, required=False, type=float, help="Prolongation global learning rate used on all levels")
    ap.add_argument("-epochs", "--epochs", default=epochs_def,  required=False, type=int, help="Maximum number of epochs")
    ap.add_argument("-acc", "--accuracy", default=acc_def,  required=False, type=float, help="Validation accuracy used for stopping")

    ap.add_argument("-lev", "--num_levels", default=lev_def, required=False, type=int, help="Numbber of levels used for generating ML hierarchy")
    ap.add_argument("-smoothing_its", "--smoothing_its", default=smoothing_its_def, required=False, type=int, help="Number of smoothing steps")
    ap.add_argument("-coarse_its", "--coarse_its", default=coarse_its_def, required=False, type=int, help="Number of coarse grid steps")

    ap.add_argument("-out", "--output_name", default=out_def, required=False, type=str, help="Name of output file for storing csv stats")
    ap.add_argument("-lr_decay", "--lr_decay_type", default=lr_decay_def,  required=False, type=str, help="Type of learning rate decay")

    ap.add_argument("-seed", "--seed", default=seed, required=False, type=int, help="Seed for rand. execs")
    ap.add_argument("-gamma_corr", "--gamma_corr", default=gamma_corr, required=False, type=float, help="gamma_corr value")
    ap.add_argument("-obj", "--objective", default=obj_def, required=False, type=str, help="Subdomain objective type (additive, multiplicative, etc.)")
    ap.add_argument("-dd_type", "--domain_decomp_type", default=dd_type_def, required=False, type=str, help="Domain decomposition type.")



    args = ap.parse_args()

    print("lr_rate", args.lr_rate)
    print("llr_rate", args.local_lr_rate)
    print("glr_rate", args.global_lr_rate)
    print("pglr_rate", args.prolongation_global_lr_rate)
    print("epochs", args.epochs)
    print("reg_parameter_coarse", args.reg_parameter_coarse)
    print("final_time", args.final_time)
    print("residual_blocks_coarse", args.residual_blocks_coarse)
    print("mini_batch_size", args.mini_batch_size)
    print("num_levels", args.num_levels)
    print("no_nodes_per_layer", args.no_nodes_per_layer)
    print("epochs", args.epochs)
    print("accuracy", args.accuracy)
    print("smoothing_its", args.smoothing_its)
    print("coarse_its", args.coarse_its)
    print("lr_decay_type", args.lr_decay_type)
    print("seed", args.seed)
    print("gamma_corr", gamma_corr)
    print("dataset_name", args.dataset_name)
    print("sample_sizes", args.sample_sizes)
    print("use_line_search", args.use_line_search)
    print("print_local", args.print_local)
    print("local_grad_desc_steps", args.local_grad_desc_steps)
    print("VR_inner_its", args.vr_inner_its)
    print("objective", args.objective)
    print("domain decomposition type", args.domain_decomp_type)

    if(args.output_name!=''):
        args.output_name = args.output_name +'_lr' + str(args.lr_rate) +'_reg' + str(args.reg_parameter_coarse) +'_T' + str(args.final_time) +'_rb' \
                            + str(args.residual_blocks_coarse) +'_mbs' + str(args.mini_batch_size) +'_lev' + str(args.num_levels) +'_filters' \
                            + str(args.no_nodes_per_layer) +'_its' + str(args.epochs) +'_acc' + str(args.accuracy) +'_sm' \
                            + str(args.smoothing_its) +'_sm_it' + str(args.coarse_its) + '_lr_decay' + args.lr_decay_type


    print("output_name", args.output_name)

    return args
