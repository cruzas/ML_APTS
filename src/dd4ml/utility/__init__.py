# dd4ml/utility/__init__.py

from .apts_utils import (
    apts_ip_restore_params,
    clone_model,
    flatten_params,
    mark_trainable,
    print_params_norm,
    restore_params,
    trainable_grads_to_vector,
    trainable_params_to_vector,
)
from .dist_utils import (
    check_gpus_per_rank,
    detect_environment,
    dprint,
    find_free_port,
    is_main_process,
    prepare_distributed_environment,
    receive_shape,
    send_shape,
)
from .factory import criterion_factory, dataset_factory, optimizer_factory
from .ml_utils import closure, get_device, is_function_module, list_flattener
from .optimizer_utils import (
    ensure_tensor,
    get_apts_hparams,
    get_loc_tr_hparams,
    get_loc_tradam_hparams,
    get_lssr1_loc_tr_hparams,
    get_lssr1_tr_hparams,
    get_state_dict,
    get_tr_hparams,
    solve_tr_first_order,
    solve_tr_second_order,
)
from .trainer_setup import generic_run

# .utils and .dist_utils both define a `broadcast_dict` with different
# signatures (per-rank None placeholder vs. a value broadcast from every
# rank). The old `from .dist_utils import *` / `from .utils import *` star
# imports silently resolved this to .utils's version, since .utils was
# imported last; every real caller already expects that one (trainer_setup.py
# even imports it locally as `from .utils import broadcast_dict`).
from .utils import CfgNode, broadcast_dict, set_seed
from .wandb_utils import (
    best_learning_rates,
    compute_best_lr_per_batch_size,
    compute_metrics,
    fetch_run_data,
)
from .pinn_poisson_loss import PoissonPINNLoss
from .pinn_poisson2d_loss import Poisson2DPINNLoss
from .pinn_poisson3d_loss import Poisson3DPINNLoss
from .pinn_allencahn_loss import AllenCahnPINNLoss
from .pinn_allencahn_time_loss import AllenCahnTimePINNLoss
