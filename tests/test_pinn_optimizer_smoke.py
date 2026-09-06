"""Smoke test: run the APTS optimizers end to end on a real PINN problem.

This exists to catch regressions from refactors that touch the optimizer import
surface (e.g. converting ``from .apts_base import *`` to explicit imports) or the
optimizer/model/dataset/loss wiring in general. It deliberately uses dd4ml's
actual PINN classes (``Poisson1DDataset``, ``PINNFFNN``, ``PoissonPINNLoss``)
rather than a synthetic proxy, so a broken import, a missing constructor
argument, or a dtype mismatch in the PDE-residual computation fails here.

This does not assert the network fits the true solution of the PDE -- with 30
steps on an 8-hidden-layer network that would be flaky. It only asserts the
optimizer runs, drives the residual down, and leaves the model finite.
"""

import pytest
import torch
import torch.distributed as dist

from dd4ml.datasets.pinn_poisson import Poisson1DDataset
from dd4ml.models.ffnn.pinn_ffnn import PINNFFNN
from dd4ml.optimizers.apts_d import APTS_D
from dd4ml.optimizers.apts_p import APTS_P
from dd4ml.optimizers.tr import TR
from dd4ml.utility.optimizer_utils import get_loc_tr_hparams, get_tr_hparams
from dd4ml.utility.pinn_poisson_loss import PoissonPINNLoss

SEED = 0
N_STEPS = 30


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    """A real single-rank gloo group.

    APTS_D/APTS_P issue real collective calls (a no-op on one rank, but the
    code path is exercised); file-based rendezvous avoids picking a TCP port,
    which would be flaky on a shared CI runner.
    """
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("dist") / "rendezvous"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


class _HParams:
    """Minimal stand-in for the config object the hparam helpers read."""

    delta = 0.1
    max_delta = 2.0
    min_delta = 1e-4
    norm_type = 2
    tol = 1e-10
    glob_second_order = False
    glob_dogleg = False
    loc_second_order = False
    loc_dogleg = False


def _make_problem():
    """1D Poisson collocation points plus the PINN model/loss dd4ml's own factory wires up."""
    dataset = Poisson1DDataset(Poisson1DDataset.get_default_config())
    inputs = dataset.data.clone()
    boundary_flag = torch.tensor(
        [[1.0] if i >= len(dataset.x_interior) else [0.0] for i in range(len(dataset))]
    )

    torch.manual_seed(SEED)
    model = PINNFFNN(PINNFFNN.get_default_config())
    criterion = PoissonPINNLoss()
    return inputs, boundary_flag, model, criterion


def _residual_loss(criterion, model, inputs, boundary_flag):
    """Evaluate the PDE-residual loss.

    Coordinates must have grad tracking enabled *before* the forward pass, not
    after: the loss differentiates the output w.r.t. the input coordinates, so
    a graph that never recorded ``inputs`` as requiring grad has nothing to
    differentiate through. This mirrors ``Trainer.evaluate``'s own ordering in
    ``src/dd4ml/trainer.py``.
    """
    inputs.requires_grad_(True)
    u_pred = model(inputs)
    criterion.current_x = inputs
    return float(criterion(u_pred, boundary_flag).detach())


@pytest.mark.distributed
@pytest.mark.parametrize("cls", [APTS_D, APTS_P], ids=["apts_d", "apts_p"])
def test_apts_reduces_pinn_residual(process_group, cls):
    """APTS must run end to end on the real PINN loss without diverging."""
    inputs, boundary_flag, model, criterion = _make_problem()

    kwargs = {
        "params": model.parameters(),
        "model": model,
        "criterion": criterion,
        "device": "cpu",
        "nr_models": 1,
        "glob_opt": TR,
        "glob_opt_hparams": get_tr_hparams(_HParams),
        "loc_opt": TR,
        "loc_opt_hparams": get_loc_tr_hparams(_HParams),
        "glob_pass": True,
        "norm_type": 2,
        "max_loc_iters": 3,
        "max_glob_iters": 1,
        "tol": 1e-10,
        "delta": 0.1,
        "min_delta": 1e-4,
        "max_delta": 2.0,
    }
    if cls is APTS_D:
        # First-order consistency correction; APTS_P does not take this flag.
        kwargs["foc"] = True

    optimizer = cls(**kwargs)

    initial_loss = _residual_loss(criterion, model, inputs, boundary_flag)

    for _ in range(N_STEPS):
        criterion.current_x = inputs
        optimizer.step(inputs, boundary_flag)

    final_loss = _residual_loss(criterion, model, inputs, boundary_flag)

    assert final_loss == final_loss, "loss became NaN"
    assert final_loss < initial_loss, f"{initial_loss} -> {final_loss}"
    assert all(torch.isfinite(p).all() for p in model.parameters()), (
        "model parameters went non-finite"
    )
