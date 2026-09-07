#!/usr/bin/env python3
"""Quickstart: solve a 1D Poisson problem with a physics-informed network.

Run it with no arguments and no infrastructure -- no Docker, no wandb, no
cluster:

    python3 experiments/examples/pinn_poisson_exact.py

The problem is

    -u''(x) = sin(pi x)   on (0, 1),      u(0) = u(1) = 0

which is worth using as a demo because it has a closed-form solution,

    u(x) = sin(pi x) / pi^2

so the network's output can be scored against the true answer rather than
against its own training loss. A loss that goes down tells you very little; a
relative L2 error against an exact solution tells you whether the PDE was
actually solved.

The same network, initialization and iteration budget are given to a
first-order trust-region method and to the second-order variant that uses a
limited-memory SR1 model of the curvature. The second-order method reaches a
markedly lower error for the same number of iterations -- typically by a factor
of a few to over an order of magnitude, depending on the seed.
"""

from __future__ import annotations

import argparse
import math
import time

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
import torch

from dd4ml.datasets.pinn_poisson import Poisson1DDataset
from dd4ml.models.ffnn.pinn_ffnn import PINNFFNN
from dd4ml.optimizers.tr import TR
from dd4ml.utility.optimizer_utils import get_tr_hparams
from dd4ml.utility.pinn_poisson_loss import PoissonPINNLoss


class _HParams:
    """Trust-region settings. `glob_second_order` is what we are comparing."""

    delta = 1.0
    max_delta = 10.0
    min_delta = 1e-8
    norm_type = 2
    tol = 1e-14
    glob_dogleg = False

    def __init__(self, second_order: bool):
        self.glob_second_order = second_order


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    """u(x) = sin(pi x) / pi^2, the analytical solution of the problem above."""
    return torch.sin(math.pi * x) / math.pi**2


def build_problem(seed: int):
    """Collocation points, boundary mask, an untrained network and the PDE loss."""
    dataset = Poisson1DDataset(Poisson1DDataset.get_default_config())
    x = dataset.data.clone()
    is_boundary = torch.tensor(
        [[1.0] if i >= len(dataset.x_interior) else [0.0] for i in range(len(dataset))]
    )
    torch.manual_seed(seed)  # same initialization for both methods
    model = PINNFFNN(PINNFFNN.get_default_config())
    return x, is_boundary, model, PoissonPINNLoss()


def train(second_order: bool, iterations: int, seed: int):
    x, is_boundary, model, criterion = build_problem(seed)

    def closure(compute_grad: bool = False):
        # The loss differentiates the network output with respect to x to form
        # the PDE residual, so x must carry grad *before* the forward pass and
        # the whole thing must run with autograd enabled.
        x.requires_grad_(True)
        criterion.current_x = x
        loss = criterion(model(x), is_boundary)
        if compute_grad:
            model.zero_grad()
            loss.backward()
        return loss.detach()

    optimizer = TR(model.parameters(), **get_tr_hparams(_HParams(second_order)))

    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.step(closure)
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        predicted = model(x)
        truth = exact_solution(x)
        rel_l2 = (torch.norm(predicted - truth) / torch.norm(truth)).item()

    return {
        "x": x.detach().squeeze(),
        "u": predicted.squeeze(),
        "rel_l2": rel_l2,
        "loss": float(closure()),
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="pinn_poisson_exact.png")
    args = parser.parse_args()

    print("-u''(x) = sin(pi x) on (0, 1), u(0) = u(1) = 0")
    print("exact solution: u(x) = sin(pi x) / pi^2")
    print(f"{args.iterations} iterations per method, seed {args.seed}\n")

    results = {}
    for label, second_order in (("first-order", False), ("second-order", True)):
        results[label] = train(second_order, args.iterations, args.seed)
        r = results[label]
        print(
            f"  {label:13s} relative L2 error {r['rel_l2']:.3e}"
            f"   residual loss {r['loss']:.2e}   {r['seconds']:.1f}s"
        )

    ratio = results["first-order"]["rel_l2"] / results["second-order"]["rel_l2"]
    print(f"\n  second-order is {ratio:.1f}x more accurate for the same budget")

    x = results["first-order"]["x"]
    order = torch.argsort(x)
    plt.figure(figsize=(7, 4))
    plt.plot(x[order], exact_solution(x)[order], "k-", lw=2, label="exact")
    for label, style in (("first-order", "--"), ("second-order", ":")):
        r = results[label]
        plt.plot(
            x[order],
            r["u"][order],
            style,
            lw=2,
            label=f"{label} (rel. $L_2$ = {r['rel_l2']:.1e})",
        )
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.title("1D Poisson via a physics-informed network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
