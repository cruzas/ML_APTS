#!/usr/bin/env python3
"""Quickstart: when does a second-order trust-region method beat Adam and SGD?

Run it with no arguments and no infrastructure -- no Docker, no wandb, no
cluster:

    python3 experiments/examples/quickstart.py

The test problem is a convex quadratic with a known minimizer,

    f(w) = 0.5 (w - w*)^T A (w - w*),      A = Q diag(lambda) Q^T

so the optimal value is exactly zero and any method can be scored against the
truth rather than against its own loss curve. Two knobs decide the outcome, and
between them they explain when this library's optimizers are worth reaching for.

1. Whether the ill-conditioning is aligned with the coordinate axes.

   With Q = I the Hessian is diagonal, and Adam is superb: its per-parameter
   second-moment scaling *is* a diagonal preconditioner, so a diagonal Hessian
   is its best case, not a hard one. Rotating the same spectrum by a random
   orthogonal Q leaves the condition number identical but destroys that
   advantage, because no per-coordinate rescaling can undo a rotation.

2. Whether the limited-memory SR1 model has enough pairs to capture the
   curvature.

   SR1 built from n curvature pairs reproduces the Hessian of an n-dimensional
   quadratic exactly, at which point the trust-region step is a Newton step.
   Below that the low-rank model simply misses directions, and the method is no
   better than a first-order one.

Both baselines are given a learning-rate sweep and scored at their best, so the
comparison is not decided by a badly chosen step size.

The same reasoning explains where the trust-region method does *not* win: on a
large network with a well-conditioned loss and memory far smaller than the
parameter count, a cheap diagonal preconditioner is hard to beat.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
import torch

from dd4ml.optimizers.tr import TR
from dd4ml.utility.optimizer_utils import get_tr_hparams

LR_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


class _HParams:
    delta = 1.0
    max_delta = 1e4
    min_delta = 1e-14
    norm_type = 2
    tol = 1e-18
    glob_second_order = True
    glob_dogleg = False


def build_quadratic(n: int, kappa: float, seed: int, rotate: bool):
    """Return f and its exact minimizer for a quadratic of condition number kappa."""
    generator = torch.Generator().manual_seed(seed)
    eigenvalues = torch.logspace(0, torch.log10(torch.tensor(float(kappa))), n).double()
    if rotate:
        Q = torch.linalg.qr(
            torch.randn(n, n, generator=generator, dtype=torch.float64)
        )[0]
    else:
        Q = torch.eye(n, dtype=torch.float64)
    A = Q @ torch.diag(eigenvalues) @ Q.T
    w_star = torch.randn(n, generator=generator, dtype=torch.float64)

    def f(w: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((w - w_star) @ A @ (w - w_star))

    return f, w_star


def run_baseline(name: str, f, n: int, iterations: int):
    """Best result over a learning-rate sweep, so the baseline is not strawmanned.

    A method can diverge at every step size in the grid -- SGD does exactly that
    here -- so the best history is seeded with the first sweep rather than left
    unset, and the reported value stays inf to record the divergence.
    """
    best_value, best_history = float("inf"), None
    for lr in LR_GRID:
        w = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        optimizer = (torch.optim.SGD if name == "SGD" else torch.optim.Adam)([w], lr=lr)
        history = []
        for _ in range(iterations):
            loss = f(w)
            history.append(float(loss.detach()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        value = float(f(w).detach())
        if best_history is None:
            best_history = history
        if value == value and value < best_value:  # value == value rejects NaN
            best_value, best_history = value, history
    return best_value, best_history


def run_trust_region(f, n: int, iterations: int, memory: int):
    w = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    hparams = get_tr_hparams(_HParams)
    hparams["mem_length"] = memory
    optimizer = TR([w], **hparams)

    history = []

    def closure(compute_grad: bool = False):
        loss = f(w)
        if compute_grad:
            if w.grad is not None:
                w.grad.zero_()
            loss.backward()
        return loss.detach()

    for _ in range(iterations):
        history.append(float(f(w).detach()))
        optimizer.step(closure)
    return float(f(w).detach()), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--condition-number", type=float, default=1e6)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="quickstart.png")
    args = parser.parse_args()

    n, kappa = args.dimension, args.condition_number
    print(
        f"f(w) = 0.5 (w - w*)^T A (w - w*),  n = {n},  condition number = {kappa:.0e}"
    )
    print(f"optimal value f(w*) = 0;  {args.iterations} iterations;  seed {args.seed}")
    print("baselines are swept over learning rates and scored at their best")
    print("SGD and Adam are torch.optim; TR is this library's trust-region method\n")

    print("  1. Does the ill-conditioning align with the coordinate axes?")
    print(f"     {'Hessian':<24} {'SGD':>10} {'Adam':>10} {'TR':>12}")
    curves = {}
    for rotate, label in ((False, "diagonal"), (True, "rotated")):
        f, _ = build_quadratic(n, kappa, args.seed, rotate)
        sgd, sgd_hist = run_baseline("SGD", f, n, args.iterations)
        adam, adam_hist = run_baseline("Adam", f, n, args.iterations)
        tr, tr_hist = run_trust_region(f, n, args.iterations, memory=n)
        print(f"     {label:<24} {sgd:>10.2e} {adam:>10.2e} {tr:>12.2e}")
        if rotate:
            curves = {"SGD": sgd_hist, "Adam": adam_hist, "TR": tr_hist}
    print("\n     Both reach machine precision when the Hessian is diagonal: Adam's")
    print("     per-parameter scaling is exactly a diagonal preconditioner, so that")
    print("     is its best case. The rotation leaves the condition number unchanged")
    print("     but no per-coordinate rescaling can undo it, and Adam stalls while")
    print("     the trust-region method is unaffected. SGD diverges at every step")
    print("     size in the grid, on both problems.\n")

    print("  2. Does the SR1 memory capture the curvature? (rotated problem)")
    f, _ = build_quadratic(n, kappa, args.seed, rotate=True)
    print(f"     {'SR1 memory':<24} {'final f(w)':>12}")
    for memory in (5, 10, n):
        value, _ = run_trust_region(f, n, args.iterations, memory)
        note = "  <- memory = n, SR1 is exact for a quadratic" if memory == n else ""
        print(f"     {memory:<24} {value:>12.2e}{note}")

    # SGD diverges past 1e300 here, which would flatten everything else on a
    # shared log axis, so the view is clipped to a readable window. Curves that
    # leave it simply run off the top or bottom edge.
    low, high = 1e-25, 1e8
    legend_names = {"TR": "TR (trust-region)"}
    plt.figure(figsize=(7, 4.5))
    for label, history in curves.items():
        series = [min(max(v, low), high) if v == v else float("nan") for v in history]
        plt.semilogy(
            series, lw=2, label=f"{legend_names.get(label, label)}, best of sweep"
        )
    plt.ylim(low, high)
    plt.xlabel("iteration")
    plt.ylabel("$f(w)$   (optimum $= 0$)")
    plt.title(
        f"Rotated quadratic, $n={n}$, $\\kappa=10^{{{int(torch.log10(torch.tensor(kappa)))}}}$"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
