"""Regression test: every dd4ml module must be importable on its own.

A circular import between ``dd4ml.utility`` and ``dd4ml.pmw`` once made 10 of
the package's modules -- the whole pmw core, four optimizers and the OBS solver
-- fail to import unless some other dd4ml module had been imported first. It
went unnoticed because the usual entry points happen to import
``dd4ml.utility`` early, which primes ``sys.modules`` and hides the cycle.

That is exactly why each import here runs in its own fresh interpreter. Calling
``importlib.import_module`` from inside pytest would prove nothing: by the time
this module runs, the collection of the other test modules has already
populated ``sys.modules``, so every import would succeed no matter how tangled
the real dependency graph is.

The subprocesses are run concurrently -- they spend nearly all their time
importing torch, so the wall time is a fraction of the serial cost.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Vendored from karpathy/minGPT, and excluded from linting for the same reason.
EXCLUDED_PREFIXES = ("dd4ml.models.gpt.nanogpt",)

IMPORT_TIMEOUT_S = 120


def _module_names():
    """Every importable module and subpackage under src/dd4ml."""
    for path in sorted((SRC / "dd4ml").rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in EXCLUDED_PREFIXES
        ):
            continue
        yield name


def _import_alone(module: str) -> tuple[str, str]:
    """Import `module` as the very first dd4ml import. Returns (module, error)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(SRC)]
    )
    env["WANDB_MODE"] = "disabled"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return module, f"timed out after {IMPORT_TIMEOUT_S}s"
    if proc.returncode == 0:
        return module, ""
    last_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    return module, last_line


@pytest.mark.slow
def test_every_module_imports_standalone():
    modules = list(_module_names())
    assert modules, "no dd4ml modules discovered -- is the src layout intact?"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_import_alone, modules))

    failures = [(mod, err) for mod, err in results if err]
    assert not failures, "modules that cannot be imported on their own:\n" + "\n".join(
        f"  {mod}: {err}" for mod, err in failures
    )
