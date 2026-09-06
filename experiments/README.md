## Usage

This directory holds the experiment drivers: run configurations, SLURM job
templates, sweep submission scripts and analysis helpers. `tests/` is reserved
for the pytest suite.

In a ***local*** environment (e.g. PC), you can, for example, run the following:
```bash
open -a Docker
wandb server start
python3 ./experiments/run_config_file.py --sweep_config="./experiments/config_files/<config_file>.yaml"
```
See `experiments/config_files/` for the available configurations.

In a ***cluster*** environment, e.g. managed by SLURM, you can run:
```bash
cd experiments
./submit_jobs.sh --config <config_name>
```
`<config_name>` must match a file in `experiments/job_configs/` (without the `.conf` extension); it defines the optimizers, datasets, models and sweep parameters for the job. `submit_jobs.sh` and `submit_jobs_common.sh` take care of the job name, wandb usage, and picking the right cluster job template (`rosa.job` or `daintalps.job`, selected automatically based on the environment) — there's nothing to edit by hand. Change the time/number of GPUs/etc. directly in the relevant `.conf` file or job template if needed.

### Cluster prerequisites

The job templates source these files from your home directory on the cluster.
Create them once per machine:

| File | Defines | Purpose |
| --- | --- | --- |
| `~/.tests_runpath` | `TESTS_RUNPATH` | Absolute path to this repo's `experiments/` directory. The job `cd`s there before launching the run. |
| `~/.slurm_env` | `SLURM_ACCOUNT` | The SLURM account to charge (`daintalps.job` only). |
| `~/.wandb_env` | wandb credentials (e.g. `WANDB_API_KEY`) | Sourced only when the job runs with wandb enabled. |

For example:
```bash
echo 'export TESTS_RUNPATH=/path/to/DD4ML/experiments' > ~/.tests_runpath
```

Note on the name: `TESTS_RUNPATH` dates from when these scripts lived under
`tests/`, so it reads as though it points at the pytest suite. It must point at
`experiments/`. If you would rather the name matched, define
`EXPERIMENTS_RUNPATH` in `~/.experiments_runpath` and update the two `source`
lines in `rosa.job` and `daintalps.job` to match.
