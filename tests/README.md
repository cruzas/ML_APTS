## Usage

It is relatively straightforward to execute a test.

In a ***local*** environment (e.g. PC), you can, for example, run the following:
```bash
open -a Docker
wandb server start
python3 ./tests/run_config_file.py --sweep_config="./tests/config_files/<config_file>.yaml"
```
See `tests/config_files/` for the available configurations.

In a ***cluster*** environment, e.g. managed by SLURM, you can run:
```bash
cd tests
./submit_jobs.sh --config <config_name>
```
`<config_name>` must match a file in `tests/job_configs/` (without the `.conf` extension); it defines the optimizers, datasets, models and sweep parameters for the job. `submit_jobs.sh` and `submit_jobs_common.sh` take care of the job name, wandb usage, and picking the right cluster job template (`rosa.job` or `daintalps.job`, selected automatically based on the environment) — there's nothing to edit by hand. Change the time/number of GPUs/etc. directly in the relevant `.conf` file or job template if needed.
