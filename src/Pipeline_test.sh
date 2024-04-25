#!/bin/bash

#SBATCH --account=c24
#SBATCH --job-name=pipeline_test
#SBATCH --output=pipeline_test.out
#SBATCH --error=pipeline_test.err
#SBATCH --nodes=3
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --exclusive

SCRIPT_NAME="Pipeline_test.py"

echo "Loading daint-gpu `date`"
module load daint-gpu
echo "Finished loading daint-gpu `date`"

conda activate pytorch_daint_multigpu

echo "Calling $SCRIPT_NAME `date`"
srun python -u $SCRIPT_NAME
echo "Finished with $SCRIPT_NAME `date`"
