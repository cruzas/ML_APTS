#!/bin/bash
#SBATCH --account=c24
#SBATCH --job-name=test
#SBATCH --output=test_output.txt
#SBATCH --error=test_error.txt
#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --exclusive

module load daint-gpu

source /users/scruzale/anaconda3/etc/profile.d/conda.sh
conda activate llms
echo "Calling main `date`"
srun python -u deepspeed_example.py
echo "Finished with main `date`"
