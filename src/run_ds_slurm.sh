#!/bin/bash
#SBATCH --account=c24
#SBATCH --job-name=deepspeed
#SBATCH --output=ds_output.txt
#SBATCH --error=ds_error.txt
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
conda activate deepspeed
echo "Calling deepspeed `date`"
srun deepspeed --bind_cores_to_rank deepspeed_example.py --deepspeed $@
echo "Deepspeed finished `date`"
