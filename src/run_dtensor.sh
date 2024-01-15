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

echo "Loading daint-gpu `date`"
module load daint-gpu
echo "Finished loading daint-gpu `date`"

echo "Sourcing conda.sh `date`"
source /users/scruzale/anaconda3/etc/profile.d/conda.sh
echo "Finished sourcing conda.sh `date`"

echo "Activating deepspeed `date`"
conda activate deepspeed
echo "Finished activating deepspeed `date`"

echo "Calling main `date`"
srun python -u deepspeed_example.py
echo "Finished with main `date`"
