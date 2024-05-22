#!/bin/bash
#SBATCH --account=c24
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --exclusive

module load daint-gpu

export GPUS_PER_NODE=1
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

source /users/scruzale/anaconda3/etc/profile.d/conda.sh
conda activate deepspeed
echo "Calling deepspeed `date`"

srun bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
cifar10_deepspeed.py --deepspeed'

echo "Deepspeed finished `date`"
exit 0
