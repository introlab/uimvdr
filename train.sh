#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:4         # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=4    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G      
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

# Copie du code et des datasets
cd $SLURM_TMPDIR/
cp -r  ~/dev/weakseparation .
cp -r /home/jacobk17/projects/def-fgrondin/jacobk17/dataset/FSD50K ./weakseparation/library/dataset/

# Création de l'environnement virtuel
cd $SLURM_TMPDIR
module load python/3.8 opencv cuda cudnn
virtualenv --no-download env
source env/bin/activate
pip install --no-index -r ./weakseparation/requirements.txt 

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

#Execution de l'entrainement
cd weakseparation/library/weakseparation/src
srun python main.py --train --epochs 2005 --log_path /home/jacobk17/dev/weakseparation/logs \
 --dataset_path $SLURM_TMPDIR/weakseparation/library/dataset --num_of_workers 4 --run_id $SLURM_JOB_ID "$@"
