#!/bin/bash

#SBATCH --job-name=hyperparmaeter
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --account=rmm7011_a_gpu
#SBATCH --gpus=1
#SBATCH --mail-user=hojin.kim@psu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


module load anaconda
conda activate pyg_env

cd /storage/group/rmm7011/default/hjkim/Recon_GNN

python -u train.py