#!/bin/bash
#SBATCH --job-name=adafactor
#SBATCH --output=adafactor.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=bigbatch
#SBATCH --partition=afkm
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --ntasks=1
#SBATCH --time=14-0:00:00

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate hf
export PYTHONPATH=./
srun python src/train.py
