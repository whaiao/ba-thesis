#!/bin/bash
#SBATCH --job-name=dgpt-small
#SBATCH --output=dgpt-sm.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=bigbatch
#SBATCH --time 5-0:00:00
#SBATCH --mem=16GB
#SBATCH --partition=afkm
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate hf
export PYTHONPATH=./
srun python src/models/dialoGPT/conditional_generation.py
