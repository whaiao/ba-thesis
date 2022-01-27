#!/bin/bash
#SBATCH --job-name=dialogpt-online
#SBATCH --output=gpt-online.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=batch
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate hf
export PYTHONPATH=./
srun python src/models/dialoGPT/fine_tuning_medium_online.py
