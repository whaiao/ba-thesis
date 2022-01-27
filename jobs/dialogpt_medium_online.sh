#!/bin/bash
#SBATCH --job-name=m-dialogpt
#SBATCH --output=medium-gpt-online.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=batch
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:2
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate hf
export PYTHONPATH=./
srun python src/models/dialoGPT/fine_tuning_medium_online.py
