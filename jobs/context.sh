#!/bin/bash
#SBATCH --job-name=ctx-bert
#SBATCH --output=ctx-bert.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=batch
#SBATCH --time 3-0:00:00
#SBATCH --mem=32g
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate hf
export PYTHONPATH=./
srun python src/models/bert/context_encoder.py
