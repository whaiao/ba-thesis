#!/bin/bash
#SBATCH --job-name=atomic
#SBATCH --output=atomic-tagger.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --qos=batch
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128000

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate thesis
export PYTHONPATH=./
srun python src/data/atomic.py
