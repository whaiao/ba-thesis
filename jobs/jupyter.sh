#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.log
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#
#SBATCH --qos=batch
#SBATCH --partition=students
#SBATCH --time=3-0:00:00
#SBATCH --gres=gpu:mem11g:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=90000
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate thesis
srun jupyter notebook --no-browser --port=9999 --ip 0.0.0.0
