#!/bin/bash
#SBATCH --job-name=soc_srl_cpu
#SBATCH --output=soc_srl_cpu.log
#SBATCH --time=72:00:00
#SBATCH --mail-user=beilharz@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --qos=batch
#SBATCH --partition=compute
#SBATCH --ntasks=1

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/.bashrc
conda activate thesis
export PYTHONPATH=./
srun python src/data/social_chemistry.py
