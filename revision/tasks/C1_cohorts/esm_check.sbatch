#!/usr/bin/env bash
#SBATCH --job-name=c1_esmchk
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
source revision/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -I revision/tasks/C1_cohorts/esm_embed.py check --threads 16
