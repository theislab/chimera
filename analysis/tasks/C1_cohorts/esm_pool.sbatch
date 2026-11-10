#!/usr/bin/env bash
#SBATCH --job-name=c1_esm
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=23:50:00
#SBATCH --exclude=cpusrv09,cpusrv18
# One model, forked workers. Usage: sbatch [--array=0-3] esm_pool.sbatch <species,species>
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -I analysis/tasks/C1_cohorts/esm_embed.py pool --species "$1" --workers 8 --threads 4
