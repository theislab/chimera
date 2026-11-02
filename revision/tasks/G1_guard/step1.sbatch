#!/usr/bin/env bash
#SBATCH --job-name=g1_step1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=cpusrv114
#SBATCH --output=results/revision/G1/logs/%x_%j.out
source revision/slurm/_body.sh
command -v mamba >/dev/null || { echo "no mamba on $(hostname)"; exit 3; }
mamba run --no-capture-output -n chimera_env python -m revision.tasks.G1_guard.step1_scan --runs revision/tasks/G1_guard/step1_runs.tsv --out results/revision/G1
