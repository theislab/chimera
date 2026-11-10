#!/usr/bin/env bash
#SBATCH --job-name=b2pool
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch -c 32 --mem=128G --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/pool.sbatch <b|c>
source analysis/slurm/_body.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B2_benchmark.pool "$1"
