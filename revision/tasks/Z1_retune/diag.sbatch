#!/usr/bin/env bash
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Usage: sbatch [--job-name=x --output=...] revision/tasks/Z1_retune/diag.sbatch <runs.tsv> <out.tsv>
source revision/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python revision/tasks/Z1_retune/diagnostics.py --runs-tsv "$1" --out "$2"
