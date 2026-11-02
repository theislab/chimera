#!/usr/bin/env bash
#SBATCH --job-name=d1_fwd
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=03:00:00
# Usage (from chimera/): sbatch --array=0-9 -c 4 --mem=12G --output=results/revision/D1/logs/%x_%A_%a.out revision/tasks/D1_sparse_genes/fwd.sbatch
source revision/slurm/_body.sh
set -e
C=(liv_hmouse liv_hmonkey liv_hzeb liv_4sp test_hmouse test_hmac test_3sp kidney islet bone_marrow)
mamba run --no-capture-output -n chimera_env python -m revision.tasks.D1_sparse_genes.fwd --cohort "${C[$SLURM_ARRAY_TASK_ID]}"
