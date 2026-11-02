#!/usr/bin/env bash
#SBATCH --job-name=d1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=08:00:00
# Usage (from chimera/): sbatch --array=<rows> -c <cpus> --mem=<mem> --output=results/revision/D1/logs/%x_%A_%a.out revision/tasks/D1_sparse_genes/array.sbatch
source revision/slurm/_body.sh
export PATH=/ictstr01/groups/ml01/workspace/kemal.inecik/_home/tools/apps/mamba/bin:$PATH
command -v mamba >/dev/null || { echo "no mamba on $(hostname)"; exit 3; }
set -e
I=$SLURM_ARRAY_TASK_ID
PY="mamba run --no-capture-output -n chimera_env python"
eval "$($PY -m revision.tasks.D1_sparse_genes.recipe --index "$I")"
scores=results/revision/D1/scores/$RUN.parquet
if [[ -f "$scores" ]]; then echo "already scored: $RUN"; exit 0; fi
h5=data/revision/D1/h5ad/$RUN.h5ad
if [[ ! -f "$h5" ]]; then $PY -m revision.tasks.D1_sparse_genes.objects --index "$I"; fi
set +e
# shellcheck disable=SC2086
train_run "$RUN" $ARGS --adata "$h5" --seed "$SEED"
set -e
$PY -m revision.tasks.D1_sparse_genes.score_fit --index "$I" --h5ad "$h5"
rm -f "$h5"
rm -rf "training/runs/$RUN/model" "training/runs/$RUN/lightning" "training/runs/$RUN/alpha_snapshots" "training/runs/$RUN/cache"
