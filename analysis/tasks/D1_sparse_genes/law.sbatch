#!/usr/bin/env bash
#SBATCH --job-name=d1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=08:00:00
# Arm L: as array.sbatch with the anchor scaled by the object's depth.
source analysis/slurm/_body.sh
command -v mamba >/dev/null || { echo "no mamba on $(hostname)"; exit 3; }
set -e
I=$SLURM_ARRAY_TASK_ID
PY="mamba run --no-capture-output -n chimera_env python"
eval "$($PY -m analysis.tasks.D1_sparse_genes.recipe --index "$I")"
scores=results/analysis/D1/scores/$RUN.parquet
if [[ -f "$scores" ]]; then echo "already scored: $RUN"; exit 0; fi
h5=data/analysis/D1/h5ad/$RUN.h5ad
if [[ ! -f "$h5" ]]; then $PY -m analysis.tasks.D1_sparse_genes.objects --index "$I"; fi
LAW=$($PY -m analysis.tasks.D1_sparse_genes.anchor_arg --index "$I" --h5ad "$h5" | tail -1)
echo "$RUN: $LAW"
set +e
# shellcheck disable=SC2086
train_run "$RUN" $ARGS $LAW --adata "$h5" --seed "$SEED"
set -e
$PY -m analysis.tasks.D1_sparse_genes.score_fit --index "$I" --h5ad "$h5"
rm -f "$h5"
rm -rf "training/runs/$RUN/model" "training/runs/$RUN/lightning" "training/runs/$RUN/alpha_snapshots" "training/runs/$RUN/cache"
