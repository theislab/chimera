#!/usr/bin/env bash
#SBATCH --job-name=b2
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N%K -c <cpus> --mem=<mem> --nice=10 --output=results/revision/B2/logs/%x_%A_%a.out \
#   revision/tasks/B2_benchmark/array.sbatch revision/tasks/B2_benchmark/grid_<cohort>.tsv
source revision/slurm/_body.sh
set -e
M="$1"; I=$SLURM_ARRAY_TASK_ID
PY="mamba run --no-capture-output -n chimera_env python"
eval "$($PY -m revision.tasks.B2_benchmark.recipe --manifest "$M" --index "$I")"
scores=results/revision/B2/scores/$RUN.parquet
if [[ -f "$scores" ]]; then echo "already scored: $RUN"; exit 0; fi
h5=data/revision/B2/h5ad/$RUN.h5ad
if [[ ! -f "$h5" ]]; then $PY -m revision.tasks.B2_benchmark.step --manifest "$M" --index "$I"; fi
if [[ "$FIT" == "1" ]]; then
  set +e
  # shellcheck disable=SC2086
  train_run "$RUN" $ARGS --adata "$h5" --seed "$SEED"
  set -e
fi
$PY -m revision.tasks.B2_benchmark.score_run --manifest "$M" --index "$I" --h5ad "$h5"
rm -f "$h5"
rm -rf "training/runs/$RUN/model" "training/runs/$RUN/lightning" "training/runs/$RUN/alpha_snapshots" "training/runs/$RUN/cache"
