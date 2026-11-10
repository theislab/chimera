#!/usr/bin/env bash
#SBATCH --job-name=b1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N%K -c <cpus> --mem=<mem> --output=results/analysis/B1/logs/%x_%A_%a.out \
#   analysis/tasks/B1_ground_truth/array.sbatch analysis/tasks/B1_ground_truth/grid_<cohort>.tsv
source analysis/slurm/_body.sh
set -e
M="$1"; I=$SLURM_ARRAY_TASK_ID
T=analysis/tasks/B1_ground_truth
PY="mamba run --no-capture-output -n chimera_env python"
run=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['run'])")
scores=results/analysis/B1/scores/$run.parquet
if [[ -f "$scores" ]]; then echo "already scored: $run"; exit 0; fi
h5=data/analysis/B1/h5ad/$run.h5ad
# build unless the fit already finished and kept its object
if [[ ! -f "$h5" ]]; then $PY -m analysis.tasks.B1_ground_truth.step --manifest "$M" --index "$I"; fi
args=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['recipe'])")
seed=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['seed'])")
set +e
( train_run "$run" $args --adata "$h5" --seed "$seed" ) || echo "fit did not finish: $run; direct readouts are scored all the same"
set -e
$PY -m analysis.tasks.B1_ground_truth.score_run --manifest "$M" --index "$I" --h5ad "$h5"
# keep the state and the summaries; drop what the scores no longer need
rm -f "$h5"
rm -rf "training/runs/$run/model" "training/runs/$run/lightning" "training/runs/$run/alpha_snapshots" "training/runs/$run/cache"
