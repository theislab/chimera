#!/usr/bin/env bash
#SBATCH --job-name=g1_arm
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
# G1 step 3, guard arm. sbatch --array=<rows>%K --mem=<mem> --output=results/analysis/G1/logs/%x_%A_%a.out analysis/tasks/G1_guard/step3.sbatch
# Training goes through analysis/tasks/T12_2b_debug/nan_probe.py (per-step maxima, first non-finite batch), which wraps the package's forward and changes nothing in it.
source analysis/slurm/_body.sh
command -v mamba >/dev/null || { echo "no mamba on $(hostname)"; exit 3; }
I=$SLURM_ARRAY_TASK_ID
PY="mamba run --no-capture-output -n chimera_env python"
set -e
$PY -m analysis.tasks.G1_guard.step3_prepare --index "$I" --build | tail -3
eval "$($PY -m analysis.tasks.G1_guard.step3_prepare --index "$I" | tail -1)"
set +e
if [[ -f "training/runs/$RUN/summary.json" ]]; then echo "finished: $RUN"; else
  if [[ -e "training/runs/$RUN" ]]; then mkdir -p training/runs/_failed; mv "training/runs/$RUN" "training/runs/_failed/${RUN}_${SLURM_JOB_ID}"; fi
  [[ -n "$H5" ]] && ADATA="--adata $H5" || ADATA=""
  # shellcheck disable=SC2086
  provenance "$RUN" $ARGS $ADATA
  # shellcheck disable=SC2086
  $PY "$TRAIN" --run-name "$RUN" --accelerator cpu $ARGS $ADATA
  echo "train exit $?"
fi
$PY -m analysis.tasks.G1_guard.step3_health --index "$I"
[[ -n "$H5" ]] && rm -f "$H5"
rm -rf "training/runs/$RUN/cache"
