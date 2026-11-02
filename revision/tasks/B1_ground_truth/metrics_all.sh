#!/usr/bin/env bash
# Metrics for every scored altered fit not yet done, on one node. Usage: bash metrics_all.sh <n_parallel>
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
PY=/home/icb/kemal.inecik/tools/apps/mamba/envs/chimera_env/bin/python
cd results/revision/B1/scores
ls | grep '^b1-' | sed 's/.parquet$//' > /dev/null
cd - > /dev/null
for r in $(ls results/revision/B1/scores | grep '^b1' | grep -v -- '-none-s' | sed 's/.parquet$//'); do
  # rerun when newer tool scores exist
  m=results/revision/B1/metrics/$r.parquet
  if [[ ! -f $m || -n $(find results/revision/B1/tool_scores -name "*__$r.parquet" -newer "$m" 2>/dev/null) || results/revision/B1/scores/$r.parquet -nt $m ]]; then echo "$r"; fi
done | xargs -P "${1:-8}" -I{} "$PY" -m revision.tasks.B1_ground_truth.score_metrics --run {} 2>&1 | grep -E "done|Traceback|Error" | tail -3
"$PY" -m revision.tasks.B1_ground_truth.aggregate 2>&1 | grep -vi warn | tail -3
