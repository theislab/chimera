#!/usr/bin/env bash
# Tops up B1 tasks from queue.tsv, in its order, so that B1 holds about CPU_CAP CPUs (run plus pending, counted with squeue -r) and the
# user's whole queue stays under LIMIT tasks. Touches no other job. Run from chimera/. Usage: bash feeder.sh [CPU_CAP] [LIMIT]
CAP=${1:-170}; LIMIT=${2:-95}
Q=revision/tasks/B1_ground_truth/queue.tsv
T=revision/tasks/B1_ground_truth
L=results/revision/B1/logs
n=$(($(wc -l < $Q) - 1)); i=${B1_START:-0}
while (( i < n )); do
  IFS=$'\t' read -r kind manifest index cpus mem wall done_file tier <<<"$(sed -n "$((i + 2))p" $Q)"
  if [[ -f "$done_file" ]]; then i=$((i + 1)); continue; fi
  mine=$(squeue -u "$USER" -r -h -o "%j %C" | awk '$1 ~ /^b1/{s+=$2}END{print s+0}')
  all=$(squeue -u "$USER" -r -h | wc -l)
  if (( mine + cpus > CAP || all >= LIMIT )); then sleep 60; continue; fi
  i=$((i + 1))
  case $kind in
    fit) sbatch -J "b1_$(basename "$manifest" .tsv | sed 's/grid_//')" --array="$index" -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%A_%a.out" $T/array.sbatch "$manifest" ;;
    saturn|camex) sbatch -J "b1tool_$kind" --array="$index" -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%A_%a.out" $T/tool_array.sbatch "$kind" "$manifest" ;;
    placebo) sbatch -J "b1plc" -p cpu_p --qos=cpu_normal -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%j.out" --wrap="source revision/slurm/_body.sh; mamba run --no-capture-output -n chimera_env python -m revision.tasks.B1_ground_truth.placebo $manifest" ;;
    trackd) sbatch -J b1trackd --array="$index" -o "$L/%x_%A_%a.out" $T/track_d/job.sbatch ;;
  esac
  sleep 2
done
echo "feeder done at $(date -Is)"
