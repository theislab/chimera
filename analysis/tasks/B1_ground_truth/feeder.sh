#!/usr/bin/env bash
# Tops up this folder's jobs from queue.tsv, in its order, to at most CPU_CAP CPUs and MEM_CAP GB (run plus pending, counted with
# squeue -r): the per-user limits are 512 CPUs and 2T. Touches no other job. Run from chimera/. Usage: bash feeder.sh [CPU_CAP] [MEM_CAP_GB]
CAP=${1:-170}; MEMCAP=${2:-680}
Q=analysis/tasks/B1_ground_truth/queue.tsv
T=analysis/tasks/B1_ground_truth
L=results/analysis/B1/logs
EX=${B1_EXCLUDE:+--exclude=$B1_EXCLUDE}  # nodes to leave out, if any: B1_EXCLUDE=<node list>
n=$(($(wc -l < $Q) - 1))
STATE=results/analysis/B1/feeder.state
i=$(cat "$STATE" 2>/dev/null || echo 0)
while (( i < n )); do
  IFS=$'\t' read -r kind manifest index cpus mem wall done_file tier <<<"$(sed -n "$((i + 2))p" $Q)"
  if [[ -f "$done_file" ]]; then i=$((i + 1)); continue; fi
  read -r mine mineg <<<"$(squeue -u "$USER" -r -h -o "%j %C %m" | awk '$1 ~ /^b1/{c+=$2; m=$3; v=m+0; if(m~/T/)v*=1024; if(m~/M/)v/=1024; g+=v}END{printf "%d %d", c+0, g+0}')"
  if (( mine + cpus > CAP || mineg + ${mem%G} > MEMCAP )); then sleep 60; continue; fi
  i=$((i + 1))
  case $kind in
    fit) sbatch $EX -J "b1_$(basename "$manifest" .tsv | sed 's/grid_//')" --array="$index" -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%A_%a.out" $T/array.sbatch "$manifest" ;;
    saturn|camex|genespectra|cacimar|samap) sbatch $EX -J "b1tool_$kind" --array="$index" -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%A_%a.out" $T/tool_array.sbatch "$kind" "$manifest" ;;
    placebo) sbatch $EX -J "b1plc" -p cpu_p --qos=cpu_normal -c "$cpus" --mem="$mem" --time="$wall" -o "$L/%x_%j.out" --wrap="source analysis/slurm/_body.sh; mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.placebo $manifest" ;;
    trackd) sbatch $EX -J b1trackd --array="$index" -o "$L/%x_%A_%a.out" $T/track_d/job.sbatch ;;
  esac
  echo "$i" > "$STATE"
  sleep 2
done
echo "feeder done at $(date -Is)"
