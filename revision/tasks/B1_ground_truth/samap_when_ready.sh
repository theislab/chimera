#!/usr/bin/env bash
# Submits SAMap on the tier-3 rows once B2's BLAST graph exists; gives up at the deadline (date -d form). Run from chimera/.
DEADLINE=${1:-"2026-11-04 13:00"}
G=data/revision/B2/tools/samap/graph
until [[ -d $G && -n "$(ls -A $G 2>/dev/null)" ]]; do
  (( $(date +%s) > $(date -d "$DEADLINE" +%s) )) && { echo "no graph by $DEADLINE"; exit 0; }
  sleep 300
done
sleep 600  # let the graph writer finish
sbatch --exclude=cpusrv114 -J b1tool_samap --array="$(cat results/revision/B1/samap_tier3_index.txt)%6" -c 16 --mem=100G --time=12:00:00 \
  -o results/revision/B1/logs/%x_%A_%a.out revision/tasks/B1_ground_truth/tool_array.sbatch samap revision/tasks/B1_ground_truth/grid_tools_v2.tsv
