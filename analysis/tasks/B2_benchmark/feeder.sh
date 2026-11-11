#!/usr/bin/env bash
# Submit the lines of a queue file in order, keeping our tasks (names b2*, running and pending) within a CPU and a memory share, about a third of the per-user caps (512 CPUs, 2T), so other work of the same user keeps moving.
# Usage (from chimera/): nohup setsid bash analysis/tasks/B2_benchmark/feeder.sh <queue file> [cpu share] [memory share in G] &
sbatch() { command sbatch ${CHIMERA_SBATCH_EXCLUDE:+--exclude=$CHIMERA_SBATCH_EXCLUDE} "$@"; }  # nodes to leave out, if any: CHIMERA_SBATCH_EXCLUDE=<node list>
Q="$1"; SHARE="${2:-170}"; MEMS="${3:-680}"; DONE="$Q.done"; touch "$DONE"
trap 'echo "$(date -Is) feeder exit, status $?"' EXIT
trap 'echo "$(date -Is) feeder got a signal"; exit 143' TERM INT HUP
while :; do
  # a cap file, if present, overrides the arguments without a restart: "<cpu share> <memory share in G> <user total CPU limit>"
  CAPF="$(dirname "$Q")/feeder_cap"; USERMAX=100000
  if [[ -f "$CAPF" ]]; then read -r SHARE MEMS USERMAX < "$CAPF"; USERMAX=${USERMAX:-100000}; fi
  cpu=$(squeue -u "$USER" -r -h -o "%j %C" | awk '$1 ~ /^b2/ {s+=$2} END {print s+0}')
  utot=$(squeue -u "$USER" -r -h -o "%C" | awk '{s+=$1} END {print s+0}')
  mem=$(squeue -u "$USER" -r -h -o "%j %m" | awk '$1 ~ /^b2/ {v=$2; g=v+0; if (v ~ /M$/) g/=1024; s+=g} END {print int(s)}')
  mapfile -t todo < <(grep -vxFf "$DONE" "$Q")
  (( ${#todo[@]} == 0 )) && { echo "$(date -Is) queue empty"; break; }
  for line in "${todo[@]}"; do
    c=$(grep -oE -- ' -c [0-9]+' <<<"$line" | head -1 | awk '{print $2}'); c=${c:-8}
    g=$(grep -oE -- '--mem=[0-9]+' <<<"$line" | head -1 | cut -d= -f2); g=${g:-16}
    (( cpu + c > SHARE || mem + g > MEMS || utot + c > USERMAX )) && break
    if eval "$line"; then echo "$line" >> "$DONE"; cpu=$(( cpu + c )); mem=$(( mem + g )); utot=$(( utot + c )); else echo "$(date -Is) FAILED SUBMIT: $line"; fi
    sleep 1
  done
  echo "$(date -Is) cpu=$cpu mem=${mem}G user=$utot cap=$SHARE/${MEMS}G/$USERMAX pending_lines=${#todo[@]}"
  sleep 180
done
