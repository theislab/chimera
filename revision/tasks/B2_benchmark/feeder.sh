#!/usr/bin/env bash
# Submit the lines of a queue file in order, keeping our tasks (names b2*, running and pending) within a CPU and a memory share, about a third of the per-user caps (512 CPUs, 2T), so other work of the same user keeps moving.
# Usage (from chimera/): nohup setsid bash revision/tasks/B2_benchmark/feeder.sh <queue file> [cpu share] [memory share in G] &
Q="$1"; SHARE="${2:-170}"; MEMS="${3:-680}"; DONE="$Q.done"; touch "$DONE"
while :; do
  cpu=$(squeue -u "$USER" -r -h -o "%j %C" | awk '$1 ~ /^b2/ {s+=$2} END {print s+0}')
  mem=$(squeue -u "$USER" -r -h -o "%j %m" | awk '$1 ~ /^b2/ {v=$2; g=v+0; if (v ~ /M$/) g/=1024; s+=g} END {print int(s)}')
  mapfile -t todo < <(grep -vxFf "$DONE" "$Q")
  (( ${#todo[@]} == 0 )) && { echo "$(date -Is) queue empty"; break; }
  for line in "${todo[@]}"; do
    c=$(grep -oE -- ' -c [0-9]+' <<<"$line" | head -1 | awk '{print $2}'); c=${c:-8}
    g=$(grep -oE -- '--mem=[0-9]+' <<<"$line" | head -1 | cut -d= -f2); g=${g:-16}
    (( cpu + c > SHARE || mem + g > MEMS )) && break
    if eval "$line"; then echo "$line" >> "$DONE"; cpu=$(( cpu + c )); mem=$(( mem + g )); else echo "$(date -Is) FAILED SUBMIT: $line"; fi
    sleep 1
  done
  echo "$(date -Is) cpu=$cpu mem=${mem}G pending_lines=${#todo[@]}"
  sleep 180
done
