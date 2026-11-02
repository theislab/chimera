#!/usr/bin/env bash
# Submit the lines of a queue file in order, keeping our tasks (names b2*, running and pending) at or below CPU_SHARE CPUs, about a third of the per-user cap, so other work of the same user keeps moving.
# Usage (from chimera/): nohup setsid bash revision/tasks/B2_benchmark/feeder.sh <queue file> [cpu share] &
Q="$1"; SHARE="${2:-170}"; DONE="$Q.done"; touch "$DONE"
while :; do
  cpu=$(squeue -u "$USER" -r -h -o "%j %C" | awk '$1 ~ /^b2/ {s+=$2} END {print s+0}')
  mapfile -t todo < <(grep -vxFf "$DONE" "$Q")
  (( ${#todo[@]} == 0 )) && { echo "$(date -Is) queue empty"; break; }
  for line in "${todo[@]}"; do
    c=$(grep -oE -- ' -c [0-9]+' <<<"$line" | head -1 | awk '{print $2}'); c=${c:-8}
    (( cpu + c > SHARE )) && break
    if eval "$line"; then echo "$line" >> "$DONE"; cpu=$(( cpu + c )); else echo "$(date -Is) FAILED SUBMIT: $line"; fi
    sleep 1
  done
  echo "$(date -Is) cpu=$cpu pending_lines=${#todo[@]}"
  sleep 180
done
