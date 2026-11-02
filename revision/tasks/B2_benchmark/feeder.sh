#!/usr/bin/env bash
# Keep at most SHARE of our tasks in the queue, never pushing the user's total above LIMIT, counted per array element (squeue -r). Lines of QUEUE are sbatch commands, run from chimera/.
# Usage (from chimera/): nohup setsid bash revision/tasks/B2_benchmark/feeder.sh <queue file> [share] [limit] &
Q="$1"; SHARE="${2:-35}"; LIMIT="${3:-100}"; DONE="$Q.done"; touch "$DONE"
while :; do
  n=$(squeue -u "$USER" -r -h -o %j | grep -c '^b2')
  tot=$(squeue -u "$USER" -r -h | wc -l)
  room=$(( SHARE - n )); left=$(( LIMIT - tot )); (( left < room )) && room=$left
  pending=$(grep -vxFf "$DONE" "$Q" | wc -l)
  (( pending == 0 )) && { echo "$(date -Is) queue empty"; break; }
  if (( room > 0 )); then
    grep -vxFf "$DONE" "$Q" | head -n "$room" | while IFS= read -r line; do
      if eval "$line"; then echo "$line" >> "$DONE"; else echo "$(date -Is) FAILED SUBMIT: $line"; fi
      sleep 1
    done
  fi
  echo "$(date -Is) ours=$n total=$tot room=$room pending=$pending"
  sleep 180
done
