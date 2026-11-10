#!/usr/bin/env bash
# Release the held liver arrays once D1 runs at or under 24 CPUs elsewhere, so the total stays at 40.
while true; do
  n=$(squeue -u $USER -h -t R -o "%C %j" | grep -E " d1_(test|4sp)" | awk '{s+=$1} END{print s+0}')
  if [ "$n" -le 24 ]; then scontrol release 42470737 42477978; echo "released at $(date) with $n"; break; fi
  sleep 60
done
