#!/usr/bin/env bash
# Set the feeder's CPU share by what other work of the user needs: 400 when no other task of the user waits for resources, priority or a QOS limit, 250 otherwise. User total guard 490 throughout.
F=results/revision/B2/feeder_cap
while :; do
  # held jobs and jobs waiting on a dependency do not count
  other_pd=$(squeue -u "$USER" -r -h -t PD -o '%j %r' | awk '$1 !~ /^b2/ && $2 ~ /^(Resources|Priority|QOSMax)/ {n++} END {print n+0}')
  if (( other_pd == 0 )); then want="400 1600 490"; else want="250 1000 490"; fi
  [[ "$(cat $F 2>/dev/null)" != "$want" ]] && { echo "$want" > $F; echo "$(date -Is) cap -> $want (other pending $other_pd)"; }
  sleep 300
done
