#!/usr/bin/env bash
# Build the two competitor envs (CPU) and clone the tools at pinned commits.
# Clones live in master_chimera/external_tools/, outside every repo.
# Run from chimera/. Env solves on the slow filesystem take tens of minutes.
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
ROOT=$(cd .. && pwd)
TOOLS=$ROOT/external_tools
mkdir -p "$TOOLS" && chmod 700 "$TOOLS"

[[ -d $TOOLS/SATURN ]] || git clone https://github.com/snap-stanford/SATURN "$TOOLS/SATURN"
git -C "$TOOLS/SATURN" checkout 6906abf
[[ -d $TOOLS/CAMEX ]] || git clone https://github.com/zhanglabtools/CAMEX "$TOOLS/CAMEX"
git -C "$TOOLS/CAMEX" checkout 17053c9

mamba env list | grep -q '^saturn_env ' || mamba env create -y -f analysis/envs/saturn_env.yml
mamba env list | grep -q '^camex_env ' || mamba env create -y -f analysis/envs/camex_env.yml
# CAMEX is used from the clone on sys.path (run_camex.py); no install into the env.

for e in saturn_env camex_env; do
  mamba run -n "$e" python -m pip freeze --all > "analysis/envs/$e.pip_freeze.txt"
done
