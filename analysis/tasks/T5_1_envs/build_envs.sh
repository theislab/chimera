#!/usr/bin/env bash
# Build the environments of SATURN and CAMEX (CPU), saturn_env and camex_env.
# The tools themselves are cloned at their pinned commits by analysis/external_tools/fetch.py,
# called below; where the clones live and which commits are pinned is defined there.
# Run from the repository root. Solving an environment on a slow file system takes tens of minutes.
set -euo pipefail
command -v mamba >/dev/null 2>&1 || PATH="$(dirname "${MAMBA_EXE:-${CONDA_EXE:-/nonexistent/mamba}}"):$PATH"
command -v mamba >/dev/null 2>&1 || { echo "mamba not found: put its bin folder on PATH" >&2; exit 2; }

python3 analysis/external_tools/fetch.py SATURN CAMEX

mamba env list | grep -q '^saturn_env ' || mamba env create -y -f analysis/envs/saturn_env.yml
mamba env list | grep -q '^camex_env ' || mamba env create -y -f analysis/envs/camex_env.yml
# CAMEX is used from the clone on sys.path (run_camex.py); no install into the env.

for e in saturn_env camex_env; do
  mamba run -n "$e" python -m pip freeze --all > "analysis/envs/$e.pip_freeze.txt"
done
