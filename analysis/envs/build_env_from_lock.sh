#!/usr/bin/env bash
# Build a CPU environment for Chimera from the linux-64 lock file, without the notebook.
#
# Usage, from the repository root of a checkout:
#   bash analysis/envs/build_env_from_lock.sh [env_name]      # default: chimera_env
#
# Creates the mamba/conda environment with the Python of the lock, installs every pinned
# package of analysis/envs/chimera_env.linux-64.lock.txt (CPU torch index and the PyG wheel
# page are named inside the lock), installs this checkout in editable mode without
# dependencies, and runs `pip check`. Refuses to touch an environment that already exists:
# remove it first (`mamba env remove -n <env_name> -y`), since a partly built environment is
# otherwise easy to mistake for a complete one.
set -euo pipefail

ENV_NAME="${1:-chimera_env}"
PYTHON_VERSION="3.12.16"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="$ROOT/analysis/envs/chimera_env.linux-64.lock.txt"
CONDA="${CONDA_EXE:-$(command -v mamba || command -v conda)}"

[[ -f $LOCK ]] || { echo "missing $LOCK"; exit 2; }
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "environment $ENV_NAME exists; remove it first"; exit 2
fi

"$CONDA" create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip

# The lock lists the package itself; it is installed from this checkout below instead.
REQS="$(mktemp)"
trap 'rm -f "$REQS"' EXIT
grep -v -E '^chimera==' "$LOCK" > "$REQS"

run() { "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"; }
run python -m pip install --no-cache-dir -r "$REQS"
run python -m pip install --no-cache-dir --no-deps -e "$ROOT"
run python -m pip check
run python -c "import chimera, torch, scvi; print('chimera', chimera.__file__); print('torch', torch.__version__, 'scvi', scvi.__version__)"
echo "environment $ENV_NAME ready"
