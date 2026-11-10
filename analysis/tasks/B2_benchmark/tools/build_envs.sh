#!/usr/bin/env bash
# Build the environments of GeneSpectra, CACIMAR and SAMap (CPU): genespectra_env, cacimar_env, samap_env.
# The tools themselves are cloned at their pinned commits by analysis/external_tools/fetch.py,
# called below; where the clones live and which commits are pinned is defined there.
# Usage, from the repository root: bash analysis/tasks/B2_benchmark/tools/build_envs.sh [genespectra|cacimar|samap ...]
set -euo pipefail
command -v mamba >/dev/null 2>&1 || PATH="$(dirname "${MAMBA_EXE:-${CONDA_EXE:-/nonexistent/mamba}}"):$PATH"
command -v mamba >/dev/null 2>&1 || { echo "mamba not found: put its bin folder on PATH" >&2; exit 2; }

python3 analysis/external_tools/fetch.py GeneSpectra CACIMAR SAMap
TOOLS="$(python3 analysis/external_tools/location.py)"

freeze() { mamba run -n "$1" python -m pip freeze --all > "analysis/envs/$1.pip_freeze.txt"; }
exists() { mamba env list | grep -q "^$1 "; }

for t in "${@:-genespectra cacimar samap}"; do for tool in $t; do case $tool in
  genespectra)
    # Used from the clone on sys.path (run_genespectra.py): its pyproject requires exactly python 3.9.
    exists genespectra_env || mamba env create -y -f analysis/envs/genespectra_env.yml
    freeze genespectra_env
    ;;
  cacimar)
    exists cacimar_env || mamba env create -y -f analysis/envs/cacimar_env.yml
    # R CMD INSTALL from the clone, no network: every dependency comes from the yml.
    mamba run -n cacimar_env R CMD INSTALL --no-docs --no-test-load "$TOOLS/CACIMAR"
    freeze cacimar_env
    mamba run -n cacimar_env Rscript -e 'ip <- installed.packages()[, c("Package", "Version")]; write.table(ip, "analysis/envs/cacimar_env.r_packages.txt", quote = FALSE, row.names = FALSE, sep = "\t")'
    ;;
  samap)
    exists samap_env || mamba env create -y -f analysis/envs/samap_env.yml
    mamba run -n samap_env python -m pip install "$TOOLS/SAMap"
    freeze samap_env
    # pip records the tool by the local path of the clone; record its upstream commit instead.
    sed -i "s#^sc-samap @ file://.*#sc-samap @ git+$(python3 analysis/external_tools/location.py --url SAMap)@$(python3 analysis/external_tools/location.py --commit SAMap)#" \
      analysis/envs/samap_env.pip_freeze.txt
    ;;
  *) echo "unknown tool $tool" >&2; exit 2 ;;
esac; done; done
