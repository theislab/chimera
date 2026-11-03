#!/usr/bin/env bash
# Clone GeneSpectra, CACIMAR and SAMap at pinned commits and build their envs (CPU).
# Clones live in master_chimera/external_tools/, outside every repo.
# Usage, from chimera/: bash revision/tasks/B2_benchmark/tools/build_envs.sh [genespectra|cacimar|samap ...]
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
ROOT=$(cd .. && pwd)
TOOLS=$ROOT/external_tools
mkdir -p "$TOOLS" && chmod 700 "$TOOLS"

pin() {  # pin <github path> <folder> <commit>
  [[ -d $TOOLS/$2 ]] || git clone "https://github.com/$1" "$TOOLS/$2"
  [[ $(git -C "$TOOLS/$2" rev-parse HEAD) == "$3" ]] || git -C "$TOOLS/$2" checkout -q "$3"
}
pin Papatheodorou-Group/GeneSpectra GeneSpectra 28234b4097116ea83808b5253d6472f152ef3aed
pin jiang-junyao/CACIMAR CACIMAR 492d8f7f82d735b07507f9aa34e0b3a39ec2cdb7
pin atarashansky/SAMap SAMap 8ffeaec17d738477d4a6302045098943a2726be7

freeze() { mamba run -n "$1" python -m pip freeze --all > "revision/envs/$1.pip_freeze.txt"; }
exists() { mamba env list | grep -q "^$1 "; }

for t in "${@:-genespectra cacimar samap}"; do for tool in $t; do case $tool in
  genespectra)
    # Used from the clone on sys.path (run_genespectra.py): its pyproject requires exactly python 3.9.
    exists genespectra_env || mamba env create -y -f revision/envs/genespectra_env.yml
    freeze genespectra_env
    ;;
  cacimar)
    exists cacimar_env || mamba env create -y -f revision/envs/cacimar_env.yml
    # R CMD INSTALL from the clone, no network: every dependency comes from the yml.
    mamba run -n cacimar_env R CMD INSTALL --no-docs --no-test-load "$TOOLS/CACIMAR"
    freeze cacimar_env
    mamba run -n cacimar_env Rscript -e 'ip <- installed.packages()[, c("Package", "Version")]; write.table(ip, "revision/envs/cacimar_env.r_packages.txt", quote = FALSE, row.names = FALSE, sep = "\t")'
    ;;
  samap)
    exists samap_env || mamba env create -y -f revision/envs/samap_env.yml
    mamba run -n samap_env python -m pip install "$TOOLS/SAMap"
    freeze samap_env
    ;;
  *) echo "unknown tool $tool" >&2; exit 2 ;;
esac; done; done
