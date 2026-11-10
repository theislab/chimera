#!/usr/bin/env bash
# E1 cross-tissue atlas of deployment programmes. Run from chimera/ in chimera_env. About 5 min on 12 CPUs.
set -euo pipefail
R="mamba run --no-capture-output -n chimera_env"
export PYTHONPATH=$PWD
[ -f data/analysis/E1/MANIFEST.tsv ] || bash analysis/tasks/E1_atlas/fetch.sh
$R python -m analysis.tasks.E1_atlas.sets
$R python -m analysis.tasks.E1_atlas.atlas --jobs 12
$R python -m analysis.tasks.E1_atlas.summarize
$R python -m analysis.tasks.E1_atlas.distance
$R python -m analysis.tasks.E1_atlas.genes
$R python -m analysis.tasks.E1_atlas.figures
SC=${E1_SCRATCH_MS:-$(mktemp -d)}; mkdir -p "$SC/figures/revision"
CHIMERA_MANUSCRIPT=$SC $R python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/E1_atlas/specs_e1.py --all
mkdir -p results/analysis/E1/figures && cp "$SC"/figures/revision/fig_e1*_composed.pdf results/analysis/E1/figures/ && cp results/analysis/F1/fig_e1*_composed.png results/analysis/E1/figures/
