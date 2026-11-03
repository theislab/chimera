#!/usr/bin/env bash
# E4 pipeline, from chimera/: bash revision/tasks/E4_translational/run_all.sh   (about 25 min on 16 local cores; no SLURM)
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
R="mamba run -n chimera_env python -m revision.tasks.E4_translational"
bash revision/tasks/E4_translational/fetch.sh
$R.annot
$R.analysis tables q1 q23
$R.names
$R.summarize
$R.q1_followup
$R.figures
S=${E4_SCRATCH:-$(mktemp -d)}; mkdir -p "$S/figures/revision" results/revision/E4/figures
CHIMERA_MANUSCRIPT=$S mamba run --no-capture-output -n chimera_env python -m revision.tasks.F1_figures.compose --spec revision/tasks/E4_translational/specs_e4.py --all
cp "$S"/figures/revision/fig_e4*_composed.pdf results/revision/E4/figures/
cp results/revision/F1/fig_e4*_composed.png results/revision/E4/figures/
