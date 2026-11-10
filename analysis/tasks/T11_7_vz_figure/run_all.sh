#!/usr/bin/env bash
# T11.5-T11.7, one pass, re-runnable: forward passes and residuals for any D seed without output, then
# readout, integration scoring, glaucoma readouts, figure and summary. Run from chimera/; baselines for a new dataset go through
# analysis/tasks/T11_4_vz_integration/baselines.tsv first (score.py reports missing latents).
set -euo pipefail
export PYTHONPATH=$PWD
R="mamba run --no-capture-output -n chimera_env python"
$R analysis/tasks/T11_5_vz_genes/forward.py --missing --threads 12
$R analysis/tasks/T11_6_vz_glaucoma/residuals.py --threads 12
$R analysis/tasks/T11_5_vz_genes/readout.py
$R analysis/tasks/T11_4_vz_integration/score.py
$R analysis/tasks/T11_6_vz_glaucoma/glaucoma.py
$R analysis/tasks/T11_7_vz_figure/figure.py
