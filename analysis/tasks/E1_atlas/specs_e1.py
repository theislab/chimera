"""E1 candidate figure for F1_figures/compose.py. Composed outside the manuscript (CHIMERA_MANUSCRIPT points at a scratch folder);
the composed PDFs are copied to results/analysis/E1/figures/.

    mkdir -p <scratch>/figures/revision
    CHIMERA_MANUSCRIPT=<scratch> PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/E1_atlas/specs_e1.py --all
"""
from __future__ import annotations

from analysis.tasks.E1_atlas import rules as RU
from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = RU.OUT / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    "fig_e1": Figure(Col(P("e1_atlas", "a"), Row(P("e1_distance", "b"), width=330), gap=3)),
    "fig_e1_cont": Figure(P("e1_tissue", "c"), width=420),
    "fig_e1_genes": Figure(P("e1_genes", "d"), width=190),
}
