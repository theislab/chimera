"""E6 candidate figure for F1_figures/compose.py (composed outside the manuscript; CHIMERA_MANUSCRIPT points at a scratch folder).

    mkdir -p <scratch>/figures/revision
    CHIMERA_MANUSCRIPT=<scratch> PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/E6_known_differences/specs_e6.py --all
"""
from __future__ import annotations

from revision.tasks.E6_known_differences import rules as RU
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = RU.OUT / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    "fig_e6": Figure(Col(P("e6_matched", "a"), Row(P("e6_conserved", "b"), width=360), gap=4)),
}
