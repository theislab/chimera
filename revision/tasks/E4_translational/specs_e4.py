"""E4 candidate figures for F1_figures/compose.py. Composed outside the manuscript (CHIMERA_MANUSCRIPT points at a scratch folder);
the composed PDFs are copied to results/revision/E4/figures/.

    mkdir -p <scratch>/figures/revision
    CHIMERA_MANUSCRIPT=<scratch> PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/E4_translational/specs_e4.py --all
"""
from __future__ import annotations

from revision.tasks.E4_translational import rules as RU
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = RU.OUT / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    "fig_e4": Figure(Col(P("e4_heat", "a"), Row(P("e4_partial", "b"), P("e4_q1", "c")), gap=4)),
    "fig_e4_targets": Figure(P("e4_targets", "d")),
}
