"""E5 candidate figure for F1_figures/compose.py (exploratory; not placed in the manuscript).

    CHIMERA_MANUSCRIPT=$PWD/results/revision/E5/ms_stage PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/E5_regulatory/specs_e5.py fig_e5
"""
from __future__ import annotations

from revision.common import gene_ids
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = gene_ids.REPO_ROOT / "results" / "revision" / "E5" / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    "fig_e5": Figure(Col(Row(P("e5_properties", "a"), P("e5_regulons", "b"), same_scale=True),
                         Row(P("e5_dose", "c"), width=230), gap=3)),
}
