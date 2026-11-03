"""E2 candidate figure for F1_figures/compose.py (exploratory; not placed in the manuscript).

    CHIMERA_MANUSCRIPT=$PWD/results/revision/E2/ms_stage PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/E2_protein/specs_e2.py fig_e2
"""
from __future__ import annotations

from revision.common import gene_ids
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = gene_ids.REPO_ROOT / "results" / "revision" / "E2" / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    "fig_e2": Figure(Col(Row(P("e2_properties", "a"), P("e2_domains", "b"), same_scale=True),
                         Row(P("e2_age_dup", "c"), width=205), gap=3)),
}
