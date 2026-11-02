"""D1 figure for F1_figures/compose.py. Panels live in results/revision/D1/panels/; the composed PDF goes where compose.py writes (CHIMERA_MANUSCRIPT may point elsewhere).

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.F1_figures.compose --spec revision/tasks/D1_sparse_genes/specs_d1.py supp_d1_sparse
"""
from __future__ import annotations

from revision.tasks.D1_sparse_genes import rules as RU
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = RU.RES / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {"supp_d1_sparse": Figure(Row(P("d1_masked", "a"), Col(P("d1_depth", "b"), P("d1_heldout", "c"))))}
