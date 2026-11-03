"""E3 candidate figures for F1_figures/compose.py; panels in results/revision/E3/fig_ms/figures/revision (outside the manuscript).

    CHIMERA_MANUSCRIPT=$PWD/results/revision/E3/fig_ms PYTHONPATH=$PWD python -m revision.tasks.F1_figures.compose --spec revision/tasks/E3_cells/fig_spec.py --all
"""
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row
from revision.tasks.T8_4_figures import common as cm

P = lambda name, label=None, **kw: Panel(cm.FIG_OUT / f"{name}.pdf", label=label, **kw)  # noqa: E731
FIGS = {
    "fig_e3_compartments": Figure(Col(P("e3a_compartments", "a"), Row(P("e3b_pattern_vs_level", "b"), width=280), gap=3)),
    "fig_e3_lineages": Figure(Col(P("e3c_lineages", "a"), P("e3d_named", "b"), gap=3)),
    "fig_e3_primates": Figure(Row(P("e3e_branch", "a"), P("e3f_mtg", "b"), P("e3g_human_lineage", "c"))),
}
