"""X8 supplementary figures for F1_figures/compose.py. Panels in results/revision/X8/panels/; composed PDFs and previews are written to
results/revision/X8/composed/ by compose_x8.py (the manuscript folder is not touched; the narrative pass moves them).
"""
from __future__ import annotations

from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row
from revision.tasks.X8_gate_divergence import rules as RU

D = RU.OUT / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

FIGS = {
    # Where the two read-outs place a gene differently: size and noise, neighbourhoods, covariates.
    "supp_x8_gate_gap": Figure(Col(Row(P("x8_a_scatter", "a"), P("x8_b_cohorts", "b")), Row(P("x8_c_neighbours", "c"), P("x8_d_covariates", "d")), gap=3)),
    # What the gap is: tissues, gene classes, independent references, sequence, named genes.
    "supp_x8_gate_gap_cont": Figure(Col(Row(P("x8_e_cross", "e"), P("x8_f_sets", "f")), Row(P("x8_g_golds", "g"), P("x8_h_sequence", "h")), gap=3)),
    # Named examples that pass every check.
    "supp_x8_gate_gap_named": Figure(P("x8_i_named", "i"), width=330),
}
