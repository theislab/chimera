"""H1 figures for F1_figures/compose.py. Panels live in results/revision/H1/panels/; only the composed PDFs go to the manuscript.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.F1_figures.compose --spec revision/tasks/H1_harmonise/specs_h1.py --all
"""
from __future__ import annotations

from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row
from revision.tasks.H1_harmonise import rules as RU

D = RU.OUT / "panels"
P = lambda name, label=None, **kw: Panel(D / f"{name}.pdf", label=label, **kw)  # noqa: E731

# Y1's six panels, row by row from its page; its letters are masked, ours run on from (b). Rows stay whole: the labels of (f) reach over the axes of (e).
# Each row is cropped from a page on which only that row is drawn (same box), so no hidden text of other rows is embedded.
y = lambda k, region, **lab: Row(P(f"y1_masked_row{k}", lab, region=region), same_scale=True, width=425)  # noqa: E731


def tissue(stem: str) -> Figure:
    q = lambda k: P(f"{stem}_{k}", k)  # noqa: E731
    return Figure(Col(Row(q("a"), q("b"), q("c"), same_scale=True), Row(q("d"), q("e"), same_scale=True), Row(q("f"), q("g"), q("h"), same_scale=True), Row(q("i"), q("j"), q("k"), same_scale=True), gap=3))


FIGS = {
    # Three floats of one figure (eight tissues no longer fit two pages): (a) axis and (b) named genes; (c) programmes; (d)-(i) Y1, gene by tissue.
    "fig_h1": Figure(Col(P("fig_h1_axis", "a"), P("fig_h1_genes", "b"), gap=3)),
    "fig_h1_cont": Figure(P("fig_h1_recurrence", "c"), width=470),
    "fig_h1_cont2": Figure(Col(y(1, (0, 0, 512, 119), d=150, e=400), y(2, (0, 120, 512, 283), f=150, g=395), y(3, (0, 283, 512, 429), h=145, i=385), gap=3)),
    "supp_h1_bone_marrow": tissue("supp_h1_bone_marrow"),
    "supp_h1_kidney": tissue("supp_h1_kidney"),
    "supp_h1_islet": tissue("supp_h1_islet"),
    "supp_h1_eye": tissue("supp_h1_eye"),
    "supp_h1_cortex": tissue("supp_h1_cortex"),
    "supp_h1_lung": tissue("supp_h1_lung"),
    "supp_h1_muscle": tissue("supp_h1_muscle"),
    "supp_h1_tumour": tissue("supp_h1_tumour"),
    "supp_h1_recurrence_all": Figure(P("supp_h1_recurrence_all"), width=440),
    "supp_h1_lineage": Figure(P("supp_h1_lineage")),
}
