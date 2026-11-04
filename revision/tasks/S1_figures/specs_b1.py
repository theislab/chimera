"""Supplementary figure of the ground-truth benchmark (B1), composed from its panels at one scale (rule 7):
one figure over five floats, letters a-i.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/S1_figures/specs_b1.py \\
        supp_b1 supp_b1_cont1 supp_b1_cont2 supp_b1_cont3 supp_b1_cont4
"""

from __future__ import annotations

from revision.tasks.F1_figures.compose import Col, Figure, Panel


def X(src: str, label: str) -> Panel:  # noqa: N802
    """A panel saved at its exact size; the whole page is kept."""
    return Panel(src, label=label, ink=False)


FIGS = {
    "supp_b1": Figure(Col(X("b1_task_level", "a"), X("b1_identity", "b"), gap=6), note="B1 level alterations and identity tertiles"),
    "supp_b1_cont1": Figure(Col(X("b1_task_pattern", "c"), X("b1_share", "d"), gap=6), note="B1 pattern alterations and share"),
    "supp_b1_cont2": Figure(Col(X("b1_task_loss", "e"), X("b1_trackD", "f"), gap=6), note="B1 loss alterations and practical cost"),
    "supp_b1_cont3": Figure(Col(X("b1_low_expression", "g"), X("b1_type1", "h"), gap=6), note="B1 expression bands and type I error"),
    "supp_b1_cont4": Figure(Col(X("b1_spikein_raw_counts", "i")), note="B1 alterations in raw counts"),
}
