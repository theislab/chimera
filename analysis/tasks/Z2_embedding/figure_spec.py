"""python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/Z2_embedding/figure_spec.py fig_z2  with CHIMERA_MANUSCRIPT=results/analysis/Z2/compose so nothing lands in the manuscript."""
from pathlib import Path

from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = Path(__file__).resolve().parents[3] / "results/analysis/Z2/figures"
P = lambda n, l: Panel(str(D / f"z2_{n}.pdf"), l)  # noqa: E731
FIGS = {"fig_z2": Figure(Col(Row(P("a", "a"), Col(P("b", "b"))), Row(P("c", "c"), P("d", "d"))))}
