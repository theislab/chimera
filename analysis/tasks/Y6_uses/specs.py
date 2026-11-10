"""Y6: figure spec for F1_figures.compose (``--spec``). Output folder redirected with CHIMERA_MANUSCRIPT, see figures.py."""

from pathlib import Path

from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row

D = Path(__file__).resolve().parents[3] / "results" / "analysis" / "Y6" / "figures" / "panels"


def P(k):
    return Panel(D / f"{k}.pdf", k)


FIGS = {"fig_y6_uses": Figure(Col(Row(P("a"), P("b")), Row(P("c"), P("d")), Row(P("e"), P("f")), gap=6))}
