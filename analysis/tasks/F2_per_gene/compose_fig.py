"""Compose the F2 per-gene figure with F1's compose.py (same grid, label and text checks as every revision figure).

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.F2_per_gene.compose_fig
Output: chimera-manuscript/figures/revision/fig_per_gene_composed.pdf; preview and report in results/analysis/F2/.
"""

from __future__ import annotations

import sys

from analysis.tasks.F1_figures import compose as C
from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row


def X(src: str, label: str) -> Panel:  # noqa: N802
    """Panel saved at its exact size: the page is kept, so axes bottoms in a row share one baseline."""
    return Panel(src, label=label, ink=False)


FIGS = {
    "fig_per_gene": Figure(Col(Row(X("fig_per_gene_a", "a"), X("fig_per_gene_b", "b"), same_scale=True, gap=4),
                               Row(X("fig_per_gene_c", "c"), X("fig_per_gene_d", "d"), X("fig_per_gene_e", "e"),
                                   same_scale=True, gap=4), gap=4)),
}


def main() -> int:
    C.OUT = C.REPO / "results" / "analysis" / "F2"
    fails = 0
    for name, fig in FIGS.items():
        fails += not C.compose(name, fig)["ok"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
