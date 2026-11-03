"""Candidate synthesis figure: what goes with deployment beyond the protein prior (exploratory; placement decided
in the narrative pass). Panels are the E3, E2 and E5 panels as drawn by their tasks, unchanged:

    (a) E3 home compartments: specialised-cell genes above the prior, broadly used genes below (both read-outs, with
        profile correlation and level similarity at the home type as direct baselines);
    (b) E2 domain architecture: signalling and cell-surface domains above, RNA, chromatin and proteostasis below;
    (c) E5 regulatory and gene architecture: distal enhancers, regulating TFs and Polycomb marks above, housekeeping
        and CpG-island promoters below (its legend carries the read-out symbols of (b) as well).

E5's dose-response panel (e5_dose) in place of (c) leaves a white gap of about a third of the row above it at the
same scale (domains are tall, the dose-response is short), so the forest panel of the same form as (b) is used.

    mkdir -p results/revision/P2/ms_stage/figures/revision
    CHIMERA_MANUSCRIPT=$PWD/results/revision/P2/ms_stage PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/F1_figures/specs_synthesis.py --all

Re-render the source panels first if their tasks changed (E3_cells/figure.py, E2_protein/figures.py,
E5_regulatory/figures.py).
"""
from __future__ import annotations

from revision.common import gene_ids
from revision.tasks.F1_figures.compose import Col, Figure, Panel, Row

R = gene_ids.REPO_ROOT / "results" / "revision"


def _synthesis(a: str, b: str, c: str, note: str) -> Figure:
    return Figure(
        Col(Panel(R / "E3/fig_ms/figures/revision/e3a_compartments.pdf", label=a),
            Row(Panel(R / "E2/panels/e2_domains.pdf", label=b), Panel(R / "E5/panels/e5_properties.pdf", label=c),
                same_scale=True, gap=10),
            gap=4),
        note=note)


FIGS = {
    # stand-alone candidate (a supplementary figure, or a figure of its own)
    "fig_synthesis_deployment": _synthesis("a", "b", "c", "candidate synthesis: E3(a) home compartments, E2(b) domains, "
                                                          "E5(a) regulatory architecture"),
    # the same figure lettered to continue the one-reading figure (fig:one-reading has panels a-i)
    "fig_h1_cont3": _synthesis("j", "k", "l", "candidate continuation of fig:one-reading, panels j-l (same panels as "
                                              "fig_synthesis_deployment)"),
}
