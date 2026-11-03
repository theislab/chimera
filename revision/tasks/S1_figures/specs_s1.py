"""Merged main figures of the narrative pass (S1 phase 1b): eight main figures.

Step 4: validation (a-d) continued by the per-gene read-out (e-i). Step 7: the GIN figure (a-c, unchanged
fig5_composed) continued by the protein-space atlas (d-n, page 31 of the originals, relabelled).

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose --spec revision/tasks/S1_figures/specs_s1.py \\
        fig_validation fig_validation_cont fig_protein_cont
"""

from __future__ import annotations

from revision.tasks.F1_figures.compose import FIG_DIR, ORIGINALS, Col, Figure, Panel, Row

P = Panel


def X(src: str, label: str) -> Panel:  # noqa: N802
    """Panel saved at its exact size: the page is kept, axes share a baseline in a same-scale row."""
    return Panel(src, label=label, ink=False)


def O(page: int, region: tuple, label: dict, exempt_font: float | None = None) -> Panel:  # noqa: N802
    """A region of a page of the original figures; source labels stay outside the region, new ones are drawn."""
    return Panel(ORIGINALS, label=label, page=page, region=region, exempt_font=exempt_font)


# Page 31 of figures_planning_cropped.pdf with its eleven panel labels covered in white
# (atlas_unlabelled.tex; label boxes read with pdftotext -bbox), so a row can include a scatter that reaches
# into the old label band; new letters are drawn at the old label centres.
ATLAS_EXEMPT = 8.0


def A(region: tuple, label: dict, pad: tuple) -> Panel:  # noqa: N802
    """One row of the page; ``pad`` restores the page's full width (0 to 770.16 bp) so all rows share one scale."""
    return Panel(FIG_DIR / "fig_atlas_p31_unlabelled.pdf", label=label, page=1, region=region, pad=pad,
                 exempt_font=ATLAS_EXEMPT)
FIGS = {
    "fig_validation": Figure(Col(Row(P("fig7a_forest_h1_h4", "a"), P("fig7b_set_discrimination", "b"), same_scale=True),
                                 X("fig7c_head_to_head", "c"), X("fig7d_incremental_information", "d"))),
    "fig_validation_cont": Figure(Col(Row(X("fig_per_gene_a", "e"), X("fig_per_gene_b", "f"), same_scale=True, gap=4),
                                      Row(X("fig_per_gene_c", "g"), X("fig_per_gene_d", "h"), X("fig_per_gene_e", "i"),
                                          same_scale=True, gap=4), gap=4)),
    "fig_protein_cont": Figure(Col(
        A((0, 0, 770, 160.8), {"d": 73.7, "e": 267.9, "f": 462.0, "g": 653.6}, (4.4, 0, 0, 0)),
        A((0, 171.5, 770, 328.6), {"h": 201.0, "i": 496.9, "j": 676.0}, (0, 0, 42.08, 0)),
        A((0, 328.8, 770, 493.0), {"k": 90.1, "l": 240.7, "m": 399.6, "n": 622.7}, (11.12, 0, 62.0, 0)))),
}
