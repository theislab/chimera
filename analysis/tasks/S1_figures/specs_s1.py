"""Merged main figures of the narrative pass (S1 phase 1b): eight main figures.

Step 4: validation (a-d) continued by the per-gene read-out (e-i). Step 7: the GIN figure (a-c, unchanged
fig5_composed) continued by the protein-space atlas (d-n, page 31 of the originals, relabelled).

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/S1_figures/specs_s1.py \\
        fig_validation fig_validation_cont fig_protein_cont
"""

from __future__ import annotations

from analysis.tasks.F1_figures.compose import FIG_DIR, LABEL_H, ORIGINALS, Col, Figure, Panel, Row

P = Panel


def X(src: str, label: str) -> Panel:  # noqa: N802
    """Panel saved at its exact size: the page is kept, axes share a baseline in a same-scale row."""
    return Panel(src, label=label, ink=False)


def O(page: int, region: tuple, label: dict, exempt_font: float | None = None) -> Panel:  # noqa: N802
    """A region of a page of the original figures; source labels stay outside the region, new ones are drawn."""
    return Panel(ORIGINALS, label=label, page=page, region=region, exempt_font=exempt_font)


# Page 31 of figures_planning_cropped.pdf with its eleven bold panel labels removed from the content stream
# (strip_atlas_labels.py: the eleven text blocks in the label font are deleted, nothing else changes), placed once
# as a whole page so no text is embedded twice; new letters drawn at the old label positions (pdftotext -bbox of
# the original page: centre x, bottom of the label box in bp from the page top).
ATLAS_EXEMPT = 8.0
ATLAS_AT = {"d": (73.7, 171.3), "e": (267.9, 171.3), "f": (462.0, 171.3), "g": (653.6, 171.3),
            "h": (201.0, 341.7), "i": (496.9, 341.7), "j": (676.0, 341.7),
            "k": (90.1, 503.5), "l": (240.7, 503.5), "m": (399.6, 503.5), "n": (622.7, 503.5)}


class AtPanel(Panel):
    """A whole page cropped to its ink and padded back to the full page (plus the bottom label band), with labels at
    given source positions instead of under the panel. The page is embedded once."""

    def __init__(self, src, at: dict, bottom: float, **kw):
        super().__init__(src, label=None, **kw)
        self.at, self.bottom = at, bottom

    def load(self) -> None:
        super().load()
        x0, y0, x1, y1, w, h = self.box
        self.pad = (x0, self.bottom - y1, w - x1, y0)

    def place(self, x: float, y: float, w: float) -> None:
        super().place(x, y, w)
        s, (l, b, r, t) = self.scale, self.pad
        x0, y0 = self.box[:2]
        self.labels_at = [(k, x + (l + cx - x0) * s, y + (t + yb - 1.5 - y0) * s - LABEL_H + 1.2)
                          for k, (cx, yb) in self.at.items()]


FIGS = {
    "fig_validation": Figure(Col(Row(P("fig7a_forest_h1_h4", "a"), P("fig7b_set_discrimination", "b"), same_scale=True),
                                 X("fig7c_head_to_head", "c"), X("fig7d_incremental_information", "d"))),
    "fig_validation_cont": Figure(Col(Row(X("fig_per_gene_a", "e"), X("fig_per_gene_b", "f"), same_scale=True, gap=4),
                                      Row(X("fig_per_gene_c", "g"), X("fig_per_gene_d", "h"), X("fig_per_gene_e", "i"),
                                          same_scale=True, gap=4), gap=4)),
    # Fig 8 first float (unchanged layout of F1's fig5) from page 23 without its old letters.
    "fig5": Figure(Col(Panel(FIG_DIR / "fig_gin_p23_unlabelled.pdf", label={"a": 182.7, "b": 463}, page=1,
                             region=(0, 0, 571, 156), exempt_font=5.0),
                       P("fig5_controls", "c"))),
    "fig_protein_cont": Figure(AtPanel(FIG_DIR / "fig_atlas_p31_unlabelled.pdf", ATLAS_AT, bottom=505.0, exempt_font=ATLAS_EXEMPT)),
}
