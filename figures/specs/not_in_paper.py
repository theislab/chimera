"""Composed figures that are not part of the paper.

They can be composed (``python -m figures.compose <key>``) and are not in ``figures.pdf``, ``order.txt`` or the index.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, P, Row, SRC

IN_PAPER = False

FIGS = {
    "extra_new_use_cases": Figure(
        Col(Row(P("extra_new_use_cases/near_identical", "a"), P("extra_new_use_cases/heldout_regions", "b")),
            Row(P("extra_new_use_cases/transfer", "c"), P("extra_new_use_cases/tumor_mixing", "d"), P("extra_new_use_cases/embedding", "e"))),
        title="Further use cases on added cohorts"),
    "extra_sparse_genes": Figure(
        Col(Row(P("extra_sparse_genes/masked", "a"), P("extra_sparse_genes/heldout", "b")), P("extra_sparse_genes/depth", "c")),
        title="Sparse genes: masked and held-out counts, depth"),
    "extra_uses": Figure(
        Col(Row(P("extra_uses/a", "a"), P("extra_uses/b", "b")),
            Row(P("extra_uses/c", "c"), P("extra_uses/d", "d")),
            Row(P("extra_uses/e", "e"), P("extra_uses/f", "f"))),
        title="Uses of the readouts"),
    "extra_embedding": Figure(
        Col(Row(P("extra_embedding/a", "a"), Col(P("extra_embedding/b", "b"))),
            Row(P("extra_embedding/c", "c"), P("extra_embedding/d", "d"))),
        title="The cell embedding"),
    # pending: joins the paper at integration. The residue audit over two floats: the first entry replaces the layout of
    # "supp_residue_audit" in specs/supp_protein.py (panels a to d), the second becomes its continuation float there.
    "pending_supp_residue_audit": Figure(
        Col(Row(SRC("assembled/page37", "a", region=(0, 0, 375, 384)),
                SRC("assembled/page37", "b", region=(375, 0, 724, 384))),
            Row(SRC("assembled/page37", "c", region=(724, 0, 958, 188)),
                SRC("assembled/page37", "d", region=(724, 200, 958, 384))),
            gap=6),
        label="fig:supp-main-gin-protein",
        title="Residue-level attribution audit",
        note="regions of the hand-assembled page 37; (a) and (b) stay at scale 0.67 until they are redrawn"),
    "supp_residue_audit_2": Figure(
        Col(Row(NB("ortholog_qc/i3c_cross_tissue_disagreement_portrait_grayscale", "e"),
                Col(SRC("assembled/page37", "f", region=(258, 397, 520, 511)),
                    SRC("assembled/page37", "g", region=(258, 524, 520, 650)))),
            SRC("assembled/page37", "h", region=(520, 397, 958, 650)),
            gap=6),
        label="fig:supp-main-gin-protein", part=2,
        title="Residue-level attribution audit, continued",
        note="regions of the hand-assembled page 37 and one notebook panel"),
}
