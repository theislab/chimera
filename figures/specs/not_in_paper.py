"""Composed figures that are not part of the paper.

They can be composed (``python -m figures.compose <key>``) and are not in ``figures.pdf``, ``order.txt`` or the index.
"""

from __future__ import annotations

from figures.compose import Col, Figure, P, Row

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
}
