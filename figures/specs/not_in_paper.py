"""Composed figures that are not part of the paper.

They can be composed (``python -m figures.compose <key>``) and are not in ``figures.pdf``, ``order.txt`` or the index.
"""

from __future__ import annotations

from figures.compose import Col, Figure, P, Row, X

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
    # pending: joins the paper at integration. The two tall figures on the difference between the readouts, each over
    # two floats: a "pending_" entry replaces the layout of the entry of that name in specs/supp_validation.py, and the
    # "_2" entry becomes its continuation float there (same label, part 2).
    "pending_supp_gate_gap": Figure(
        Row(X("supp_gate_gap/scatter", "a"), X("supp_gate_gap/cohorts", "b")),
        label="fig:supp-gate-gap",
        title="Where the two readouts agree and differ",
        shows=("Chimera can read each gene in two ways: through the default protein-feature gate, where a network sets "
               "the gene's value from its protein features, and through the companion per-gene gate, where each gene "
               "gets its own value from its counts. Panel a takes one cohort, kidney from human and mouse, and plots the "
               "two readouts against each other for the 9,042 genes with count evidence in both fits. Most genes lie "
               "along the diagonal (Spearman 0.63); the dark ones differ by more than the noise between training seeds, "
               "430 above the diagonal (higher with the per-gene gate) and 803 below, and five genes of the named-gene "
               "figure are marked. Panel b repeats this for all 26 cohorts: how reproducible the difference is when the "
               "training seeds are split in two halves (0.47 to 0.99, against a gray band around zero when one gate's "
               "seeds are split; two cohorts have one seed and no split), how strongly the two readouts agree (0.30 to "
               "0.79), and what share of genes is called lower (bars to the left) and higher (bars to the right), 0 to "
               "32 % in all. Panel c asks whether proteins that look alike to a protein language model (the 15 nearest "
               "neighbors in ESM-2 space) are read alike: they share little of the per-gene readout (hexagons, median "
               "0.09) and much more of the protein-feature readout (circles, median 0.19, up to 0.70), so the default "
               "gate reads a protein family as a unit and the per-gene gate keeps its members apart. Panel d, with the "
               "cohorts in the rows of panel c, shows which gene properties go with the difference: the strongest is "
               "uneven detection between the species (partial correlation -0.23 to -0.56 in 21 of 26 cohorts), and all "
               "seven properties together explain 12 to 49 % of it.")),
    "supp_gate_gap_2": Figure(
        Row(X("supp_gate_gap/neighbors", "c"), X("supp_gate_gap/covariates", "d")),
        label="fig:supp-gate-gap", part=2,
        title="Where the two readouts agree and differ",
        shows="Panels c and d of the figure above."),
    "pending_supp_gate_gap_classes": Figure(
        Col(Row(X("supp_gate_gap_classes/cross_tissue", "a"), X("supp_gate_gap_classes/sets", "b")),
            X("supp_gate_gap_classes/references", "c")),
        label="fig:supp-gate-gap-cont",
        title="The difference between the readouts by tissue, gene class and independent reference",
        shows=("This figure asks what the difference between the two readouts of a gene is made of. Panel a compares "
               "cohorts in pairs (45 pairs within a tissue, 280 between tissues): each readout is similar between "
               "cohorts of the same tissue (median Spearman 0.54 for the protein-feature gate, 0.57 for the per-gene "
               "gate) and still related between tissues (0.25 and 0.20), while the difference is related within a tissue "
               "(0.24) and almost not between tissues (0.03). A gene accounts for 8 % of the variance of the difference "
               "across eight tissues, against 22 to 23 % for either readout (the numbers in parentheses in the key): the "
               "difference belongs to a gene in a tissue. Panel b looks at whole gene classes. Mitochondrial proteins, "
               "the translation machinery, proliferation programs and protein homeostasis read higher with the per-gene "
               "gate than with the protein-feature gate in three or four of eight tissues; transcription factors, "
               "zinc-finger factors and receptors read lower. The two columns at the right print the mean of each "
               "readout: both place these classes on the same side of the prior, the protein-feature gate further out. "
               "Panel c tests each readout against independent data while holding the other fixed, one row per cohort "
               "and reference under tissue headings: the per-gene readout adds agreement on 26 of 32 gene-level "
               "references and on all nine held-out cortical regions (0.16 to 0.28), and the protein-feature readout "
               "still adds agreement on liver pattern and on fine kidney cell types. Panel d shows that the difference "
               "is unrelated to how constrained a gene's sequence is (dN/dS and LOEUF, all values between -0.07 and "
               "+0.10).")),
    "supp_gate_gap_classes_2": Figure(
        X("supp_gate_gap_classes/sequence", "d"),
        label="fig:supp-gate-gap-cont", part=2,
        title="The difference between the readouts by tissue, gene class and independent reference",
        shows="Panel d of the figure above."),
}
