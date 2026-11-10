"""Supplementary figures on validation: independent references, the two readouts, published species differences, the benchmarks.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, LANDSCAPE, P, Row, X

FIGS = {
    "supp_bone_marrow_reference": Figure(
        Col(Row(P("supp_bone_marrow_reference/a", "a"), P("supp_bone_marrow_reference/b", "b"),
                P("supp_bone_marrow_reference/c", "c"), same_scale=True),
            Row(P("supp_bone_marrow_reference/d", "d"), P("supp_bone_marrow_reference/e", "e"),
                P("supp_bone_marrow_reference/f", "f"), same_scale=True)),
        label="fig:supp-bone-marrow",
        title="Bone marrow against sorted-lineage conservation of expression",
        shows=""),
    "supp_gate_gap": Figure(
        Col(Row(P("supp_gate_gap/scatter", "a"), P("supp_gate_gap/cohorts", "b")),
            Row(P("supp_gate_gap/neighbors", "c"), P("supp_gate_gap/covariates", "d")), gap=3),
        label="fig:supp-gate-gap",
        title="Where the two readouts agree and differ",
        shows=("Chimera can read each gene in two ways: through the default protein-feature gate, where a "
               "network sets the gene's value from its protein features, and through the companion per-gene "
               "gate, where each gene gets its own value from its counts. Panel a takes one cohort, kidney from "
               "human and mouse, and plots the two readings against each other for the 9,042 genes with count "
               "evidence in both fits. Most genes lie along the diagonal (Spearman 0.63); the colored ones "
               "differ by more than the noise between training seeds, 430 higher from their own counts and 803 "
               "lower. Panel b repeats this for all 26 cohorts: how reproducible the difference is when the "
               "training seeds are split in two halves (0.47 to 0.99, against a band centered on zero when one "
               "gate's seeds are split), how strongly the two readings agree (0.30 to 0.79), and what share of "
               "genes is called different (0 to 32 %). Panel c asks whether proteins that look alike to a "
               "protein language model (the 15 nearest neighbors in ESM-2 space) are read alike: they share "
               "little of the per-gene reading (median 0.09) and much more of the protein-feature reading "
               "(median 0.19, up to 0.70), so the default gate reads a protein family as a unit and the per-gene "
               "read-out keeps its members apart. Panel d shows which gene properties go with the difference: "
               "the strongest is uneven detection between the species (partial correlation -0.23 to -0.56 in 21 "
               "of 26 cohorts), and all seven properties together explain 12 to 49 % of it.")),
    "supp_gate_gap_classes": Figure(
        Col(Row(P("supp_gate_gap_classes/cross_tissue", "e"), P("supp_gate_gap_classes/sets", "f")),
            Row(P("supp_gate_gap_classes/references", "g"), P("supp_gate_gap_classes/sequence", "h")), gap=3),
        label="fig:supp-gate-gap-cont",
        title="The difference between the readouts by tissue, gene class and independent reference",
        shows=("This figure asks what the difference between the two readings of a gene is made of. Panel e "
               "compares cohorts in pairs: each reading is similar between cohorts of the same tissue (median "
               "Spearman 0.57 for the per-gene read-out, 0.54 for the protein-feature gate) and still related "
               "between tissues (0.20 and 0.25), while the difference is related within a tissue (0.24) and "
               "almost not between tissues (0.03). A gene accounts for 8 % of the variance of the difference "
               "across eight tissues, against 22 to 23 % for either reading: the difference belongs to a gene in "
               "a tissue. Panel f looks at whole gene classes. Mitochondrial proteins, the translation "
               "machinery, proliferation programs and protein homeostasis read higher from their counts than "
               "from their protein family in three or four of eight tissues; transcription factors, zinc-finger "
               "factors and receptors read lower. The two columns at the right show that both readings place "
               "these classes on the same side of the prior, the protein-feature gate further out. Panel g tests "
               "each reading against independent data while holding the other reading fixed: the per-gene read-out "
               "adds agreement on 26 of 32 gene-level references and on all nine held-out cortical regions "
               "(0.16 to 0.28), and the protein-feature gate still adds agreement on liver pattern and on fine "
               "kidney cell types. Panel h shows that the difference is unrelated to how constrained a gene's "
               "sequence is (dN/dS and LOEUF, all values between -0.07 and +0.10).")),
    "supp_gate_gap_named": Figure(
        P("supp_gate_gap_named/named", "i"),
        width=330,
        label="fig:supp-gate-gap-named",
        title="Named genes where the two readouts differ",
        shows=("This figure gives named examples of genes whose own counts and whose protein family disagree. "
               "Each row is a gene and each column a tissue. The color is the difference between the gene's per-gene "
               "reading and its protein-feature reading: blue means the gene's own counts place it lower "
               "than its protein features predict, red higher; a dot marks a tissue in which the difference "
               "passes the test against seed noise. At the right of each row are the gene's three nearest "
               "neighbors in the space of the protein language model, that is, the proteins it most resembles. "
               "The upper rows are near-identical, broadly expressed core proteins that read lower from their "
               "counts than their paralog families in three to five tissues: MARK2 (next to MARK1, MARK3, "
               "MARK4), PPP2CB (PPP2CA, PPP4C, PPP6C), ARF5, GSK3B, XRCC6 and TERF2IP. The lower rows are "
               "metabolic and nucleolar enzymes that read higher: HADH, ALDH6A1, QDPR, MSRB3 and fibrillarin "
               "(FBL). The FKBP family shows both at once: the protein model groups FKBP4, FKBP5 and FKBP7 "
               "together, while the counts separate them, FKBP4 and FKBP5 lower and FKBP7 higher.")),
    "supp_published_differences": Figure(
        Col(P("supp_published_differences/matched", "a"),
            Row(P("supp_published_differences/conserved", "b"), width=360), gap=4),
        shrink=0.85,
        label="fig:supp-e6",
        title="Published species differences on the count evidence",
        shows=""),
    "supp_benchmark_overview": Figure(
        X("supp_benchmark_overview/overview"),
        width=LANDSCAPE,
        label="fig:supp-b2-overview",
        title="Gene-level benchmark: overview",
        shows=""),
    "supp_benchmark_tasks": Figure(
        X("supp_benchmark_overview/overview", page=2),
        width=LANDSCAPE,
        label="fig:supp-b2-tasks",
        title="Gene-level benchmark: every task",
        shows=("Each row is a way of scoring a gene across species: four readouts of Chimera, seven direct "
               "calculations that need no model (for example the correlation of a gene's cell-type profile "
               "between species), three measures from the protein or DNA sequence, and five other cross-species "
               "tools. Each column is a family of tests. Track A changes the counts of chosen genes on purpose "
               "and asks whether a method finds them. Track B damages the data (fewer cells, wrong labels, "
               "different cell-type mix) and asks whether the scores stay the same. Track C compares the scores "
               "with independent data from other studies and species. Track D measures run time and memory. A "
               "large dark circle means the method ranked near the top in that family; a hatched circle means "
               "the test does not apply to the method. The bar on the right is the overall rank.")),
    "supp_ground_truth": Figure(
        Col(X("supp_ground_truth/task_level", "a"), X("supp_ground_truth/identity", "b")),
        label="fig:supp-b1",
        title="Ground-truth benchmark: level alterations and protein identity",
        shows=("To test whether a score can find a real change, the counts of chosen genes were altered on "
               "purpose in one species, and the model was trained again on the altered data. In this first part "
               "the alteration is a change of level: the gene's counts are thinned 2-, 4- or 8-fold in one "
               "species. The large table gives, for every score, how well it separates the altered genes from "
               "untouched genes of the same kind (0.5 is chance, 1 is perfect), in liver, testis and kidney and "
               "in four groups of genes from weakly to strongly expressed. The table beside it gives the share "
               "of altered genes that a score finds when 5 % false alarms are allowed. The three line plots show "
               "how detection grows with the size of the change. Panel (b) repeats the comparison for genes with "
               "low, medium and high protein identity.")),
    "supp_ground_truth_2": Figure(
        Col(X("supp_ground_truth/task_pattern", "c"), X("supp_ground_truth/share", "d")),
        label="fig:supp-b1", part=2,
        title="Ground-truth benchmark: pattern alterations and share of the move",
        shows=("Here the planted change is one of pattern: a share of a gene's counts is moved from the cell "
               "type where it is most used to the one where it is least used, in one species, so the total stays "
               "the same and only where the gene is used changes. Panel (c) shows, as before, how well each "
               "score separates the altered genes from untouched ones and how detection grows as more counts are "
               "moved. Panel (d) asks whether it matters how many genes are altered at once (2, 5 or 10 % of all "
               "genes): the lines are nearly flat, so the scores do not depend on the number of altered genes.")),
    "supp_ground_truth_3": Figure(
        Col(X("supp_ground_truth/task_loss", "e"), X("supp_ground_truth/practical_cost", "f")),
        label="fig:supp-b1", part=3,
        title="Ground-truth benchmark: loss alterations and practical cost",
        shows=("In panel (e) the planted change is a loss: a gene's counts are removed, in part or completely, "
               "from the one cell type where the gene is most used, in one species. The tables and lines read as "
               "in the earlier panels. Panel (f) shows what a run costs: the time one run takes on 16 processor "
               "cores and the memory it needs, for Chimera, for the direct calculations and for two other tools, "
               "on the liver, testis and kidney data, and how the time grows with the number of cells.")),
    "supp_ground_truth_4": Figure(
        Col(X("supp_ground_truth/low_expression", "g"), X("supp_ground_truth/type1_error", "h")),
        label="fig:supp-b1", part=4,
        title="Ground-truth benchmark: expression bands and false-positive rate",
        shows=("Panel (g) asks how detection depends on how strongly a gene is expressed. The top row gives the "
               "share of altered genes found, for genes in four groups from weakly to strongly expressed; the "
               "bottom row shows why Chimera's count evidence finds none of the weakly expressed ones: for those "
               "genes the model has too few counts and sets the count evidence to zero. Panel (h) tests for "
               "false alarms with three kinds of data in which no gene was changed in the way the test looks "
               "for: random gene sets, a reshuffling of counts that keeps the cell-type pattern, and a thinning "
               "applied to both species alike. A score that stays near the dashed line raises no false alarms; "
               "on the first test every score does.")),
    "supp_ground_truth_5": Figure(
        Col(X("supp_ground_truth/raw_counts", "i")),
        label="fig:supp-b1", part=5,
        title="Ground-truth benchmark: alterations in raw counts",
        shows=("Three example genes show what the planted changes look like in the raw data of the kidney. Each "
               "bar pair is one cell type: gray is the average count per cell before the change, blue after. For "
               "GOT2 all counts in mouse are thinned, so every blue bar is much lower (a level change). For "
               "RAB12 the counts are removed in one mouse cell type only, the fibroblasts (a loss). For ORC4 "
               "counts are moved in human from the cell type that uses the gene most to the one that uses it "
               "least, so one bar falls and another rises (a pattern change). The plots on the right show the "
               "same genes in the other species, where nothing was changed.")),
}
