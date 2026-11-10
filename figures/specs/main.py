"""The main figures.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, COLUMNWIDTH, Figure, LabelledPage, P, Row, SRC, X

# Page 31 of the first assembled pages with its panel letters removed; the letters of this figure are drawn at the
# positions the old ones had (center x, bottom of the letter box, in bp from the top left of that page).
ATLAS_LETTERS = {"d": (73.7, 171.3), "e": (267.9, 171.3), "f": (462.0, 171.3), "g": (653.6, 171.3),
                 "h": (201.0, 341.7), "i": (496.9, 341.7), "j": (676.0, 341.7),
                 "k": (90.1, 503.5), "l": (240.7, 503.5), "m": (399.6, 503.5), "n": (622.7, 503.5)}

FIGS = {
    # Drawn by figures/redraw/overview/overview.py at print size; the five panels are placed as whole pages.
    "main_overview": Figure(
        Col(Row(X("main_overview/problem", "a"), X("main_overview/inputs", "b"), X("main_overview/model", "c")),
            Row(X("main_overview/outputs", "d"), X("main_overview/applications", "e"))),
        label="fig:overview",
        title="Overview: what the model takes, does and returns",
        shows=("A protein can stay almost the same between species while the cells use its gene differently, and "
               "a protein can change while the cells use it alike. Panel (a) shows these two cases with two toy "
               "genes: protein sequence on top, expression across cell states below. Panel (b) lists what the "
               "model is given: single-cell counts from several species for genes that have exactly one "
               "counterpart in each species, how similar the proteins are (the protein prior), and a description "
               "of each gene (protein language model, protein similarity, expression statistics). Panel (c) is "
               "the model. Cells of all species are placed on one map. Two decoders explain the counts together: "
               "one does not know the species, the other does. A gate per gene decides how much of the species-aware "
               "decoder is let in. The gate starts where the protein similarity puts it and moves only "
               "when the counts say so; that count evidence is called S_g. Panel (d) shows what comes out: the "
               "cell map, and for every gene the count evidence, the gate, and how far the gate ended from its "
               "protein-based start (Δα). Above the prior means the cells of the species share the gene's "
               "pattern; below means the pattern or the level differs. Genes with too few counts get a value "
               "from genes with similar proteins, shown in gray. Panel (e) lists what the paper does with these "
               "readouts.")),
    "main_alpha_vs_prior": Figure(
        SRC("assembled/page04", ink=False),
        width=1.10 * COLUMNWIDTH,
        label="fig:alpha-intro",
        title="The gate value α against the protein prior",
        shows=("Every dot or bar is a gene. The model starts each gene at a value set by how similar its protein "
               "is between the species (the protein prior). During training the gene's value α can move away "
               "from that start when the counts say so. (a) α against the prior: most genes stay near the "
               "diagonal and many leave it. (b) The move itself, Δα = α minus the prior: about half of the genes "
               "move up and half move down. (c) The distribution of α beside that of the prior: the same overall "
               "shape, shifted by the data.")),
    "main_integration": Figure(
        Row(X("main_integration/mixing", "a"), X("main_integration/conservation", "b"),
            X("main_integration/composite", "c"), same_scale=True, gap=4),
        label="fig:integration",
        title="Integration benchmark on the seven liver and testis cohorts",
        shows=("Rows are methods that align cells across species; columns are the seven liver and testis "
               "cohorts. Each cell is a score from 0 (lowest method on that cohort) to 1 (highest). (a) How well "
               "cells of different species mix. (b) How well cell types stay apart. (c) A weighted mean of the "
               "two. Methods are grouped by whether they see cell-type labels during training.")),
    "main_delta_alpha_liver": Figure(
        Row(SRC("assembled/page16", {"a": 194.7, "b": 342.6}, region=(0, 0, 414, 402)),
            P("main_delta_alpha_liver/prior_landscape", "c")),
        label="fig:dalpha-biology",
        title="Δα by gene category and across the protein prior, four-species liver",
        shows=("(a) For twelve large gene categories, the spread of Δα over the genes of the category in the "
               "four-species liver model. Categories at the top sit above their protein prior, those at the "
               "bottom below it. (b) The same as effect sizes: how far each category is shifted (left column) "
               "and whether its genes move more than the average gene (right column). (c) Every gene as a dot, "
               "protein prior on the x axis and Δα on the y axis. Named genes: liver enzyme genes (urea cycle, "
               "gluconeogenesis) and mitochondrially encoded genes far below the prior, genes of supporting cell "
               "types above it. A dagger marks a gene whose position comes from its protein features and not "
               "from its own counts.")),
    "main_delta_alpha_liver_2": Figure(
        Col(Row(P("main_delta_alpha_liver/celltype_drivers", "d"),
                Col(X("main_delta_alpha_liver/margin_support", "e"), X("main_delta_alpha_liver/sequence_axis", "f"))),
            P("main_delta_alpha_liver/cross_tissue", "g")),
        label="fig:dalpha-biology", part=2,
        title="Cell-type sources of the count evidence, margin support, tissue comparison and sequence measures",
        shows=("(d) For each named gene, the cell types its count evidence comes from: hepatocytes carry most of "
               "it. (e) For each cohort, the share of genes in the upper and in the lower tail whose Δα is "
               "backed by count evidence of the same sign. (f) How α and Δα correlate with protein identity and "
               "dN/dS in 16 cohorts: identity and dN/dS go together, Δα follows neither. (g) Δα of the same genes "
               "in liver against testis for two species pairs: no agreement beyond chance, so the reading belongs "
               "to the tissue.")),
    "main_validation": Figure(
        Col(Row(P("main_validation/prespecified_tests", "a"), P("main_validation/gene_sets", "b"), same_scale=True),
            X("main_validation/head_to_head", "c"), X("main_validation/incremental", "d")),
        label="fig:validation",
        title="Pre-specified tests against independent data and comparison with direct baselines",
        shows=("(a) Four tests fixed before the analysis: does Δα agree with independent data on expression "
               "level and on cell-type pattern, in liver and in testis? Points right of zero mean agreement; the "
               "dotted line is the bar set in advance. Liver pattern passes. (b) Sets of genes with a known "
               "expectation, tested against matched background genes. (c) Δα against simpler direct scores and "
               "against the gene scores of SATURN and CAMEX: right of zero, Δα is closer to the independent "
               "data. (d) What each score adds when all the others are known: the count evidence S_g adds on "
               "pattern.")),
    "main_validation_2": Figure(
        Col(Row(X("main_validation/gates_tests", "e"), X("main_validation/gates_testis_level", "f"), same_scale=True,
                gap=4),
            Row(X("main_validation/gates_vs_count_evidence", "g"), X("main_validation/gates_named_findings", "h"),
                X("main_validation/gates_sequence_measures", "i"), same_scale=True, gap=4),
            gap=4),
        label="fig:validation", part=2,
        title="The tests and two named findings under the protein-feature gate and the per-gene gate",
        shows=("The model can read Δα in two ways: through the protein-feature gate (the default) or gene by "
               "gene from each gene's own counts (the per-gene gate). (e) The four tests of panel a for both "
               "gates: with the per-gene gate Δα agrees with independent pattern data in liver and in testis. "
               "Right: the difference between the gates. (f) Testis: Δα against independent expression level; "
               "the per-gene readout follows the count evidence. (g) Δα against the count evidence of the same "
               "fit: loosely related with the protein-feature gate, almost identical with the per-gene gate. (h) "
               "Two named findings, spermatocyte genes above the prior and hepatocyte catabolic genes below it, "
               "hold under both gates. (i) Under both gates Δα stays unrelated to dN/dS and to the protein "
               "prior.")),
    "main_tissues": Figure(
        Col(P("main_tissues/axis", "a"), P("main_tissues/genes", "b"), gap=3),
        label="fig:one-reading",
        title="Δα against protein identity and dN/dS, and named genes, in eight tissues",
        shows=("The same reading in eight tissues, one column or block per tissue. (a) Top row: how far genes "
               "move from the prior. Middle and bottom rows: Δα against protein identity and against dN/dS. The "
               "orange trend line is flat, so deployment does not follow sequence conservation in any tissue. "
               "(b) For each tissue, the genes furthest above the prior (orange), the tissue's own lineage genes "
               "(bold) and the genes furthest below (blue).")),
    "main_tissues_2": Figure(
        P("main_tissues/recurrence", "c"),
        width=470,
        label="fig:one-reading", part=2,
        title="Gene programs and classes across eight tissues",
        shows=("Rows are gene programs and gene classes, columns the eight tissues. A red cell means the program "
               "sits above the protein prior in that tissue, a blue cell below; a star marks significance. Most "
               "programs point the same way in every tissue: receptors, cell contacts and signaling above; "
               "translation, mitochondria and chromatin below. The columns at the right show that the count "
               "evidence and direct profile correlation agree with this, and that protein identity and dN/dS do "
               "not predict it.")),
    "main_tissues_3": Figure(
        Col(Row(P("main_tissues/gene_by_tissue_row1", {"d": 150, "e": 400}, region=(0, 0, 512, 119)), same_scale=True,
                width=425),
            Row(P("main_tissues/gene_by_tissue_row2", {"f": 150, "g": 395}, region=(0, 120, 512, 283)),
                same_scale=True, width=425),
            Row(P("main_tissues/gene_by_tissue_row3", {"h": 145, "i": 385}, region=(0, 283, 512, 429)),
                same_scale=True, width=425),
            gap=3),
        label="fig:one-reading", part=3,
        title="Gene and tissue components of the readouts",
        shows=("(d) How much of each readout belongs to the gene in every tissue, to the gene in one tissue, or "
               "to noise. (e) Each tissue's readout against independent data of every tissue: the matching "
               "tissue (outlined) agrees best. (f) The advantage of the matching tissue over the others. (g) "
               "Within one gene, differences between tissues in the readout follow differences between tissues "
               "in the independent data. (h) For genes whose reading flips between tissues, what is low in the "
               "low tissue: mostly a gap in expression level. (i) Six example genes.")),
    "main_tissues_4": Figure(
        Col(P("supp_cell_compartments/compartments", "j"),
            Row(P("supp_protein_architecture/domains", "k"), P("supp_regulatory_architecture/properties", "l"),
                same_scale=True, gap=10),
            gap=4),
        label="fig:one-reading", part=4,
        title="Home compartments, protein domains and regulatory features",
        shows=("(j) Genes grouped by the kind of cell that uses them most. Genes of specialized cells sit above "
               "their protein prior in nearly every cohort; broadly used genes sit below. Direct level "
               "similarity (right column) shows the opposite order. (k) Protein families: signaling and cell-surface "
               "domains above the prior; RNA, chromatin and protein-turnover domains below. (l) "
               "Regulatory features: genes with many enhancers and many regulating transcription factors above; "
               "housekeeping genes and CpG-island promoters below.")),
    "main_disease": Figure(
        Col(Row(P("main_disease/kidney_transfer", "a"), P("main_disease/stage_transfer", "b"), same_scale=True),
            Row(P("main_disease/healthy_tissue_test", "c"), P("main_disease/transfer_vs_null", "d"),
                P("main_disease/injury_genes", "e"), same_scale=True)),
        label="fig:disease",
        title="Transfer of disease programs between human and mouse",
        shows=("(a) Kidney: for each cell type, how many disease programs that are active in human acute kidney "
               "injury also change the same way in the mouse injury model. (b) Early (2 days) against late (4 to "
               "20 weeks) mouse injury: the early stage reproduces more of the human programs. (c) The test "
               "fixed in advance: does Δα from healthy tissue predict which genes respond alike in human and "
               "mouse disease? The estimates sit at zero, inside the grey null bands. (d) The number of "
               "transferring programs against its null in kidney, liver and islet. (e) Published lists of shared "
               "and human-specific injury genes.")),
    "main_protein": Figure(
        Col(SRC("assembled/page23_unlabelled", {"a": 182.7, "b": 463}, region=(0, 0, 571, 156)),
            P("main_protein/controls", "c")),
        label="fig:gin-esm2",
        title="Inputs of the gate network and control trainings",
        shows=("(a) Four ways of asking which inputs the gate network uses. Protein-language features (ESM-2) "
               "lead in three of them; expression features lead when whole groups of inputs are removed. (b) For "
               "each gene, the part of the correction that comes from protein features against the part that "
               "comes from expression features. The two are nearly unrelated and both are needed. (c) Control "
               "trainings: gene features removed, randomized or set to zero; counts shuffled between genes. Each "
               "dot compares one control run with normal training. The count evidence S_g (right) stays the same "
               "when features are changed and collapses when counts are shuffled. Δα (left) shows how much "
               "follows the features and how much follows the counts.")),
    "main_protein_2": Figure(
        LabelledPage("src:assembled/page31_unlabelled", at=ATLAS_LETTERS, bottom=505.0),
        label="fig:gin-esm2", part=2,
        title="Protein-space atlas and gene archetypes",
        shows=("A map of all genes arranged by protein similarity: each dot is a gene, and dots that lie close "
               "have similar protein-language features. (d to g) The map coloured by α, by Δα, by the size of "
               "the gate's correction and by gene category. (h) The median Δα of twelve protein neighbourhoods "
               "in four liver models. (i to k) The gate network sorts genes into three groups (archetypes) that "
               "lie below, at and above the prior. (l to n) The three groups on the protein map, and checks that "
               "expression level alone does not explain them.")),
}
