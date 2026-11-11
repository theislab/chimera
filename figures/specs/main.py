"""The main figures.

One entry per figure: the layout, the label it belongs to, a short title and what it shows. Every main figure is one
float on one page (no continuation floats). See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import COLUMNWIDTH, Col, Figure, P, Row, X

FIGS = {
    # Drawn by figures/redraw/overview/overview.py at print size; the four panel files are placed as whole pages.
    # Panels (b) and (c) are one file, because arrows lead from the inputs into the model: its two letters stand
    # under the centers of the two cards (LETTER_X in the script).
    "main_overview": Figure(
        Col(Row(X("main_overview/problem", "a"), X("main_overview/inputs_model", {"b": 47.2, "c": 234.2})),
            Row(X("main_overview/outputs", "d"), X("main_overview/applications", "e"))),
        label="fig:overview",
        title="Overview: what the model takes, does and returns",
        shows=("A protein can stay almost the same between species while the cells use its gene at a different "
               "level, and a protein can change while the cells use it alike. Panel (a) shows these two cases with "
               "two toy genes: protein sequence on the left, expression across cell states on the right. Panel (b) lists "
               "what the model is given: single-cell counts from several species for genes that have exactly one "
               "counterpart in each species, how similar the proteins are (the protein prior), and a description "
               "of each gene (protein language model, protein similarity, expression statistics); an arrow leads "
               "each input to the part of the model that reads it. Panel (c) is the model. Cells of all species "
               "are placed on one map. Two decoders explain the counts together: one does not know the species, "
               "the other does. A gate per gene, drawn as a dial, decides how much of the species-aware decoder "
               "is let in. The gate starts where the protein similarity puts it (the tick of the dial) and moves "
               "(the needle) only when the counts say so; that count evidence is called S_g, and the distance "
               "from tick to needle is Δα. Panel (d) shows what comes out: the cell map, and for every gene the "
               "count evidence, the gate, and how far the gate ended from its protein-based start (Δα). Above "
               "the prior means the cells of the species share the gene's pattern; below means the pattern or "
               "the level differs (the two small profiles beside the histogram are the two toy genes of panel "
               "(a)). Genes with too few counts get a value from genes with similar proteins, shown "
               "in gray. Panel (e) lists what the paper does with these readouts; its strip shows for each of "
               "eight tissues that Δα does not follow protein similarity, while dN/dS, a second measure of "
               "sequence conservation, does.")),
    # Drawn by figures/redraw/main_figures.py at print size: four pages with the same axes box.
    "main_alpha_vs_prior": Figure(
        Row(X("main_alpha_vs_prior/scatter", "a"), X("main_alpha_vs_prior/shift", "b"),
            X("main_alpha_vs_prior/distribution", "c"), X("main_alpha_vs_prior/training", "d"), same_scale=True),
        label="fig:alpha-intro",
        title="The gate value α against the protein prior",
        shows=("Every dot, bar or line is a gene. The model starts each gene at a value set by how similar its "
               "protein is between the species (the protein prior). During training the gene's value α can move "
               "away from that start when the counts say so. A gene that ends more than 0.05 above its prior is "
               "drawn in vermilion, one that ends more than 0.05 below in blue, the others in gray. (a) α against "
               "the prior: the ordering is kept (Spearman correlation 0.77), most genes stay near the diagonal "
               "and many leave it, in both directions. (b) The move itself, Δα = α minus the prior: about half "
               "of the genes move up and half move down. The dotted lines stand at +0.05 and −0.05; 16% of the "
               "genes end above the first and 22% below the second. (c) The distribution of α beside that of the "
               "prior: the same overall shape, shifted by the data. (d) Δα of 120 of the genes over the epochs "
               "of training: every gene starts at its prior, at zero, and is moved from it.")),
    # One column wide. Drawn by analysis/tasks/T8_4_figures/fig3_integration_ext.py (composite_strip) from the stored table
    # of the benchmark on eleven cohorts.
    "main_integration": Figure(
        X("main_integration/composite_strip"),
        width=COLUMNWIDTH,
        label="fig:integration",
        title="Composite integration score of ten methods on eleven cohorts",
        shows=("Before genes are read, the cells of the species must be lined up. Each row is one of eleven "
               "cohorts (species sets of liver, testis, bone marrow, kidney, islet and eye); each mark is one of "
               "ten methods that align cells across species. The position of a mark is the method's composite "
               "score on that cohort: a weighted mean of how well cells of different species mix (40%) and how "
               "well cell types stay apart (60%), where every measurement behind it is scaled within the cohort "
               "from 0 (the lowest method) to 1 (the highest). Chimera is drawn in color, last and largest; "
               "the nine other methods are gray, and two of them, SATURN (dark gray) and CAMEX (open), have "
               "marks of their own and are named in the key. The shape of a mark says whether the method is given cell-type "
               "labels (circle: none; triangle: clusters found within each species; square: cell-type labels); "
               "Chimera uses none on the liver cohorts and labels on the others. Chimera lies in the upper "
               "group on most cohorts: the alignment is good enough to read genes from, which is what the paper "
               "needs from it. All three scores of every method, by name: the supplementary integration "
               "figures.")),
    # Panels a and b drawn by figures/redraw/main_figures.py, panel c by analysis/tasks/T8_4_figures/fig4_additions.py:
    # three pages of one height with the axes in one band.
    "main_delta_alpha_liver": Figure(
        Row(X("main_delta_alpha_liver/category_boxes", "a"), X("main_delta_alpha_liver/category_effect_sizes", "b"),
            X("main_delta_alpha_liver/prior_landscape", "c"), same_scale=True),
        label="fig:dalpha-biology",
        title="Δα by gene category and across the protein prior, four-species liver",
        shows=("(a) For twelve large gene categories, the spread of Δα over the genes of the category in the "
               "four-species liver model; the column n gives the number of genes. Categories at the top sit "
               "above their protein prior, those at the bottom below it. A box is vermilion when the category is "
               "shifted above the prior, blue when below, and gray when the shift is not significant. (b) The "
               "same as effect sizes: how far each category is shifted (left column) and whether its genes move "
               "more than the average gene (right column); an asterisk marks a significant value. (c) Every gene "
               "as a dot, protein prior on the x axis and Δα on the y axis. Named genes, each group labeled in "
               "its color: hepatocyte metabolic enzymes (urea cycle, gluconeogenesis, breakdown of "
               "phenylalanine) and mitochondrially encoded genes far below the prior, genes of supporting cell "
               "types above it, conserved control genes near it. Two named genes that lie at one place share a "
               "marker with one half for each. "
               "A dagger marks a gene whose position comes from its protein features and not from its own "
               "counts. Three hepatocyte enzymes that pass every check (the same sign in all five seeds, margin "
               "support, counts in every species) are named in bold: HAL, ARG2 and CYP7A1. Small triangles at "
               "the left edge are the few genes whose prior lies below the axis.")),
    # Drawn by analysis/tasks/H1_harmonise/figures.py (main_axis_summary, main_axis_clouds) from the stored table of
    # correlations and the stored gene tables of the eight tissues.
    "main_axis_tissues": Figure(
        Col(X("main_axis_tissues/summary", "a"), X("main_axis_tissues/clouds", "b")),
        label="fig:axis-tissues",
        title="Protein identity, dN/dS, count evidence and Δα in eight tissues",
        shows=("Is how a gene is used across cell states just another view of how conserved its protein is? Eight "
               "tissues, one model each. (a) Each mark is one tissue; the axis is a rank correlation across genes "
               "(−1 or +1: two measures order the genes the same way; 0: they are unrelated). Top row: protein "
               "identity against dN/dS, two measures of sequence conservation. They agree strongly in every "
               "tissue (−0.76 to −0.89; negative because a conserved protein has high identity and low dN/dS). "
               "The four rows below: the count evidence S_g and Δα, each against protein identity and against "
               "dN/dS. All marks sit close to zero (0.19 at most), so the model's readouts do not follow sequence "
               "conservation. The gray bar in each row runs from zero over the eight marks. (b) The same "
               "gene by gene: every dot is a gene, Δα on the y axis, protein identity (top row) or dN/dS (bottom "
               "row) on the x axis, one column per tissue. Blue dots are genes whose value rests on their own "
               "counts, gray dots genes whose value comes from their protein features. The orange line, the "
               "median of Δα along the x axis, is flat in every tissue. Small gray triangles on the frame mark "
               "genes beyond an axis.")),
    # Panels a and b (one file, two letters): analysis/tasks/Y3_findings/figures.py; c: T8_4_figures/fig4_additions.py
    # (panel_drivers_enzymes); d and e: F2_per_gene/panels.py (panel_readouts, panel_named_compact).
    "main_validation": Figure(
        Col(Row(X("main_validation/worked_genes", {"a": 87, "b": 249}), X("main_validation/celltype_drivers_enzymes", "c"),
                same_scale=True),
            Row(X("main_validation/readouts_vs_references", "d"), X("main_validation/named_findings", "e"), same_scale=True)),
        label="fig:validation",
        title="Two worked genes, the cell types behind the count evidence, and the readouts against independent data",
        shows=("(a) and (b) show two example genes in the raw counts, before any model. Each line is one species "
               "and each point the expression of the gene in one cell type or stage (the counts of all cells of "
               "that type summed, on a log scale). (a) ZPBP2, a sperm protein whose sequence differs between "
               "species: its expression peaks at the same stage, spermatocytes (shaded), in human, mouse and "
               "macaque. The protein changed, its pattern of use did not, and the model places the gene above "
               "its protein prior (Δα +0.17). SG, SC, rSD, eSD: spermatogonia, spermatocytes, round and "
               "elongated spermatids. (b) ALAS1, the first enzyme of heme synthesis, a conserved protein: it is "
               "highest in hepatocytes in all three mammals, but the mouse expresses it 4.6 log2 units (about "
               "24-fold) above human. The model places it below its prior (Δα −0.24). The corner of each plot "
               "gives Δα and the count evidence S_g of the gene; the protein identities of the two genes are "
               "measured on different sets of species and are given in the caption, not on the panels. (c) For "
               "seven such liver enzymes, the cell types the count evidence comes from: "
               "nearly all of it is contributed by hepatocytes (the black tick is the total). (d) Do the "
               "readouts agree with independent data, from samples the model never saw? Rows are four "
               "references: cell-state pattern and expression level, in liver and in testis. Marks right of zero "
               "mean agreement; the shaded side clears the bar that was fixed before the analysis. The count "
               "evidence S_g (squares) and Δα read gene by gene (hexagons) agree with independent pattern in "
               "both tissues. Δα from the default gate (circles) carries the outcome of the test fixed in "
               "advance, written beside each row under the words \"pre-specified test\": supported for liver "
               "pattern, null for testis pattern and liver level, reversed for testis level. The key of the "
               "three marks stands in the upper left corner of (d) and holds for (e) as well. (e) The two "
               "findings as gene sets, under both ways of reading Δα: genes that peak in spermatocytes sit above "
               "matched genes, the hepatocyte enzyme program below.")),
    # Drawn by analysis/tasks/H1_harmonise/figures.py (main_recurrence, compact form).
    "main_tissues": Figure(
        X("main_tissues/recurrence"),
        label="fig:one-reading",
        title="Gene programs and classes across eight tissues",
        shows=("Rows are gene programs and gene classes, columns the eight tissues. A red cell means the program "
               "sits above the protein prior in that tissue, a blue cell below; a star marks significance, a gray "
               "cell a program with too few genes. The rows stand in four blocks, each with a heading and its "
               "number of programs: 26 above the prior in tissue after tissue (receptors, cell contacts and "
               "signaling), 9 below it (translation, mitochondria and chromatin), 3 found in one tissue, and 3 "
               "with opposite signs in different tissues. At the right, two columns of small bars count the tissues "
               "in which the count evidence and direct profile correlation place the program above the prior "
               "(bar to the right) or below it (bar to the left): they agree with the map. The two columns of "
               "numbers show that protein identity and dN/dS do not predict it. The values behind the colors "
               "are printed in the supplementary figure of all recurrent sets.")),
    # Two panels of their own, drawn by analysis/tasks/E3_cells/figure.py (main_compartments, main_pattern_vs_level) from the
    # table that the two panels of the supplementary figure supp_cell_compartments read.
    "main_specialized_cells": Figure(
        Row(X("main_specialized_cells/compartments", "a"), X("main_specialized_cells/pattern_vs_level", "b"), same_scale=True),
        label="fig:above-prior",
        title="Home compartments of genes: Δα under both gates, profile correlation and level similarity",
        shows=("(a) Every gene is given a \"home\": the cell type in which human cells use it most, or no home if it "
               "is used in all cells. Home cell types are grouped into four compartments: the specialized cells "
               "of a tissue (neurons, kidney tubule cells, germ cells), stromal cells, blood-vessel cells and "
               "immune cells. For each compartment and each of the 26 cohorts the figure asks whether its genes "
               "score higher or lower than comparison genes with the same protein identity, expression, "
               "detection and cell-type specificity. Four scores stand side by side: Δα with the protein-feature "
               "gate, Δα with the per-gene gate, the correlation of cell-type expression profiles between "
               "species, and the similarity of expression level at the home cell type. One mark is one cohort "
               "(filled when the difference is significant), the bar is the median, and the numbers beside a row "
               "count the cohorts above zero. Genes of specialized cells sit above their protein prior in 23 of 24 and "
               "24 of 24 cohorts, while their expression level at the home cell type is the least similar "
               "between species (2 of 24 above zero); genes used in every cell sit below the prior in 25 and 26 "
               "of 26 cohorts while their level is similar (23 of 26 above zero). (b) The same at the "
               "resolution of single home cell types, one mark each: Δα with the per-gene gate against level "
               "similarity. The specialized cell types (filled) lie above zero on Δα and left of zero on level "
               "similarity (spermatids, excitatory neurons, trabecular meshwork); the stromal, vascular and "
               "immune types (open) lie around zero on level.")),
    # One panel, drawn by analysis/tasks/E4_translational/figures.py (p_targets).
    "main_drug_targets": Figure(
        X("supp_drug_targets/targets"),
        label="fig:drug-targets",
        title="Approved drug targets against the protein prior",
        shows=("Each of the four plots is one tissue: bone marrow, cortex, eye and kidney. Every gene with count "
               "evidence is a light gray dot, placed by the rank of its protein identity between the species "
               "(left, less conserved; right, more; the genes at the upper bound of the prior have one identity "
               "and are drawn as a narrow band, named in the first plot) and by its Δα with the per-gene gate (up, above what the protein predicts; down, "
               "below). The targets of approved drugs are drawn on top. Vermilion marks targets that sit above "
               "the prior on both gates of the model, blue marks those below on both, and gray marks targets "
               "without a call; a diamond means that the drug is used for a disease of that organ system. Named "
               "examples above the prior are immune receptors in bone marrow whose proteins evolve fast (IL6R, "
               "IL4R, IFNGR1, CD86), neuronal channels and receptors in cortex (SCN8A, GABRA1, OPRM1) and "
               "transporters in kidney (SCNN1G, SLC9A3, SLC22A12). Named examples below the prior are conserved "
               "proteins that the mouse deploys differently, such as PDE10A, FAAH and KCNQ1 in cortex and ROCK2 "
               "in the eye. The two lines under the title of each plot give the number of approved targets "
               "called above and below the prior, each with the number expected from genes of equal identity and "
               "expression.")),
    # Drawn by figures/redraw/main_figures.py at print size: four square maps in the upper row; under them the cluster
    # panel and, at the right, the key of the category map.
    "main_protein": Figure(
        Col(Row(X("main_protein/atlas_alpha", "a"), X("main_protein/atlas_delta_alpha", "b"),
                X("main_protein/atlas_correction", "c"), X("main_protein/atlas_categories", "d"), same_scale=True),
            Row(X("main_protein/atlas_clusters", "e"), X("main_protein/atlas_categories_key"), same_scale=True)),
        label="fig:gin-esm2",
        title="Protein-space atlas",
        shows=("A map of all genes arranged by protein similarity: each dot is a gene, and dots that lie close "
               "have similar protein-language features; the two directions of the map have no unit. (a) The map "
               "colored by the gate value α. (b) Colored by Δα, blue below the prior and red above it; the colors "
               "end at −0.15 and +0.15, as the color bar says. Neighboring genes tend to share the sign; three "
               "categories of (d) whose genes lie together are named. (c) Colored by the size of the correction "
               "the gate network makes to the prior, light where it is small. (d) Twelve gene categories on the "
               "map, at most 30 genes each, with the three names of (b) at the same places; the key is at the "
               "right of the lower row, in the order of the rows of the category figure. (e) The genes of each "
               "of four liver cohorts sorted into twelve protein clusters; each cell is the median Δα of one "
               "cluster in units of 0.01, sorted from lowest to highest within the cohort. The spread is widest "
               "in the four-species cohort.")),
}
