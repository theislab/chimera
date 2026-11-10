"""Supplementary figures on the behavior of the gate and on integration.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, Row, SRC, TEXTWIDTH, X

FIGS = {
    "supp_alpha_training": Figure(
        Col(Row(X("supp_alpha_training/drift", "a"), X("supp_alpha_training/tails", "b"), X("supp_alpha_training/trajectories", "c"), same_scale=True),
            Row(X("supp_alpha_training/distribution", "d"), X("supp_alpha_training/loss_terms", "e"), same_scale=True, width=321.64)),
        label="fig:supp-alpha-training-dynamics",
        title="Training dynamics of α",
        shows=("The model gives every gene a number α between 0 and 1: the share of the gene's counts that is "
                "explained by the part of the model shared by all species. α starts at the value predicted from "
                "protein similarity (the prior) and then moves. The figure follows that movement during training on "
                "liver cells of human and monkey: (a) how much α still changes between stored epochs, with the "
                "stability floor; (b) the share of genes that end clearly above or clearly below their prior; (c) "
                "the paths of 120 single genes, the mean of all genes and the mean prior; (d) the spread of α over "
                "all genes at five epochs; (e) the two terms that set α: the pull back to the prior and the reward "
                "for genes that the shared part explains well."),
        note=("panels: figures/redraw/alpha.py; the second row keeps the panel size of the first (2 x 157.32 + 7 bp) "
              "and is centered")),
    "supp_alpha_robustness": Figure(
        Col(Row(X("supp_alpha_robustness/seeds_42_44", "a"), X("supp_alpha_robustness/seeds_42_43", "b"),
                X("supp_alpha_robustness/seeds_43_44", "c"), same_scale=True),
            Row(X("supp_alpha_robustness/liver_4sp_hmouse", "d"), X("supp_alpha_robustness/liver_4sp_hmonkey", "e"),
                X("supp_alpha_robustness/liver_4sp_hzeb", "f"), same_scale=True),
            Row(X("supp_alpha_robustness/testis_3sp_hmac", "g"), X("supp_alpha_robustness/testis_3sp_hmouse", "h"),
                X("supp_alpha_robustness/testis_hmac_hmouse", "i"), same_scale=True),
            Row(X("supp_alpha_robustness/tissues_hmonkey_hmac", "j"), X("supp_alpha_robustness/tissues_hmouse", "k"),
                same_scale=True, width=321.67),
            gap=4),
        label="fig:supp-alpha-robustness",
        title="Program-level Δα across seeds, species sets and tissues",
        shows=("Genes are grouped into about 40 named programs, such as ribosomal proteins, cytochrome P450 "
               "enzymes or cell-cycle genes. For each program the figure takes the average Δα of its genes, that "
               "is, how far the program's shared use across species sits above or below what its proteins' "
               "conservation predicts. Every panel compares these program averages between two fits of the "
               "model; a point on the dashed diagonal is a program with the same value in both. Panels a to c "
               "compare three fits of the liver Human+Monkey data that differ only in the random starting point: "
               "the programs line up along the diagonal (rank correlation 0.77 to 0.88). Panels d to f compare "
               "the four-species liver fit with each two-species liver fit: agreement is strong with Human+Mouse "
               "(0.71), moderate with Human+Monkey (0.46) and weak with Human+Zebrafish (0.27). Panels g to i do "
               "the same for testis, where the three fits agree closely (0.89 to 0.97). Panels j and k compare "
               "testis with liver for the same pair of species: here the order of programs is unrelated (-0.06 "
               "and -0.20), so which programs sit above their protein prior depends on the tissue. Every program "
               "is drawn. One program, the genes encoded on the mitochondrial genome, lies far from the others in "
               "every panel; to show it without squeezing the rest, each axis is compressed five-fold in its outer, "
               "shaded part, and the program is named there."),
        note=("panels: figures/redraw/robustness.py; all panels have one page size and one axes position; the last row "
              "keeps the panel size of the others (2 x 157.33 + 7 bp) and is centered")),
    "supp_alpha_decorrelation": Figure(
        Row(X("supp_alpha_decorrelation/detection_rate", "a"), X("supp_alpha_decorrelation/mean_count", "b"), X("supp_alpha_decorrelation/count_variance", "c"), same_scale=True),
        label="fig:supp-alpha-decorrelation",
        title="Δα against expression magnitude statistics",
        shows=("Each dot is a gene of the liver model of human and monkey. Its height is Δα, how far the model "
                "moved the gene from its protein-based starting value. Its horizontal position is (a) the share of "
                "cells in which the gene is detected, (b) its mean count, (c) the variance of its count. The clouds "
                "are flat (rank correlations between −0.03 and −0.05): on this cohort Δα does not repeat how "
                "strongly a gene is expressed. Genes below the Δα axis are marked at its lower edge."),
        note=("panels: figures/redraw/alpha.py")),
    "supp_integration_seven_methods": Figure(
        Col(X("supp_integration_seven_methods/composite", "a"), X("supp_integration_seven_methods/mean_score", "b")),
        width=330,
        label="fig:supp-main-integration",
        title="Integration scores, seven-method panel",
        shows=("Before genes are read, the cells of the species must be lined up. The figure scores how well seven "
                "methods do that on the seven liver and testis cohorts. The score combines whether cells of "
                "different species mix (40%) and whether cell types stay apart (60%), each measurement scaled within "
                "a cohort. (a) One cell per cohort and method; the best score of a cohort is bold with a star, and "
                "the Chimera column is outlined. (b) The mean score of each method, with its median rank and the "
                "number of cohorts in which it scores best."),
        note=("panels: figures/redraw/integration_scores.py; heat map and bars have the same page width and the same "
              "seven columns")),
    "supp_integration_metrics": Figure(
        Col(Row(X("supp_integration_metrics/knn_species_purity", "a"), X("supp_integration_metrics/knn_mixing_entropy", "b"), same_scale=True),
            Row(X("supp_integration_metrics/silhouette_species", "c"), X("supp_integration_metrics/knn_celltype_purity", "d"), same_scale=True),
            Row(X("supp_integration_metrics/silhouette_celltype", "e"), X("supp_integration_metrics/label_transfer_accuracy", "f"), same_scale=True)),
        label="fig:supp-integration-metrics",
        title="Raw integration metrics, seven-method panel",
        shows=("The six measurements behind the composite score, each on its own, for the same seven methods and "
                "seven cohorts, with the value in every cell. Blue maps ask whether species mix: (a) how often the "
                "neighbors of a cell are of its own species (lower is better), (b) how mixed the neighbors are "
                "(higher is better), (c) how far the species sit apart as groups (lower is better). Orange maps ask "
                "whether cell types are kept: (d) how often neighbors share the cell type, (e) how well cell types "
                "separate, (f) how well the cell type of one species is predicted from the others (higher is "
                "better). The darker end of each color scale is the better one."),
        note=("panels: figures/redraw/integration_scores.py")),
    "supp_integration_umap_testis": Figure(
        X("supp_integration_umap_testis/grid"),
        label="fig:supp-integration-testis-umap",
        title="UMAP by method, three-species testis",
        shows=("Each map shows all cells of the three-species testis cohort as dots; cells with similar profiles "
                "lie close together. Each row is one method (Chimera, PCA, scANVI, scVI with donors as batches, "
                "sysVI). The same map is drawn three times: colored by donor, by species and by cell type, with one "
                "legend and the cell counts under each column. In a well-integrated map the colors of the first two "
                "columns are mixed and the colors of the third stay apart."),
        note=("panel: figures/redraw/integration_umap.py (stored UMAP coordinates, nothing is recomputed); donors and "
              "species are drawn in one fixed random order of the cells, cell types from the largest to the smallest")),
    "supp_integration_umap_liver": Figure(
        X("supp_integration_umap_liver/grid"),
        label="fig:supp-integration-liver-umap",
        title="UMAP by method, liver Human+Monkey",
        shows=("Each map shows all cells of the liver cohort of human and monkey as dots; cells with similar "
                "profiles lie close together. Each row is one method (Chimera, PCA, scANVI, scVI with donors as "
                "batches, sysVI). The same map is drawn three times: colored by donor, by species and by cell type, "
                "with one legend and the cell counts under each column. In a well-integrated map the colors of the "
                "first two columns are mixed and the colors of the third stay apart."),
        note=("panel: figures/redraw/integration_umap.py (stored UMAP coordinates, nothing is recomputed); donors and "
              "species are drawn in one fixed random order of the cells, cell types from the largest to the smallest")),
    "supp_integration_ten_methods": Figure(
        NB("baselines_master_ext/composite_mixing_bio_10models"),
        label="fig:supp-integration-purity",
        title="Integration: mixing and conservation, ten-method panel",
        shows=("The four blocks score the same ten integration methods on the seven liver and testis datasets. "
               "The first block is a raw measure, before any scaling: among the nearest neighbors of a cell, "
               "what fraction comes from the cell's own species (1.0 means species stay apart, lower means they "
               "mix). The second and third blocks are the scaled scores for species mixing and for keeping cell "
               "types apart, and the fourth is their weighted sum. Methods are grouped by whether they are given "
               "cell-type labels in training.")),
    "supp_integration_raw_metrics": Figure(
        NB("baselines_master_ext/raw_metrics_10models"),
        label="fig:supp-integration-raw",
        title="Raw integration metrics, ten-method panel",
        shows=("Six measures of how well a method combines cells of different species, each before any scaling, "
               "for ten methods on the seven liver and testis datasets. The top row measures species mixing: how "
               "often a cell's nearest neighbors are of its own species, how evenly species are spread among "
               "neighbors, and how separate the species are as groups. The bottom row measures whether cell "
               "types are kept: how often neighbors share the cell type, how separate cell types are as groups, "
               "and how accurately a cell-type label can be carried from one species to another.")),
    "supp_integration_eleven_cohorts": Figure(
        Col(Row(X("supp_integration_eleven_cohorts/mixing", "a"),
                X("supp_integration_eleven_cohorts/conservation", "b"), same_scale=True, gap=4),
            Row(X("supp_integration_eleven_cohorts/composite", "c"), width=281)),
        shrink=0.93,
        label="fig:supp-integration-ext",
        title="Integration benchmark on eleven cohorts",
        shows=("Ten methods that place cells of different species in one shared map are scored on eleven "
               "datasets: the seven liver and testis datasets of the main figure and four more tissues (bone "
               "marrow, kidney, pancreatic islet, eye). Panel (a) scores how well species mix, panel (b) how "
               "well cell types stay apart, panel (c) the combined score. Each column is scaled so that the best "
               "method of that dataset has 1. Methods are grouped by how much cell-type annotation they use in "
               "training, because methods that are given the labels have an advantage on (b).")),
}
