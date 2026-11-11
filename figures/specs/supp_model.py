"""Supplementary figures on the behavior of the gate and on integration.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, Row, TEXTWIDTH, X

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
            gap=4),
        label="fig:supp-alpha-robustness",
        title="Program-level Δα across seeds and species sets",
        shows=("Genes are grouped into about 40 named programs, such as ribosomal proteins, cytochrome P450 "
               "enzymes or cell-cycle genes. For each program the figure takes the average Δα of its genes, that "
               "is, how far the program's shared use across species sits above or below what its proteins' "
               "conservation predicts. Every panel compares these program averages between two fits of the "
               "model; a point on the dashed diagonal is a program with the same value in both. Panels a to c "
               "compare three fits of the liver Human+Monkey data that differ only in the random starting point: "
               "the programs line up along the diagonal (rank correlation 0.77 to 0.88). Panels d to f compare "
               "the four-species liver fit with each two-species liver fit: agreement is strong with Human+Mouse "
               "(0.71), moderate with Human+Monkey (0.46) and weak with Human+Zebrafish (0.27). Panels g to i do "
               "the same for testis, where the three fits agree closely (0.89 to 0.97). Every program "
               "is drawn. One program, the genes encoded on the mitochondrial genome, lies far from the others in "
               "every panel; to show it without squeezing the rest, each axis is compressed five-fold in its outer, "
               "shaded part, and the program is named there."),
        note=("panels: figures/redraw/robustness.py; all panels have one page size and one axes position")),
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
        Col(Row(X("supp_integration_ten_methods/purity", "a"), X("supp_integration_ten_methods/mixing", "b"), same_scale=True, gap=5.85),
            Row(X("supp_integration_ten_methods/conservation", "c"), X("supp_integration_ten_methods/composite", "d"), same_scale=True, gap=5.85)),
        width=351.35,
        label="fig:supp-integration-purity",
        title="Integration: mixing and conservation, ten-method panel",
        shows=("The four panels score the same ten integration methods (rows) on the seven liver and testis datasets "
               "(columns). Panel (a) is a raw measure, before any scaling: among the nearest neighbors of a cell, "
               "what fraction comes from the cell's own species (1.0 means species stay apart, lower means they "
               "mix). Panels (b) and (c) are the scaled scores for species mixing and for keeping cell types apart, "
               "and panel (d) is their weighted sum. Methods are grouped by whether they are given cell-type labels "
               "in training; Chimera is given none on liver and uses them on testis, where its cells have a dashed "
               "outline. In every panel a darker cell is a better value, and the best method of each dataset is "
               "in bold.")),
    "supp_integration_raw_metrics": Figure(
        Col(Row(X("supp_integration_raw_metrics/knn_species_purity", "a"), X("supp_integration_raw_metrics/knn_mixing_entropy", "b"),
                X("supp_integration_raw_metrics/silhouette_species", "c"), same_scale=True, gap=5.85),
            Row(X("supp_integration_raw_metrics/knn_celltype_purity", "d"), X("supp_integration_raw_metrics/silhouette_celltype", "e"),
                X("supp_integration_raw_metrics/label_transfer_accuracy", "f"), same_scale=True, gap=5.85)),
        label="fig:supp-integration-raw",
        title="Raw integration metrics, ten-method panel",
        shows=("Six measures of how well a method combines cells of different species, each before any scaling, "
               "for ten methods (rows) on the seven liver and testis datasets (columns). The top row measures "
               "species mixing: (a) how often a cell's nearest neighbors are of its own species, (b) how evenly "
               "species are spread among neighbors, and (c) how separate the species are as groups. The bottom row "
               "measures whether cell types are kept: (d) how often neighbors share the cell type, (e) how separate "
               "cell types are as groups, and (f) how accurately a cell-type label can be carried from one species "
               "to another. The value is printed in every cell. The arrow after the name of a measure says whether "
               "lower or higher is better, and each panel is colored over its own range with the better end dark.")),
    "supp_integration_eleven_cohorts": Figure(
        Col(Row(X("supp_integration_eleven_cohorts/mixing", "a"),
                X("supp_integration_eleven_cohorts/conservation", "b"), same_scale=True, gap=4, width=478.64),
            Row(X("supp_integration_eleven_cohorts/composite", "c"), width=276.78)),
        label="fig:supp-integration-ext",
        title="Integration benchmark on eleven cohorts",
        shows=("Ten methods that place cells of different species in one shared map are scored on eleven "
               "datasets: the seven liver and testis datasets of the main figure and four more tissues (bone "
               "marrow, kidney, pancreatic islet, eye). Panel (a) scores how well species mix, panel (b) how "
               "well cell types stay apart, panel (c) the combined score. Each column is scaled so that the best "
               "method of that dataset has 1. Methods are grouped by how much cell-type annotation they use in "
               "training, because methods that are given the labels have an advantage on (b)."),
        note=("panels: analysis/tasks/T8_4_figures/fig3_integration_ext.py; the widths are those of the panel pages, "
              "so every panel is placed at scale 1")),
}
