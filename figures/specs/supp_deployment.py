"""Supplementary figures on deployment relative to the protein prior: gene categories, pathways, sequence measures, relations to other gene measures, reliability.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, P, Row, SRC, TEXTWIDTH, X

FIGS = {
    "supp_categories_broad": Figure(
        Col(Row(NB("alpha_biology_simpler/a1_delta_alpha_by_category__panc_4sp__box", "a"),
                NB("alpha_biology_simpler/a3_effect_size__panc_4sp", "b")),
            Row(NB("alpha_biology_simpler/a1_delta_alpha_by_category__test_hmac__box", "c"),
                NB("alpha_biology_simpler/a3_effect_size__test_hmac", "d")),
            gap=4),
        width=430,
        label="fig:supp-dalpha-broad-categories",
        title="Δα by broad gene category, liver and testis",
        shows=("Genes are sorted into ten broad functional groups: cytoskeleton and extracellular matrix, "
               "signaling pathways, immunity and inflammation, transcription factors, cell cycle, protein "
               "homeostasis, chromatin and DNA processing, mitochondria, metabolism, and translation. Panel a "
               "takes the liver fit with four species and shows, for each group, the spread of Δα over its genes "
               "as a box; the dashed line marks zero, where a gene's shared use is exactly what its protein "
               "conservation predicts. Groups are sorted from the highest to the lowest median, and "
               "mitochondrial, metabolic and translation genes sit furthest below the line. Panel b gives the "
               "same groups as two numbers each: \"shift\" is how far the group's average lies from zero, measured "
               "in units of the spread of all genes, and \"amplitude\" asks whether the group's genes lie farther "
               "from zero than other genes do; a star marks a statistically supported entry. Panels c and d "
               "repeat this for testis of human and macaque, where nine of the ten groups lie above zero, led by "
               "cytoskeleton and mitochondria, and translation sits at zero. The overall difference in sign "
               "between the tissues reflects where the protein prior sits in each tissue (in testis it is capped "
               "at 0.85), so the order of the groups within a tissue is the reading to take.")),
    "supp_categories_liver": Figure(
        Row(X("supp_categories_liver/boxes", "a"), X("supp_categories_liver/effect_sizes", "b"), same_scale=True),
        label="fig:supp-dalpha-liver-detailed",
        title="Δα by gene category, four-species liver, all 38 categories",
        shows=("Genes are sorted into 38 groups by what they do. For the liver model with four species, (a) shows "
                "for each group where its genes sit relative to their protein-based starting value: boxes right of "
                "the dashed line are groups placed above the prior, boxes left of it below; each row gives the "
                "number of genes, the standardized shift d and a star where the shift is statistically supported. "
                "(b) gives two numbers per group on the same rows: the shift, and the amplitude, which says whether "
                "the genes of the group are moved more strongly than other genes in either direction. A triangle at "
                "the left edge marks rows with genes beyond the axis."),
        note=("panels: figures/redraw/categories.py; both pages have the same row positions, so each category is on "
              "one line")),
    "supp_categories_testis": Figure(
        Row(P("supp_categories_testis/box", "a"), P("supp_categories_testis/effect_sizes", "b")),
        label="fig:supp-dalpha-testis-detailed",
        title="Δα by gene category, testis Human+Macaque",
        shows=("This is the detailed version of the broad-category figure for testis of human and macaque, with "
               "41 gene categories instead of ten. In panel a each row is one category, with its number of genes "
               "in brackets; the box shows the spread of Δα across the category's genes, the blue line the "
               "median, and the dashed line zero, where a gene's shared use is what its protein conservation "
               "predicts. Rows run from the highest median at the top to the lowest at the bottom. The number at "
               "the right of each row is the category's average shift from zero, in units of the spread of all "
               "genes, with a star where the shift is statistically supported. Panel b shows the same shift as a "
               "colored table, next to \"amplitude\", which asks whether the genes of a category lie farther from "
               "zero than other genes. One category stands apart: the 11 genes encoded on the mitochondrial "
               "genome lie more than four units above the prior. Collagens, actin and microtubule genes follow "
               "at about one unit, and the mitochondrial ribosome is the only category clearly below zero. Most "
               "testis categories lie above zero because the protein prior is capped in testis, so the order of "
               "the rows is the reading to take.")),
    "supp_hallmark": Figure(
        Col(Row(P("supp_hallmark/scatter", "a"), P("supp_hallmark/counts", "b")), P("supp_hallmark/cross_tissue", "c")),
        label="fig:supp-dalpha-hallmark",
        title="Hallmark pathway enrichment of Δα",
        shows=("Hallmark is a standard collection of 50 gene sets, each standing for a well-known biological "
               "process. Gene set enrichment analysis takes a ranking of all genes and asks which sets gather at "
               "its top or bottom; the normalized enrichment score (NES) gives the strength and the direction. "
               "In panel a, for liver of human and mouse, each dot is one gene set: its horizontal position is "
               "its score when genes are ranked by the protein prior alone, its vertical position its score when "
               "genes are ranked by Δα. Colored dots are sets that are significant under the Δα ranking, red "
               "toward the top of the ranking and blue toward the bottom; epithelial-mesenchymal transition and "
               "Hedgehog signaling stand out, and xenobiotic metabolism changes sign between the two rankings. "
               "Panel b counts the significant sets in each cohort and splits them by the ranking that finds "
               "them: in liver the sets found only by Δα grow from 2 to 12 as the species become more distant, "
               "while in testis nearly all sets are found by the prior ranking. Panel c compares the tissues "
               "directly by ranking genes on the difference of Δα between liver and testis: liver metabolic sets "
               "(xenobiotic metabolism, bile acid metabolism, peroxisome) lie on the side where Δα is higher in "
               "testis, and allograft rejection and Hedgehog signaling on the side where it is higher in liver.")),
    "supp_sequence_axis": Figure(
        P("supp_sequence_axis/sequence_axis"),
        label="fig:supp-sequence-axis",
        title="Protein identity, dN/dS and Δα gene by gene",
        shows=""),
    "supp_known_measures": Figure(
        P("supp_known_measures/known_measures"),
        label="fig:supp-known-measures",
        title="Δα and count evidence among known gene measures",
        shows=""),
    "supp_worked_genes": Figure(
        P("supp_worked_genes/worked_genes"),
        label="fig:supp-worked-genes",
        title="Two worked genes in the counts",
        shows=""),
    "supp_functional_layer": Figure(
        Row(P("supp_functional_layer/a", "a"), P("supp_functional_layer/b_left", "b"),
            Col(P("supp_functional_layer/b_right", "b"), P("supp_functional_layer/c", "c"))),
        label="fig:supp-functional-layer",
        title="Pathway and network coherence and conservation",
        shows=""),
    "supp_covariate_audit": Figure(
        P("supp_covariate_audit/covariate_audit"),
        width=330,
        label="fig:supp-covariate-audit",
        title="Covariate audit of Δα in seven cohorts",
        shows=""),
    "supp_seed_reliability": Figure(
        P("supp_seed_reliability/seed_reliability"),
        width=340,
        label="fig:supp-seed-reliability",
        title="Agreement between training seeds",
        shows=""),
    "supp_celltype_closure": Figure(
        P("supp_celltype_closure/closure"),
        label="fig:supp-celltype-closure",
        title="Closure of the cell-type decomposition of the count evidence",
        shows=("Chimera's count evidence for a gene is an average over all cells. That average can be split "
               "exactly into the part contributed by each cell type. This figure checks that the split adds up: "
               "for every gene, the value rebuilt from the cell-type parts (horizontal axis) is plotted against "
               "the value the model stored when training ended (vertical axis), in each of the seven liver and "
               "testis datasets. The points lie on the diagonal, with correlations from 0.969 to 0.997.")),
    "supp_celltype_seeds": Figure(
        P("supp_celltype_seeds/seed_stability"),
        width=420,
        label="fig:supp-celltype-seeds",
        title="Seed stability of cell-type contributions to the count evidence",
        shows=("The model is trained three times with different random starts. For each cell type of the liver, "
               "the figure asks whether the three runs agree on how much that cell type contributes to each "
               "gene's count evidence. Every dot is one pair of runs; 1.0 means perfect agreement. The number in "
               "brackets is the cell type's share of the total; cell types with a share of at least 0.10 are "
               "drawn in blue. Hepatocytes and endothelial cells, which carry most of the signal, agree at 0.95 "
               "or higher; rare cell types agree less.")),
    "supp_relations_heatmap": Figure(
        P("supp_relations_heatmap/heatmap"),
        label="fig:supp-relations-heatmap",
        title="Relations of Δα and count evidence to sequence, expression and direct baselines",
        shows=""),
    "supp_relations_density": Figure(
        P("supp_relations_density/density"),
        width=413,
        label="fig:supp-relations-density",
        title="The same relations gene by gene",
        shows=""),
}
