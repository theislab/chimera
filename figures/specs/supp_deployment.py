"""Supplementary figures on deployment relative to the protein prior: gene categories, pathways, sequence measures, relations to other gene measures, reliability.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, P, Row, SRC, TEXTWIDTH, X

FIGS = {
    "supp_categories_broad": Figure(
        Col(Row(X("supp_categories_broad/liver_boxes", "a"), X("supp_categories_broad/liver_effect_sizes", "b"), same_scale=True),
            Row(X("supp_categories_broad/testis_boxes", "c"), X("supp_categories_broad/testis_effect_sizes", "d"), same_scale=True),
            gap=4),
        label="fig:supp-dalpha-broad-categories",
        title="Δα by broad gene category, liver and testis",
        shows=("Genes are sorted into ten broad functional groups: cytoskeleton and extracellular matrix, "
               "signaling pathways, immunity and inflammation, transcription factors, cell cycle, protein "
               "homeostasis, chromatin and DNA processing, mitochondria, metabolism, and translation. Panel a "
               "takes the liver fit with four species and shows, for each group, the spread of Δα over its genes "
               "as a box, with the number of genes in parentheses after the name; the dashed line marks zero, "
               "where a gene's shared use is exactly what its protein conservation predicts. Groups are sorted "
               "from the highest to the lowest median, and mitochondrial, metabolic and translation genes sit "
               "furthest below the line. A small gray triangle at the left edge of a row says that the row has "
               "genes below the end of the axis at -0.4 (four rows, twelve genes in all). Panel b gives the "
               "same groups as two numbers each: \"shift\" is how far the group's average lies from zero, measured "
               "in units of the spread of all genes, and \"amplitude\" asks whether the group's genes lie farther "
               "from zero than other genes do; a star marks a statistically supported entry. Panels c and d "
               "repeat this for testis of human and macaque, where nine of the ten groups lie above zero, led by "
               "cytoskeleton and mitochondria, and translation sits at zero. The values of the testis model span "
               "a smaller range, so the axis of panel c is narrower (-0.1 to 0.2 instead of -0.4 to 0.4) and "
               "holds every gene. The overall difference in sign between the tissues reflects where the protein "
               "prior sits in each tissue (in testis it is capped at 0.85), so the order of the groups within a "
               "tissue is the reading to take.")),
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
        X("supp_sequence_axis/sequence_axis"),
        label="fig:supp-sequence-axis",
        title="Protein identity, dN/dS and Δα gene by gene",
        shows=("Each dot is one gene, and each column is one cohort (liver of human and mouse, liver of four species, "
               "testis of human and mouse). The top row compares two classic measures taken from sequence alone: how "
               "similar the protein is between species (protein identity) and how strongly the gene's coding sequence "
               "has been constrained (dN/dS). They follow each other closely. The second row shows the model's raw "
               "estimate α against protein identity; the dashed line marks where the two would be equal, and most genes "
               "stay near it because the model starts from the protein. The third and fourth rows show Chimera's readout "
               "Δα, which is how far the counts move a gene above or below what its protein predicts, against the same "
               "two sequence measures. Those clouds are flat: the orange line of medians stays near zero. Blue genes "
               "have enough expressing cells for the counts to speak; for gray genes the readout is filled in from "
               "protein and expression features. Small triangles on the frame of a plot are genes that lie beyond its "
               "axis.")),
    "supp_known_measures": Figure(
        X("supp_known_measures/known_measures"),
        label="fig:supp-known-measures",
        title="Δα and count evidence among known gene measures",
        shows=("The figure places Chimera's two gene scores among ten measures that biologists already use. In the top "
               "row, each grid is one cohort; every square shows how strongly two measures rise and fall together across "
               "genes (red together, blue opposite, white unrelated, gray not available). The measures are protein "
               "identity, dN/dS, tolerance to damaging variants in humans (LOEUF), distance between the two species' "
               "proteins in a protein language model, how stable a gene's expression level is across mammals (EVEE), how "
               "similar its expression profile across cell types is between species, how similar its cell-type "
               "specificity is, expression level, cell-type specificity and coding length. The columns follow the order "
               "of the rows; the last two rows and the two wide columns at the right are Chimera's Δα and its count "
               "evidence, with the values printed in the columns. The second row places seven familiar gene groups, such "
               "as housekeeping genes and transcription factors, by their typical protein identity and typical Δα; bars "
               "cover the middle half of each group. All groups sit near zero except the handful of genes encoded in the "
               "mitochondrial genome. The lower rows show Δα gene by gene against three of the measures; the clouds are "
               "flat. Small triangles on the frame of a plot are genes that lie beyond its axis.")),
    "supp_worked_genes": Figure(
        X("supp_worked_genes/worked_genes"),
        label="fig:supp-worked-genes",
        title="Two worked genes in the counts",
        shows=("The figure shows two example genes as they appear in the raw data, before any model. For each "
               "gene the counts of all cells of one type are summed per species and put on a common scale, so "
               "each line is one species and each point is one cell type. On the left is ZPBP2, a sperm protein "
               "whose sequence differs clearly between species (protein identity 0.82). Its expression rises to "
               "a peak in spermatocytes and falls afterward in human, mouse and macaque alike: the protein "
               "has changed, the pattern of use has not. On the right is ALAS1, the first enzyme of heme "
               "synthesis, whose protein is well conserved (protein identity 0.87). It is highest in "
               "hepatocytes in all three mammals, but the mouse expresses it far more strongly than human "
               "(about 24-fold in hepatocytes): the protein is kept, the level of use differs.")),
    "supp_functional_layer": Figure(
        Row(X("supp_functional_layer/pathways", "a"), X("supp_functional_layer/interactions", "b"),
            Col(X("supp_functional_layer/interologs", "c"), X("supp_functional_layer/regulatory", "d"))),
        label="fig:supp-functional-layer",
        title="Pathway and network coherence and conservation",
        shows=("The figure asks whether Δα is organized by biology beyond single genes, in four ways. Panel a "
               "tests pathways: if genes of the same pathway have similar values, a coherence index rises "
               "above 1. It does so for the protein prior, for the count evidence and for Δα after known "
               "covariates are removed (Δα, adjusted), and it stays above the gray band of chance after the "
               "protein-language-model features are removed as well (Δα, adjusted + ESM-2), in two liver "
               "datasets and in testis. Panel b tests protein interaction networks: Moran's I measures "
               "whether interacting proteins have similar Δα. On the STRING network the fully adjusted value "
               "lies beyond the gray null band in every dataset; on the HuRI network it does not; open "
               "circles repeat the test without the most similar protein pairs. Panel c asks about "
               "conservation instead of coherence: whether genes whose interactions are retained in mouse "
               "differ in Δα; all four estimates lie inside the null band. Panel d asks the same for gene "
               "regulation, using transcription-factor binding that is conserved between human and mouse: "
               "the correlations are small, and the difference between liver and testis (open diamonds) is "
               "not distinguishable from zero for the fully adjusted Δα.")),
    "supp_covariate_audit": Figure(
        X("supp_covariate_audit/covariate_audit"),
        width=350,
        label="fig:supp-covariate-audit",
        title="Covariate audit of Δα in seven cohorts",
        shows=("The figure asks how much of Chimera's reading Δα can be explained by simple, known properties of "
               "a gene. Each column is one of the seven liver and testis datasets. Each of the first eight rows "
               "is one property: how strongly and how widely the gene is expressed and how variable its "
               "expression is, how specific it is to a cell type, how conserved its protein is (the protein "
               "prior), the model's own count evidence, and the length and base composition of the coding "
               "sequence. The number in a cell is the rank correlation of that property with Δα; red means "
               "they rise together, blue the opposite. The ninth row shows that the count evidence stays "
               "related to Δα after expression is accounted for. The two gray rows at the bottom give the "
               "share of the variation of Δα that can be predicted from three expression features alone and "
               "from all 72 inputs of the protein-feature gate (darker means a larger share).")),
    "supp_seed_reliability": Figure(
        P("supp_seed_reliability/seed_reliability"),
        width=340,
        label="fig:supp-seed-reliability",
        title="Agreement between training seeds",
        shows=""),
    "supp_celltype_closure": Figure(
        X("supp_celltype_closure/closure"),
        label="fig:supp-celltype-closure",
        title="Closure of the cell-type decomposition of the count evidence",
        shows=("Chimera's count evidence for a gene is an average over all cells. That average can be split "
               "exactly into the part contributed by each cell type. This figure checks that the split adds up: "
               "for every gene, the value rebuilt from the cell-type parts in one pass over all cells "
               "(horizontal axis) is plotted against the value the model stored when training ended (vertical "
               "axis), in each of the seven liver and testis datasets. Every gene is drawn, and each plot has "
               "its own range. The points lie on the diagonal, with correlations from 0.969 to 0.997.")),
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
        X("supp_relations_heatmap/heatmap"),
        label="fig:supp-relations-heatmap",
        title="Relations of Δα and count evidence to sequence, expression and direct baselines",
        shows=("Each number says how closely one of Chimera's gene scores follows a known property of genes, in "
               "one of the seven liver and testis datasets (columns). The properties (rows) are sequence "
               "constraint (dN/dS), protein identity between species, tolerance to damaging variants in humans "
               "(LOEUF), expression level, cell-type specificity, coding length, and two direct comparisons of "
               "expression between species (profile correlation and specificity similarity). Red means the "
               "score rises with the property, blue that it falls, pale that the two are unrelated; a gray "
               "cell without a number means the property is not defined for that dataset. The top pair of "
               "maps is for Δα, the middle pair for Δα measured against the protein identity before it is "
               "capped, the bottom pair for the count evidence. In each pair the left map is the plain rank "
               "correlation and the right map the correlation after protein identity, expression, detection "
               "and coding length are held fixed. For the four liver datasets the middle maps repeat the top "
               "maps, because their prior is not capped. The rows of the sequence measures are pale; the "
               "expression row is strong in four-species liver and in testis; the count evidence follows the "
               "profile comparison moderately.")),
    "supp_relations_density": Figure(
        X("supp_relations_density/density"),
        label="fig:supp-relations-density",
        title="The same relations gene by gene",
        shows=("This figure shows, gene by gene, the relations that the previous figure gives as single "
               "numbers. Each small plot takes all genes of one dataset and places them by a known gene "
               "property (horizontal) and by one of Chimera's two scores (vertical). Each dataset (liver "
               "Human+Mouse, four-species liver, testis Human+Mouse) has two rows of plots: Δα above, the "
               "count evidence below. Each column is one property, in the order of the rows of the previous "
               "figure: sequence constraint (dN/dS), protein identity, tolerance to damaging variants "
               "(LOEUF), expression level, cell-type specificity, coding length and two direct comparisons "
               "of expression between species. Darker blue means more genes in that spot; the orange line "
               "follows the typical value from low to high values of the property; the rank correlation and "
               "the number of genes stand above the plots. Against the sequence measures the clouds are flat. "
               "Against expression they slope, downward in four-species liver and upward in testis. The dark "
               "horizontal band in the count-evidence plots is the set of genes with too few expressing "
               "cells, whose count evidence is exactly zero.")),
}
