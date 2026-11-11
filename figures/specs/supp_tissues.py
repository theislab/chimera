"""Supplementary figures on the tissues: one figure per tissue, recurring programs, cell types, disease genes and drug targets.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, P, Panel, Row, X


def tissue(family: str) -> Col:
    """The eleven panels of a tissue figure, a to k, in four rows; each panel is a page of its exact size."""
    def q(k: str) -> Panel:
        return X(f"{family}/{k}", k)

    return Col(Row(q("a"), q("b"), q("c"), same_scale=True), Row(q("d"), q("e"), same_scale=True),
               Row(q("f"), q("g"), q("h"), same_scale=True), Row(q("i"), q("j"), q("k"), same_scale=True), gap=3)


TISSUE_SHOWS = ("The reading of {tissue} ({species}) with the panels the main figures use for liver and testis. (a) to (c) "
                "compare, for every gene, the value the model gives it (α) with the value its protein conservation alone "
                "predicts (the protein prior), and show how far genes move from the prior (Δα). (d) places every gene by "
                "prior and Δα and marks textbook marker genes of the tissue's cell lineages; the names of crowded genes "
                "stand in a row along the top. (e) and (f) ask the same for groups of genes: ten broad categories, and the "
                "Hallmark gene sets, where triangles are sets whose q value is below the plotted range. (g) and (h) show "
                "which cell types supply the evidence in the counts, for the marker genes and for the genes farthest above "
                "and below the prior. (i) compares Δα between the groups of genes found inside the model's protein network. "
                "(j) shows that Δα rises with the evidence in the counts. {k} Small gray triangles mark genes that lie "
                "beyond an axis.")
K_SUPPORT = "(k) gives the share of genes far above and far below the prior that are backed by count evidence of the same sign."
K_MODELS = "(k) shows the distribution of Δα in every model of this tissue, one row per model, the model of this figure first."

FIGS = {
    # 476 bp wide: at the text width the float with its caption is as tall as the page (smallest text 5.1 pt here).
    "supp_tissues_named_genes": Figure(
        Col(P("main_tissues/axis", "a"), P("main_tissues/genes", "b"), gap=3),
        width=476,
        label="fig:supp-tissue-genes",
        title="Δα against protein identity and dN/dS, and named genes, in eight tissues",
        shows=("The same reading in eight tissues, one column or block per tissue. (a) Top row: how far genes "
               "move from the prior, with the spread and the shares of genes clearly above and below. Middle and "
               "bottom rows: Δα against protein identity and against dN/dS, with the correlations under each "
               "cloud. The orange trend line is flat, so deployment does not follow sequence conservation in any "
               "tissue. Small gray triangles mark genes beyond an axis. (b) For each tissue, the genes furthest "
               "above the prior (orange), the tissue's own lineage genes (bold) and the genes furthest below "
               "(blue), every bar at its full length.")),
    "supp_tissue_bone_marrow": Figure(
        tissue("supp_tissue_bone_marrow"),
        label="fig:supp-h1-bone-marrow",
        title="The reading in bone marrow",
        shows=TISSUE_SHOWS.format(tissue="bone marrow", species="human, mouse lemur and mouse", k=K_SUPPORT)),
    "supp_tissue_kidney": Figure(
        tissue("supp_tissue_kidney"),
        label="fig:supp-h1-kidney",
        title="The reading in kidney",
        shows=TISSUE_SHOWS.format(tissue="kidney", species="human and mouse", k=K_SUPPORT)),
    "supp_tissue_islet": Figure(
        tissue("supp_tissue_islet"),
        label="fig:supp-h1-islet",
        title="The reading in pancreatic islet",
        shows=TISSUE_SHOWS.format(tissue="pancreatic islet", species="human and mouse", k=K_SUPPORT)),
    "supp_tissue_eye": Figure(
        tissue("supp_tissue_eye"),
        label="fig:supp-h1-eye",
        title="The reading in the eye",
        shows=TISSUE_SHOWS.format(tissue="the eye", species="human and mouse", k=K_MODELS)),
    "supp_tissue_cortex": Figure(
        tissue("supp_tissue_cortex"),
        label="fig:supp-h1-cortex",
        title="The reading in cerebral cortex",
        shows=TISSUE_SHOWS.format(tissue="cerebral cortex (primary motor cortex)", species="human and mouse", k=K_MODELS)),
    "supp_tissue_lung": Figure(
        tissue("supp_tissue_lung"),
        label="fig:supp-h1-lung",
        title="The reading in lung",
        shows=TISSUE_SHOWS.format(tissue="lung", species="human, mouse lemur and mouse", k=K_SUPPORT)),
    "supp_tissue_muscle": Figure(
        tissue("supp_tissue_muscle"),
        label="fig:supp-h1-muscle",
        title="The reading in skeletal muscle",
        shows=TISSUE_SHOWS.format(tissue="skeletal muscle", species="human and mouse", k=K_SUPPORT)),
    "supp_tissue_tumor_myeloid": Figure(
        tissue("supp_tissue_tumor_myeloid"),
        label="fig:supp-h1-tumour",
        title="The reading in lung-tumor myeloid cells",
        shows=TISSUE_SHOWS.format(tissue="lung-tumor myeloid cells", species="human and mouse", k=K_SUPPORT)),
    # 448 bp wide (as it printed before): with the heading of its subsection the float fits the page (smallest text 5.1 pt).
    "supp_tissues_recurrence": Figure(
        P("supp_tissues_recurrence/recurrence_all"),
        width=448,
        label="fig:supp-h1-recurrence-all",
        title="All gene sets that recur across tissues",
        shows=("Each row is a set of genes with a common function, for example receptors, ion channels or ribosome and "
               "translation genes. Each column is a tissue. A red cell means that in this tissue the genes of the set sit "
               "above their protein prior more than comparable genes do; a blue cell means the opposite; the number is the "
               "z value, a star marks significance, and a gray cell is a set with too few genes in that tissue. Colors stop "
               "at 8, the numbers do not. The rows stand in three blocks, each with a heading: sets above the prior in "
               "tissue after tissue (receptors, cell-surface, signaling and inflammation sets), sets below it "
               "(mitochondrial, translation, chromatin and proteostasis sets), and three sets found in one tissue only. The "
               "columns at the right, explained by the key under the figure, give the test of recurrence and show how the "
               "same sets behave on the count evidence, on a direct profile comparison and on two sequence measures.")),
    "supp_tissues_lineage": Figure(
        P("supp_tissues_lineage/lineage"),
        label="fig:supp-h1-lineage",
        title="Lineage marker sets across tissues",
        shows=("Each row is a small set of well-known marker genes of one cell lineage, such as B cells of the bone marrow, "
               "the proximal tubule of the kidney or inhibitory neurons of the cortex; the tissue the lineage belongs to is "
               "given in parentheses. Each column is a tissue. A red cell means that in this tissue the marker genes sit "
               "above comparable genes: on the left for Δα, the reading against the protein prior, and on the right for the "
               "count evidence behind it. The outline marks the lineage's own tissue. A light gray cell means that fewer "
               "than five of the marker genes have enough counts in that tissue to be scored. Most values lie in the "
               "outlined cells, and all of those are red.")),
    "supp_programs_recurrent": Figure(
        Col(X("supp_programs_recurrent/atlas", "a"), X("supp_programs_recurrent/distance", "b"), gap=3),
        label="fig:supp-e1",
        title="Gene programs that recur across eight tissues",
        shows=("Panel (a) asks which groups of genes, such as pathways, protein classes and cell compartments, sit above "
               "or below what their protein conservation predicts, and whether this is the same in every tissue. Each "
               "row is one gene program, with its source in parentheses (GO, UniProt keywords, Reactome, the Hallmark "
               "collection, or the gene categories of this paper), and each column a tissue. Red means the program's "
               "genes are used more alike across species than their proteins predict (above the prior), blue less "
               "(below); an asterisk marks a tissue in which the difference is significant. The colors end at 1 SD, so "
               "the few larger effects (up to 1.7) keep the darkest red. The left block is read with the default "
               "protein-feature gate, the right block with the per-gene gate, and they show the same picture. The rows "
               "are sorted into four groups, named at the left and separated by rules, by two properties: whether the "
               "proteins evolve fast or are conserved, and whether the program sits above or below the prior. Secreted "
               "and cell-surface proteins evolve fast yet sit above in all eight tissues; the nuclear RNA machinery is "
               "highly conserved yet sits below. The three number columns at the right give, for each program, how its "
               "protein identity, its dN/dS and its direct profile correlation compare with matched genes, as one z "
               "value combined over the eight tissues. Panel (b) takes the two-species models of four tissues (symbol "
               "and color) and orders them by the evolutionary distance of the partner species from human, from "
               "chimpanzee (6 million years) to zebrafish (429). Filled symbols and solid lines are the protein-feature "
               "gate, open symbols and dashed lines the per-gene gate. The size of the programs above and below the "
               "prior (left) and the agreement of each model's program order with the other tissues (right) stay flat "
               "with distance: the program-level reading is already present between human and chimpanzee.")),
    "supp_programs_by_tissue": Figure(
        X("supp_programs_by_tissue/tissue"),
        label="fig:supp-e1-cont",
        title="Tissue programs in each tissue",
        shows=("For each of eight tissues a short list of gene programs that stand for the tissue's known biology was "
               "chosen before the analysis: bile acid and drug metabolism for liver, meiosis and cilia for testis, "
               "immune and heme programs for bone marrow, transporters for kidney, hormone secretion for islet, synapses "
               "for cortex, and so on. Each row is one of these 35 programs, grouped by its tissue (bold, at the left) "
               "and with its source in parentheses; each column is a tissue. Red means that the program's genes sit "
               "above comparable genes in that tissue. The colors end at a z of 12, and a gray cell is a program with "
               "fewer than 15 genes in that tissue's model, which gets no z. The cell of the program's own tissue is "
               "outlined: with a solid line when the own tissue is significantly above the other seven, with a dashed "
               "line when it is not. The three blocks give the same test with three readings: Δα with the default "
               "protein-feature gate, Δα with the per-gene gate, and the direct correlation of expression profiles "
               "between species. With both gates a tissue's own programs lead in seven of eight tissues (20 of 35 "
               "programs significantly, for each gate): synapse in cortex, transport in kidney, immune and heme programs "
               "in bone marrow, hormone programs in islet. In bone marrow, and for most islet programs, the direct "
               "profile correlation does not single them out (19 of 35 programs overall). Liver is the exception: its "
               "metabolic programs come first on profile correlation but not on Δα.")),
    "supp_programs_named_genes": Figure(
        X("supp_programs_named_genes/genes"),
        label="fig:supp-e1-genes",
        title="Named genes that recur across tissues",
        shows=("Single genes that land on the same side of the protein prior in several tissues, with both gates. Each "
               "row is a gene and each column one of eight tissues. A large vermilion dot means the gene is called above "
               "the prior in that tissue (its use across cell states is shared between species more than its protein "
               "predicts), a large blue dot means below, a small gray dot means tested but not called, and no dot means "
               "the gene could not be tested in that tissue. The left block lists the 24 genes that recur most often "
               "above the prior: the endothelial receptor HEG1 is called in seven of eight tissues; many of the others "
               "build cell junctions and the cell cortex (LIMA1, CADM1, AKAP12, TJP1, MACF1, EPB41L2, FERMT2, JAM2), and "
               "two are cytokine receptors (IL6R, IFNGR1). The right block lists the 14 genes that recur most often "
               "below the prior, mostly conserved nuclear RNA and DNA genes (XRCC5, UPF3A, RRP1, RPL3, TNRC6C). The two "
               "columns beside each block give the gene's protein identity between the species and in how many of the "
               "cohorts in which it was tested it is called (HEG1: 13 of 26).")),
    "supp_gene_by_tissue": Figure(
        Col(Row(P("main_tissues/gene_by_tissue_row1", {"a": 150, "b": 400}, region=(0, 0, 512, 119)), same_scale=True),
            Row(P("main_tissues/gene_by_tissue_row2", {"c": 150, "d": 395}, region=(0, 120, 512, 283)),
                same_scale=True),
            Row(P("main_tissues/gene_by_tissue_row3", {"e": 145, "f": 385}, region=(0, 283, 512, 429)),
                same_scale=True),
            gap=3),
        label="fig:supp-gene-by-tissue",
        title="Gene and tissue components of the readouts",
        shows=("(a) How much of each readout belongs to the gene in every tissue, to the gene in one tissue, or "
               "to noise. (b) Each tissue's readout against independent data of every tissue: the matching "
               "tissue (outlined) agrees best. (c) The advantage of the matching tissue over the others. (d) "
               "Within one gene, differences between tissues in the readout follow differences between tissues "
               "in the independent data. (e) For genes whose reading flips between tissues, what is low in the "
               "low tissue: mostly a gap in expression level. (f) Six example genes.")),
    "supp_cell_compartments": Figure(
        Col(X("supp_cell_compartments/compartments", "a"),
            Row(X("supp_cell_compartments/pattern_vs_level", "b"), width=300), gap=3),
        label="fig:supp-e3-compartments",
        title="Home compartments of genes against Δα",
        shows=("Every gene is given a \"home\": the cell type in which human cells use it most, or no home if it "
               "is used in all cells. Home cell types are grouped into four compartments: the specialized cells "
               "of a tissue (neurons, kidney tubule cells, germ cells), stromal cells, blood-vessel cells and "
               "immune cells. Panel (a) asks, for each compartment and each of the 26 cohorts, whether its genes "
               "score higher or lower than comparison genes with the same protein identity, expression, "
               "detection and cell-type specificity. Four scores stand side by side: Δα with the protein-feature "
               "gate, Δα with the per-gene gate, the correlation of cell-type expression profiles between "
               "species, and the similarity of expression level at the home cell type. One mark is one cohort "
               "(filled when the difference is significant), the bar is the median, and the gray numbers count "
               "the cohorts above zero. Genes of specialized cells sit above their protein prior in 23 of 24 and "
               "24 of 24 cohorts, while their expression level at the home cell type is the least similar "
               "between species (2 of 24 above zero); genes used in every cell sit below the prior in 25 and 26 "
               "of 26 cohorts while their level is similar (23 of 26 above zero). Panel (b) splits the "
               "compartments into 70 finer cell groups from nine cohorts (one per tissue, and skeletal muscle) "
               "and plots how much the level differs between species against the excess of Δα with the per-gene "
               "gate: specialized cell groups gather where both are positive, and the liver sinusoidal "
               "endothelium is the most shared program of all (+0.79 SD).")),
    "supp_cell_lineages": Figure(
        Col(X("supp_cell_lineages/lineages", "a"), X("supp_cell_lineages/named", "b"), gap=3),
        label="fig:supp-e3-lineages",
        title="Cell types that carry shared deployment",
        shows=("Panel (a) takes four tissues for which several cohorts were trained independently (liver 4, eye "
               "6, cortex 7, testis 3) and asks, for the genes whose home is a given cell type, how far they sit "
               "above or below comparison genes with the same protein identity, expression and specificity. Each "
               "hexagon is Δα with the per-gene gate in one cohort, each triangle the correlation of expression "
               "profiles between species in the same cohort, and the short bars are medians. Genes of the liver "
               "sinusoidal endothelium (the lining of the liver's blood channels) are the most shared program of "
               "any tissue, +0.79 to +1.19 SD in all four liver cohorts, while hepatocyte genes sit near their "
               "prior. In the eye, genes of the trabecular meshwork (the tissue through which fluid leaves the "
               "eye, central to glaucoma), ciliary muscle and Schwann cells are above; in the cortex, the "
               "neurons; in the testis, spermatocytes and spermatids. Panel (b) names seven genes for four of "
               "these cell types and prints their Δα with the per-gene gate as a z-score in every cohort of the "
               "tissue: a row that is red in every column is a gene above its prior whichever species set was "
               "used. The color scale ends at 3 and larger values keep the darkest red; a gray cell is a gene "
               "without count evidence in that cohort. Examples are HEG1 and PLXND1 in the liver endothelium, "
               "the glaucoma gene MYOC in the trabecular meshwork, and ANK3 and GAP43 in neurons.")),
    "supp_primate_branches": Figure(
        Row(X("supp_primate_branches/branch", "a"), X("supp_primate_branches/mtg", "b"),
            X("supp_primate_branches/human_lineage", "c")),
        label="fig:supp-e3-primates",
        title="Species-specific departures along the primate phylogeny",
        shows=("One model was fitted to prefrontal cortex cells of four primates at once: human, chimpanzee, "
               "macaque and marmoset. For every gene and cell type the model gives each species its own evidence "
               "value, and a species \"departs\" when its value is clearly lower than those of the other three. "
               "Panel (a) shows how often each species departs, with the species in the order of the length of "
               "their own branch of the primate tree (printed under each name, in million years): about 9 to "
               "11 % of gene-by-cell-type pairs for human and chimpanzee (6.4), 13 % for macaque (28.8) and 24 % "
               "for marmoset (43). Chimera with either gate and a direct comparison of expression levels give "
               "the same order. Panel (b) tests this in a second cortical region from another laboratory that "
               "the model never saw: genes that depart more on the marmoset branch than on the macaque branch "
               "also show the larger expression gap there (ρ = 0.17), and departures on the human branch agree "
               "with a human-specific gap (ρ = 0.07), with the chimpanzee branch as a control. Panel (c) shows "
               "five genes that an earlier study reported as changed on the human lineage: in the cell type "
               "named by that study the human bar lies below those of the other three primates, for example "
               "FOXP2 in microglia.")),
    # Program transfer between human disease and its mouse models (panels: analysis/tasks/T8_4_figures/fig8_disease.py).
    "supp_disease_transfer": Figure(
        Col(Row(P("main_disease/kidney_transfer", "a"), P("main_disease/stage_transfer", "b"), same_scale=True),
            Row(P("main_disease/healthy_tissue_test", "c"), P("main_disease/transfer_vs_null", "d"),
                P("main_disease/injury_genes", "e"), same_scale=True)),
        label="fig:disease",
        title="Transfer of disease programs between human and mouse",
        shows=("(a) Kidney: for each cell type, how many disease programs that are active in human acute kidney "
               "injury also change the same way in the mouse injury model. (b) Early (2 days) against late (4 to "
               "20 weeks) mouse injury: the early stage reproduces more of the human programs. (c) The test "
               "fixed in advance: does Δα from healthy tissue predict which genes respond alike in human and "
               "mouse disease? The estimates sit at zero, inside the gray null bands. (d) The number of "
               "transferring programs against its null in kidney, liver and islet. (e) Published lists of shared "
               "and human-specific injury genes.")),
    "supp_disease_genes": Figure(
        Col(X("supp_disease_genes/heat", "a"),
            Row(X("supp_disease_genes/partial", "b"), X("supp_disease_genes/q1", "c")), gap=4),
        label="fig:supp-e4",
        title="Disease genes and drug targets by organ system",
        shows=("For each of nine tissues with a human and mouse model, two gene lists were taken from public "
               "resources: genes in which a mutation causes an inherited disease of that organ system, and the "
               "targets of approved drugs used for diseases of that organ system. Panel (a) asks, for each list "
               "(rows) and each tissue's model (columns), whether the list sits above (red) or below (blue) "
               "comparison genes of the same protein identity and expression, for three scores: Δα with the "
               "protein-feature gate, Δα with the per-gene gate, and the correlation of expression profiles "
               "between species. The outlined cell in each row is the list's own tissue, and a solid outline "
               "means that the list is higher there than in the other tissues; values beyond 6 keep the end "
               "colors. Disease genes sit above the prior in their own tissue in seven of nine tissues under "
               "the protein-feature gate, led by bone marrow, testis and kidney, with liver at the prior and "
               "cortex below. Panel (b) repeats the own-tissue comparison after also accounting for how "
               "cell-type-specific and how well studied each gene is (filled marks: the interval excludes "
               "zero): bone-marrow drug targets stay above on both gates (+0.23 and +0.25 SD) where profile "
               "correlation shows nothing (-0.05), and cortex drug targets, eye disease genes and the "
               "bone-marrow and testis disease genes also remain. Panel (c) asks a different question, whether "
               "a higher score makes it more likely that a mouse mutant of the gene is a recorded model of the "
               "same human disease. The two resources point in opposite directions (odds ratios 0.87 to 0.90 "
               "for curated models, 1.04 to 1.06 for phenotype matches), so the reading does not predict which "
               "mouse mutants model the disease.")),
}
