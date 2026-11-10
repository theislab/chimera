"""Supplementary figures on the tissues: one figure per tissue, recurring programs, cell types, disease genes and drug targets.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, P, Panel, Row


def tissue(family: str) -> Col:
    """The eleven panels of a tissue figure, a to k, in four rows."""
    def q(k: str) -> Panel:
        return P(f"{family}/{k}", k)

    return Col(Row(q("a"), q("b"), q("c"), same_scale=True), Row(q("d"), q("e"), same_scale=True),
               Row(q("f"), q("g"), q("h"), same_scale=True), Row(q("i"), q("j"), q("k"), same_scale=True), gap=3)

FIGS = {
    "supp_tissue_bone_marrow": Figure(
        tissue("supp_tissue_bone_marrow"),
        label="fig:supp-h1-bone-marrow",
        title="The reading in bone marrow",
        shows=""),
    "supp_tissue_kidney": Figure(
        tissue("supp_tissue_kidney"),
        label="fig:supp-h1-kidney",
        title="The reading in kidney",
        shows=""),
    "supp_tissue_islet": Figure(
        tissue("supp_tissue_islet"),
        label="fig:supp-h1-islet",
        title="The reading in pancreatic islet",
        shows=""),
    "supp_tissue_eye": Figure(
        tissue("supp_tissue_eye"),
        label="fig:supp-h1-eye",
        title="The reading in the eye",
        shows=""),
    "supp_tissue_cortex": Figure(
        tissue("supp_tissue_cortex"),
        label="fig:supp-h1-cortex",
        title="The reading in cerebral cortex",
        shows=""),
    "supp_tissue_lung": Figure(
        tissue("supp_tissue_lung"),
        label="fig:supp-h1-lung",
        title="The reading in lung",
        shows=""),
    "supp_tissue_muscle": Figure(
        tissue("supp_tissue_muscle"),
        label="fig:supp-h1-muscle",
        title="The reading in skeletal muscle",
        shows=""),
    "supp_tissue_tumor_myeloid": Figure(
        tissue("supp_tissue_tumor_myeloid"),
        label="fig:supp-h1-tumour",
        title="The reading in lung-tumor myeloid cells",
        shows=""),
    "supp_tissues_recurrence": Figure(
        P("supp_tissues_recurrence/recurrence_all"),
        width=440,
        label="fig:supp-h1-recurrence-all",
        title="All gene sets that recur across tissues",
        shows=("Each row is a set of genes with a common function, for example receptors, ion channels or "
               "ribosome and translation genes. Each column is a tissue. A red cell means that in this tissue "
               "the genes of the set sit above their protein prior more than comparable genes do: their use "
               "across cell types is shared between species more than the conservation of their proteins "
               "predicts. A blue cell means the opposite. The same sets are red in tissue after tissue "
               "(receptors, cell-surface, signaling and inflammation sets) and the same sets are blue "
               "(mitochondrial, translation, chromatin and proteostasis sets). The columns at the right give the "
               "test of recurrence and show how the same sets behave on the count evidence, on a direct profile "
               "comparison and on two sequence measures.")),
    "supp_tissues_lineage": Figure(
        P("supp_tissues_lineage/lineage"),
        label="fig:supp-h1-lineage",
        title="Lineage marker sets across tissues",
        shows=("Each row is a small set of well-known marker genes of one cell lineage, such as B cells of the "
               "bone marrow, the proximal tubule of the kidney or inhibitory neurons of the cortex; the tissue "
               "the lineage belongs to is given in brackets. Each column is a tissue. A red cell means that in "
               "this tissue the marker genes sit above comparable genes: on the left for Δα, the reading against "
               "the protein prior, and on the right for the count evidence behind it. The black box marks the "
               "lineage's own tissue. A dash means that fewer than five of the marker genes have enough counts "
               "in that tissue to be scored. Most values lie in the boxes, and they are red.")),
    "supp_programs_recurrent": Figure(
        Col(P("supp_programs_recurrent/atlas", "a"), Row(P("supp_programs_recurrent/distance", "b"), width=330), gap=3),
        label="fig:supp-e1",
        title="Gene programs that recur across eight tissues",
        shows=("Panel a asks which groups of genes, such as pathways, protein classes and cell compartments, sit "
               "above or below what their protein conservation predicts, and whether this is the same in every "
               "tissue. Each row is one gene program and each column a tissue; red means the program's genes are "
               "used more alike across species than their proteins predict (above the prior), blue less (below), "
               "and a dot marks a significant tissue. The left block is read with the default protein-feature "
               "gate, the right block with the per-gene read-out, and they show the same picture. The rows are "
               "sorted into four groups by two properties: whether the proteins evolve fast or are conserved, "
               "and whether the program sits above or below the prior. Secreted and cell-surface proteins evolve "
               "fast yet sit above in all eight tissues; the nuclear RNA machinery is highly conserved yet sits "
               "below. The three number columns give, for each program, how its protein identity, its dN/dS and "
               "its direct profile correlation compare with matched genes. Panel b takes the two-species models "
               "of four tissues and orders them by the evolutionary distance of the partner species from human, "
               "from chimpanzee (6 million years) to zebrafish (429). The size of the above and below programs "
               "(left) and the agreement of each model's program order with the other tissues (right) stay flat "
               "with distance: the program-level reading is already present between human and chimpanzee.")),
    "supp_programs_by_tissue": Figure(
        P("supp_programs_by_tissue/tissue", "c"),
        width=420,
        label="fig:supp-e1-cont",
        title="Tissue programs in each tissue",
        shows=""),
    "supp_programs_named_genes": Figure(
        P("supp_programs_named_genes/genes", "d"),
        width=190,
        label="fig:supp-e1-genes",
        title="Named genes that recur across tissues",
        shows=""),
    "supp_cell_compartments": Figure(
        Col(P("supp_cell_compartments/compartments", "a"),
            Row(P("supp_cell_compartments/pattern_vs_level", "b"), width=280), gap=3),
        label="fig:supp-e3-compartments",
        title="Home compartments of genes against Δα",
        shows=("Every gene is given a \"home\": the cell type in which human cells use it most, or no home if it "
               "is used in all cells. Home cell types are grouped into four compartments: the specialized cells "
               "of a tissue (neurons, kidney tubule cells, germ cells), stromal cells, blood-vessel cells and "
               "immune cells. Panel (a) asks, for each compartment and each of the 26 cohorts, whether its genes "
               "score higher or lower than comparison genes with the same protein identity, expression, "
               "detection and cell-type specificity. Four scores stand side by side: Chimera's Δα with the "
               "default protein-feature gate, Δα read gene by gene, the correlation of cell-type expression "
               "profiles between species, and the similarity of expression level at the home cell type. Genes of "
               "specialized cells sit above their protein prior in 23 of 24 and 24 of 24 cohorts, while their "
               "expression level at the home cell type is the least similar between species (2 of 24 above "
               "zero); genes used in every cell show the reverse in 26 of 26 cohorts. Panel (b) splits the "
               "compartments into 70 finer cell groups from nine cohorts and plots how much the level differs "
               "between species against the per-gene Δα excess: specialized cell groups gather where both are "
               "positive, and the liver sinusoidal endothelium is the most shared program of all (+0.79 SD).")),
    "supp_cell_lineages": Figure(
        Col(P("supp_cell_lineages/lineages", "a"), P("supp_cell_lineages/named", "b"), gap=3),
        label="fig:supp-e3-lineages",
        title="Cell types that carry shared deployment",
        shows=("Panel (a) takes four tissues for which several models were trained independently (liver 4, eye "
               "6, cortex 7, testis 3) and asks, for the genes whose home is a given cell type, how far they sit "
               "above or below comparison genes with the same protein identity, expression and specificity. Each "
               "circle is Chimera's per-gene Δα in one model, each triangle the correlation of expression "
               "profiles between species in the same model, and the short bars are medians. Genes of the liver "
               "sinusoidal endothelium (the lining of the liver's blood channels) are the most shared program of "
               "any tissue, +0.79 to +1.19 SD in all four liver models, while hepatocyte genes sit near their "
               "prior. In the eye, genes of the trabecular meshwork (the tissue through which fluid leaves the "
               "eye, central to glaucoma), ciliary muscle and Schwann cells are above; in the cortex, the "
               "neurons; in the testis, spermatocytes and spermatids. Panel (b) names seven genes for four of "
               "these cell types and shows their per-gene Δα as a z-score in every model of the tissue: a row "
               "that is red in every column is a gene above its prior whichever species set was used. Examples "
               "are HEG1 and PLXND1 in the liver endothelium, the glaucoma gene MYOC in the trabecular meshwork, "
               "and ANK3 and GAP43 in neurons.")),
    "supp_primate_branches": Figure(
        Row(P("supp_primate_branches/branch", "a"), P("supp_primate_branches/mtg", "b"),
            P("supp_primate_branches/human_lineage", "c")),
        label="fig:supp-e3-primates",
        title="Species-specific departures along the primate phylogeny",
        shows=("One model was fitted to prefrontal cortex cells of four primates at once: human, chimpanzee, "
               "macaque and marmoset. For every gene and cell type the model gives each species its own evidence "
               "value, and a species \"departs\" when its value is clearly lower than those of the other three. "
               "Panel (a) plots how often each species departs against the length of its own branch of the "
               "primate tree: about 9 to 11 % of gene-by-cell-type pairs for human and chimpanzee (6.4 million "
               "years), 13 % for macaque (28.8) and 24 % for marmoset (43). Both gates of Chimera and a direct "
               "comparison of expression levels give the same order. Panel (b) tests this in a second cortical "
               "region from another laboratory that the model never saw: genes that depart more on the marmoset "
               "branch than on the macaque branch also show the larger expression gap there (ρ = 0.17), and "
               "departures on the human branch agree with a human-specific gap (ρ = 0.07), with the chimpanzee "
               "branch as a control. Panel (c) shows five genes that an earlier study reported as changed on the "
               "human lineage: in the cell type named by that study the human bar lies below those of the other "
               "three primates, for example FOXP2 in microglia.")),
    "supp_disease_genes": Figure(
        Col(P("supp_disease_genes/heat", "a"),
            Row(P("supp_disease_genes/partial", "b"), P("supp_disease_genes/q1", "c")), gap=4),
        shrink=0.9,
        label="fig:supp-e4",
        title="Disease genes and drug targets by organ system",
        shows=""),
    "supp_drug_targets": Figure(
        P("supp_drug_targets/targets", "d"),
        label="fig:supp-e4-targets",
        title="Approved drug targets against the protein prior",
        shows=""),
}
