"""Supplementary figures on the protein side: the gate network, the protein-space atlas, the residue audit, protein and regulatory architecture.

One entry per float of the paper: the layout, the label it belongs to, a short title and what it shows.
A continuation float has the key of its figure with ``_2``, ``_3`` and the same label. See ``figures/README.md``.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, P, Row, SRC, TEXTWIDTH, X

FIGS = {
    "supp_gin_linear_baseline": Figure(
        Row(X("supp_gin_linear_baseline/r2_by_input_set", "a"), X("supp_gin_linear_baseline/spearman_single_inputs", "b"), same_scale=True),
        label="fig:supp-gin-esm2-linear-baseline",
        title="Linear surrogate and scalar-correlation controls of the gate correction",
        shows=("The gate network turns 72 numbers per gene (64 from a protein language model and 8 single numbers "
                "such as expression level and sequence identity) into one correction of the gene's starting value. "
                "(a) How much of that correction a straight-line model rebuilds from parts of the input: the "
                "expression numbers alone, the eight single numbers, the protein-language numbers alone, both "
                "together, and all 72. (b) How each of the eight single numbers alone relates to the correction "
                "across genes. Four-species liver model."),
        note=("panels: figures/redraw/gate_network.py (loads the stored gate network; nothing is trained)")),
    "supp_gin_attribution_suite": Figure(
        Col(Row(X("supp_gin_attribution_suite/ablation_change_by_group", "a"), X("supp_gin_attribution_suite/gradient_input_by_group", "b"), same_scale=True),
            Row(X("supp_gin_attribution_suite/gradient_input_signed_single", "c"), X("supp_gin_attribution_suite/ablation_correlation_by_group", "d"), same_scale=True),
            Row(Col(X("supp_gin_attribution_suite/weight_path_by_group", "e"),
                    NB("gin_attention_overall/b3_integrated_gradients_completeness", "g", region=(0, 11, 274.41, 191.28))),
                Col(X("supp_gin_attribution_suite/weight_path_top20", "f"), X("supp_gin_attribution_suite/ig_by_group", "h")))),
        label="fig:supp-gin-esm2-attribution-suite",
        title="Attribution of the gate correction to input groups, all methods",
        shows=("Four standard tests of which inputs the trained gate network uses, on the four-species liver model. "
                "The 72 inputs are put in five groups: the protein-language numbers, their difference between "
                "species, the mean sequence identity (the quantity the prior is made of), the expression numbers, "
                "and the spread of identity. (a) and (d): switch a group off and see how much the output changes and "
                "how well it still follows the full output. (b) and (c): how strongly the output reacts to a small "
                "change of an input, per group and with sign for the eight single numbers. (e) and (f): how much "
                "weight the connections of the network give to each group and to the 20 heaviest single inputs. (g): "
                "a check that integrated gradients add up to the change of the output. (h): integrated gradients per "
                "group."),
        note=("panels: figures/redraw/gate_network.py; (g) is the notebook panel without its title line, because the "
              "error it prints is a rounding-level quantity whose last digits depend on the machine")),
    "supp_gin_method_consensus": Figure(
        Col(X("supp_gin_method_consensus/consensus_bars", "a"), X("supp_gin_method_consensus/consensus_heatmap", "b")),
        width=420,
        label="fig:supp-gin-esm2-method-consensus",
        title="Consensus of the attribution methods",
        shows=("One table shown twice: for each of the four tests of the previous figure, the share of the total "
                "that falls on each of the five groups of inputs. (a) as bars, (b) as numbers. Three tests put most "
                "of the total on the protein-language numbers; the test that switches inputs off puts the most on "
                "expression; the identity number that defines the prior gets very little in all four. Four-species "
                "liver model."),
        note=("panels: figures/redraw/gate_network.py")),
    "supp_gin_orthogonality": Figure(
        Row(X("supp_gin_orthogonality/r2_components_from_expression", "a"), X("supp_gin_orthogonality/delta_alpha_inputs_zeroed", "b"), same_scale=True),
        label="fig:supp-gin-esm2-orthogonality",
        title="ESM-2 and expression inputs: orthogonality and ablation",
        shows=("Two questions about the two kinds of input of the gate network, on the four-species liver model. "
                "(a) Do the protein-language numbers simply contain the expression numbers? For each of the 64 "
                "components, the share that the expression numbers predict: close to zero for almost all. (b) What "
                "happens to Δα when one kind of input is switched off: without the protein-language numbers the "
                "values gather near zero; without the expression numbers the whole distribution moves to the left."),
        note=("panels: figures/redraw/gate_network.py")),
    "supp_gin_per_pc": Figure(
        Col(Row(X("supp_gin_per_pc/ig_by_component", "a"), X("supp_gin_per_pc/spearman_by_component", "b"), same_scale=True),
            Row(X("supp_gin_per_pc/component_rank1", "c"), X("supp_gin_per_pc/component_rank2", "d"), X("supp_gin_per_pc/component_rank3", "e"),
                same_scale=True)),
        label="fig:supp-gin-esm2-per-pc",
        title="Contribution of single ESM-2 principal components",
        shows=("The 64 protein-language numbers one by one, on the four-species liver model. (a) How much the "
                "output of the gate network depends on each component: several matter, none dominates. (b) How each "
                "component alone relates to Δα across genes: some positively, some negatively, none strongly. (c) to "
                "(e) Δα against the three most important components, one dot per gene: weak trends in wide clouds."),
        note=("panels: figures/redraw/gate_network.py")),
    "supp_atlas_spatial_autocorrelation": Figure(
        Col(Row(X("supp_atlas_spatial_autocorrelation/coherence_alpha", "a"), X("supp_atlas_spatial_autocorrelation/coherence_delta_alpha", "b"), X("supp_atlas_spatial_autocorrelation/coherence_prior", "c"), same_scale=True),
            Row(X("supp_atlas_spatial_autocorrelation/moran_alpha", "d"), X("supp_atlas_spatial_autocorrelation/moran_delta_alpha", "e"), X("supp_atlas_spatial_autocorrelation/moran_prior", "f"), same_scale=True),
            X("supp_atlas_spatial_autocorrelation/legend")),
        label="fig:supp-esm2-umap-spatial-autocorr",
        title="Neighborhood coherence of α and Δα on the ESM-2 graph",
        shows=("Proteins can be placed in a space in which similar proteins are neighbors. The figure asks whether "
                "genes that are neighbors there also get similar values from the model, for four liver cohorts from "
                "a close pair of species to four species. Solid bars are the measured agreement between a gene and "
                "its 15 protein neighbors; the other bars are two chance levels (values shuffled among genes; random "
                "numbers in place of the protein description). Top row: agreement as a rank correlation for (a) α, "
                "(b) Δα, (c) the protein prior. Bottom row: the same with a second statistic, Moran's I, for (d) α, "
                "(e) Δα, (f) the prior."),
        note=("panels: figures/redraw/atlas.py (the neighbor search is repeated as in the notebook; see the note on "
              "ties in that script)")),
    "supp_atlas_archetype_trajectory": Figure(
        P("supp_atlas_archetype_trajectory/trajectory"),
        width=238,
        label="fig:supp-esm2-umap-archetype-trajectory",
        title="Δα of the archetypes during training",
        shows=("The gate network sorts genes into three groups by its own internal description of them (the "
                "archetypes A0, A1, A2 of the main atlas figure). The figure follows the mean Δα of each group while "
                "the four-species liver model trains. All three start at zero, because the gate starts at the prior, "
                "and separate after about 20 epochs: A0 ends clearly below the prior, A1 slightly below, A2 above."),
        note=("panel: figures/redraw/atlas.py")),
    "supp_atlas_archetype_enrichment": Figure(
        Row(X("supp_atlas_archetype_enrichment/categories", "a"), X("supp_atlas_archetype_enrichment/broad_categories", "b"), same_scale=True),
        width=399,
        label="fig:supp-esm2-umap-archetype-enrichment",
        title="Gene categories enriched in the archetypes",
        shows=("The three gene groups found inside the gate network (archetypes A0, A1, A2) are compared with "
                "groups of genes defined by function, on the four-species liver model. Red means that a functional "
                "group has more genes in an archetype than expected by chance, blue fewer; a star marks the cells "
                "that pass the statistical test. (a) 35 fine groups, (b) ten broad groups; each map has its own "
                "color scale."),
        note=("panels: figures/redraw/atlas.py; two pages of 196 bp and one 7 bp gap, placed at scale 1")),
    "supp_residue_audit": Figure(
        Col(Row(SRC("assembled/page37", "a", region=(0, 0, 375, 384)),
                SRC("assembled/page37", "b", region=(375, 0, 724, 384))),
            Row(SRC("assembled/page37", "c", region=(724, 0, 958, 188)),
                SRC("assembled/page37", "d", region=(724, 200, 958, 384))),
            gap=6),
        label="fig:supp-main-gin-protein",
        title="Residue-level attribution audit",
        shows=("The default gate of the model reads features of each protein that a protein language model computes "
               "from its sequence. For 70 well-studied proteins the figure asks which amino-acid positions those "
               "features draw on most when the model sets a gene's correction, and whether that pattern follows known "
               "properties of protein structure. (a) Twenty comparisons of kinds of residues (for example exposed "
               "against buried, or hydrophobic against charged) over all 36,832 residues; a bar to the right means "
               "the first kind receives more attribution. (b) The same comparisons made inside every protein and "
               "then tested across proteins, which removes effects that only reflect which proteins contribute many "
               "residues: six remain, led by hydrophobic against positively charged residues. (c) Each structural "
               "property explains well under one percent of the variation on its own. (d) Every comparison placed by "
               "its pooled and its within-protein evidence."),
        note="regions of the hand-assembled page 37, enlarged over two floats; (a) and (b) stay at scale 0.67 until they are redrawn"),
    "supp_residue_audit_2": Figure(
        Col(Row(NB("ortholog_qc/i3c_cross_tissue_disagreement_portrait_grayscale", "e"),
                Col(SRC("assembled/page37", "f", region=(258, 397, 520, 511)),
                    SRC("assembled/page37", "g", region=(258, 524, 520, 650)))),
            SRC("assembled/page37", "h", region=(520, 397, 958, 650)),
            gap=6),
        label="fig:supp-main-gin-protein", part=2,
        title="Residue-level attribution audit, continued",
        shows=("(e) The genes shared by liver and testis: mean Δα in liver against mean Δα in testis, shaded by the "
               "quadrant a gene falls in. (f, g) Three hand-curated groups of proteins: their overall attribution does "
               "not differ, and the contrast of hydrophobic against polar residues is clearest in the first group. "
               "(h) Six properties that are often assumed to drive such maps (disorder, glycine, proline, conservation "
               "and others) look strong when residues are pooled and disappear within proteins, except for contact "
               "density."),
        note="regions of the hand-assembled page 37 and one notebook panel"),
    "supp_residue_mixed_effects": Figure(
        P("supp_residue_mixed_effects/forest"),
        width=300,
        label="fig:supp-gin-protein-residue-beta",
        title="Residue-level mixed-effects regression",
        shows=("For 70 proteins the sensitivity of the model can be traced to single amino-acid positions (36,832 "
                "in all). A statistical model asks which properties of a position go with higher sensitivity when "
                "all other properties are held equal and differences between proteins are set aside. Each bar is one "
                "property with its 95% interval; bars to the right mean higher sensitivity; dark bars are "
                "statistically supported. The effects are small."),
        note=("panel: figures/redraw/residue.py")),
    "supp_residue_cliffs_delta": Figure(
        P("supp_residue_cliffs_delta/forest"),
        label="fig:supp-gin-protein-cliffs-delta",
        title="Protein-replicated effect sizes of residue features",
        shows=("Twenty comparisons between kinds of amino-acid positions (for example exposed against buried, "
                "hydrophobic against charged), each asking which kind receives more of the model's sensitivity. The "
                "bar is the size of the difference with its interval. Two tests are given per row: one that pools "
                "all positions, and a stricter one that compares the two kinds inside each protein and asks whether "
                "the proteins agree. Filled bars pass the stricter test."),
        note=("panel: figures/redraw/residue.py")),
    "supp_residue_attention_by_feature": Figure(
        P("supp_residue_attention_by_feature/distributions", label={"a": 65.0, "b": 166.25, "c": 280.5, "d": 412.5},
          pad=(0, 2.5, 0, 0)),
        label="fig:supp-gin-protein-attention-by-feature",
        title="Per-residue attribution by structural feature",
        shows=("The size of the model's sensitivity at each of the 36,832 amino-acid positions of 70 proteins, as "
                "box plots (line: median, box: middle half), grouped four ways: (a) by the local shape of the chain, "
                "(b) by how buried the position is, (c) by how confident the predicted structure is there, (d) by "
                "the chemistry of the amino acid. The boxes overlap almost completely; the number of positions is "
                "given under each box."),
        note=("panel: figures/redraw/residue.py; one page with four axes, the letters stand under the centers of the "
              "axes (bp on the page)")),
    "supp_residue_card_etv1": Figure(
        P("supp_residue_card_etv1/card"),
        label="fig:supp-gin-protein-etv1",
        title="Per-gene card: ETV1",
        shows=("One protein, ETV1 (a protein that the model leaves at its prior, with protein-language and "
                "expression inputs in balance), read along its chain from the first to the last amino acid. From top "
                "to bottom: where the proteins of macaque, mouse and zebrafish differ from the human one; how much "
                "the protein language model sees the species differ at each position; how unusual each position is "
                "within the protein; a combined relevance score; and how strongly each position pushes the model's "
                "value for this gene up or down. The strip at the bottom gives the share of the model's sensitivity "
                "for this gene by kind of input."),
        note=("panel: figures/redraw/residue.py")),
    "supp_residue_card_hprt1": Figure(
        P("supp_residue_card_hprt1/card"),
        label="fig:supp-gin-protein-hprt1",
        title="Per-gene card: HPRT1",
        shows=("One protein, HPRT1 (a strongly conserved reference protein that the model leaves at its prior), "
                "read along its chain from the first to the last amino acid. From top to bottom: where the proteins "
                "of macaque, mouse and zebrafish differ from the human one; how much the protein language model sees "
                "the species differ at each position; how unusual each position is within the protein; a combined "
                "relevance score; and how strongly each position pushes the model's value for this gene up or down. "
                "The strip at the bottom gives the share of the model's sensitivity for this gene by kind of input."),
        note=("panel: figures/redraw/residue.py")),
    "supp_residue_card_runx1": Figure(
        P("supp_residue_card_runx1/card"),
        label="fig:supp-gin-protein-runx1",
        title="Per-gene card: RUNX1",
        shows=("One protein, RUNX1 (a protein that the model places above its prior, with the expression inputs "
                "leading), read along its chain from the first to the last amino acid. From top to bottom: where the "
                "proteins of macaque, mouse and zebrafish differ from the human one; how much the protein language "
                "model sees the species differ at each position; how unusual each position is within the protein; a "
                "combined relevance score; and how strongly each position pushes the model's value for this gene up "
                "or down. The strip at the bottom gives the share of the model's sensitivity for this gene by kind "
                "of input."),
        note=("panel: figures/redraw/residue.py")),
    "supp_protein_architecture": Figure(
        Col(Row(P("supp_protein_architecture/properties", "a"), P("supp_protein_architecture/domains", "b"),
                same_scale=True),
            Row(P("supp_protein_architecture/age_duplication", "c"), width=205), gap=3),
        label="fig:supp-e2",
        title="Protein architecture and gene history against Δα",
        shows=""),
    "supp_regulatory_architecture": Figure(
        Col(Row(P("supp_regulatory_architecture/properties", "a"), P("supp_regulatory_architecture/regulons", "b"),
                same_scale=True),
            Row(P("supp_regulatory_architecture/dose", "c"), width=230), gap=3),
        label="fig:supp-e5",
        title="Regulatory and gene-architecture features against Δα",
        shows=""),
}
