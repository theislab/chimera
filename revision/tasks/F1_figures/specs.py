"""Layout of every composed revision figure. See compose.py for the tree and the checks."""

from __future__ import annotations

from revision.tasks.F1_figures.compose import ORIGINALS, Col, Figure, Panel, Row

P = Panel


def O(page: int, region: tuple, label=None, exempt_font: float | None = None, pad: tuple = (0, 0, 0, 0)) -> Panel:  # noqa: N802
    """A region of a page of figures_planning_cropped.pdf; the source labels stay outside the region."""
    return Panel(ORIGINALS, label=label, page=page, region=region, exempt_font=exempt_font, pad=pad)


def X(src: str, label: str | None = None) -> Panel:  # noqa: N802
    """A panel saved at its exact size, axes on the grid: the page is kept, not cropped to ink."""
    return Panel(src, label=label, ink=False)


FIGS = {
    "fig3": Figure(Row(X("fig3a_integration_mixing", "a"), X("fig3b_integration_biology", "b"),
                       X("fig3c_integration_composite", "c"), same_scale=True, gap=4)),
    "fig4": Figure(Row(O(16, (0, 0, 414, 402), {"a": 194.7, "b": 342.6}, exempt_font=6.0),
                       P("fig4c_prior_landscape", "c"))),
    "fig4_cont": Figure(Col(Row(P("fig4_celltype_drivers", "d"),
                                Col(X("fig4_margin_support", "e"), X("fig7e_sequence_axis", "g"))),
                            P("fig4_cross_tissue", "f"))),
    "fig5": Figure(Col(O(23, (0, 0, 571, 156), {"a": 182.7, "b": 463}, exempt_font=5.0), P("fig5_controls", "c"))),
    "fig7": Figure(Col(Row(X("fig7a_forest_h1_h4", "a"), X("fig7b_set_discrimination", "b")),
                       X("fig7c_head_to_head", "c"), X("fig7d_incremental_information", "d"))),
    "fig8": Figure(Col(Row(P("fig8_disease_a", "a"), P("fig8_disease_b", "b"), same_scale=True),
                       Row(P("fig8_disease_c", "c"), P("fig8_disease_d", "d"), P("fig8_disease_e", "e"), same_scale=True))),
    # Supplementary figures.
    "supp_robustness": Figure(Col(*[Row(*[P(f"supp_robustness_{k}", k) for k in r], same_scale=True) for r in ("abc", "def", "ghi")],
                                  Row(P("supp_robustness_j", "j"), P("supp_robustness_k", "k"), width=266, same_scale=True),
                                  gap=4), width=405),
    "supp_broad_categories": Figure(Col(Row(P("supp_broad_categories_a", "a"), P("supp_broad_categories_b", "b")),
                                        Row(P("supp_broad_categories_c", "c"), P("supp_broad_categories_d", "d")), gap=4),
                                    width=430),
    "supp_testis_categories": Figure(Row(P("supp_testis_categories_a", "a"), P("supp_testis_categories_b", "b"))),
    "supp_hallmark": Figure(Col(Row(P("supp_hallmark_a", "a"), P("supp_hallmark_b", "b")), P("supp_hallmark_c", "c"))),
    # Unchanged panels of page 37, reflowed so that they print above their size on that page; (e) re-rendered.
    "supp_residue_audit": Figure(Col(
        Row(O(37, (0, 0, 375, 384), "a", 6.5), O(37, (375, 0, 724, 384), "b", 6.5), width=438),
        Row(O(37, (724, 0, 958, 188), "c", 6.5, pad=(0, 0, 0, 90)), O(37, (724, 200, 958, 384), "d", 6.5, pad=(0, 0, 0, 90)),
            P("supp_residue_audit_e", "e"), width=440),
        Row(Col(O(37, (258, 397, 520, 511), "f", 6.5), O(37, (258, 524, 520, 650), "g", 6.5)),
            O(37, (520, 397, 958, 650), "h", 6.5)), gap=4),
        floor=4.7, note="unchanged panels of page 37: its 7 pt text prints at 4.7 pt in (g), its 5-6 pt counts lower"),
    "supp_sequence_axis": Figure(P("fig_sequence_axis")),
    "supp_known_measures": Figure(P("fig_known_measures")),
    "supp_worked_genes": Figure(P("supp_worked_genes")),
    "supp_functional_layer": Figure(Row(P("supp_functional_layer_a", "a"), P("supp_functional_layer_b_left", "b"),
                                        Col(P("supp_functional_layer_b_right", "b"), P("supp_functional_layer_c", "c")))),
    "supp_integration_purity": Figure(P("supp_integration_purity_view")),
    "supp_integration_raw": Figure(P("supp_integration_raw_metrics")),
    "supp_covariate_audit": Figure(P("supp_covariate_audit"), width=330),
    "supp_seed_reliability": Figure(P("supp_seed_reliability"), width=340),
    "supp_celltype_closure": Figure(P("supp_celltype_closure")),
    "supp_celltype_seed_stability": Figure(P("supp_celltype_seed_stability"), width=420),
    "supp_relations_heatmap": Figure(P("supp_relations_heatmap")),
    "supp_relations_density": Figure(P("supp_relations_density"), width=413),
    "supp_bone_marrow": Figure(Col(Row(*[P(f"supp_bone_marrow_{k}", k) for k in "abc"], same_scale=True),
                                   Row(*[P(f"supp_bone_marrow_{k}", k) for k in "def"], same_scale=True))),
}
