"""Composed figures that are not part of the paper.

They can be composed (``python -m figures.compose <key>``) and are not in ``figures.pdf``, ``order.txt`` or the index.
``EXTRA``: figures drawn beside the paper. ``RETIRED``: figures and panels that were in the paper and left it because a
later analysis of the same question on the same data replaced them; each entry says what it showed, what replaced it
(figure key and analysis task) and when. Their panel files stay in ``figures/panels/`` and their scripts still draw them.
"""

from __future__ import annotations

from figures.compose import Col, Figure, NB, P, Row, X

IN_PAPER = False

EXTRA = {
    # The three scores of the ten methods on the seven liver and testis cohorts as heat maps without method names. The
    # same scores, with the names, are supp_integration_ten_methods; the main figure shows the composite as a strip.
    "extra_integration_seven_cohorts": Figure(
        Row(X("main_integration/mixing", "a"), X("main_integration/conservation", "b"), X("main_integration/composite", "c"),
            same_scale=True, gap=4),
        title="Integration scores of ten methods on the seven liver and testis cohorts"),
    "extra_new_use_cases": Figure(
        Col(Row(P("extra_new_use_cases/near_identical", "a"), P("extra_new_use_cases/heldout_regions", "b")),
            Row(P("extra_new_use_cases/transfer", "c"), P("extra_new_use_cases/tumor_mixing", "d"), P("extra_new_use_cases/embedding", "e"))),
        title="Further use cases on added cohorts"),
    "extra_sparse_genes": Figure(
        Col(Row(P("extra_sparse_genes/masked", "a"), P("extra_sparse_genes/heldout", "b")), P("extra_sparse_genes/depth", "c")),
        title="Sparse genes: masked and held-out counts, depth"),
    "extra_uses": Figure(
        Col(Row(P("extra_uses/a", "a"), P("extra_uses/b", "b")),
            Row(P("extra_uses/c", "c"), P("extra_uses/d", "d")),
            Row(P("extra_uses/e", "e"), P("extra_uses/f", "f"))),
        title="Uses of the readouts"),
    "extra_embedding": Figure(
        Col(Row(P("extra_embedding/a", "a"), Col(P("extra_embedding/b", "b"))),
            Row(P("extra_embedding/c", "c"), P("extra_embedding/d", "d"))),
        title="The cell embedding"),
}

RETIRED = {
    # Showed: Δα against detection rate, mean count and count variance of every gene, liver Human+Monkey only.
    # Replaced by: supp_covariate_audit, the same three statistics on seven cohorts (task T2_4_covariates).
    # Retired: 2026-11-10.
    "supp_alpha_decorrelation": Figure(
        Row(X("supp_alpha_decorrelation/detection_rate", "a"), X("supp_alpha_decorrelation/mean_count", "b"), X("supp_alpha_decorrelation/count_variance", "c"), same_scale=True),
        title="Δα against expression magnitude statistics (retired)",
        note=("panels: figures/redraw/alpha.py")),
    # Showed: the composite integration score of seven methods on the seven liver and testis cohorts.
    # Replaced by: the benchmark of ten methods on the same cohorts and metrics: main_integration,
    # supp_integration_ten_methods, supp_integration_eleven_cohorts (tasks T5_5_benchmark, B2_benchmark).
    # Retired: 2026-11-10.
    "supp_integration_seven_methods": Figure(
        Col(X("supp_integration_seven_methods/composite", "a"), X("supp_integration_seven_methods/mean_score", "b")),
        width=330,
        title="Integration scores of seven methods (retired)",
        note=("panels: figures/redraw/integration_scores.py")),
    # Showed: the six raw integration metrics of seven methods on the seven liver and testis cohorts.
    # Replaced by: supp_integration_raw_metrics, the same six metrics for ten methods (task T5_5_benchmark).
    # Retired: 2026-11-10.
    "supp_integration_metrics": Figure(
        Col(Row(X("supp_integration_metrics/knn_species_purity", "a"), X("supp_integration_metrics/knn_mixing_entropy", "b"), same_scale=True),
            Row(X("supp_integration_metrics/silhouette_species", "c"), X("supp_integration_metrics/knn_celltype_purity", "d"), same_scale=True),
            Row(X("supp_integration_metrics/silhouette_celltype", "e"), X("supp_integration_metrics/label_transfer_accuracy", "f"), same_scale=True)),
        title="Raw integration metrics of seven methods (retired)",
        note=("panels: figures/redraw/integration_scores.py")),
    # Showed: program means of Δα in liver against testis for the same species pair, against the clamped prior (the
    # former panels j and k of supp_alpha_robustness; panel files supp_alpha_robustness/tissues_hmonkey_hmac and
    # supp_alpha_robustness/tissues_hmouse).
    # Replaced by: the tissue comparison against the raw prior, centered, genes matched by identifier: panel d of
    # supp_delta_alpha_sources (task T2_3_tissue).
    # Retired: 2026-11-10.
    "retired_tissue_programs_clamped_prior": Figure(
        Row(X("supp_alpha_robustness/tissues_hmonkey_hmac", "a"), X("supp_alpha_robustness/tissues_hmouse", "b"), same_scale=True, width=321.67),
        title="Program-level Δα, liver against testis, clamped prior (retired)",
        note=("panels: figures/redraw/robustness.py")),
    # Showed: the genes shared by liver and testis by sign class of Δα against the clamped prior (the former panel e of
    # supp_residue_audit_2; the panel stored by notebook 08, notebooks/figures/ortholog_qc/).
    # Replaced by: panel d of supp_delta_alpha_sources (task T2_3_tissue).
    # Retired: 2026-11-10.
    "retired_tissue_sign_classes_clamped_prior": Figure(
        NB("ortholog_qc/i3c_cross_tissue_disagreement_portrait_grayscale"),
        width=228,
        title="Liver against testis by sign class of Δα, clamped prior (retired)"),
}

FIGS = {**EXTRA, **RETIRED}
