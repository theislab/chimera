"""Build ``10b_gin_attribution_figures.ipynb`` — the fast-iteration notebook.

This notebook is a **figure-only counterpart** to
``10_gin_attribution_protein_structure.ipynb``.  It loads every cached
§L analysis result from disk and re-renders the §A / §B / §C / §D / §E
paper-grade figures by calling functions in ``_summary_figures.py``.
No model load, no captum, no per-gene 3D generation, no upstream §A–§K
compute.  End-to-end execution should be **under 10 seconds** on a
clean kernel — purpose-built for fast figure design iteration.

Run from `chimera_env`::

    mamba run -n chimera_env python chimera/notebooks/_gin_attribution_figures_builder.py
    mamba run -n chimera_env jupyter nbconvert --to notebook --execute \\
        --inplace chimera/notebooks/10b_gin_attribution_figures.ipynb

Workflow for figure iteration:

1.  Edit the rendering function in
    ``chimera/notebooks/_summary_figures.py``  (single source of truth
    for every paper-grade figure rendered here).
2.  Re-run the two commands above.  Total round-trip ~5–10 seconds.

If you change the *upstream compute* (residue table, mixed model,
LOGO sweep), invalidate the cache by deleting the relevant
``cache/gin_attribution/L_analysis/*.{pkl,json}`` files and re-run
``10_*.ipynb`` once to regenerate them; ``10b`` will then pick the
fresh artifacts up automatically.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "10b_gin_attribution_figures.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).lstrip("\n").rstrip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).lstrip("\n").rstrip() + "\n")


cells: list[nbf.NotebookNode] = []


# ============ Header ============
cells.append(md("""
    # GIN attribution — paper-grade figures notebook

    *Sibling of `10_gin_attribution_protein_structure.ipynb`.  This
    notebook does **no upstream compute**: it loads every §L cached
    DataFrame from disk and re-renders every paper-grade figure by
    calling functions in [`_summary_figures.py`](_summary_figures.py).
    End-to-end execution: ~5–10 seconds.*

    The figures are organised by **level of evidence**, in the order a
    reader should consume them:

    - **§A — POOLED.**  Per-residue distributions, per-residue scatter
      against continuous covariates, and a pooled Cliff's δ forest
      across every contrast.  Aggregates 36,832 residues across 70
      genes; tells you what differs at the *raw residue level*.
    - **§B — PER-GENE.**  Per-gene paired Wilcoxon scorecard, the
      pooled-vs-per-gene composition-artefact diagnostic, per-gene
      forests for the chemistry contrasts, and the leave-one-gene-out
      sensitivity of the headline.  Tells you *which* of the §A
      effects are real and which are between-gene composition
      artefacts.
    - **§C — RESIDUE-CONTROLLED.**  The mixed-effects fixed-effect
      forest and the variance partition (ΔR² when each feature is
      dropped).  Tells you which features carry *unique* explanatory
      power once everything else is held fixed.
    - **§D — DIRECTION-AWARE.**  M-flag / F-flag / reference-class
      breakdown of the chemistry headline + the per-gene |Δα| × mean
      |attention| coupling, faceted by direction class.  Tests the
      mechanistic novelty: is the GIN's attention shape the same
      across the three α-correction regimes?
    - **§E — MASTER.**  Two master figures:  the evidence-chain
      summary (every contrast, ranked, both p-values per row) and
      the negative-controls scorecard (six features the GIN does
      *not* attend to, side-by-side pooled-vs-per-gene bars).

    Edit `_summary_figures.py` and rerun this notebook to iterate.

    ## Required caches

    Produced by the full notebook (run it once first if any are missing):

    - `cache/gin_attribution/L_analysis/residue_table.pkl`
    - `cache/gin_attribution/L_analysis/effect_df.pkl`
    - `cache/gin_attribution/L_analysis/mixedlm_coef.pkl`
    - `cache/gin_attribution/L_analysis/mixedlm_meta.json`
    - `cache/gin_attribution/L_analysis/variance_partition.pkl`
    - `cache/gin_attribution/L_analysis/paired_results.pkl`
    - `cache/gin_attribution/L_analysis/logo_df.pkl`
    - `cache/gin_attribution/L_analysis/corr_df.pkl`
    - `cache/gin_attribution/L_analysis/delta_alpha_attention.pkl`
    - `cache/gin_attribution/L_analysis/direction_breakdown.pkl`
    - `cache/gin_attribution/L_analysis/direction_chem_interaction_p.json`
    - `cache/gin_attribution/L_analysis/summary_contrasts.pkl`  (20 contrasts)
"""))


# ============ Setup ============
cells.append(md("""
    ## §0 — Lightweight imports + cache directory + design system
"""))
cells.append(code(r"""
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Notebook folder: $CHIMERA_ROOT/notebooks if set, else located from the working directory.
    import os
    NB_DIR = (Path(os.environ["CHIMERA_ROOT"]) / "notebooks" if os.environ.get("CHIMERA_ROOT")
              else Path.cwd() if (Path.cwd() / "_figure_design.py").exists()
              else Path.cwd() / "notebooks")
    assert (NB_DIR / "_figure_design.py").exists(), f"no notebooks folder at {NB_DIR}; set CHIMERA_ROOT"
    if str(NB_DIR) not in sys.path:
        sys.path.insert(0, str(NB_DIR))

    import _figure_design as fd
    import _summary_figures as sf

    fd.apply_design()

    CACHE_DIR  = NB_DIR / "cache" / "gin_attribution" / "L_analysis"
    FIG_DIR    = fd.set_figure_dir(NB_DIR / "figures" / "gin_attribution")
    print(f"cache dir : {CACHE_DIR}")
    print(f"figure dir: {FIG_DIR}")
"""))


# ============ Cache loaders ============
cells.append(md("""
    ## §1 — Load every §L cache in one shot

    `pd.read_pickle` for DataFrames, `json.loads` for scalars.  All
    files are produced by the full pipeline notebook's §L cells.
    Total load time: a few hundred milliseconds.
"""))
cells.append(code(r"""
    def _pq(name: str) -> "pd.DataFrame | None":
        p = CACHE_DIR / f"{name}.pkl"
        if not p.exists():
            print(f"  MISSING  {name}.pkl  — run the full notebook to regenerate")
            return None
        return pd.read_pickle(p)

    def _json(name: str) -> "dict | None":
        p = CACHE_DIR / f"{name}.json"
        if not p.exists():
            print(f"  MISSING  {name}.json")
            return None
        return json.loads(p.read_text())

    t0 = time.time()
    RESIDUE_TABLE       = _pq("residue_table")
    EFFECT_DF           = _pq("effect_df")
    coef_df             = _pq("mixedlm_coef")
    mixedlm_meta        = _json("mixedlm_meta") or {}
    VARPART_DF          = _pq("variance_partition")
    PAIRED_RESULTS      = _pq("paired_results")
    LOGO_DF             = _pq("logo_df")
    CORR_DF             = _pq("corr_df")
    DA_DF               = _pq("delta_alpha_attention")
    DIRECTION_BREAKDOWN = _pq("direction_breakdown")
    SUMMARY_CONTRASTS   = _pq("summary_contrasts")
    INTERACTION_LRT     = _json("direction_chem_interaction_p") or {}
    elapsed = time.time() - t0

    if coef_df is not None and "term" in coef_df.columns:
        coef_df = coef_df.set_index("term")

    p_lrt = float(INTERACTION_LRT.get("p_lrt", float("nan")))

    print(f"\nloaded all §L caches in {elapsed*1000:.0f} ms")
    print(f"  residue table        : {0 if RESIDUE_TABLE is None else len(RESIDUE_TABLE):>7,} residues")
    print(f"  effect_df            : {0 if EFFECT_DF is None else len(EFFECT_DF):>7,} contrasts")
    print(f"  coef_df              : {0 if coef_df is None else len(coef_df):>7,} fixed effects")
    print(f"  variance partition   : {0 if VARPART_DF is None else len(VARPART_DF):>7,} features")
    print(f"  paired_results       : {0 if PAIRED_RESULTS is None else len(PAIRED_RESULTS):>7,} contrasts")
    print(f"  logo_df              : {0 if LOGO_DF is None else len(LOGO_DF):>7,} LOGO splits")
    print(f"  corr_df              : {0 if CORR_DF is None else len(CORR_DF):>7,} genes")
    print(f"  delta_alpha_attention: {0 if DA_DF is None else len(DA_DF):>6,} genes")
    print(f"  direction_breakdown  : {0 if DIRECTION_BREAKDOWN is None else len(DIRECTION_BREAKDOWN):>7,} direction classes")
    print(f"  summary_contrasts    : {0 if SUMMARY_CONTRASTS is None else len(SUMMARY_CONTRASTS):>7,} pooled contrasts")
    print(f"  direction × chem LRT p = {p_lrt:.3e}" if np.isfinite(p_lrt) else "  direction × chem LRT p missing")
    if mixedlm_meta:
        print(f"  mixedlm_meta         : ICC = {mixedlm_meta.get('icc', 0)*100:.1f} %  "
              f"R²_marg = {mixedlm_meta.get('r2_marginal', float('nan')):.3f}  "
              f"R²_cond = {mixedlm_meta.get('r2_conditional', float('nan')):.3f}")
"""))


# ============ §A — POOLED ============
cells.append(md("""
    ## §A — Pooled (per-residue, all 36,832 residues)

    Three figures.  Together they answer: *what does the residue-level
    attention look like across the entire pooled corpus?*  The pooled
    view sets a baseline; §B then re-tests every effect at the
    per-gene level so we know what survives between-gene composition
    controls.
"""))


cells.append(md("""
    ### §A.1 — Per-residue $|$attention$|$ distributions, by category

    Box-and-whisker summaries of the per-residue $|$attention$|$
    distribution split by four orthogonal categories: secondary
    structure, relative SASA bin, AlphaFold pLDDT bin, and side-chain
    chemistry.  Whiskers are 5–95 % per category; outliers suppressed
    so the bulk distribution is comparable across panels.  Categories
    are encoded by gray ramp (light → dark = increasing class
    sequence), not by hue.
"""))
cells.append(code(r"""
    sf.render_A1_residue_distributions(
        residue_table=RESIDUE_TABLE,
        save_figure=fd.save_figure,
    )
"""))


cells.append(md("""
    ### §A.2 — Pooled Cliff's $\\delta$ forest (every contrast)

    All 20 pairwise contrasts in `summary_contrasts.pkl` on a single
    horizontal-bar chart, ordered by panel (SS / SASA / pLDDT / Chem
    / Disorder / Contact / Special / Cons.) and within-panel by
    Cliff's δ.  Bars are filled black if the pooled Mann-Whitney U
    test reaches $p < 0.05$, hollow gray otherwise — see §B.2 for
    which of these survive the per-gene control.
"""))
cells.append(code(r"""
    a2_df = sf.render_A2_pooled_effect_forest(
        summary_contrasts=SUMMARY_CONTRASTS,
        save_figure=fd.save_figure,
    )
    print(a2_df[["panel", "label", "cliffs_d", "ci_lo", "ci_hi", "mwu_p"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
"""))


cells.append(md("""
    ### §A.3 — Per-residue scatter against continuous covariates

    Hex-bin density (log-count, monochrome `Greys` colormap) of
    per-residue $|$attention$|$ vs four continuous covariates:
    hydropathy, conservation entropy, $C_{\\alpha}$ contact density,
    and relative SASA.  A solid black line traces the binned median
    so the trend reads at a glance; per-panel Spearman ρ + p + n in
    the upper right.
"""))
cells.append(code(r"""
    sf.render_A3_residue_scatter_continuous(
        residue_table=RESIDUE_TABLE,
        save_figure=fd.save_figure,
    )
"""))


# ============ §B — PER-GENE ============
cells.append(md("""
    ## §B — Per-gene (70 genes)

    The pooled effects in §A include both *real* within-gene effects
    and *artefactual* effects driven by between-gene composition (e.g.
    one gene contributing 5,000 residues with high SASA can carry the
    pooled signal entirely on its own).  The §B panels strip the
    per-gene scaling out: we collapse each gene to its within-gene
    median per category, then test paired across the 70 genes.  Any
    contrast that survives §B is genuinely a within-gene effect.
"""))


cells.append(md("""
    ### §B.1 — Per-gene paired Wilcoxon scorecard

    Per-contrast $-\\log_{10}(p)$ from a paired Wilcoxon test on the
    within-gene category medians, $n_g \\leq 70$.  Bars are solid
    black when significant ($p < 0.05$), hollow when not.  The
    chemistry contrasts dominate the survivors.
"""))
cells.append(code(r"""
    b1_df = sf.render_B1_per_gene_paired_scorecard(
        paired_results=PAIRED_RESULTS,
        summary_contrasts=SUMMARY_CONTRASTS,
        save_figure=fd.save_figure,
    )
"""))


cells.append(md("""
    ### §B.2 — Pooled vs per-gene composition-artefact diagnostic

    Each contrast plotted at $(-\\log_{10}(\\text{pooled MWU } p),\\ -\\log_{10}(\\text{per-gene paired } p))$.
    Points along the diagonal = effect agrees at both levels (real
    within-gene effect).  Points far above the x-axis but near the
    y-axis baseline = pooled-only effects that collapse under the
    per-gene control.  Filled circle = real, hollow square = composition
    artefact.  This is the single visual that justifies the §B → §C
    chain.
"""))
cells.append(code(r"""
    b2_df = sf.render_B2_pooled_vs_pergene_diagnostic(
        summary_contrasts=SUMMARY_CONTRASTS,
        save_figure=fd.save_figure,
    )
"""))


cells.append(md("""
    ### §B.3 — Per-gene chemistry forest (hydrophobic vs polar / positive / negative)

    Three side-by-side per-gene forests, one per contrast.  Each row
    is a gene; the bar is
    $\\text{median}(|\\text{attn}|\\,|\\,\\text{hydrophobic}) -
       \\text{median}(|\\text{attn}|\\,|\\,b)$.
    Bars are sorted by direction class first (M / F / ref) and within
    each class by signed effect.  Direction shading uses a 3-step gray
    ramp.  This figure tells you the chemistry headline is *not*
    driven by 1-2 outlier genes — the support is broad.
"""))
cells.append(code(r"""
    b3_df = sf.render_B3_per_gene_forest_chemistry(
        residue_table=RESIDUE_TABLE,
        paired_results=PAIRED_RESULTS,
        save_figure=fd.save_figure,
    )
    print(b3_df.to_string(index=False))
"""))


cells.append(md("""
    ### §B.4 — Leave-one-gene-out (LOGO) robustness

    Re-run the headline (hydrophobic vs positive) per-gene paired
    Wilcoxon excluding each gene in turn (n = 70 splits).  If the
    headline survived only because of one gene, removing that gene
    would push $p$ above 0.05.  Bars are solid when the split keeps
    significance; the corner badge reports the survival fraction and
    worst-case $p$.
"""))
cells.append(code(r"""
    b4_df = sf.render_B4_logo_summary(
        logo_df=LOGO_DF,
        save_figure=fd.save_figure,
    )
"""))


# ============ §C — RESIDUE-CONTROLLED ============
cells.append(md("""
    ## §C — Residue-controlled mixed-effects model

    Two figures from a single mixed-effects regression:

    $$
    \\log|\\text{attn}|_{ij} \\sim
        \\text{SS} + \\text{SASA bin} + \\text{pLDDT bin}
        + \\text{hydropathy} + \\text{chemistry} + \\text{entropy}
        + \\text{ESM-2 PC frac} + \\text{rel.\\,position}
        + (1 \\,|\\, \\text{gene}_i)
    $$

    fit on $n = 36{,}832$ residues across 70 genes.  Gene is a
    random intercept; everything else is a fixed effect.
"""))


cells.append(md("""
    ### §C.1 — Mixed-effects fixed-effect forest

    $\\beta \\pm 95\\,\\%$ CI for every fixed effect, sorted by
    $\\beta$.  Bars are solid black when $p < 0.05$, hollow grey
    otherwise.  Star annotations on the right margin report the
    Wald-test significance.  The corner badge reports model-level
    statistics: ICC, marginal R², conditional R².
"""))
cells.append(code(r"""
    c1_df = sf.render_C1_mixed_effects_forest(
        coef_df=coef_df,
        mixedlm_meta=mixedlm_meta,
        save_figure=fd.save_figure,
    )
    print(c1_df[["display", "coef", "ci_lo", "ci_hi", "p"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
"""))


cells.append(md("""
    ### §C.2 — Variance partition (also saved as `l_variance_partition.pdf`)

    For each feature, drop it from the mixed-effects model and record
    the resulting $\\Delta R^2_{\\mathrm{cond}}$.  This is the feature's
    *unique* explanatory power, with all other features held fixed.
    Bars are solid when the unique contribution is $\\geq 0.01$ %,
    hollow gray otherwise.

    The figure is saved twice: once with the new naming
    (`fig_C2_variance_partition.pdf`) and once under the legacy
    name (`l_variance_partition.pdf`) so existing paper figure
    references continue to resolve.
"""))
cells.append(code(r"""
    c2_df = sf.render_C2_variance_partition(
        varpart_df=VARPART_DF,
        save_figure=fd.save_figure,
    )
    print(c2_df[["feature", "delta_r2_pct"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
"""))


# ============ §D — DIRECTION-AWARE ============
cells.append(md("""
    ## §D — Direction-aware (M-flag / F-flag / reference)

    The novelty hook for the manuscript: is the chemistry-attention
    signature shared across the three α-correction regimes, or is it
    direction-class-specific?  Three figures: (1) direction × chemistry
    LRT-supported breakdown, (2) Δα coupling per direction, (3)
    per-direction attention magnitude distribution.
"""))


cells.append(md("""
    ### §D.1 — Direction × chemistry breakdown

    Per-direction Cliff's $\\delta$ for hydrophobic vs polar.  Bars
    are solid black when the within-direction per-gene paired
    Wilcoxon survives $p < 0.05$, hollow otherwise.  The bottom
    annotation reports the LRT $p$ for the direction × chemistry
    interaction term — the formal test of "are these three
    distributions the same?"
"""))
cells.append(code(r"""
    d1_df = sf.render_D1_direction_chemistry(
        direction_breakdown=DIRECTION_BREAKDOWN,
        p_lrt=p_lrt,
        save_figure=fd.save_figure,
    )
    print(d1_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
"""))


cells.append(md("""
    ### §D.2 — Per-gene $|\\Delta\\alpha|$ × mean $|$attention$|$, by direction

    Three faceted scatters.  $x$ = $|\\Delta\\alpha|$ per gene
    (= $|\\alpha - \\text{prior}|$), $y$ = mean per-gene
    $|$attention$|$.  Per-facet Spearman ρ + $p$ + $n_g$.  This is
    the answer to *does the GIN's residue-level attention magnitude
    track the gene-level $\\alpha$-correction signal?*  The empirical
    answer is: not really, in any direction class — the attention is
    structurally driven, not Δα-driven.
"""))
cells.append(code(r"""
    d2_df = sf.render_D2_delta_alpha_split(
        da_df=DA_DF,
        save_figure=fd.save_figure,
    )
    print(d2_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
"""))


cells.append(md("""
    ### §D.3 — Per-gene mean $|$attention$|$ by direction class

    Box plot + jittered strip of the per-gene mean
    $|$attention$|$, split by direction.  Pairwise Mann-Whitney
    annotations on the upper-right.  Reference-class housekeeping
    genes carry systematically larger attention magnitudes than
    M-flag and F-flag genes — a finding that supports the §D.2
    result that magnitude $\\neq$ Δα coupling.
"""))
cells.append(code(r"""
    d3_df = sf.render_D3_direction_attention_distribution(
        da_df=DA_DF,
        save_figure=fd.save_figure,
    )
    print(d3_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
"""))


# ============ §E — MASTER ============
cells.append(md("""
    ## §E — Master / evidence-chain summaries

    Two summary figures that fit on a single supplementary page each:
    the **evidence chain** (every contrast on one bar chart with both
    $p$-values per row) and the **negative-controls scorecard**
    (six features the GIN is *not* driven by, side-by-side
    pooled-vs-per-gene bars).
"""))


cells.append(md("""
    ### §E.1 — Evidence chain (also saved as `l_evidence_chain_summary.pdf`)

    Every contrast in `summary_contrasts.pkl` ranked by Cliff's δ.
    Bars are solid when the per-gene paired Wilcoxon survives,
    hollow otherwise.  Right-margin per-row annotation reports both
    the pooled MWU $p$ and the per-gene paired Wilcoxon $p$, so the
    reader can verify both levels at a glance.

    The figure is saved twice: once with the new naming
    (`fig_E1_evidence_chain.pdf`) and once under the legacy name
    (`l_evidence_chain_summary.pdf`) so existing paper figure
    references continue to resolve.
"""))
cells.append(code(r"""
    e1_df = sf.render_E1_evidence_chain(
        summary_contrasts=SUMMARY_CONTRASTS,
        save_figure=fd.save_figure,
    )
"""))


cells.append(md("""
    ### §E.2 — Negative-controls scorecard

    Six candidate features the GIN's attention is sometimes assumed
    to track: intrinsic disorder (IDR vs ordered), $C_{\\alpha}$
    contact-density-hub residues, aromatic FYW, glycine, proline,
    conservation entropy.  Solid bars = pooled MWU $-\\log_{10}(p)$;
    hatched bars = per-gene paired Wilcoxon $-\\log_{10}(p)$.  In every
    panel the pooled bar is tall and the per-gene bar collapses near
    or below the $p = 0.05$ line — these are between-gene composition
    artefacts.
"""))
cells.append(code(r"""
    e2_df = sf.render_E2_negative_controls(
        summary_contrasts=SUMMARY_CONTRASTS,
        save_figure=fd.save_figure,
    )
    print(e2_df[["panel", "label", "cliffs_d", "mwu_p", "pg_p"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.3e}"))
"""))


# ============ Footer ============
cells.append(md("""
    ## §F — Output inventory
"""))
cells.append(code(r"""
    families = [
        ("§A — pooled",            "fig_A*"),
        ("§B — per-gene",          "fig_B*"),
        ("§C — residue-controlled","fig_C*"),
        ("§D — direction-aware",   "fig_D*"),
        ("§E — master",            "fig_E*"),
        ("legacy backwards-compat","l_variance_partition*"),
        ("legacy backwards-compat","l_evidence_chain_summary*"),
    ]
    print(f"figures in {FIG_DIR}:\n")
    seen = set()
    for label, glob in families:
        files = sorted(p for p in FIG_DIR.glob(glob + ".pdf") if p.name not in seen)
        if not files: continue
        print(f"--- {label} ---")
        for p in files:
            print(f"  {p.name:50s} {p.stat().st_size/1024:6.1f} KB")
            seen.add(p.name)
        print()
"""))


# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (chimera_env)",
        "language": "python",
        "name": "chimera_env",
    },
    "language_info": {
        "name": "python", "pygments_lexer": "ipython3",
        "mimetype": "text/x-python", "file_extension": ".py",
    },
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH}  ({len(nb['cells'])} cells)")
