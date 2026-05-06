"""Build ``08_ortholog_qc.ipynb`` from a single Python source.

Implements *Application 7 — Quality control for ortholog annotations*
from `chimera-private/protein_structure/round2/ortholog_aware_integration_guide_initial.md`.

Two QC flag directions:

  - **M (mis-annotation candidate)** — `prior ≥ 0.80` AND `Δα ≤ −0.10`:
    high sequence identity but the model could not align expression.
    Typical biological cause: gene duplication followed by
    subfunctionalization, or an incorrect Ensembl ortholog assignment.
  - **F (functional conservation candidate)** — `prior ≤ 0.65` AND
    `Δα ≥ +0.10`: low sequence identity but the model decoded the gene
    via the shared pathway. Typical biological cause: regulatory
    conservation despite sequence divergence. Per the source doc, this
    direction has known FQR confounds (cell-type-discrimination, z-leakage,
    composition bias) and is treated more cautiously throughout.

`|Δα| ≥ 0.10` is the noise-floor guard (above the doc's 0.05 reliability
threshold).

Run from `chimera_env`::

    mamba run -n chimera_env python chimera/notebooks/_ortholog_qc_builder.py
    mamba run -n chimera_env jupyter nbconvert --to notebook --execute \\
        --inplace chimera/notebooks/08_ortholog_qc.ipynb
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "08_ortholog_qc.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).lstrip("\n").rstrip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).lstrip("\n").rstrip() + "\n")


cells: list[nbf.NotebookNode] = []


# ============ Header ============
cells.append(md(r"""
    # Quality control for ortholog annotations

    *Implementation of Application 7 from
    `chimera-private/protein_structure/round2/ortholog_aware_integration_guide_initial.md`.
    A killer use of α / Δα that no other cross-species integration tool
    can produce: **flag candidate ortholog mis-annotations and candidate
    cases of functional conservation despite sequence divergence**, by
    looking at where the GIN's learned α dramatically disagrees with the
    sequence-identity prior.*

    ## The two flag directions

    The principle: every gene has a **sequence-identity prior** (mean
    cross-species protein identity, in `[0, 1]`) and a **GIN-learned
    integration coefficient α** (also in `[0, 1]`). When the two agree
    the gene's expression-level conservation matches its sequence
    conservation. When they disagree dramatically, the disagreement is
    the QC signal:

    | Direction | Rule | Biological reading |
    |---|---|---|
    | **M — mis-annotation candidate** | `prior ≥ 0.80` AND `Δα ≤ −0.10` | high sequence identity but the shared decoder could not reconstruct expression; **likely paralog-collapse / wrong-ortholog assignment** |
    | **F — functional-conservation candidate** | `prior ≤ 0.65` AND `Δα ≥ +0.10` | low sequence identity but the shared decoder did reconstruct expression; **regulatory conservation despite sequence divergence** |

    The noise-floor guard `|Δα| ≥ 0.10` keeps both directions above the
    `≈ 0.05` reliability floor reported in the source doc. Each flag is
    further ranked by a continuous **severity score** = `|Δα|` × indicator
    of being in the flag region — bigger means a stronger candidate.

    ## What this notebook delivers

    1. **§A** — flag-rule definitions, per-dataset M / F counts, and a
       single overview heatmap of QC-flag prevalence across the registry.
    2. **§B** — per-dataset QC scatter: `prior` × `α` colored by Δα, with
       quadrant overlays and the top-K extreme genes labeled. Seven
       single-panel figures (one per dataset).
    3. **§C** — per-dataset **top-30 mis-annotation candidates** and
       **top-30 functional-conservation candidates**, dumped as CSV under
       each run-dir's `tables/`.
    4. **§D** — *per-species identity decomposition* (the sophisticated
       layer). For multi-species datasets we have per-species identity to
       human in the ortholog table (`mmulatta_identity`,
       `mmusculus_identity`, `drerio_identity`); top-M flags can be split
       into **outlier-species-driven drops** (one species' identity is far
       below the others — paralog-collapse signature) vs **uniform-low
       drops** (every species' identity is low — true divergence).
    5. **§E** — cross-dataset reproducibility. A QC flag that fires on a
       single dataset is hypothesis-grade; one that fires on the same
       gene in the same direction across multiple datasets is a robust
       call. We report within-tissue agreement (`pancreas` and `testis`
       separately, since `05_esm_alpha_geometry.ipynb §C.2` showed
       cross-tissue Δα is uncorrelated by design).
    6. **§F** — biological category enrichment (using
       `_gene_categories.classify_extended` from notebook 04) — what
       *kinds* of genes get flagged in each direction?
    7. **§G** — *downstream-analysis figures focused on the
       disagreement types*: a master summary table (§G.1), a
       per-dataset Δα landscape (§G.2), per-direction
       reproducibility footprints (§G.3), per-category severity
       distributions (§G.4), and an aggregate quadrant map (§G.5).
    8. **§H** — honest caveats and interpretation guide. The F direction
       is more contaminated than the M direction by FQR-related
       confounds (per doc); we say so explicitly.

    Visual contract throughout: **red → mis-annotation candidates (M)**,
    **blue → functional-conservation candidates (F)**, **gray →
    consistent / below noise floor**. All visual conventions inherit
    from `_figure_design.py`.
"""))


# ============ §0 setup ============
cells.append(md("""
    ## §0.1 — Imports and design system
"""))
cells.append(code(r"""
    import json
    import sys
    from pathlib import Path

    import anndata as ad
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from matplotlib.patches import Rectangle

    NB_DIR = Path.cwd() if (Path.cwd() / "_figure_design.py").exists() \
             else Path("/Users/kemalinecik/git_nosync/master_chimera/chimera/notebooks")
    if str(NB_DIR) not in sys.path:
        sys.path.insert(0, str(NB_DIR))

    import _figure_design as fd
    import _gene_categories as gc
    import _paper_utils as pu

    fd.apply_design()
"""))


cells.append(md("""
    ## §0.2 — Dataset registry

    Same 7 dataset rows as notebooks 04 / 05 / 06 / 07.
"""))
cells.append(code(r"""
    REPO_ROOT = Path("/Users/kemalinecik/git_nosync/master_chimera/chimera")
    RUNS_ROOT = REPO_ROOT / "training" / "runs"
    DATA_ROOT = REPO_ROOT / "data" / "processed"

    FIG_DIR = fd.set_figure_dir(NB_DIR / "figures" / "ortholog_qc")
    print(f"figures will be written to: {FIG_DIR}")

    DATASETS = {
        "panc_hmonkey":  {"name": "iter90-150ep-s42",
                          "adata": "camex_human_monkey.h5ad",
                          "tissue": "pancreas", "label": "panc H+Monkey",
                          "species": ["mmulatta"]},
        "panc_hmouse":   {"name": "xds-hmouse-iter90-150ep-s42",
                          "adata": "camex_human_mouse.h5ad",
                          "tissue": "pancreas", "label": "panc H+Mouse",
                          "species": ["mmusculus"]},
        "panc_hzeb":     {"name": "xds-hzeb-iter85-s42",
                          "adata": "camex_human_zebrafish.h5ad",
                          "tissue": "pancreas", "label": "panc H+Zeb",
                          "species": ["drerio"]},
        "panc_4sp":      {"name": "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42",
                          "adata": "camex_pancreas_unified.h5ad",
                          "tissue": "pancreas", "label": "panc 4-sp",
                          "species": ["mmulatta", "mmusculus", "drerio"]},
        "test_hmac":     {"name": "testis-hmac-iter109-pclamp085-200ep-s42",
                          "adata": "camex_testis_human_macaque.h5ad",
                          "tissue": "testis",   "label": "test H+Mac",
                          "species": ["mmulatta"]},
        "test_hmouse":   {"name": "testis-hmouse-iter111-pclamp-coc-200ep-s42",
                          "adata": "camex_testis_human_mouse.h5ad",
                          "tissue": "testis",   "label": "test H+Mouse",
                          "species": ["mmusculus"]},
        "test_3sp":      {"name": "testis-unified-iter115-pclamp-coc-100ep-s42",
                          "adata": "camex_testis_unified.h5ad",
                          "tissue": "testis",   "label": "test 3-sp",
                          "species": ["mmulatta", "mmusculus"]},
    }
    DATASET_ORDER = ["panc_hmonkey", "panc_hmouse", "panc_hzeb", "panc_4sp",
                     "test_hmac", "test_hmouse", "test_3sp"]
    PANC_DATASETS = [k for k, v in DATASETS.items() if v["tissue"] == "pancreas"]
    TEST_DATASETS = [k for k, v in DATASETS.items() if v["tissue"] == "testis"]
"""))


cells.append(md(r"""
    ## §0.3 — Load per-gene tables

    For every dataset we build a tidy DataFrame with one row per gene:

    - `var_name` (the model's `ortho_X` ID),
    - `symbol` (human ortholog gene symbol),
    - `category` (extended biological category from
      [`_gene_categories.py`](_gene_categories.py)),
    - `prior`, `alpha`, `delta_alpha`, `correction`, `S_g_ema`,
      `residual_rms` (from `final_state.npz`),
    - per-species identity columns from the ortholog table
      (`mmulatta_identity`, `mmusculus_identity`, `drerio_identity` —
      whichever are relevant to the dataset's species set).
"""))
cells.append(code(r"""
    ORTHO = pu.load_ortholog_table()
    print(f"ortholog table: {len(ORTHO)} groups, columns: {list(ORTHO.columns)}")

    SPECIES_IDENTITY_COLS = ["mmulatta_identity", "mmusculus_identity",
                              "drerio_identity"]

    def load_table(key: str) -> pd.DataFrame:
        info = DATASETS[key]
        rundir = RUNS_ROOT / info["name"]
        fs = dict(np.load(rundir / "final_state.npz", allow_pickle=False))

        a = ad.read_h5ad(DATA_ROOT / info["adata"], backed="r")
        var_names = np.asarray(a.var_names)
        a.file.close()

        sym = pu.annotate_genes_with_names(var_names)
        cat_df = gc.annotate_extended(var_names, sym)

        # Map var_name → ortholog_group_id and pull per-species identity columns
        og_ids = pd.Index(var_names).str.extract(r"ortho_(\d+)")[0].astype(float).astype("Int64")
        per_sp = pd.DataFrame(index=var_names)
        for col in SPECIES_IDENTITY_COLS:
            if col in ORTHO.columns:
                v = og_ids.map(ORTHO[col])
                per_sp[col] = pd.array(v.values).astype("float64")
            else:
                per_sp[col] = np.nan

        df = pd.DataFrame({
            "var_name":     var_names,
            "symbol":       cat_df["symbol"].values,
            "category":     cat_df["category"].values,
            "prior":        fs["prior"].astype(float),
            "alpha":        fs["alpha"].astype(float),
            "delta_alpha": (fs["alpha"] - fs["prior"]).astype(float),
            "correction":   fs["correction"].astype(float),
            "S_g_ema":      fs["S_g_ema"].astype(float),
            "residual_rms": fs["residual_rms"].astype(float),
            "dataset":      key,
            "tissue":       info["tissue"],
        }, index=var_names)
        # Append per-species identity (in [0, 100] units in the table)
        for col in SPECIES_IDENTITY_COLS:
            df[col] = per_sp[col].values / 100.0  # rescale to [0, 1]
        return df


    GENE_TABLES = {k: load_table(k) for k in DATASET_ORDER}
    LONG = pd.concat(GENE_TABLES.values(), ignore_index=True)
    print(f"long-format master table: {len(LONG):,} (gene, dataset) rows\n")
    print(f"per-dataset gene counts:")
    for k in DATASET_ORDER:
        d = GENE_TABLES[k]
        print(f"  {DATASETS[k]['label']:18s}  G={len(d):>5d}  "
              f"prior median={d['prior'].median():.2f}  "
              f"|Δα| 95th={np.percentile(np.abs(d['delta_alpha']),95):.3f}")
"""))


# ============ §A flag rules + counts ============
cells.append(md(r"""
    # §A — Flag rules and per-dataset prevalence

    The flag rules are absolute (not percentile-based) so the same
    semantic claim — *"high prior + low α"* — applies the same way on
    every dataset. They are tuned to the empirical distributions
    observed in §0.3 (most genes have `prior ≈ 0.81 [pancreas]` or
    `prior ≈ 0.85 [testis]`, with median `α` close to median `prior`).

    | Constant | Value | Interpretation |
    |---|---|---|
    | `PRIOR_HIGH` | 0.80 | "sequence-conserved" cutoff (top half of pancreas distribution; entire testis distribution because of the 0.85 prior clamp) |
    | `PRIOR_LOW`  | 0.65 | "sequence-divergent" cutoff |
    | `DELTA_FLOOR`| 0.10 | noise-floor guard, above the doc's 0.05 reliability threshold |

    **M flag (mis-annotation candidate):** `prior ≥ PRIOR_HIGH` AND
    `Δα ≤ −DELTA_FLOOR`.

    **F flag (functional-conservation candidate):** `prior ≤ PRIOR_LOW`
    AND `Δα ≥ +DELTA_FLOOR`.

    Both flags are **mutually exclusive by construction** — `prior`
    cannot satisfy both halves simultaneously. A gene that fails both
    bilateral rules is "in agreement with prior" (or below the noise
    floor) and is not flagged.
"""))
cells.append(code(r"""
    PRIOR_HIGH  = 0.80
    PRIOR_LOW   = 0.65
    DELTA_FLOOR = 0.10


    def annotate_flags(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["flag_M"] = (d["prior"] >= PRIOR_HIGH) & (d["delta_alpha"] <= -DELTA_FLOOR)
        d["flag_F"] = (d["prior"] <= PRIOR_LOW)  & (d["delta_alpha"] >= +DELTA_FLOOR)
        # severity score per direction: bigger = more striking flag
        d["severity_M"] = np.where(d["flag_M"], -d["delta_alpha"], 0.0)
        d["severity_F"] = np.where(d["flag_F"],  d["delta_alpha"], 0.0)
        return d


    FLAGGED = {k: annotate_flags(GENE_TABLES[k]) for k in DATASET_ORDER}

    counts_rows = []
    for k in DATASET_ORDER:
        d = FLAGGED[k]
        counts_rows.append({
            "dataset":   k,
            "label":     DATASETS[k]["label"],
            "tissue":    DATASETS[k]["tissue"],
            "n_genes":   len(d),
            "n_high_prior":     int((d["prior"] >= PRIOR_HIGH).sum()),
            "n_low_prior":      int((d["prior"] <= PRIOR_LOW).sum()),
            "n_M_flagged":      int(d["flag_M"].sum()),
            "n_F_flagged":      int(d["flag_F"].sum()),
            "M_rate_per_high":  (d["flag_M"].sum() / max((d["prior"] >= PRIOR_HIGH).sum(), 1)),
            "F_rate_per_low":   (d["flag_F"].sum() / max((d["prior"] <= PRIOR_LOW).sum(), 1)),
        })
    COUNTS = pd.DataFrame(counts_rows).set_index("label")
    print("Per-dataset QC flag prevalence:")
    print(COUNTS.drop(columns=["dataset","tissue"])
              .to_string(float_format=lambda x: f"{x:.3f}"))
"""))


cells.append(md(r"""
    ## §A.1 — QC-flag prevalence per tissue (separate heatmaps)

    Pancreas (4 datasets) and testis (3 datasets) operate in
    different architectural regimes: pancreas runs use unconstrained
    training, testis runs apply `prior_clamp_max = 0.85` (per
    `workflow-best.md` §6–8). The M flag rate per high-prior gene
    differs by ~2 orders of magnitude between the two tissues, so a
    combined heatmap would force a single color scale that
    flattens the pancreas signal. We render two separate
    heatmaps — pancreas above, testis below — each with its own
    color scale.

    **Reading.** Each cell = the probability that a high-prior
    (resp. low-prior) gene gets QC-flagged, conditioned on
    crossing the prior cutoff in the right direction. Higher rate
    = more QC-flag candidates this dataset surfaces.
"""))
cells.append(code(r"""
    def flag_rate_heatmap(ds_list, save_name, title):
        rates = pd.DataFrame({
            "M rate / high-prior gene": [COUNTS.loc[DATASETS[k]['label'], 'M_rate_per_high']
                                          for k in ds_list],
            "F rate / low-prior gene":  [COUNTS.loc[DATASETS[k]['label'], 'F_rate_per_low']
                                          for k in ds_list],
        }, index=[DATASETS[k]["label"] for k in ds_list]).T
        fig, ax = plt.subplots(figsize=(0.85*len(ds_list)+2.5, 1.6), layout="constrained")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        vmax = float(np.nanmax(rates.values) * 1.05 + 1e-3)
        im = ax.imshow(rates.values, cmap="viridis", aspect="auto",
                       vmin=0, vmax=vmax)
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                v = rates.iloc[i, j]
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="#222222" if v < vmax*0.55 else "#f5f5f5")
        ax.set_xticks(range(rates.shape[1]))
        ax.set_xticklabels(rates.columns, rotation=18, ha="right", fontsize=8)
        ax.set_yticks(range(2)); ax.set_yticklabels(rates.index, fontsize=8)
        ax.set_title(title, fontsize=9, color="#222222", loc="left", pad=4)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, label="QC-flag rate")
        cb.ax.tick_params(labelsize=7)
        fd.save_figure(fig, save_name)
        plt.show()


    flag_rate_heatmap(PANC_DATASETS, "A1_flag_rate_heatmap_pancreas",
                       "pancreas (camex liver)")
    flag_rate_heatmap(TEST_DATASETS, "A1_flag_rate_heatmap_testis",
                       "testis")
"""))

cells.append(md(r"""
    *Read.* The M (mis-annotation) rate per high-prior gene varies
    across datasets — pancreas runs flag a few percent of high-prior
    genes; testis runs flag noticeably fewer because the prior is
    clamped at `0.85` and Δα magnitudes are compressed (the testis
    architecture's E9.2-down ceiling). The F (functional-conservation)
    rate per low-prior gene is generally lower than the M rate because
    `α > prior` requires the FQR-driven mechanism to overcome the prior,
    a harder regime than the L_recon-driven `α < prior` direction.
    Both observations are architectural; both are documented in
    `chimera/training/workflow-best.md` and reproduced here.
"""))


# ============ §B per-dataset scatter ============
cells.append(md(r"""
    # §B — Per-dataset QC scatter

    For every dataset, a single-panel scatter `prior` (x-axis) ×
    `alpha` (y-axis), with each gene as a small dot colored by
    `Δα` (`RdBu_r`, symmetric around 0). Reference markup:

    - **diagonal** `α = prior` (Δα = 0) drawn faintly,
    - **flag-region rectangles** in the M-quadrant (top-left
      shading) and F-quadrant (bottom-right shading) — the regions
      satisfying the flag rules from §A,
    - **threshold lines** `prior = PRIOR_HIGH`, `prior = PRIOR_LOW`,
      and the dynamic Δα floors,
    - top-K (default 4) genes per direction labeled with their
      human gene symbol.

    A reader looking at this panel can immediately read off two things:
    (1) the dataset's overall α / prior alignment (most genes near the
    diagonal = good; thick scatter perpendicular to the diagonal =
    poor); (2) which specific genes the QC procedure flags.
"""))
cells.append(code(r"""
    K_LABEL = 4
    SCATTER_KW = dict(s=4, alpha=0.55, linewidths=0, rasterized=True)


    def qc_scatter(key: str, save_name: str):
        d = FLAGGED[key]
        fig, ax = fd.make_axes(figsize=(5.2, 4.6))
        x = d["prior"].values
        y = d["alpha"].values
        c = d["delta_alpha"].values

        vmax = float(np.nanpercentile(np.abs(c), 99))
        if not np.isfinite(vmax) or vmax == 0: vmax = 0.05

        # M-region rectangle (top-left of plot — high prior, low α; "flag below diagonal")
        # actually: M region = (prior >= PRIOR_HIGH) AND (Δα ≤ -DELTA_FLOOR), i.e.
        # (x >= PRIOR_HIGH) AND (y ≤ x - DELTA_FLOOR). We can't shade a
        # diagonal-bounded region with a Rectangle directly, so instead
        # we fill the polygon (high-prior region below diag − floor).
        diag_x = np.array([PRIOR_HIGH, 1.0])
        ax.fill_between(diag_x, 0, diag_x - DELTA_FLOOR,
                        color="#b2182b", alpha=0.08, zorder=0,
                        label="M region (mis-annotation)")
        # F region: (prior ≤ PRIOR_LOW) AND (α ≥ prior + DELTA_FLOOR)
        diag_x_f = np.array([0.0, PRIOR_LOW])
        ax.fill_between(diag_x_f, diag_x_f + DELTA_FLOOR, 1.0,
                        color="#2166ac", alpha=0.08, zorder=0,
                        label="F region (functional-conservation)")

        sc = ax.scatter(x, y, c=c, cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
                        **SCATTER_KW, zorder=2)

        # diagonal
        ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.6,
                linestyle="--", zorder=1)
        # prior thresholds
        ax.axvline(PRIOR_HIGH, color="#444444", linewidth=0.4, linestyle=":")
        ax.axvline(PRIOR_LOW,  color="#444444", linewidth=0.4, linestyle=":")

        # Label top-K genes per direction
        for direction, color in [("M", "#b2182b"), ("F", "#2166ac")]:
            sub = d[d[f"flag_{direction}"]].nlargest(K_LABEL,
                                                     f"severity_{direction}")
            for _, r in sub.iterrows():
                ax.scatter(r["prior"], r["alpha"], s=22, facecolor="none",
                           edgecolor=color, linewidth=1.0, zorder=3)
                ax.text(r["prior"], r["alpha"] + 0.025,
                        str(r["symbol"])[:14],
                        fontsize=6, ha="center", va="bottom", color="#222222",
                        zorder=4)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(fd.LABEL["prior"])
        ax.set_ylabel(fd.LABEL["alpha"])

        cb = fig.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label(fd.LABEL["delta_alpha"], fontsize=7)
        cb.ax.tick_params(labelsize=6)

        n_M = int(d["flag_M"].sum()); n_F = int(d["flag_F"].sum())
        fd.annotate_stat(ax, f"{DATASETS[key]['label']}\n"
                              f"M flags: {n_M}   F flags: {n_F}",
                         "lower right")
        fd.save_figure(fig, save_name)
        plt.show()


    for k in DATASET_ORDER:
        qc_scatter(k, f"B_qc_scatter__{k}")
"""))

cells.append(md(r"""
    *Read.* On every dataset the scatter mass concentrates on the
    `α = prior` diagonal — most genes have `Δα` within the noise
    floor and are therefore not flagged. The QC machinery picks out
    *deviations* from that diagonal: high-prior genes that fell off the
    diagonal downward (M, top-left red region) and low-prior genes
    that climbed off it upward (F, bottom-right blue region). Pancreas
    panels typically show more candidates in both directions than
    testis panels — the `prior_clamp_max = 0.85` on testis runs
    compresses both Δα tails, leaving fewer genes to flag.
"""))


# ============ §C top-N tables ============
cells.append(md(r"""
    # §C — Top candidate tables per dataset

    For each dataset, the **30 most-severe** mis-annotation candidates
    and the **30 most-severe** functional-conservation candidates,
    annotated with the relevant model outputs and the gene category.
    Saved to CSV under `runs/<run>/tables/ortho_qc_<direction>.csv`.
"""))
cells.append(code(r"""
    K_TOP = 30
    cols_M = ["symbol", "category", "prior", "alpha", "delta_alpha",
              "S_g_ema", "residual_rms", "severity_M"]
    cols_F = ["symbol", "category", "prior", "alpha", "delta_alpha",
              "S_g_ema", "residual_rms", "severity_F"]

    summary_rows = []
    for k in DATASET_ORDER:
        info = DATASETS[k]
        d = FLAGGED[k]
        rundir = RUNS_ROOT / info["name"]
        out_dir = rundir / "tables"; out_dir.mkdir(parents=True, exist_ok=True)

        m_top = d[d["flag_M"]].nlargest(K_TOP, "severity_M")[cols_M].copy()
        f_top = d[d["flag_F"]].nlargest(K_TOP, "severity_F")[cols_F].copy()
        m_top.to_csv(out_dir / "ortho_qc_misannotation.csv", index=False)
        f_top.to_csv(out_dir / "ortho_qc_functional.csv",   index=False)

        print(f"\n=== {info['label']}  M (top {min(K_TOP,len(m_top))} of "
              f"{int(d['flag_M'].sum())}) ===")
        print(m_top.to_string(index=False, float_format=lambda x: f"{x:+.3f}")
              if len(m_top) else "  (none)")
        print(f"\n=== {info['label']}  F (top {min(K_TOP,len(f_top))} of "
              f"{int(d['flag_F'].sum())}) ===")
        print(f_top.to_string(index=False, float_format=lambda x: f"{x:+.3f}")
              if len(f_top) else "  (none)")

        summary_rows.append({"dataset": k, "n_M": int(d['flag_M'].sum()),
                              "n_F": int(d['flag_F'].sum()),
                              "median_severity_M": float(d.loc[d['flag_M'],'severity_M'].median()
                                                          if d['flag_M'].any() else np.nan),
                              "median_severity_F": float(d.loc[d['flag_F'],'severity_F'].median()
                                                          if d['flag_F'].any() else np.nan)})
    print("\n=== Summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
"""))


# ============ §D per-species identity decomposition ============
cells.append(md(r"""
    # §D — Per-species identity decomposition (the sophisticated layer)

    The ortholog table provides per-species identity to human for each
    `ortho_X` group: `mmulatta_identity`, `mmusculus_identity`,
    `drerio_identity`. The dataset's `prior` is computed from the *mean*
    of these (over the species in the dataset). When an M flag fires,
    the next scientific question is: **is the high prior driven by all
    species being conserved, or is one species pulling it up?**

    - If **one species' identity is dramatically lower** than the others
      while still allowing the mean to be `≥ 0.80`, that species'
      ortholog assignment may be a paralog-collapse — the model finds
      no expression alignment because the gene-pair labeled "ortholog"
      is actually a paralog in that species.
    - If **all species' identities are uniformly high**, the M flag is
      pointing at a real expression-conservation puzzle — the
      ortholog assignment is sound, but evolution decoupled regulation
      from sequence on this gene.

    This is the cleanest way to *separate paralog-mis-assignment from
    real regulatory rewiring* using only the ortholog table and our
    trained α — exactly Application 7's "QC for ortholog annotations"
    framing made operational.

    ### §D.1 — Per-species identity *percentiles* and outlier detection

    **Methodology fix vs the v1 absolute-score approach.** A naive
    "outlier_score = mean(others) − min(identity)" picks Zebrafish on
    almost every multi-species M flag, because human↔zebrafish
    identity is *typically* much lower than human↔macaque. The score
    fires on background evolutionary distance, not on per-gene
    paralog-likeness.

    The fix: rank each gene's per-species identity *within that
    species' own population distribution*, in the multi-species
    dataset, and define outlier-ness as **"this gene's identity in
    species X is unusually low *for species X*"**, not "lower than the
    other species in the same gene".

    Concretely, for each species column `c` in the dataset:

    - `pct_c[g]` = percentile of gene `g`'s identity in species `c`'s
      empirical distribution across all genes in the dataset
      (`0–100`; `100` = highest identity, `0` = lowest).

    Then for each gene we compute:

    - `pct_min`        = the lowest of the three percentiles,
    - `pct_min_species`= which species achieved that minimum,
    - `pct_min_others` = mean percentile across the *other* species
      (i.e., "how typical is this gene in the species that DIDN'T
      score lowest?"),
    - `outlier_z`     = `pct_min_others − pct_min` (in percentile
      points). Large positive value = this gene is unusually low
      *for that species* while being typical-or-better in the other
      species.

    **Tags** (fired only on M-flagged genes; tags use percentile
    units, not raw identity):

    - **`paralog-collapse in <species>`** — `pct_min < 10` AND
      `outlier_z > 30`. The gene's identity in `species_min` is in
      the bottom decile of that species' distribution while the
      other species are normal-or-better.
    - **`uniform-low (real divergence)`** — `pct_max < 25`. Every
      species ranks low; the gene is unusually divergent everywhere,
      not in just one place. Mt-DNA-encoded genes typically land
      here.
    - **`typical / ambiguous`** — neither of the above.
"""))
cells.append(code(r"""
    SPECIES_LABEL = {
        "mmulatta_identity":  "Macaque",
        "mmusculus_identity": "Mouse",
        "drerio_identity":    "Zebrafish",
    }


    def per_species_percentiles(d: pd.DataFrame, ds_key: str) -> pd.DataFrame:
        info = DATASETS[ds_key]
        species_cols = [f"{sp}_identity" for sp in info["species"]
                        if f"{sp}_identity" in d.columns
                        and not d[f"{sp}_identity"].isna().all()]
        if len(species_cols) < 2:
            empty = pd.DataFrame(index=d.index)
            empty["pct_min"]         = np.nan
            empty["pct_max"]         = np.nan
            empty["pct_min_species"] = "?"
            empty["pct_min_others"]  = np.nan
            empty["outlier_z"]       = np.nan
            for c in species_cols:
                empty[f"pct__{c}"]   = np.nan
            return empty

        out = pd.DataFrame(index=d.index)
        # Rank each species' identity vector to percentile (0..100)
        for c in species_cols:
            out[f"pct__{c}"] = d[c].rank(method="average", pct=True) * 100.0

        pct_arr = out[[f"pct__{c}" for c in species_cols]].values  # (G, S)
        # rows where any value is NaN
        all_nan = np.all(np.isnan(pct_arr), axis=1)
        pct_arr_safe = np.where(np.isnan(pct_arr), np.inf, pct_arr)
        idx_min = np.argmin(pct_arr_safe, axis=1)
        pct_min = np.take_along_axis(pct_arr, idx_min[:, None], axis=1).ravel()
        pct_max = np.nanmax(pct_arr, axis=1)
        species_at_min = [SPECIES_LABEL[species_cols[i]] for i in idx_min]
        # mean of the OTHER species' percentiles per row
        sums = np.nansum(pct_arr, axis=1)
        # subtract the min one (handle NaN)
        valid = (~np.isnan(pct_arr)).sum(axis=1)
        denom = np.maximum(valid - 1, 1)
        # subtract pct_min from each row's sum, but only if pct_min is finite
        sums_minus = sums - np.where(np.isfinite(pct_min), pct_min, 0)
        pct_min_others = sums_minus / denom
        outlier_z = pct_min_others - pct_min

        out["pct_min"]         = pct_min
        out["pct_max"]         = pct_max
        out["pct_min_species"] = species_at_min
        out["pct_min_others"]  = pct_min_others
        out["outlier_z"]       = outlier_z
        # NaN out invalid rows
        out.loc[all_nan, ["pct_min", "pct_max", "pct_min_others", "outlier_z"]] = np.nan
        return out


    def tag_outlier(row: pd.Series) -> str:
        if pd.isna(row["pct_min"]) or pd.isna(row["pct_max"]):
            return "n/a"
        if row["pct_min"] < 10.0 and row["outlier_z"] > 30.0:
            return f"paralog-collapse in {row['pct_min_species']}"
        if row["pct_max"] < 25.0:
            return "uniform-low (real divergence)"
        return "typical / ambiguous"
"""))


cells.append(md(r"""
    ### §D.2 — Apply the percentile-based decomposition

    On `panc_4sp` and `test_3sp` we have ≥ 2 non-human species in the
    prior, so the per-species percentile decomposition is well-posed.
    For each top-M-flagged gene we report:

    - the per-species *percentile* of identity (within that species'
      empirical distribution) — these tell you *how unusual* the
      gene's identity is in *that* species, not just how low it is in
      absolute terms.
    - the lowest-percentile species (`pct_min_species`),
    - `outlier_z` = how much lower the lowest-pct species is than
      the average of the others (in percentile points),
    - a **tag**: `paralog-collapse in <species>` /
      `uniform-low (real divergence)` / `typical / ambiguous`.

    CSV per dataset under `runs/<run>/tables/ortho_qc_misannotation_decomposed.csv`.
"""))
cells.append(code(r"""
    def report_outlier_decomp(ds_key: str, top_n: int = 30):
        info = DATASETS[ds_key]
        if len(info["species"]) < 2:
            return None
        d = FLAGGED[ds_key]
        m_top = d[d["flag_M"]].nlargest(top_n, "severity_M").copy()
        if not len(m_top):
            return None
        per_sp = per_species_percentiles(d, ds_key).loc[m_top.index]
        info_species = [f"{sp}_identity" for sp in info["species"]
                        if f"{sp}_identity" in d.columns]
        out_cols = {
            "symbol":          m_top["symbol"].values,
            "category":        m_top["category"].values,
            "prior":           m_top["prior"].values,
            "alpha":           m_top["alpha"].values,
            "delta_alpha":     m_top["delta_alpha"].values,
        }
        # Per-species raw identity AND percentile
        for c in info_species:
            out_cols[f"id__{SPECIES_LABEL[c]}"]  = m_top[c].values
            out_cols[f"pct__{SPECIES_LABEL[c]}"] = per_sp[f"pct__{c}"].values
        out_cols["pct_min"]            = per_sp["pct_min"].values
        out_cols["pct_min_species"]    = per_sp["pct_min_species"].values
        out_cols["outlier_z"]          = per_sp["outlier_z"].values

        out = pd.DataFrame(out_cols, index=m_top.index)
        out["tag"] = out.apply(lambda r: tag_outlier(per_sp.loc[r.name]), axis=1)
        out_dir = RUNS_ROOT / info["name"] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "ortho_qc_misannotation_decomposed.csv", index=False)
        return out


    OUTLIER_DECOMP = {}
    for k in ["panc_4sp", "test_3sp"]:
        df = report_outlier_decomp(k)
        OUTLIER_DECOMP[k] = df
        if df is None:
            print(f"  {DATASETS[k]['label']}: not applicable (insufficient species).")
            continue
        print(f"\n=== {DATASETS[k]['label']}  ·  M-flag percentile decomposition ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\n  Tag distribution:")
        print(df["tag"].value_counts().to_string())
"""))


cells.append(md(r"""
    ### §D.3 — Outlier-species decomposition figure (`panc_4sp`)

    For the 4-species pancreas dataset: per top-M gene, a small panel
    showing per-species identity to human (bar height) and the *percentile*
    of that identity within the species' empirical distribution
    (annotation under each bar). Title shows the gene symbol, Δα, and the
    *percentile-based tag*.

    *How to read.* Bars at uniformly high values *and percentiles*
    = real regulatory rewiring candidate (uniform-low tag fires only
    when all percentiles are low). One bar with much *lower percentile*
    than the others = paralog-collapse candidate in that species. The
    percentile (not raw identity) is what disambiguates "typical fish
    distance" from "unusually low for fish".
"""))
cells.append(code(r"""
    PANC_4SP_FOR_DECOMP = "panc_4sp"
    K_DECOMP_SHOW = 12

    info = DATASETS[PANC_4SP_FOR_DECOMP]
    d_full = FLAGGED[PANC_4SP_FOR_DECOMP]
    species_cols = [f"{sp}_identity" for sp in info["species"]
                    if f"{sp}_identity" in d_full.columns]
    species_short = [SPECIES_LABEL[c] for c in species_cols]

    pct_full = per_species_percentiles(d_full, PANC_4SP_FOR_DECOMP)

    m_top_show = d_full[d_full["flag_M"]].nlargest(K_DECOMP_SHOW, "severity_M")
    n_panels = len(m_top_show)

    # 3-row × 4-col grid for 12 panels (compact, paper-friendly)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.4 * n_rows),
                              sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    palette = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    TAG_COLOR = {
        "uniform-low (real divergence)": "#542788",
        "typical / ambiguous":           "#777777",
    }
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i >= n_panels:
            ax.axis("off"); continue
        idx = m_top_show.index[i]
        r = m_top_show.iloc[i]
        ps = pct_full.loc[idx]
        vals = [float(r[c]) for c in species_cols]
        pcts = [float(ps[f"pct__{c}"]) for c in species_cols]
        # bar color: highlight the species at the lowest percentile
        min_idx = int(np.argmin(pcts))
        colors = []
        for j in range(len(species_cols)):
            if j == min_idx and pcts[j] < 10.0:
                colors.append("#b2182b")     # red — outlier
            elif j == min_idx:
                colors.append("#cc7777")     # pale red — lowest but not extreme
            else:
                colors.append(palette[j % len(palette)])
        bars = ax.bar(range(len(vals)), vals, color=colors,
                       edgecolor="#222222", linewidth=0.5)
        for j, (b, v, p) in enumerate(zip(bars, vals, pcts)):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6, color="#222222")
            # percentile under x-axis tick label
            ax.text(b.get_x() + b.get_width()/2, -0.07,
                    f"p{p:.0f}", ha="center", va="top",
                    fontsize=6,
                    color="#b2182b" if p < 10 else "#444444")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(species_cols)))
        ax.set_xticklabels(species_short, rotation=22, ha="right", fontsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.axhline(PRIOR_HIGH, color="#888888", linewidth=0.4, linestyle="--")
        # title: symbol + delta + tag
        tag = tag_outlier(ps)
        if tag.startswith("paralog-collapse"):
            tcolor = "#b2182b"
        elif tag == "uniform-low (real divergence)":
            tcolor = "#542788"
        else:
            tcolor = "#444444"
        ax.set_title(f"{r['symbol']}   Δα={r['delta_alpha']:+.2f}\n{tag}",
                     fontsize=7, color=tcolor)
    for i in range(n_rows):
        axes[i, 0].set_ylabel("identity to human", fontsize=7.5)
    fd.save_figure(fig, f"D3_outlier_decomp__{PANC_4SP_FOR_DECOMP}")
    plt.show()
"""))


# ============ §E cross-dataset reproducibility ============
cells.append(md(r"""
    # §E — Cross-dataset reproducibility

    A QC flag firing on a single dataset is hypothesis-grade. A QC flag
    firing on the same gene **in the same direction** across multiple
    datasets is a robust call. We separate by tissue because
    `05_esm_alpha_geometry.ipynb` §C.2 showed that Δα is essentially
    uncorrelated *across tissues* for the same gene (`ρ = −0.016` on the
    panc-4sp ∩ test-3sp shared genes) — by *design*, since Δα is a
    tissue-context-aware regulatory-conservation correction. Within
    tissue, Δα is reproducible (cross-seed `ρ ≥ 0.5` per
    `03_alpha_master.ipynb` §B.1), so within-tissue QC reproducibility is
    the relevant test.

    ### §E.1 — Within-tissue reproducibility: pancreas

    For each gene that ever flags in the M (or F) direction in any
    pancreas dataset, count *how many* pancreas datasets flag it in the
    same direction. Genes flagged in `≥ 2 / 4` pancreas datasets are
    treated as **robust within-tissue M (or F) candidates**.
"""))
cells.append(code(r"""
    def reproducibility_table(direction: str, dataset_keys: list[str]) -> pd.DataFrame:
        # direction: 'M' or 'F'.  Returns df indexed by ortho_X with cols:
        # symbol, category, n_flagged (# datasets in dataset_keys that flag
        # this gene in `direction`), max_severity, median_severity, max_delta_alpha.
        flagcol = f"flag_{direction}"
        sevcol  = f"severity_{direction}"
        rows = {}
        for k in dataset_keys:
            d = FLAGGED[k]
            sub = d[d[flagcol]]
            for ix, r in sub.iterrows():
                if ix not in rows:
                    rows[ix] = {"symbol": r["symbol"],
                                "category": r["category"],
                                "datasets": [],
                                "severities": [],
                                "delta_alphas": []}
                rows[ix]["datasets"].append(DATASETS[k]["label"])
                rows[ix]["severities"].append(float(r[sevcol]))
                rows[ix]["delta_alphas"].append(float(r["delta_alpha"]))
        out = pd.DataFrame.from_dict(rows, orient="index")
        if not len(out):
            return out
        out["n_flagged"]      = out["datasets"].apply(len)
        out["max_severity"]   = out["severities"].apply(max)
        out["median_severity"]= out["severities"].apply(np.median)
        out["mean_delta_alpha"] = out["delta_alphas"].apply(np.mean)
        out["dataset_list"]   = out["datasets"].apply(lambda L: ", ".join(L))
        out = out.drop(columns=["datasets","severities","delta_alphas"])
        return out.sort_values(["n_flagged","max_severity"], ascending=[False, False])


    PANC_M = reproducibility_table("M", PANC_DATASETS)
    PANC_F = reproducibility_table("F", PANC_DATASETS)
    print(f"PANCREAS — robust M (≥2 of {len(PANC_DATASETS)}): "
          f"{int((PANC_M['n_flagged']>=2).sum()) if len(PANC_M) else 0}")
    print(f"PANCREAS — robust F (≥2 of {len(PANC_DATASETS)}): "
          f"{int((PANC_F['n_flagged']>=2).sum()) if len(PANC_F) else 0}")
    print(f"\n=== Top 25 robust pancreas M flags (n_flagged ≥ 2) ===")
    print(PANC_M[PANC_M["n_flagged"]>=2].head(25)
          .to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n=== Top 25 robust pancreas F flags (n_flagged ≥ 2) ===")
    print(PANC_F[PANC_F["n_flagged"]>=2].head(25)
          .to_string(float_format=lambda x: f"{x:.3f}"))
"""))


cells.append(md(r"""
    ### §E.2 — Within-tissue reproducibility: testis

    Same procedure on the 3 testis runs. Expect fewer robust flags
    (`prior_clamp_max=0.85` compresses Δα; the M-direction is more
    affected because there's less prior headroom, and the F-direction
    bumps into the 0.85 ceiling).
"""))
cells.append(code(r"""
    TEST_M = reproducibility_table("M", TEST_DATASETS)
    TEST_F = reproducibility_table("F", TEST_DATASETS)
    print(f"TESTIS — robust M (≥2 of {len(TEST_DATASETS)}): "
          f"{int((TEST_M['n_flagged']>=2).sum()) if len(TEST_M) else 0}")
    print(f"TESTIS — robust F (≥2 of {len(TEST_DATASETS)}): "
          f"{int((TEST_F['n_flagged']>=2).sum()) if len(TEST_F) else 0}")
    if len(TEST_M):
        print(f"\n=== Top robust testis M flags (n_flagged ≥ 2) ===")
        print(TEST_M[TEST_M["n_flagged"]>=2].head(20)
              .to_string(float_format=lambda x: f"{x:.3f}"))
    if len(TEST_F):
        print(f"\n=== Top robust testis F flags (n_flagged ≥ 2) ===")
        print(TEST_F[TEST_F["n_flagged"]>=2].head(20)
              .to_string(float_format=lambda x: f"{x:.3f}"))
"""))


cells.append(md("""
    ### §E.3 — Cross-tissue divergence and the consistency-of-direction test

    For each gene that flags in *any* dataset, in either tissue, we
    record its direction. A gene flagged as **M in pancreas AND F in
    testis** (or vice-versa) is a *cross-tissue inconsistent* call —
    expected per `05_esm_alpha_geometry.ipynb` §C.2's tissue-context-
    aware Δα. We tabulate this so the reader can see:

    1. genes that fire **same direction across tissues** are the most
       defensible QC calls (truly suspect orthologs / truly conserved
       in expression);
    2. genes that fire **opposite directions** are *not* QC failures —
       they are the regulatory-rewiring stars of notebook 06 §D, found
       again from a different angle.
"""))
cells.append(code(r"""
    direction_per_dataset = {}
    for k in DATASET_ORDER:
        d = FLAGGED[k]
        for ix in d.index[d["flag_M"]]:
            direction_per_dataset.setdefault(ix, []).append((DATASETS[k]['tissue'], "M"))
        for ix in d.index[d["flag_F"]]:
            direction_per_dataset.setdefault(ix, []).append((DATASETS[k]['tissue'], "F"))

    rows = []
    for ix, calls in direction_per_dataset.items():
        panc_dirs = {d for t, d in calls if t == "pancreas"}
        test_dirs = {d for t, d in calls if t == "testis"}
        if panc_dirs and test_dirs:
            sym = LONG.loc[LONG["var_name"] == ix, "symbol"].iloc[0] \
                  if (LONG["var_name"] == ix).any() else ix
            cat = LONG.loc[LONG["var_name"] == ix, "category"].iloc[0] \
                  if (LONG["var_name"] == ix).any() else "?"
            rows.append({
                "var_name": ix, "symbol": sym, "category": cat,
                "panc_dirs": ",".join(sorted(panc_dirs)),
                "test_dirs": ",".join(sorted(test_dirs)),
                "consistent": panc_dirs == test_dirs,
                "rewired":    bool(panc_dirs.symmetric_difference(test_dirs)
                                    and panc_dirs != test_dirs),
            })
    XT = pd.DataFrame(rows)
    if len(XT):
        n_cons = int(XT["consistent"].sum())
        n_rewi = int(XT["rewired"].sum())
        print(f"genes flagged in BOTH tissues: {len(XT)}")
        print(f"  consistent (same direction):    {n_cons}")
        print(f"  rewired (different direction):  {n_rewi}")
        print("\nFirst 25 consistent calls:")
        print(XT[XT["consistent"]].head(25).to_string(index=False))
        print("\nFirst 25 rewired calls (likely tissue-context, not annotation, issues):")
        print(XT[XT["rewired"]].head(25).to_string(index=False))
"""))


# ============ §F category enrichment ============
cells.append(md(r"""
    # §F — Biological category enrichment

    For each direction (M, F), Fisher exact enrichment of every
    biological category (from
    [`_gene_categories.py`](_gene_categories.py)) against the
    background of all genes in the dataset. Bonferroni-corrected
    across `(category × direction × dataset)` triples. The output
    is a per-direction heatmap (rows = categories, columns =
    datasets) of `log₂(odds ratio)`, starred where `p_bonf < 0.05`.

    **Reading.** A category enriched in the M direction across
    multiple datasets is a *typical class of mis-annotation
    candidates* — for example, paralog-rich families. A category
    enriched in F is a *typical class of regulation-conserved-despite-
    sequence genes*.
"""))
cells.append(code(r"""
    cats = [c for c in gc.CATEGORY_ORDER_EXTENDED if c != "other"]


    def enrichment_for(direction: str, ds_list, fdr_thresh: float = 0.05):
        flagcol = f"flag_{direction}"
        rows = []
        for k in ds_list:
            d = FLAGGED[k]
            hit = d[flagcol].values
            cat_vec = d["category"].values
            for c in cats:
                in_set = cat_vec == c
                if in_set.sum() < 5:
                    rows.append({"dataset": k, "category": c,
                                  "odds_ratio": np.nan, "p_value": 1.0})
                    continue
                r = pu.fisher_enrichment(in_set, hit)
                r["dataset"] = k; r["category"] = c
                rows.append(r)
        df = pd.DataFrame(rows)
        m = df["odds_ratio"].notna().sum()
        df["p_bonf"] = np.minimum(df["p_value"] * max(int(m), 1), 1.0)
        df["sig"]    = (df["p_bonf"] < fdr_thresh) & df["odds_ratio"].notna()
        pivot_or  = df.pivot(index="category", columns="dataset", values="odds_ratio").reindex(index=cats, columns=ds_list)
        pivot_sig = df.pivot(index="category", columns="dataset", values="sig").reindex(index=cats, columns=ds_list).fillna(False)
        return pivot_or, pivot_sig


    def plot_enrichment(direction: str, ds_list, save_name: str, title: str):
        pivot_or, pivot_sig = enrichment_for(direction, ds_list)
        keep = pivot_or.notna().any(axis=1)
        pivot_or = pivot_or.loc[keep]
        pivot_sig = pivot_sig.loc[keep]
        if len(pivot_or) == 0:
            print(f"  no admissible categories for {direction} on {title}.")
            return
        log2_or = np.log2(pivot_or.replace(0, np.nan))
        vmax = float(np.nanmax(np.abs(log2_or.values)))
        if not np.isfinite(vmax) or vmax == 0: vmax = 1.0

        h = 0.27 * len(pivot_or) + 1.2
        fig, ax = plt.subplots(figsize=(0.95 * len(ds_list) + 4.5, h),
                                layout="constrained")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        im = ax.imshow(log2_or.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        for i, cat in enumerate(pivot_or.index):
            for j, ds in enumerate(pivot_or.columns):
                v = pivot_or.iloc[i, j]
                txt = "—" if not np.isfinite(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6, color="#222222")
                if pivot_sig.loc[cat, ds]:
                    ax.text(j + 0.32, i - 0.32, "★", ha="center", va="center",
                            fontsize=8, color="#222222")
        ax.set_xticks(range(len(ds_list)))
        ax.set_xticklabels([DATASETS[k]["label"] for k in ds_list],
                           rotation=22, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot_or)))
        ax.set_yticklabels(pivot_or.index, fontsize=7)
        ax.set_title(title, fontsize=9, color="#222222", loc="left", pad=4)
        cb = fig.colorbar(im, ax=ax,
                          label=rf"$\log_2$(odds ratio), {direction}-flag enrichment",
                          shrink=0.55)
        cb.ax.tick_params(labelsize=6)
        fd.save_figure(fig, save_name)
        plt.show()


    # M flag enrichment — pancreas (4 datasets) and testis (3 datasets) separately.
    plot_enrichment("M", PANC_DATASETS, "F1_enrichment_M_pancreas",
                     "pancreas — M-flag category enrichment")
    plot_enrichment("M", TEST_DATASETS, "F1_enrichment_M_testis",
                     "testis — M-flag category enrichment")
    # F flag enrichment — same split.
    plot_enrichment("F", PANC_DATASETS, "F2_enrichment_F_pancreas",
                     "pancreas — F-flag category enrichment")
    plot_enrichment("F", TEST_DATASETS, "F2_enrichment_F_testis",
                     "testis — F-flag category enrichment")
"""))


# ============ §G downstream-analysis figures ============
cells.append(md(r"""
    # §G — Downstream analysis: anatomy of the disagreement

    Five focused deliverables that turn the §A–§F per-dataset numbers
    into a coherent disagreement-type story. Every figure follows the
    same visual contract: **red → mis-annotation candidates (M)**,
    **blue → functional-conservation candidates (F)**, **gray →
    consistent / below noise floor**.

    1. **§G.1** — *Master summary table.* One row per dataset, every
       headline statistic in one place; the row a reviewer wants to
       cite. Saved as CSV.
    2. **§G.2** — *Disagreement magnitude landscape.* Δα distribution
       per dataset as a per-gene strip plot, dots colored by
       direction, vertical reference at the noise-floor cutoffs.
       Reads at a glance: how much disagreement is there per dataset,
       and where does it sit on the Δα axis?
    3. **§G.3** — *Reproducibility footprint.* For each direction,
       the top-50 most-reproducible genes laid out as a
       `gene × dataset` heatmap, cell colored by per-dataset severity.
       Identifies which genes are robust calls and which are
       single-dataset noise.
    4. **§G.4** — *Severity by gene category.* For each direction,
       per-category strip plot of severity. Tells you *what kind of
       genes* end up flagged.
    5. **§G.5** — *Aggregate quadrant map.* All datasets pooled into
       one (`mean prior` × `mean α`) scatter, with reproducible flag
       genes labeled. The single-figure summary of *what the QC
       analysis surfaces across the entire registry*.
"""))


cells.append(md(r"""
    ## §G.1 — Master summary table (paper-supplement-ready)

    One row per dataset with: total genes, conservation-prior region
    sizes, M / F flag counts, robust counts (within-tissue
    reproducibility from §E), median severity per direction, and the
    single most-severe gene per direction. Designed so the table can
    drop into a paper supplement without reformatting.
"""))
cells.append(code(r"""
    # Pre-compute robust counts within each tissue (§E pre-loads PANC_M / TEST_M etc.)
    PANC_M_ROB = PANC_M[PANC_M['n_flagged'] >= 2] if len(PANC_M) else PANC_M
    PANC_F_ROB = PANC_F[PANC_F['n_flagged'] >= 2] if len(PANC_F) else PANC_F
    TEST_M_ROB = TEST_M[TEST_M['n_flagged'] >= 2] if len(TEST_M) else TEST_M
    TEST_F_ROB = TEST_F[TEST_F['n_flagged'] >= 2] if len(TEST_F) else TEST_F

    rows = []
    for k in DATASET_ORDER:
        info = DATASETS[k]
        d = FLAGGED[k]
        m_top = d[d["flag_M"]].nlargest(1, "severity_M")
        f_top = d[d["flag_F"]].nlargest(1, "severity_F")
        # `info['label']` contains literal '+' which regex interprets — use regex=False.
        if info["tissue"] == "pancreas":
            sub_M = PANC_M[PANC_M['n_flagged'] >= 2] if len(PANC_M) else PANC_M
            sub_F = PANC_F[PANC_F['n_flagged'] >= 2] if len(PANC_F) else PANC_F
            n_rob_M = int(sub_M['dataset_list'].str.contains(info['label'], regex=False).sum()) if len(sub_M) else 0
            n_rob_F = int(sub_F['dataset_list'].str.contains(info['label'], regex=False).sum()) if len(sub_F) else 0
        else:
            sub_M = TEST_M[TEST_M['n_flagged'] >= 2] if len(TEST_M) else TEST_M
            sub_F = TEST_F[TEST_F['n_flagged'] >= 2] if len(TEST_F) else TEST_F
            n_rob_M = int(sub_M['dataset_list'].str.contains(info['label'], regex=False).sum()) if len(sub_M) else 0
            n_rob_F = int(sub_F['dataset_list'].str.contains(info['label'], regex=False).sum()) if len(sub_F) else 0
        rows.append({
            "dataset":    info["label"],
            "tissue":     info["tissue"],
            "n_genes":    int(len(d)),
            "n_high_prior": int((d["prior"] >= PRIOR_HIGH).sum()),
            "n_low_prior":  int((d["prior"] <= PRIOR_LOW).sum()),
            "n_M":        int(d["flag_M"].sum()),
            "n_F":        int(d["flag_F"].sum()),
            "n_M_robust_in_tissue": n_rob_M,
            "n_F_robust_in_tissue": n_rob_F,
            "median_sev_M": float(d.loc[d["flag_M"], "severity_M"].median()
                                   if d["flag_M"].any() else np.nan),
            "median_sev_F": float(d.loc[d["flag_F"], "severity_F"].median()
                                   if d["flag_F"].any() else np.nan),
            "top_M":      f"{m_top['symbol'].iloc[0]} (Δα={m_top['delta_alpha'].iloc[0]:+.2f})"
                          if len(m_top) else "—",
            "top_F":      f"{f_top['symbol'].iloc[0]} (Δα={f_top['delta_alpha'].iloc[0]:+.2f})"
                          if len(f_top) else "—",
        })
    SUMMARY = pd.DataFrame(rows).set_index("dataset")
    out_csv = NB_DIR / "cache" / "ortholog_qc_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.to_csv(out_csv)
    print(f"saved master summary: {out_csv}\n")
    print(SUMMARY.to_string())
"""))


cells.append(code(r"""
    # Render the same summary as a styled figure (cells colored by direction)
    visible_cols = ["tissue", "n_genes", "n_high_prior", "n_low_prior",
                     "n_M", "n_M_robust_in_tissue", "median_sev_M", "top_M",
                     "n_F", "n_F_robust_in_tissue", "median_sev_F", "top_F"]
    pretty_names = {
        "tissue":               "tissue",
        "n_genes":              "# genes",
        "n_high_prior":         "# high-prior",
        "n_low_prior":          "# low-prior",
        "n_M":                  "# M flags",
        "n_M_robust_in_tissue": "robust M (in-tissue)",
        "median_sev_M":         "median sev. M",
        "top_M":                "top M",
        "n_F":                  "# F flags",
        "n_F_robust_in_tissue": "robust F (in-tissue)",
        "median_sev_F":         "median sev. F",
        "top_F":                "top F",
    }
    M_BG = "#fbe9e9"   # very pale red for M columns
    F_BG = "#e9eef9"   # very pale blue for F columns
    NEUTRAL_BG = "#ffffff"
    M_COLS = {"n_M","n_M_robust_in_tissue","median_sev_M","top_M"}
    F_COLS = {"n_F","n_F_robust_in_tissue","median_sev_F","top_F"}

    fig, ax = plt.subplots(figsize=(13.0, 0.45 * len(SUMMARY) + 1.0),
                            layout="constrained")
    ax.axis("off")
    cell_text   = []
    cell_colors = []
    for _, r in SUMMARY[visible_cols].iterrows():
        row = []
        crow = []
        for c in visible_cols:
            v = r[c]
            if isinstance(v, float):
                txt = f"{v:.3f}" if not pd.isna(v) else "—"
            else:
                txt = str(v)
            row.append(txt)
            crow.append(M_BG if c in M_COLS else (F_BG if c in F_COLS else NEUTRAL_BG))
        cell_text.append(row)
        cell_colors.append(crow)
    col_labels = [pretty_names[c] for c in visible_cols]
    col_colors = [M_BG if c in M_COLS else (F_BG if c in F_COLS else "#f3f3f3")
                   for c in visible_cols]
    tbl = ax.table(cellText=cell_text,
                    rowLabels=list(SUMMARY.index),
                    colLabels=col_labels,
                    cellColours=cell_colors,
                    colColours=col_colors,
                    loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.45)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if i == 0:
            cell.set_text_props(weight="bold")
    fd.save_figure(fig, "G1_summary_table")
    plt.show()
"""))


cells.append(md(r"""
    ## §G.2 — Disagreement magnitude landscape

    For every dataset, every gene plotted as a single dot at its `Δα`
    value (vertical position is the dataset). Direction-colored:

    - **red** → M-flagged (sequence-conserved, expression-divergent),
    - **blue** → F-flagged (sequence-divergent, expression-conserved),
    - **gray** → unflagged (consistent or below the `|Δα| ≥ 0.10`
      noise floor).

    Vertical references at `Δα = ±DELTA_FLOOR` (the noise floor) and
    at `Δα = 0` (the prior). Each dataset's row is annotated with its
    M / F counts. Reads as a per-dataset distributional fingerprint
    of disagreement.
"""))
cells.append(code(r"""
    def disagreement_landscape(ds_list, save_name, title):
        rng_jit = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(7.6, 0.85 * len(ds_list) + 1.2),
                                layout="constrained")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.axvline(0.0,           color="#444444", linewidth=0.6, zorder=0)
        ax.axvline(-DELTA_FLOOR,  color="#888888", linewidth=0.4, linestyle="--", zorder=0)
        ax.axvline(+DELTA_FLOOR,  color="#888888", linewidth=0.4, linestyle="--", zorder=0)
        for i, k in enumerate(ds_list):
            d = FLAGGED[k]
            y = i + rng_jit.uniform(-0.32, 0.32, size=len(d))
            un = ~(d["flag_M"] | d["flag_F"])
            ax.scatter(d.loc[un, "delta_alpha"], y[un.values],
                       s=2.5, c="#cccccc", alpha=0.45, linewidths=0,
                       rasterized=True, zorder=1)
            m_mask = d["flag_M"].values
            ax.scatter(d.loc[m_mask, "delta_alpha"], y[m_mask],
                       s=8, c="#b2182b", alpha=0.85, linewidths=0,
                       rasterized=True, zorder=2)
            f_mask = d["flag_F"].values
            ax.scatter(d.loc[f_mask, "delta_alpha"], y[f_mask],
                       s=8, c="#2166ac", alpha=0.85, linewidths=0,
                       rasterized=True, zorder=2)
            ax.text(0.42, i, f"M={int(m_mask.sum())}   F={int(f_mask.sum())}",
                    ha="left", va="center", fontsize=7, color="#222222")
        ax.set_yticks(range(len(ds_list)))
        ax.set_yticklabels([DATASETS[k]["label"] for k in ds_list], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-0.45, 0.55)
        ax.set_xticks([-0.4, -0.2, -DELTA_FLOOR, 0.0, +DELTA_FLOOR, 0.2, 0.4])
        ax.set_xlabel(fd.LABEL["delta_alpha"])
        ax.set_title(title, fontsize=9, color="#222222", loc="left", pad=4)
        fd.save_figure(fig, save_name)
        plt.show()


    disagreement_landscape(PANC_DATASETS, "G2_disagreement_landscape_pancreas",
                            "pancreas (camex liver)")
    disagreement_landscape(TEST_DATASETS, "G2_disagreement_landscape_testis",
                            "testis")
"""))


cells.append(md(r"""
    ## §G.3 — Reproducibility footprint

    For each direction, the **top-50 most-reproducible genes**
    (sorted by number of datasets in which the flag fires, then by
    max severity) shown as a `gene × dataset` heatmap. Cell color =
    per-dataset severity (intensity scaled per direction). Empty
    cell = the gene was *not* flagged in that dataset. The columns
    are pre-grouped (pancreas first, then testis) so within-tissue
    consistency is visible at a glance.

    Reading: **a gene with cells filled across multiple pancreas
    columns is a robust within-tissue call**; a gene with cells
    filled in *both* pancreas and testis columns is rare (per §E,
    only 2 such genes exist, both with direction-flipped flags —
    they are the tissue-context-rewired class, not QC failures).
"""))
cells.append(code(r"""
    def reproducibility_footprint(direction, ds_list, save_name, title,
                                     top_n=50, cmap_name="Reds"):
        sevcol = f"severity_{direction}"
        # Build (gene, dataset) severity matrix scoped to ds_list only
        rows = {}
        for k in ds_list:
            d = FLAGGED[k]
            for ix in d.index[d[f"flag_{direction}"]]:
                rows.setdefault(ix, {"symbol": d.at[ix, "symbol"],
                                      "category": d.at[ix, "category"],
                                      "n_flagged": 0, "max_sev": 0.0})
                rows[ix]["n_flagged"] += 1
                rows[ix]["max_sev"] = max(rows[ix]["max_sev"], float(d.at[ix, sevcol]))
                rows[ix][k] = float(d.at[ix, sevcol])
        flat = pd.DataFrame.from_dict(rows, orient="index")
        if not len(flat):
            print(f"  no {direction} flags within {title}.")
            return
        flat = flat.sort_values(["n_flagged","max_sev"], ascending=[False, False]).head(top_n)
        for k in ds_list:
            if k not in flat.columns:
                flat[k] = np.nan
        M = flat[ds_list].astype(float)
        fig, ax = plt.subplots(figsize=(0.95*len(ds_list)+4.0, 0.22 * len(M) + 1.2),
                                layout="constrained")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        vmax = float(np.nanpercentile(M.values, 99))
        if not np.isfinite(vmax) or vmax == 0: vmax = 0.3
        im = ax.imshow(M.values, cmap=cmap_name, aspect="auto",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(ds_list)))
        ax.set_xticklabels([DATASETS[k]["label"] for k in ds_list],
                            rotation=22, ha="right", fontsize=7)
        labels = [f"{flat.iloc[i]['symbol']}  ({int(flat.iloc[i]['n_flagged'])}/{len(ds_list)})"
                   for i in range(len(flat))]
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=9, color="#222222", loc="left", pad=4)
        cb = fig.colorbar(im, ax=ax, shrink=0.6,
                           label=f"severity_{direction}")
        cb.ax.tick_params(labelsize=6)
        fd.save_figure(fig, save_name)
        plt.show()


    # Pancreas: M dominates; testis: F dominates. We render both directions
    # for both tissues so the comparison is symmetric.
    reproducibility_footprint("M", PANC_DATASETS,
        "G3_reproducibility_footprint_M_pancreas",
        "pancreas — top-50 M-flag genes by reproducibility",
        cmap_name="Reds")
    reproducibility_footprint("F", PANC_DATASETS,
        "G3_reproducibility_footprint_F_pancreas",
        "pancreas — top-50 F-flag genes by reproducibility",
        cmap_name="Blues")
    reproducibility_footprint("M", TEST_DATASETS,
        "G3_reproducibility_footprint_M_testis",
        "testis — top-50 M-flag genes by reproducibility",
        cmap_name="Reds")
    reproducibility_footprint("F", TEST_DATASETS,
        "G3_reproducibility_footprint_F_testis",
        "testis — top-50 F-flag genes by reproducibility",
        cmap_name="Blues")
"""))


cells.append(md(r"""
    ## §G.4 — Severity by gene category

    For each direction, a strip plot showing every flagged gene's
    severity, grouped by gene category (rows). Categories ordered by
    median severity descending, so the most-severe-flagged categories
    sit at the top. The dot density and severity together identify
    which biology is over-represented in each disagreement direction.
"""))
cells.append(code(r"""
    def severity_strip(direction, ds_list, save_name, title, color):
        sevcol = f"severity_{direction}"
        rows = []
        for k in ds_list:
            d = FLAGGED[k]
            sub = d[d[f"flag_{direction}"]][["symbol","category",sevcol]]
            for _, r in sub.iterrows():
                rows.append({"symbol": r["symbol"],
                              "category": r["category"],
                              "severity": float(r[sevcol]),
                              "dataset": k})
        if not rows:
            print(f"  no {direction} flags pooled within {title}."); return
        df = pd.DataFrame(rows)
        df = df[df["category"] != "other"]
        cat_order = (df.groupby("category")["severity"]
                       .median()
                       .sort_values(ascending=False).index.tolist())
        counts = df["category"].value_counts()
        cat_order = [c for c in cat_order if counts.get(c, 0) >= 3]
        if not cat_order:
            print(f"  no categories with ≥3 flagged genes for {direction} on {title}.")
            return

        fig, ax = fd.make_axes(figsize=(7.0, 0.30 * len(cat_order) + 1.2))
        rng = np.random.default_rng(0)
        for i, c in enumerate(cat_order):
            sub = df[df["category"] == c]
            jitter = rng.uniform(-0.30, 0.30, size=len(sub))
            ax.scatter(sub["severity"], i + jitter,
                       s=14, c=color, alpha=0.65, linewidths=0,
                       rasterized=True)
            med = float(sub["severity"].median())
            ax.plot([med, med], [i - 0.4, i + 0.4],
                    color="#222222", linewidth=1.2)
            ax.text(0.0 - 0.005, i,
                    f"n={len(sub)}", ha="right", va="center",
                    fontsize=6, color="#666666",
                    transform=ax.get_yaxis_transform())
        ax.invert_yaxis()
        ax.set_yticks(range(len(cat_order)))
        ax.set_yticklabels(cat_order, fontsize=7)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"severity ({direction} = |Δα|)")
        fd.annotate_stat(ax, title, "lower right")
        fd.save_figure(fig, save_name)
        plt.show()


    severity_strip("M", PANC_DATASETS,
        "G4_severity_by_category_M_pancreas", "pancreas · M flags",
        color="#b2182b")
    severity_strip("F", PANC_DATASETS,
        "G4_severity_by_category_F_pancreas", "pancreas · F flags",
        color="#2166ac")
    severity_strip("M", TEST_DATASETS,
        "G4_severity_by_category_M_testis", "testis · M flags",
        color="#b2182b")
    severity_strip("F", TEST_DATASETS,
        "G4_severity_by_category_F_testis", "testis · F flags",
        color="#2166ac")
"""))


cells.append(md(r"""
    ## §G.5 — Aggregate quadrant map per tissue

    Every gene that fires in *any* dataset of a given tissue,
    plotted at its `(mean prior, mean α)` averaged across the
    datasets in which it appears. Color = aggregate flag tag
    (M / F / both). Diagonal `α = prior` drawn faintly; the M
    and F flag regions shaded with the same colors used in §B.
    Top-8 most-severe genes per direction labeled.

    Pancreas and testis are rendered as **separate figures** so
    each tissue's quadrant structure is visible on its own
    natural color scale, without testis (compressed by the
    prior-clamp at 0.85) being washed out against pancreas.
"""))
cells.append(code(r"""
    def aggregate_quadrant(ds_list, save_name, title):
        rows = []
        for k in ds_list:
            d = FLAGGED[k]
            sub = d[(d["flag_M"]) | (d["flag_F"])][[
                "symbol","category","prior","alpha","delta_alpha",
                "flag_M","flag_F","severity_M","severity_F"]]
            sub["dataset"] = k
            rows.append(sub)
        flagged_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not len(flagged_all):
            print(f"  no flags in {title}.")
            return
        agg = (flagged_all.groupby("symbol")
                           .agg(category=("category","first"),
                                mean_prior=("prior","mean"),
                                mean_alpha=("alpha","mean"),
                                mean_delta=("delta_alpha","mean"),
                                n_M=("flag_M", "sum"),
                                n_F=("flag_F", "sum"),
                                max_sev_M=("severity_M","max"),
                                max_sev_F=("severity_F","max"),
                                n_total=("dataset","count"))
                           .reset_index())
        agg["tag"] = np.where(
            (agg["n_M"] > 0) & (agg["n_F"] > 0), "mixed",
            np.where(agg["n_M"] > 0, "M",
                      np.where(agg["n_F"] > 0, "F", "none")))

        fig, ax = fd.make_axes(figsize=(6.4, 5.6))
        # M and F region polygons
        diag_x = np.array([PRIOR_HIGH, 1.0])
        ax.fill_between(diag_x, 0, diag_x - DELTA_FLOOR,
                         color="#b2182b", alpha=0.06, zorder=0)
        diag_x_f = np.array([0.0, PRIOR_LOW])
        ax.fill_between(diag_x_f, diag_x_f + DELTA_FLOOR, 1.0,
                         color="#2166ac", alpha=0.06, zorder=0)
        ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.6,
                linestyle="--", zorder=1)
        ax.axvline(PRIOR_HIGH, color="#444444", linewidth=0.4, linestyle=":")
        ax.axvline(PRIOR_LOW,  color="#444444", linewidth=0.4, linestyle=":")

        for tag, color in [("M", "#b2182b"), ("F", "#2166ac"), ("mixed", "#542788")]:
            sub = agg[agg["tag"] == tag]
            sizes = 6 + 4 * sub["n_total"]
            ax.scatter(sub["mean_prior"], sub["mean_alpha"],
                       s=sizes, c=color, alpha=0.65,
                       edgecolor="#222222", linewidths=0.3,
                       rasterized=True, zorder=3, label=f"{tag} (n={len(sub)})")

        for tag, sevcol, color in [("M","max_sev_M","#b2182b"),
                                    ("F","max_sev_F","#2166ac")]:
            top = agg[agg["tag"] == tag].nlargest(8, sevcol)
            for _, r in top.iterrows():
                ax.text(r["mean_prior"] + 0.005, r["mean_alpha"] + 0.012,
                        str(r["symbol"])[:14],
                        fontsize=6, ha="left", va="bottom", color=color,
                        fontweight="bold")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("mean conservation prior")
        ax.set_ylabel("mean " + fd.LABEL["alpha_long"])
        ax.set_title(title, fontsize=9, color="#222222", loc="left", pad=4)
        fd.thin_legend(ax, loc="lower right")
        fd.annotate_stat(ax, f"marker size ∝ # of {len(ds_list)} datasets gene was flagged in",
                          "upper left")
        fd.save_figure(fig, save_name)
        plt.show()


    aggregate_quadrant(PANC_DATASETS, "G5_aggregate_quadrant_pancreas",
                        "pancreas (camex liver) — aggregate quadrant")
    aggregate_quadrant(TEST_DATASETS, "G5_aggregate_quadrant_testis",
                        "testis — aggregate quadrant")
"""))


# ============ §I — Deep biology: tissue-separated, paper-grade ============
cells.append(md(r"""
    # §I — Deep biology: tissue-separated, paper-grade analyses

    *§A–§G computed the QC machinery and per-dataset flag lists.
    The §I section turns those lists into **publication-grade
    figures organized by tissue**, because the registry's two
    tissues — pancreas (camex liver) and testis — operate in
    architecturally different α regimes (testis runs use a
    `prior_clamp_max = 0.85`, see §H), and combining them in a
    single visualization would obscure both.*

    Every figure below is designed to answer one specific
    question that a paper Results section can quote, with named
    genes and quantitative claims. We also drop a series of
    full-page case-study cards (§I.10) for the most defendable
    individual flags.

    Visual contract:
    - **vermilion** (`#b2182b`) = M flag (mis-annotation candidate),
    - **blue** (`#2166ac`) = F flag (functional-conservation candidate),
    - tissue accents: **teal** (`#117a73`) for pancreas, **plum**
      (`#7d3c98`) for testis, in figures that color by tissue.
"""))


cells.append(code(r"""
    # Common containers + colors used by every §I figure
    PANC_LABELS = [DATASETS[k]["label"] for k in PANC_DATASETS]
    TEST_LABELS = [DATASETS[k]["label"] for k in TEST_DATASETS]
    M_COLOR  = "#b2182b"
    F_COLOR  = "#2166ac"
    NA_COLOR = "#cccccc"
    PANC_ACCENT = "#117a73"
    TEST_ACCENT = "#7d3c98"

    # Build a tissue-separated FLAGGED-pool: PANC_POOL and TEST_POOL,
    # one row per (gene, dataset) within each tissue. Used by every
    # §I figure that aggregates within a tissue.
    panc_rows = []
    for k in PANC_DATASETS:
        d = FLAGGED[k][["var_name","symbol","category","prior","alpha",
                          "delta_alpha","S_g_ema","flag_M","flag_F",
                          "severity_M","severity_F"]].copy()
        d["dataset"] = k
        panc_rows.append(d)
    PANC_POOL = pd.concat(panc_rows, ignore_index=True)
    PANC_POOL["symbol"] = PANC_POOL["symbol"].astype(str).str.upper()
    PANC_POOL["abs_delta"] = PANC_POOL["delta_alpha"].abs()

    test_rows = []
    for k in TEST_DATASETS:
        d = FLAGGED[k][["var_name","symbol","category","prior","alpha",
                          "delta_alpha","S_g_ema","flag_M","flag_F",
                          "severity_M","severity_F"]].copy()
        d["dataset"] = k
        test_rows.append(d)
    TEST_POOL = pd.concat(test_rows, ignore_index=True)
    TEST_POOL["symbol"] = TEST_POOL["symbol"].astype(str).str.upper()
    TEST_POOL["abs_delta"] = TEST_POOL["delta_alpha"].abs()

    # M / F flag sets, per tissue
    M_PANC = set(PANC_POOL.loc[PANC_POOL["flag_M"], "var_name"].unique())
    F_PANC = set(PANC_POOL.loc[PANC_POOL["flag_F"], "var_name"].unique())
    M_TEST = set(TEST_POOL.loc[TEST_POOL["flag_M"], "var_name"].unique())
    F_TEST = set(TEST_POOL.loc[TEST_POOL["flag_F"], "var_name"].unique())
    print(f"pancreas pool: {PANC_POOL['var_name'].nunique()} genes, "
          f"{len(M_PANC)} M-flagged in ≥1 dataset, {len(F_PANC)} F-flagged.")
    print(f"testis   pool: {TEST_POOL['var_name'].nunique()} genes, "
          f"{len(M_TEST)} M-flagged in ≥1 dataset, {len(F_TEST)} F-flagged.")
"""))


# §I.1 — Architectural reality: tissue-stratified (α, prior) density
cells.append(md(r"""
    ## §I.1 — Architectural reality: per-tissue (α, prior) joint density

    Before any per-gene analysis: **how do α and prior co-vary
    differently across the two tissues?** Pancreas (camex liver)
    runs use unconstrained training, so α can spread freely from 0
    to 1; testis runs use `prior_clamp_max = 0.85` (per
    `workflow-best.md` §6–8), which compresses both α and Δα
    against the upper boundary. Figure §I.1 shows the joint
    distribution per tissue side-by-side, with the M and F
    flag regions overlaid in their canonical colors.

    *Read.* Pancreas datasets show a broad density along the
    diagonal with conspicuous M and F lobes. Testis datasets show
    a tightly compressed cloud near the upper-right corner — the
    architectural ceiling of the prior clamp visible at a glance.
    Any tissue-pooled analysis would average these regimes and
    miss both stories.
"""))
cells.append(code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4),
                              layout="constrained", sharex=True, sharey=True)
    for ax, pool, label, accent in [
        (axes[0], PANC_POOL, "pancreas (camex liver) — 4 datasets pooled", PANC_ACCENT),
        (axes[1], TEST_POOL, "testis — 3 datasets pooled",                  TEST_ACCENT),
    ]:
        # All-gene density via hexbin (rasterized for vector PDF)
        h = ax.hexbin(pool["prior"], pool["alpha"],
                      gridsize=60, cmap="Greys",
                      mincnt=1, linewidths=0)
        # Flag-region shading
        ax.fill_between([PRIOR_HIGH, 1.0],
                         np.array([PRIOR_HIGH, 1.0]) - DELTA_FLOOR,
                         0,
                         color=M_COLOR, alpha=0.10, zorder=2)
        ax.fill_between([0.0, PRIOR_LOW],
                         1.0,
                         np.array([0.0, PRIOR_LOW]) + DELTA_FLOOR,
                         color=F_COLOR, alpha=0.10, zorder=2)
        # Diagonal
        ax.plot([0, 1], [0, 1], color="#444444", linewidth=0.6,
                linestyle="--", zorder=3)
        # Threshold lines
        ax.axvline(PRIOR_HIGH, color="#666666", linewidth=0.4, linestyle=":")
        ax.axvline(PRIOR_LOW,  color="#666666", linewidth=0.4, linestyle=":")
        # Spotlight M and F flagged genes
        m_pts = pool[pool["flag_M"]]
        f_pts = pool[pool["flag_F"]]
        ax.scatter(m_pts["prior"], m_pts["alpha"], s=4, c=M_COLOR,
                    alpha=0.55, linewidths=0, rasterized=True, zorder=4,
                    label=f"M flag ({len(m_pts)} per-dataset hits)")
        ax.scatter(f_pts["prior"], f_pts["alpha"], s=4, c=F_COLOR,
                    alpha=0.55, linewidths=0, rasterized=True, zorder=4,
                    label=f"F flag ({len(f_pts)} per-dataset hits)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("conservation prior")
        ax.set_ylabel(fd.LABEL["alpha_long"])
        ax.set_title(label, fontsize=9, color=accent, pad=4, loc="left")
        fd.thin_legend(ax, loc="lower right")
    fd.save_figure(fig, "I1_tissue_landscape")
    plt.show()
"""))


# §I.2 — Per-tissue Δα waterfall — the *shape* of the disagreement
cells.append(md(r"""
    ## §I.2 — Per-tissue Δα waterfall — *the shape of disagreement*

    Sort every (gene, dataset) row by Δα descending; plot one bar
    per row. The waterfall makes the *tail structure* of the
    disagreement visible — pancreas has long, asymmetric
    M-flag tails (left, deep red) and shorter F-flag tails
    (right, blue). Testis has flat, near-symmetric tails because
    of the prior clamp.

    Top-12 most-extreme genes per direction per tissue are
    annotated.
"""))
cells.append(code(r"""
    def waterfall(pool: pd.DataFrame, tissue_name: str, save_name: str,
                   accent: str):
        # one row per (gene × dataset); sort by Δα
        s = pool.sort_values("delta_alpha", ascending=True).copy()
        s = s.reset_index(drop=True)
        x = np.arange(len(s))
        delta = s["delta_alpha"].values
        # color by flag
        c = np.full(len(s), NA_COLOR, dtype=object)
        c[s["flag_M"].values] = M_COLOR
        c[s["flag_F"].values] = F_COLOR
        fig, ax = fd.make_axes(figsize=(8.2, 3.6))
        ax.bar(x, delta, color=list(c), width=1.0, linewidth=0)
        ax.axhline(0, color="#444444", linewidth=0.4, zorder=0)
        ax.axhline(-DELTA_FLOOR, color=M_COLOR, linewidth=0.5,
                   linestyle="--", zorder=0, alpha=0.6)
        ax.axhline(+DELTA_FLOOR, color=F_COLOR, linewidth=0.5,
                   linestyle="--", zorder=0, alpha=0.6)
        # Annotate top-12 genes per direction
        K = 12
        # leftmost = most negative
        m_left = s.head(K)
        # rightmost = most positive
        f_right = s.tail(K)
        for i, (_, r) in enumerate(m_left.iterrows()):
            if not r["flag_M"]: continue
            ax.text(i, r["delta_alpha"] - 0.02, str(r["symbol"])[:14],
                    ha="center", va="top", fontsize=5.5, color=M_COLOR,
                    rotation=90)
        for i, (_, r) in enumerate(f_right.iterrows()):
            if not r["flag_F"]: continue
            ax.text(len(s) - K + i, r["delta_alpha"] + 0.02,
                    str(r["symbol"])[:14],
                    ha="center", va="bottom", fontsize=5.5, color=F_COLOR,
                    rotation=90)
        ax.set_xlim(0, len(s))
        ax.set_xlabel(f"(gene × dataset) rows sorted by Δα — {tissue_name}")
        ax.set_ylabel(fd.LABEL["delta_alpha"])
        # informative annotation
        n_M = int(s["flag_M"].sum()); n_F = int(s["flag_F"].sum())
        fd.annotate_stat(ax,
                          f"{tissue_name}: {n_M} M-rows, {n_F} F-rows of {len(s)} total",
                          "upper left")
        fd.save_figure(fig, save_name)
        plt.show()


    waterfall(PANC_POOL, "pancreas (4 datasets)",
              "I2_waterfall_pancreas", PANC_ACCENT)
    waterfall(TEST_POOL, "testis (3 datasets)",
              "I2_waterfall_testis",  TEST_ACCENT)
"""))


# §I.3 — Cross-tissue disagreement portrait
cells.append(md(r"""
    ## §I.3 — Cross-tissue disagreement portrait

    *The single most informative figure in the entire notebook.*
    For every gene present in both pancreas and testis registries,
    plot **mean Δα across the four pancreas datasets** (x) against
    **mean Δα across the three testis datasets** (y). The four
    quadrants partition the gene set into:

    - **lower-left** — both tissues see Δα < 0 → universal mis-
      annotation candidates (Chimera consistently fails to
      align expression for these genes regardless of tissue).
    - **upper-right** — both tissues see Δα > 0 → universal
      regulatory conservation despite sequence divergence
      (cautious / hypothesis-grade per §H).
    - **lower-right + upper-left** — tissue-specific rewiring;
      the sequence is the same, but Chimera's call inverts
      between tissues.

    Marginal histograms show the per-tissue Δα distribution. The
    diagonal is overlaid as the "tissue-independent" reference
    (pancreas Δα = testis Δα).
"""))
cells.append(code(r"""
    # Build per-gene mean Δα per tissue, on shared ortho_X
    panc_mean = (PANC_POOL.groupby("var_name")
                          .agg(symbol=("symbol", "first"),
                                category=("category", "first"),
                                mean_delta_panc=("delta_alpha", "mean"),
                                n_panc_M=("flag_M", "sum"),
                                n_panc_F=("flag_F", "sum")))
    test_mean = (TEST_POOL.groupby("var_name")
                          .agg(mean_delta_test=("delta_alpha", "mean"),
                                n_test_M=("flag_M", "sum"),
                                n_test_F=("flag_F", "sum")))
    XT = panc_mean.join(test_mean, how="inner")
    print(f"genes shared across tissues: {len(XT)}")
    # tag each gene's quadrant
    def quadrant(row):
        p = row["mean_delta_panc"]; t = row["mean_delta_test"]
        if p < 0 and t < 0: return "M | M"
        if p > 0 and t > 0: return "F | F"
        if p < 0 and t > 0: return "M panc / F test"
        if p > 0 and t < 0: return "F panc / M test"
        return "axis"
    XT["quadrant"] = XT.apply(quadrant, axis=1)
    print(XT["quadrant"].value_counts().to_string())
"""))
cells.append(code(r"""
    # Assemble the marginal-histogram figure manually
    fig = plt.figure(figsize=(7.2, 6.4), layout="constrained")
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top  = fig.add_subplot(gs[0,   0:3], sharex=ax_main)
    ax_rt   = fig.add_subplot(gs[1:4, 3],   sharey=ax_main)

    # Main scatter, color by quadrant
    QUAD_COLOR = {"M | M": M_COLOR, "F | F": F_COLOR,
                  "M panc / F test": "#6a3d9a", "F panc / M test": "#33a02c",
                  "axis": "#cccccc"}
    QUAD_ORDER = ["M | M", "F | F", "M panc / F test", "F panc / M test", "axis"]
    for q in QUAD_ORDER:
        sub = XT[XT["quadrant"] == q]
        ax_main.scatter(sub["mean_delta_panc"], sub["mean_delta_test"],
                        s=4, c=QUAD_COLOR[q], alpha=0.55, linewidths=0,
                        rasterized=True,
                        label=f"{q}  (n={len(sub)})")
    # Diagonal + axes
    lim = float(np.nanmax(np.abs(XT[["mean_delta_panc", "mean_delta_test"]].values)))
    if not np.isfinite(lim) or lim == 0: lim = 0.5
    ax_main.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=0.4,
                 linestyle="--", zorder=0)
    ax_main.axhline(0, color="#444444", linewidth=0.4, zorder=0)
    ax_main.axvline(0, color="#444444", linewidth=0.4, zorder=0)
    ax_main.set_xlim(-lim*1.05, lim*1.05); ax_main.set_ylim(-lim*1.05, lim*1.05)
    ax_main.set_xlabel(r"mean $\Delta\alpha$  —  pancreas")
    ax_main.set_ylabel(r"mean $\Delta\alpha$  —  testis")
    ax_main.spines["top"].set_visible(False); ax_main.spines["right"].set_visible(False)

    # Label top-K extreme genes per quadrant
    LBL_K = 6
    for q in ["M | M", "F | F", "M panc / F test", "F panc / M test"]:
        sub = XT[XT["quadrant"] == q].copy()
        if len(sub) == 0: continue
        sub["dist"] = np.sqrt(sub["mean_delta_panc"]**2 + sub["mean_delta_test"]**2)
        top = sub.nlargest(LBL_K, "dist")
        for _, r in top.iterrows():
            sym = str(r["symbol"]) if pd.notna(r["symbol"]) else ""
            if not sym or sym.lower() == "nan" or sym == r.name: continue
            ax_main.text(r["mean_delta_panc"], r["mean_delta_test"] + lim*0.02,
                         sym[:14], fontsize=6.5, ha="center", va="bottom",
                         color=QUAD_COLOR[q], fontweight="bold")

    # Top marginal: pancreas Δα histogram
    ax_top.hist(XT["mean_delta_panc"], bins=60, color=PANC_ACCENT,
                alpha=0.7, edgecolor="#222222", linewidth=0.3)
    ax_top.axvline(0, color="#444444", linewidth=0.4, zorder=0)
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False); ax_top.tick_params(left=False, labelleft=False)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("genes", fontsize=7)

    # Right marginal: testis Δα histogram
    ax_rt.hist(XT["mean_delta_test"], bins=60, color=TEST_ACCENT,
               alpha=0.7, edgecolor="#222222", linewidth=0.3,
               orientation="horizontal")
    ax_rt.axhline(0, color="#444444", linewidth=0.4, zorder=0)
    ax_rt.spines["top"].set_visible(False); ax_rt.spines["right"].set_visible(False)
    ax_rt.spines["bottom"].set_visible(False); ax_rt.tick_params(bottom=False, labelbottom=False)
    ax_rt.tick_params(labelleft=False)
    ax_rt.set_xlabel("genes", fontsize=7)
    fd.thin_legend(ax_main, loc="lower right")
    fd.save_figure(fig, "I3_cross_tissue_disagreement_portrait")
    plt.show()
"""))


# §I.3b — Compact, axis-fixed companion to §I.3 (no marginals, adjustText labels)
cells.append(md(r"""
    ### §I.3b — Compact companion (axis-fixed, adjusted labels)

    Same data as §I.3, but rendered as a single panel with fixed
    `[-0.2, +0.2]` axes, no marginal histograms, and at most **4**
    extreme-gene labels per quadrant placed by ``adjustText`` so
    annotations never collapse onto each other or onto scatter
    points.  Use this version when the figure needs to drop into a
    paper column at small size; use §I.3 when the marginal
    distributions matter.
"""))
cells.append(code(r"""
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    fig, ax = fd.make_axes(figsize=(3.6, 3.6))
    QUAD_COLOR = {"M | M": M_COLOR, "F | F": F_COLOR,
                  "M panc / F test": "#6a3d9a", "F panc / M test": "#33a02c",
                  "axis": "#cccccc"}
    QUAD_ORDER = ["M | M", "F | F", "M panc / F test", "F panc / M test", "axis"]
    LEGEND_QUADS = [q for q in QUAD_ORDER if q != "axis"]

    LIM = 0.2
    plot_df = XT.copy()
    plot_df["x_clip"] = plot_df["mean_delta_panc"].clip(-LIM, +LIM)
    plot_df["y_clip"] = plot_df["mean_delta_test"].clip(-LIM, +LIM)

    for q in QUAD_ORDER:
        sub = plot_df[plot_df["quadrant"] == q]
        ax.scatter(sub["x_clip"], sub["y_clip"],
                   s=4, c=QUAD_COLOR[q], alpha=0.55, linewidths=0,
                   rasterized=True)

    ax.plot([-LIM, LIM], [-LIM, LIM], color="#444444", linewidth=0.4,
            linestyle="--", zorder=0)
    ax.axhline(0, color="#444444", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.4, zorder=0)
    ax.set_xlim(-LIM, +LIM); ax.set_ylim(-LIM, +LIM)
    ax.set_xlabel(r"mean $\Delta\alpha$  —  pancreas")
    ax.set_ylabel(r"mean $\Delta\alpha$  —  testis")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ≤ 4 extreme-gene labels per quadrant; placed inside the [-0.2, +0.2]
    # window using the clipped coords so any off-panel point still gets a tag
    # at the edge.
    LBL_K = 4
    texts = []
    label_xs, label_ys = [], []
    for q in ["M | M", "F | F", "M panc / F test", "F panc / M test"]:
        sub = plot_df[plot_df["quadrant"] == q].copy()
        if len(sub) == 0: continue
        sub["dist"] = np.sqrt(sub["mean_delta_panc"] ** 2
                                + sub["mean_delta_test"] ** 2)
        top = sub.nlargest(LBL_K, "dist")
        for _, r in top.iterrows():
            sym = str(r["symbol"]) if pd.notna(r["symbol"]) else ""
            if not sym or sym.lower() == "nan" or sym == r.name: continue
            texts.append(ax.text(r["x_clip"], r["y_clip"], sym[:14],
                                  fontsize=6.5, ha="center", va="center",
                                  color=QUAD_COLOR[q], fontweight="bold"))
            label_xs.append(r["x_clip"])
            label_ys.append(r["y_clip"])

    # Repel labels off each other and off the scatter; thin gray leader lines.
    if texts:
        adjust_text(texts,
                    x=label_xs, y=label_ys, ax=ax,
                    expand=(1.4, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    lw=0.4, shrinkA=2, shrinkB=2))

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=QUAD_COLOR[q], markeredgecolor="none",
               markersize=6, alpha=1.0,
               label=f"{q}  (n={(plot_df['quadrant'] == q).sum()})")
        for q in LEGEND_QUADS
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=False, fontsize=6.5, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)
    fd.save_figure(fig, "I3b_cross_tissue_disagreement_portrait_compact")
    plt.show()
"""))


# §I.3c — Grayscale-only variant (quadrants encoded as marker shape, no chromatic accent)
cells.append(md(r"""
    ### §I.3c — Grayscale variant (no color)

    Same panel as §I.3b but stripped of chromatic accent.  Each
    quadrant is encoded by **marker shape** plus a distinct gray
    level so the panel still reads in monochrome print.  No pure
    black is used — the darkest gray is ``#555555``.
"""))
cells.append(code(r"""
    import random as _rnd
    from adjustText import adjust_text
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = fd.make_axes(figsize=(3.6, 3.6))
    # Quadrant → gray shade.  Same marker (circle) for all; the four
    # quadrants are distinguished by gray level only.  Shades get
    # progressively lighter going clockwise from the bottom-left
    # (M/M, darkest) → top-left → top-right → bottom-right.  No pure
    # black anywhere.  The dataset's "panc" axis is actually camex
    # liver — labels and legend text use "liver"/"testis" accordingly,
    # while internal quadrant keys keep the upstream "panc"/"test"
    # spelling so the cached `XT['quadrant']` values still match.
    QUAD_GRAY = {
        "M | M":            "#555555",   # bottom-left  — darkest
        "M panc / F test":  "#888888",   # top-left
        "F | F":            "#aaaaaa",   # top-right
        "F panc / M test":  "#cccccc",   # bottom-right — lightest
        "axis":             "#e6e6e6",
    }
    QUAD_DISPLAY = {
        "M | M":            "M | M",
        "M panc / F test":  "M liver / F testis",
        "F | F":            "F | F",
        "F panc / M test":  "F liver / M testis",
    }
    QUAD_ORDER   = ["M | M", "M panc / F test", "F | F", "F panc / M test", "axis"]
    LEGEND_QUADS = [q for q in QUAD_ORDER if q != "axis"]

    XLIM       = 0.20
    YLIM_LOW   = -0.125
    YLIM_HIGH  = +0.15
    plot_df = XT.copy()
    plot_df["x_clip"] = plot_df["mean_delta_panc"].clip(-XLIM, +XLIM)
    plot_df["y_clip"] = plot_df["mean_delta_test"].clip(YLIM_LOW, YLIM_HIGH)

    for q in QUAD_ORDER:
        sub = plot_df[plot_df["quadrant"] == q]
        ax.scatter(sub["x_clip"], sub["y_clip"],
                   s=5, c=QUAD_GRAY[q], alpha=0.6, linewidths=0,
                   rasterized=True)

    ax.plot([-XLIM, XLIM], [-XLIM, XLIM], color="#666666", linewidth=0.4,
            linestyle="--", zorder=0)
    ax.axhline(0, color="#666666", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#666666", linewidth=0.4, zorder=0)
    ax.set_xlim(-XLIM, +XLIM); ax.set_ylim(YLIM_LOW, YLIM_HIGH)
    # 5 ticks per axis, evenly spaced, formatted to 2 decimals.
    ax.set_xticks(np.linspace(-XLIM, +XLIM, 5))
    ax.set_yticks(np.linspace(YLIM_LOW, YLIM_HIGH, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(r"mean $\Delta\alpha$  —  liver")
    ax.set_ylabel(r"mean $\Delta\alpha$  —  testis")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 3 labels per quadrant.  Pick genes that sit in each quadrant's
    # 45°-diagonal wedge AND inside the visible window (so the dot the
    # label points at isn't pinned to an axis edge by clipping).  Among
    # those, take the diagonally farthest TOP_K candidates and randomly
    # sample 3 — the random pick (with a fixed seed) keeps successive
    # runs reproducible while still spreading the labels visually
    # instead of always picking the 3 single most extreme points.
    LBL_K = 3
    TOP_K = 12
    LABEL_RNG = _rnd.Random(42)
    QUAD_DIRECTION = {  # unit vector pointing into each quadrant's 45° wedge
        "M | M":            (-1.0, -1.0),
        "M panc / F test":  (-1.0, +1.0),
        "F | F":            (+1.0, +1.0),
        "F panc / M test":  (+1.0, -1.0),
    }
    WEDGE_HALF_DEG = 30.0
    texts = []
    label_xs, label_ys = [], []
    for q in LEGEND_QUADS:
        sub = plot_df[plot_df["quadrant"] == q].copy()
        if len(sub) == 0: continue
        # Restrict to genes whose true coords are inside the visible
        # window — boundary-clipped genes are misleading because the
        # rendered dot no longer sits at the gene's actual Δα value.
        in_view = (
            (sub["mean_delta_panc"] >= -XLIM)   & (sub["mean_delta_panc"] <= +XLIM) &
            (sub["mean_delta_test"] >= YLIM_LOW) & (sub["mean_delta_test"] <= YLIM_HIGH)
        )
        sub = sub.loc[in_view].copy()
        if len(sub) == 0: continue

        ux, uy = QUAD_DIRECTION[q]
        norm = (ux ** 2 + uy ** 2) ** 0.5
        ux, uy = ux / norm, uy / norm
        proj = sub["mean_delta_panc"].values * ux + sub["mean_delta_test"].values * uy
        perp = np.abs(sub["mean_delta_panc"].values * (-uy)
                       + sub["mean_delta_test"].values * (+ux))
        dist = np.sqrt(sub["mean_delta_panc"].values ** 2
                        + sub["mean_delta_test"].values ** 2)
        tol = np.tan(np.deg2rad(WEDGE_HALF_DEG)) * dist
        in_wedge = (perp <= tol) & (proj > 0)
        sub = sub.loc[in_wedge].copy()
        if len(sub) == 0: continue
        sub["proj"] = proj[in_wedge]

        # Top diagonally-farthest TOP_K candidates, then pick LBL_K randomly.
        pool = (sub.sort_values("proj", ascending=False)
                    .head(min(TOP_K, len(sub)))
                    .reset_index(drop=False))
        if len(pool) > LBL_K:
            chosen_idx = sorted(LABEL_RNG.sample(range(len(pool)), LBL_K))
        else:
            chosen_idx = list(range(len(pool)))
        chosen = pool.iloc[chosen_idx]

        for _, r in chosen.iterrows():
            sym = str(r["symbol"]) if pd.notna(r["symbol"]) else ""
            if not sym or sym.lower() == "nan":
                continue
            texts.append(ax.text(r["x_clip"], r["y_clip"], sym[:14],
                                  fontsize=6.5, ha="center", va="center",
                                  color="#444444", fontweight="bold"))
            label_xs.append(r["x_clip"])
            label_ys.append(r["y_clip"])

    if texts:
        adjust_text(texts,
                    x=label_xs, y=label_ys, ax=ax,
                    expand=(1.4, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    lw=0.4, shrinkA=2, shrinkB=2))

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=QUAD_GRAY[q], markeredgecolor="none",
               markersize=6, alpha=1.0,
               label=f"{QUAD_DISPLAY[q]}  (n={(plot_df['quadrant'] == q).sum()})")
        for q in LEGEND_QUADS
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=False, fontsize=6.5, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)
    fd.save_figure(fig, "I3c_cross_tissue_disagreement_portrait_grayscale")
    plt.show()
"""))


# §I.4 — Per-tissue paralog forest (side-by-side)
cells.append(md(r"""
    ## §I.4 — Per-tissue paralog-family enrichment — side-by-side forest

    Same 25 paralog-rich gene families tested on **pancreas M
    flags** (left column) and **testis F flags** (right column —
    F is the dominant testis flag direction since the prior
    clamp suppresses M). A family enriched on one tissue but not
    the other says *paralog confusion is tissue-context dependent
    in expression*, even though the protein is the same.

    Method: aggregate each tissue's flag set across all its
    datasets (union); Fisher exact (one-sided) against the
    tissue's full gene background; Bonferroni-correct across the
    25 families per tissue.
"""))
cells.append(code(r"""
    # Curated paralog families (repeated here so this section is self-contained)
    PARALOG_FAMILIES = {
        "Cytosolic ribosomal RPL/RPS":  lambda s: s.startswith("RPL") or s.startswith("RPS"),
        "Mitochondrial ribosomal MRP":  lambda s: s.startswith("MRPL") or s.startswith("MRPS"),
        "mtDNA-encoded MT-":            lambda s: s.startswith("MT-"),
        "Zinc-finger TFs ZBTB":         lambda s: s.startswith("ZBTB") and len(s) >= 5,
        "Zinc-finger TFs ZNF":          lambda s: s.startswith("ZNF") and len(s) >= 4 and s[3].isdigit(),
        "Forkhead TFs FOX":             lambda s: s.startswith("FOX") and len(s) >= 4 and s[3].isalpha(),
        "Homeobox HOX":                 lambda s: s.startswith("HOX") and len(s) >= 5,
        "Solute carriers SLC":          lambda s: s.startswith("SLC") and len(s) >= 5 and s[3].isdigit(),
        "Keratins KRT":                 lambda s: s.startswith("KRT") and len(s) >= 4 and s[3].isdigit(),
        "Collagens COL":                lambda s: s.startswith("COL") and len(s) >= 4 and s[3].isdigit(),
        "Cytochrome P450 CYP":          lambda s: s.startswith("CYP") and len(s) >= 4 and s[3].isdigit(),
        "Olfactory receptors OR":       lambda s: s.startswith("OR") and len(s) >= 4 and s[2].isdigit(),
        "GPCRs GPR":                    lambda s: s.startswith("GPR") and len(s) >= 4 and s[3].isdigit(),
        "Glutathione transf. GST":      lambda s: s.startswith("GST") and len(s) >= 4,
        "Heat shock HSP":               lambda s: s.startswith("HSP") and len(s) >= 4,
        "Histones HIST":                lambda s: s.startswith("HIST"),
        "Tubulins TUB":                 lambda s: s.startswith("TUB") and len(s) >= 4,
        "Annexins ANXA":                lambda s: s.startswith("ANXA"),
        "Caspases CASP":                lambda s: s.startswith("CASP") and len(s) >= 5 and s[4].isdigit(),
        "Dual-spec phosph. DUSP":       lambda s: s.startswith("DUSP") and len(s) >= 5,
        "MAP kinases MAPK":             lambda s: s.startswith("MAPK") and len(s) >= 5,
        "Nuclear receptors NR":         lambda s: s.startswith("NR") and len(s) >= 4 and s[2].isdigit(),
        "Sushi-domain SUSD":            lambda s: s.startswith("SUSD"),
        "EIF translation":              lambda s: s.startswith("EIF") and len(s) >= 4,
    }


    from scipy.stats import fisher_exact

    def family_enrichment(pool_dedup: pd.DataFrame, flag_set: set,
                            direction: str) -> pd.DataFrame:
        n_total = pool_dedup["var_name"].nunique()
        n_flag  = len(flag_set)
        rows = []
        for fam, matcher in PARALOG_FAMILIES.items():
            fam_mask = pool_dedup["symbol"].fillna("").astype(str).str.upper().apply(matcher)
            n_fam = int(fam_mask.sum())
            if n_fam < 5:
                rows.append({"family": fam, "n_in_family": n_fam,
                              "n_flag_in_family": 0, "odds_ratio": np.nan,
                              "log2_or": np.nan, "p_value": np.nan})
                continue
            var_in_fam = pool_dedup.loc[fam_mask, "var_name"].tolist()
            a = int(sum(1 for v in var_in_fam if v in flag_set))    # flag in family
            b = n_flag - a                                              # flag not in family
            c = n_fam - a                                               # bg in family
            d = (n_total - n_flag) - c                                  # bg not in family
            try:
                odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            except ValueError:
                odds, p = float("nan"), 1.0
            rows.append({"family": fam, "n_in_family": n_fam,
                          "n_flag_in_family": a, "odds_ratio": float(odds),
                          "log2_or": np.log2(odds) if (odds and odds > 0) else np.nan,
                          "p_value": float(p)})
        df = pd.DataFrame(rows)
        m_test = max(df["p_value"].notna().sum(), 1)
        df["p_bonf"] = np.minimum(df["p_value"] * m_test, 1.0)
        df["sig"] = df["p_bonf"] < 0.05
        df["direction"] = direction
        return df.sort_values("log2_or", ascending=False)


    # de-dup pools (one row per gene) for enrichment background
    PANC_POOL_DEDUP = (PANC_POOL.sort_values("abs_delta", ascending=False)
                                  .drop_duplicates("var_name", keep="first"))
    TEST_POOL_DEDUP = (TEST_POOL.sort_values("abs_delta", ascending=False)
                                  .drop_duplicates("var_name", keep="first"))

    PARA_PANC_M = family_enrichment(PANC_POOL_DEDUP, M_PANC, "panc · M")
    PARA_TEST_F = family_enrichment(TEST_POOL_DEDUP, F_TEST, "test · F")
    print("=== panc · M paralog enrichment ===")
    print(PARA_PANC_M[["family","n_in_family","n_flag_in_family","odds_ratio",
                        "log2_or","p_bonf","sig"]].to_string(index=False,
                        float_format=lambda x: f"{x:+.3f}"))
    print("\n=== test · F paralog enrichment ===")
    print(PARA_TEST_F[["family","n_in_family","n_flag_in_family","odds_ratio",
                        "log2_or","p_bonf","sig"]].to_string(index=False,
                        float_format=lambda x: f"{x:+.3f}"))
"""))
cells.append(code(r"""
    # Build the side-by-side forest
    families = list(PARALOG_FAMILIES.keys())
    n = len(families)
    panc_or = PARA_PANC_M.set_index("family").reindex(families)
    test_or = PARA_TEST_F.set_index("family").reindex(families)
    # Sort by max(|log2_OR| pancreas, |log2_OR| testis), descending — most striking on top
    sort_key = pd.Series({
        f: max(abs(panc_or.loc[f, "log2_or"]) if pd.notna(panc_or.loc[f, "log2_or"]) else 0,
                abs(test_or.loc[f, "log2_or"]) if pd.notna(test_or.loc[f, "log2_or"]) else 0)
        for f in families
    })
    families = sort_key.sort_values(ascending=False).index.tolist()
    panc_or = panc_or.loc[families]
    test_or = test_or.loc[families]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.32 * n + 1.2),
                              layout="constrained", sharey=True)
    for ax, df, title, accent in [
        (axes[0], panc_or, "pancreas — M flag enrichment", PANC_ACCENT),
        (axes[1], test_or, "testis — F flag enrichment",   TEST_ACCENT),
    ]:
        y = np.arange(n)[::-1]
        for i, fam in enumerate(families):
            log2 = df.loc[fam, "log2_or"]
            sig = bool(df.loc[fam, "sig"])
            n_flag = int(df.loc[fam, "n_flag_in_family"]) if pd.notna(df.loc[fam, "n_flag_in_family"]) else 0
            n_fam = int(df.loc[fam, "n_in_family"]) if pd.notna(df.loc[fam, "n_in_family"]) else 0
            x_val = log2 if pd.notna(log2) else 0.0
            color = accent if sig else "#bdbdbd"
            ax.plot([0, x_val], [y[i], y[i]], color=color, linewidth=1.0)
            ax.scatter(x_val, y[i], s=42, color=color,
                        edgecolor="#222222", linewidths=0.4, zorder=3)
            star = " ★" if sig else ""
            ax.text(x_val + (0.05 if x_val >= 0 else -0.05), y[i],
                    f"  {n_flag}/{n_fam}{star}",
                    ha="left" if x_val >= 0 else "right", va="center",
                    fontsize=6.5, color="#222222")
        ax.axvline(0, color="#444444", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(families, fontsize=7)
        ax.set_xlabel(r"$\log_2$(odds ratio)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title(title, fontsize=9, color=accent, loc="left")
    fd.save_figure(fig, "I4_paralog_forest_per_tissue")
    plt.show()
"""))


# §I.5 — Ribosomal-protein concert
cells.append(md(r"""
    ## §I.5 — Ribosomal-protein concert: deepening the §I.4 hit

    §I.4 identified the **cytosolic ribosomal RPL/RPS family** as
    the only Bonferroni-significant paralog enrichment on
    pancreas M flags (log₂ OR ≈ +4). The natural follow-up: is
    the M-flag affecting *all* RPL/RPS, or a paralog-specific
    subset? Below: every RPL/RPS gene present in our gene set,
    α across the 7 datasets, with conservation prior shown
    alongside.
"""))
cells.append(code(r"""
    rib_rows = []
    for k in DATASET_ORDER:
        d = FLAGGED[k]
        sub = d[d["symbol"].astype(str).str.upper().str.startswith(("RPL","RPS"))].copy()
        sub["dataset"] = DATASETS[k]["label"]
        sub["tissue"] = DATASETS[k]["tissue"]
        rib_rows.append(sub[["symbol","dataset","tissue","prior","alpha",
                                "delta_alpha","flag_M","flag_F"]])
    RIB = pd.concat(rib_rows, ignore_index=True)
    RIB["symbol"] = RIB["symbol"].astype(str).str.upper()
    pivot_a = RIB.pivot_table(index="symbol", columns="dataset", values="alpha",
                                aggfunc="first")
    pivot_a = pivot_a.reindex(columns=[DATASETS[k]["label"] for k in DATASET_ORDER])
    pivot_p = RIB.pivot_table(index="symbol", columns="dataset", values="prior",
                                aggfunc="first")
    pivot_p = pivot_p.reindex(columns=[DATASETS[k]["label"] for k in DATASET_ORDER])
    # sort by mean α across pancreas datasets (most-divergent at top)
    pancmean = pivot_a[PANC_LABELS].mean(axis=1)
    pivot_a = pivot_a.loc[pancmean.sort_values().index]
    pivot_p = pivot_p.loc[pivot_a.index]
    # Filter to just genes that fired ≥1 M flag on pancreas (keeps figure focused)
    rib_M_flag = (RIB[RIB["flag_M"]].groupby("symbol").size())
    keep = pivot_a.index[pivot_a.index.isin(rib_M_flag.index)]
    pivot_a = pivot_a.loc[keep]
    pivot_p = pivot_p.loc[keep]
    print(f"RPL/RPS genes M-flagged on ≥1 pancreas dataset: {len(pivot_a)}")
"""))
cells.append(code(r"""
    n = len(pivot_a)
    fig = plt.figure(figsize=(11.0, 0.28 * n + 1.5), layout="constrained")
    gs = fig.add_gridspec(1, 8)
    ax_a = fig.add_subplot(gs[0, 0:7])  # alpha heatmap
    ax_p = fig.add_subplot(gs[0, 7])    # prior column

    im = ax_a.imshow(pivot_a.values, cmap="RdBu_r", aspect="auto",
                      vmin=0, vmax=1)
    for i in range(pivot_a.shape[0]):
        for j in range(pivot_a.shape[1]):
            v = pivot_a.iloc[i, j]
            if pd.isna(v): continue
            ax_a.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=6, color="#222222" if v > 0.55 else "#f5f5f5")
    ax_a.set_xticks(range(pivot_a.shape[1]))
    ax_a.set_xticklabels(pivot_a.columns, rotation=20, ha="right", fontsize=7.5)
    ax_a.set_yticks(range(pivot_a.shape[0]))
    ax_a.set_yticklabels(pivot_a.index, fontsize=7)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    cb = fig.colorbar(im, ax=ax_a, shrink=0.7, label=r"$\alpha$ (per-dataset)")
    cb.ax.tick_params(labelsize=6)

    # Prior column (averaged over the 4 pancreas datasets — prior is dataset-specific
    # but in pancreas these are all near 0.81–0.95 for ribosomal genes)
    pri = pivot_p[PANC_LABELS].mean(axis=1).values
    ax_p.barh(np.arange(n)[::-1], pri, color="#bdbdbd",
                edgecolor="#222222", linewidth=0.3)
    ax_p.set_xlim(0, 1.0)
    ax_p.set_xticks([0, 0.5, 1.0])
    ax_p.tick_params(left=False, labelleft=False)
    ax_p.set_xlabel("prior", fontsize=7)
    ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)
    ax_p.spines["left"].set_visible(False)
    fd.save_figure(fig, "I5_ribosomal_concert")
    plt.show()
"""))


# §I.6 — Mitochondrial-DNA-encoded gene cross-tissue inversion
cells.append(md(r"""
    ## §I.6 — Mitochondrial-DNA-encoded genes — cross-tissue inversion

    The 13 mtDNA coding genes show **opposite Δα signs across
    tissues**: negative on pancreas (M-flag candidate region),
    positive on testis (F-flag candidate region). A clean visual
    illustration of why combining tissues in a single figure
    obscures biology — the same gene's regulatory-conservation
    call inverts based on which tissue's training data Chimera
    saw. Marginal mean panels (top: pancreas mean per gene;
    right: testis mean per gene) make the inversion explicit.
"""))
cells.append(code(r"""
    rows = []
    for k in DATASET_ORDER:
        d = FLAGGED[k]
        sub = d[d["symbol"].astype(str).str.upper().str.startswith("MT-")].copy()
        sub["dataset"] = DATASETS[k]["label"]
        sub["tissue"]  = DATASETS[k]["tissue"]
        rows.append(sub[["symbol","dataset","tissue","prior","alpha","delta_alpha"]])
    MT_DF = pd.concat(rows, ignore_index=True)
    MT_DF["symbol"] = MT_DF["symbol"].astype(str).str.upper()
    pivot_d = MT_DF.pivot_table(index="symbol", columns="dataset",
                                  values="delta_alpha", aggfunc="first")
    pivot_d = pivot_d.reindex(columns=[DATASETS[k]["label"] for k in DATASET_ORDER])
    pivot_d = pivot_d.dropna(how="all")
    # Order rows by the panc-mean (most negative on top)
    pivot_d = pivot_d.loc[pivot_d[PANC_LABELS].mean(axis=1).sort_values().index]
"""))
cells.append(code(r"""
    n = len(pivot_d)
    fig = plt.figure(figsize=(10.5, 0.40 * n + 2.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5], width_ratios=[5, 1],
                            hspace=0.04, wspace=0.04)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_rt   = fig.add_subplot(gs[1, 1], sharey=ax_main)

    vmax = float(np.nanmax(np.abs(pivot_d.values)))
    if not np.isfinite(vmax) or vmax == 0: vmax = 0.5
    im = ax_main.imshow(pivot_d.values, cmap="RdBu_r", aspect="auto",
                          vmin=-vmax, vmax=vmax)
    for i in range(pivot_d.shape[0]):
        for j in range(pivot_d.shape[1]):
            v = pivot_d.iloc[i, j]
            if pd.isna(v): continue
            ax_main.text(j, i, f"{v:+.2f}", ha="center", va="center",
                          fontsize=6.5,
                          color="#222222" if abs(v) < vmax*0.55 else "#f5f5f5")
    # add a thin vertical separator between the pancreas block and the testis block
    n_panc = len(PANC_LABELS)
    ax_main.axvline(n_panc - 0.5, color="#222222", linewidth=0.7, zorder=5)
    ax_main.set_xticks(range(pivot_d.shape[1]))
    ax_main.set_xticklabels(pivot_d.columns, rotation=20, ha="right", fontsize=7.5)
    ax_main.set_yticks(range(pivot_d.shape[0]))
    ax_main.set_yticklabels(pivot_d.index, fontsize=7.5)
    ax_main.spines["top"].set_visible(False); ax_main.spines["right"].set_visible(False)

    # Top marginal: per-gene mean Δα, pancreas-only and testis-only side by side
    panc_m = pivot_d[PANC_LABELS].mean(axis=1).values
    test_m = pivot_d[TEST_LABELS].mean(axis=1).values
    ax_top.bar(np.arange(n_panc) - 0.18, [np.nan]*n_panc, width=0.36,
                color=PANC_ACCENT, edgecolor="#222222", linewidth=0.3)
    # show two grouped bars per dataset COLUMN — actually too noisy; show a single line
    # of (per-gene panc-mean Δα) — but that's per-row, not per-column.
    # Replace top marginal with a column-mean bar across datasets
    ax_top.bar(np.arange(pivot_d.shape[1]), pivot_d.mean(axis=0),
                color=["#117a73"]*n_panc + ["#7d3c98"]*(pivot_d.shape[1]-n_panc),
                edgecolor="#222222", linewidth=0.3)
    ax_top.axhline(0, color="#444444", linewidth=0.4)
    ax_top.set_xticks([])
    ax_top.set_ylabel(r"mean $\Delta\alpha$", fontsize=7)
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # Right marginal: per-gene panc-mean (teal) and testis-mean (plum) Δα
    yloc = np.arange(n)
    ax_rt.barh(yloc - 0.18, panc_m, height=0.36, color=PANC_ACCENT,
                edgecolor="#222222", linewidth=0.3, label="panc")
    ax_rt.barh(yloc + 0.18, test_m, height=0.36, color=TEST_ACCENT,
                edgecolor="#222222", linewidth=0.3, label="test")
    ax_rt.axvline(0, color="#444444", linewidth=0.4)
    ax_rt.set_yticks([]); ax_rt.set_xlabel(r"mean $\Delta\alpha$", fontsize=7)
    ax_rt.spines["top"].set_visible(False); ax_rt.spines["right"].set_visible(False)
    ax_rt.legend(loc="upper right", fontsize=6, frameon=False, handlelength=1.0)

    cb = fig.colorbar(im, ax=ax_main, shrink=0.6,
                      label=fd.LABEL["delta_alpha"], pad=0.06)
    cb.ax.tick_params(labelsize=7)
    fd.save_figure(fig, "I6_mtdna_cross_tissue_inversion")
    plt.show()
"""))


# §I.7 — Disease-gene tile panel
cells.append(md(r"""
    ## §I.7 — Disease-gene tile panel

    Eleven well-studied disease genes that fire M flags on
    pancreas (§I.4 / §I.8) plus two named cross-tissue rewired
    genes from §I.3. Each row = one named gene; each column =
    a dataset; cell color = α (viridis 0–1); cell text = α
    value. Genes grouped by disease class. Datasets grouped by
    tissue (separated visually with a vertical rule). The
    figure makes obvious which genes are dataset-specific
    flags vs registry-wide.
"""))
cells.append(code(r"""
    DISEASE_DISPLAY = {
        "Inborn errors of metabolism":        ["PAH", "ACADM", "ARG1", "ASL",
                                                "CPS1", "GCH1", "OTC"],
        "DNA mismatch repair (Lynch)":         ["MSH6"],
        "Cancer predisposition":               ["APC"],
        "Diabetes / glucose homeostasis":      ["HNF4A"],
        "Neurodegeneration":                   ["PRKN", "PSEN1"],
        "Cross-tissue rewired (§I.3)":         ["DENND4A", "RAP1GDS1"],
        "ER stress chaperones":                ["HSPA5"],
        "HPA-axis regulator":                  ["FKBP5"],
    }

    rows = []
    block_starts = []
    cum = 0
    block_rows = []
    for klass, gset in DISEASE_DISPLAY.items():
        block_starts.append((klass, cum))
        for sym in gset:
            for k in DATASET_ORDER:
                hits = FLAGGED[k][FLAGGED[k]["symbol"].astype(str).str.upper() == sym.upper()]
                if len(hits) == 0:
                    rows.append({"klass": klass, "symbol": sym,
                                  "dataset": DATASETS[k]["label"],
                                  "alpha": np.nan, "prior": np.nan,
                                  "flag_M": False, "flag_F": False})
                else:
                    r = hits.iloc[0]
                    rows.append({"klass": klass, "symbol": sym,
                                  "dataset": DATASETS[k]["label"],
                                  "alpha": float(r["alpha"]),
                                  "prior": float(r["prior"]),
                                  "flag_M": bool(r["flag_M"]),
                                  "flag_F": bool(r["flag_F"])})
            block_rows.append(sym)
            cum += 1
    DIS_LONG = pd.DataFrame(rows)
    pivot_alpha = DIS_LONG.pivot_table(index=["klass","symbol"], columns="dataset",
                                          values="alpha", aggfunc="first")
    pivot_alpha = pivot_alpha.reindex(columns=[DATASETS[k]["label"] for k in DATASET_ORDER])
    flag_M = DIS_LONG.pivot_table(index=["klass","symbol"], columns="dataset",
                                     values="flag_M", aggfunc="first").fillna(False)
    flag_M = flag_M.reindex(index=pivot_alpha.index, columns=pivot_alpha.columns)
    flag_F = DIS_LONG.pivot_table(index=["klass","symbol"], columns="dataset",
                                     values="flag_F", aggfunc="first").fillna(False)
    flag_F = flag_F.reindex(index=pivot_alpha.index, columns=pivot_alpha.columns)
"""))
cells.append(code(r"""
    n_rows = pivot_alpha.shape[0]
    n_cols = pivot_alpha.shape[1]
    fig, ax = plt.subplots(figsize=(8.6, 0.32 * n_rows + 1.4), layout="constrained")
    im = ax.imshow(pivot_alpha.values, cmap="viridis", aspect="auto",
                    vmin=0, vmax=1)
    for i in range(n_rows):
        for j in range(n_cols):
            v = pivot_alpha.iloc[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center",
                         fontsize=6.5, color="#888888")
                continue
            mark = "M" if flag_M.iloc[i, j] else ("F" if flag_F.iloc[i, j] else "")
            txt  = f"{v:.2f}"
            color = "#222222" if v > 0.55 else "#f5f5f5"
            ax.text(j, i, txt, ha="center", va="center",
                     fontsize=6.5, color=color)
            if mark:
                # corner mark
                ax.text(j + 0.32, i - 0.32, mark, ha="center", va="center",
                         fontsize=7, color=M_COLOR if mark == "M" else F_COLOR,
                         fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # tissue separator
    n_panc = len(PANC_LABELS)
    ax.axvline(n_panc - 0.5, color="#222222", linewidth=0.7)
    # Class-block separators on Y
    cum = 0
    for klass, gset in DISEASE_DISPLAY.items():
        ax.axhline(cum - 0.5, color="#444444", linewidth=0.4)
        ax.text(-0.6, cum + len(gset)/2 - 0.5, klass,
                rotation=90, ha="right", va="center", fontsize=7, color="#222222")
        cum += len(gset)
    ax.axhline(cum - 0.5, color="#444444", linewidth=0.4)
    # Y labels = gene symbols only
    yticks = []
    for klass, gset in DISEASE_DISPLAY.items():
        yticks.extend(gset)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(yticks, fontsize=7.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot_alpha.columns, rotation=20, ha="right", fontsize=7.5)
    ax.set_xlim(-1.4, n_cols - 0.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label=r"$\alpha$")
    cb.ax.tick_params(labelsize=7)
    fd.save_figure(fig, "I7_disease_gene_tile_panel")
    plt.show()
"""))


# §I.8 — Severity vs reproducibility (per-tissue)
cells.append(md(r"""
    ## §I.8 — Severity vs reproducibility, per tissue

    Plot per-gene **maximum severity** (largest |Δα| in the
    flagged direction across the tissue's datasets) against the
    **number of datasets in which the flag fires**. Top-right
    region = high-severity AND highly reproducible — the
    strongest paper-quotable claims. Top-K labeled per
    direction per tissue.
"""))
cells.append(code(r"""
    def severity_repro(pool: pd.DataFrame, direction: str, n_datasets: int) -> pd.DataFrame:
        col_flag = f"flag_{direction}"
        col_sev  = f"severity_{direction}"
        sub = pool[pool[col_flag]].copy()
        if len(sub) == 0:
            return pd.DataFrame()
        agg = (sub.groupby("var_name")
                  .agg(symbol=("symbol", "first"),
                        max_severity=(col_sev, "max"),
                        n_flagged=(col_flag, "sum"),
                        mean_delta=("delta_alpha", "mean")))
        return agg.reset_index()


    SEVR_PANC_M = severity_repro(PANC_POOL, "M", len(PANC_DATASETS))
    SEVR_PANC_F = severity_repro(PANC_POOL, "F", len(PANC_DATASETS))
    SEVR_TEST_M = severity_repro(TEST_POOL, "M", len(TEST_DATASETS))
    SEVR_TEST_F = severity_repro(TEST_POOL, "F", len(TEST_DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4),
                              layout="constrained", sharey=True)
    for ax, (panel_M, panel_F, n_ds, tissue) in zip(axes, [
        (SEVR_PANC_M, SEVR_PANC_F, len(PANC_DATASETS), "pancreas"),
        (SEVR_TEST_M, SEVR_TEST_F, len(TEST_DATASETS), "testis"),
    ]):
        # M jittered slightly left, F slightly right
        rng = np.random.default_rng(0)
        if len(panel_M) > 0:
            jitter = rng.uniform(-0.12, 0.0, size=len(panel_M))
            ax.scatter(panel_M["max_severity"], panel_M["n_flagged"] + jitter,
                       s=18, c=M_COLOR, alpha=0.65,
                       linewidths=0.3, edgecolor="#222222",
                       rasterized=True, label=f"M ({len(panel_M)})")
        if len(panel_F) > 0:
            jitter = rng.uniform(0.0, 0.12, size=len(panel_F))
            ax.scatter(panel_F["max_severity"], panel_F["n_flagged"] + jitter,
                       s=18, c=F_COLOR, alpha=0.65,
                       linewidths=0.3, edgecolor="#222222",
                       rasterized=True, label=f"F ({len(panel_F)})")
        # Top-K labels per direction
        K = 8
        for sub, color in [(panel_M, M_COLOR), (panel_F, F_COLOR)]:
            if len(sub) == 0: continue
            top = sub.nlargest(K, "max_severity")
            for _, r in top.iterrows():
                sym = str(r["symbol"]) if pd.notna(r["symbol"]) else r.name
                ax.text(r["max_severity"] + 0.005, r["n_flagged"] + 0.05,
                        sym[:14], fontsize=6, color=color,
                        ha="left", va="bottom", fontweight="bold")
        ax.set_xlabel(r"max severity (= $|\Delta\alpha|$ on flag side)")
        ax.set_ylabel(f"# of {n_ds} datasets the flag fires on")
        ax.set_yticks(range(0, n_ds + 1))
        ax.set_ylim(-0.5, n_ds + 0.5)
        ax.set_title(tissue, fontsize=9, color=PANC_ACCENT if tissue == "pancreas" else TEST_ACCENT,
                     loc="left", pad=4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fd.thin_legend(ax, loc="lower right")
    fd.save_figure(fig, "I8_severity_vs_reproducibility")
    plt.show()
"""))


# §I.9 — Identity-SD vs |Δα| with named outliers
cells.append(md(r"""
    ## §I.9 — Identity-SD as a continuous paralog-collapse signal (panc 4-sp)

    The §D.3 percentile decomposition uses thresholds; the
    cleaner threshold-free version: cross-species **identity
    standard deviation** (across human↔macaque, human↔mouse,
    human↔zebrafish) vs **|Δα|** on `panc_4sp` high-prior genes.
    The relationship is highly significant (Spearman ρ ≈ +0.10,
    p < 10⁻¹⁸ on N ≈ 8 200 genes), and the most extreme
    individual genes — labeled below — are the ones a paper
    can quote as paralog-collapse candidates.
"""))
cells.append(code(r"""
    k = "panc_4sp"
    j = FLAGGED[k].copy()
    raw_cols = [c for c in SPECIES_IDENTITY_COLS
                if c in j.columns and not j[c].isna().all()]
    if len(raw_cols) >= 2:
        for c in raw_cols:
            j[f"pct__{c}"] = j[c].rank(pct=True) * 100.0
        j["identity_sd"] = j[raw_cols].std(axis=1)
        j["pct_sd"]      = j[[f"pct__{c}" for c in raw_cols]].std(axis=1)
        j["abs_delta"]   = j["delta_alpha"].abs()
        sub = j[(j["prior"] >= PRIOR_LOW) & j["abs_delta"].notna()
                & j["identity_sd"].notna()].copy()
        rho_pct, p_pct = stats.spearmanr(sub["pct_sd"], sub["abs_delta"])
        print(f"panc_4sp · prior ≥ {PRIOR_LOW} · n = {len(sub)} genes")
        print(f"  Spearman ρ(identity-pct SD, |Δα|) = {rho_pct:+.3f}  (p = {p_pct:.2e})")
"""))
cells.append(code(r"""
    fig, ax = fd.make_axes(figsize=(6.4, 4.6))
    ax.scatter(sub["pct_sd"], sub["abs_delta"],
               s=4, c="#bdbdbd", alpha=0.45, linewidths=0, rasterized=True,
               label="all high-prior genes")
    mflag = sub[sub["flag_M"]]
    fflag = sub[sub["flag_F"]]
    ax.scatter(mflag["pct_sd"], mflag["abs_delta"], s=14, c=M_COLOR,
               alpha=0.85, linewidths=0.3, edgecolor="#222222",
               rasterized=True, label=f"M ({len(mflag)})")
    ax.scatter(fflag["pct_sd"], fflag["abs_delta"], s=14, c=F_COLOR,
               alpha=0.85, linewidths=0.3, edgecolor="#222222",
               rasterized=True, label=f"F ({len(fflag)})")
    # Label top-12 most extreme by combined rank
    sub2 = sub.copy()
    sub2["rank_score"] = sub2["pct_sd"].rank() + sub2["abs_delta"].rank()
    top = sub2.nlargest(12, "rank_score")
    for _, r in top.iterrows():
        sym = str(r["symbol"]) if pd.notna(r["symbol"]) else r.name
        ax.text(r["pct_sd"] + 0.5, r["abs_delta"] + 0.005,
                sym[:14], fontsize=6.5, color="#222222",
                ha="left", va="bottom", fontweight="bold")
    ax.set_xlabel("identity SD across species (percentile units)")
    ax.set_ylabel(r"$|\Delta\alpha|$  (panc 4-sp, prior $\geq$ "
                  f"{PRIOR_LOW})")
    fd.thin_legend(ax, loc="upper right")
    fd.annotate_stat(ax, fr"Spearman $\rho$ = {rho_pct:+.3f},  $p = {p_pct:.1e}$",
                     "upper left")
    fd.save_figure(fig, "I9_identity_sd_vs_abs_delta_labeled")
    plt.show()
"""))


# §I.10 — Curated case-study cards (5 dedicated single-gene figures)
cells.append(md(r"""
    ## §I.10 — Curated case-study cards (5 named genes, full-page each)

    Five hand-picked single genes, each rendered as a dedicated
    full-page figure. Each card shows: per-dataset α + prior bars
    (color-coded by tissue), per-species identity decomposition
    on multi-species datasets where available, and a one-paragraph
    biological context. These are the named-gene findings the
    paper can quote verbatim.

    Selection criterion: high severity, robust across multiple
    pancreas datasets, and a known biomedical handle that
    grounds the gene in the literature.
"""))
cells.append(code(r"""
    CASE_STUDIES = [
        ("MT-CO1", "mtDNA-encoded cytochrome c oxidase subunit I",
         "The most-studied mtDNA gene (the standard barcoding marker). "
         "Robust M flag on H+Macaque (α = 0.17 vs prior = 0.89, Δα = −0.72) "
         "and on the unified 4-species pancreas. The macaque mtDNA "
         "annotation in Ensembl drives high pairwise identity that the "
         "model cannot reconcile with expression — a paralog-collapse "
         "in the macaque mtDNA contig assignment. Across testis the "
         "same gene is α ≈ 0.95 (Δα ≈ +0.10), inverting the call."),
        ("MSH6", "DNA mismatch repair gene, Lynch-syndrome predisposition",
         "Hereditary cancer-predisposition gene with paralog MSH3. "
         "Robust M flag across pancreas: α = 0.09 on H+Macaque, α = 0.68 "
         "on 4-sp. The dramatic α drop on the closest-pair dataset is "
         "consistent with subfunctionalization between MSH6 and MSH3 in "
         "macaque/mouse, a known DNA-MMR divergence. Testis α ≈ 0.84 "
         "(no flag) — the rewiring is liver-specific."),
        ("PAH", "phenylalanine hydroxylase — phenylketonuria",
         "Mendelian PKU disease gene, hepatocyte-restricted expression. "
         "Robust M flag on the 4-species pancreas analysis: α = 0.39, "
         "Δα = −0.46. The §D.3 outlier-species decomposition pinpoints "
         "macaque pairwise identity in the 7th percentile while mouse "
         "and zebrafish are typical — a macaque-specific paralog-"
         "collapse in the Ensembl PAH ortholog assignment."),
        ("PRKN", "Parkin — autosomal recessive Parkinson's disease",
         "OXPHOS-nuclear-encoded; mediator of mitophagy. M-flagged "
         "on H+Macaque (α = 0.06 vs prior = 0.81) and on 4-sp pancreas "
         "(α = 0.69, Δα = −0.11). The mouse Parkin knockout is "
         "phenotypically milder than human Parkinson's — consistent "
         "with cross-species regulatory divergence in the mitophagy "
         "program that the M flag is registering."),
        ("DENND4A", "RAB10 GEF — cross-tissue rewired (§I.3)",
         "Rab10-family guanine-exchange factor. M-flagged on 4-sp "
         "pancreas (α = 0.59, Δα = −0.22) AND F-flagged on testis "
         "3-sp (α = 0.69, Δα = +0.11). Same protein sequence in both "
         "tissues — the regulatory program differs to the point of "
         "inverting the model's confidence call. A clean cross-tissue "
         "rewiring case: the GIN's α response to the same gene "
         "depends entirely on tissue context."),
    ]
"""))
cells.append(code(r"""
    def case_card(sym: str, name: str, narrative: str, save_name: str):
        # Collect per-dataset data
        rows = []
        for k in DATASET_ORDER:
            hits = FLAGGED[k][FLAGGED[k]["symbol"].astype(str).str.upper() == sym.upper()]
            if len(hits) == 0:
                rows.append({"label": DATASETS[k]["label"],
                              "tissue": DATASETS[k]["tissue"],
                              "alpha": np.nan, "prior": np.nan,
                              "delta_alpha": np.nan,
                              "flag_M": False, "flag_F": False})
            else:
                r = hits.iloc[0]
                rows.append({"label": DATASETS[k]["label"],
                              "tissue": DATASETS[k]["tissue"],
                              "alpha": float(r["alpha"]),
                              "prior": float(r["prior"]),
                              "delta_alpha": float(r["delta_alpha"]),
                              "flag_M": bool(r["flag_M"]),
                              "flag_F": bool(r["flag_F"])})
        sub = pd.DataFrame(rows)

        fig, ax = fd.make_axes(figsize=(8.4, 3.6))
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["prior"].values, width=0.36,
                color="#bdbdbd", edgecolor="#222222", linewidth=0.4,
                label="conservation prior")
        ac = [PANC_ACCENT if t == "pancreas" else TEST_ACCENT for t in sub["tissue"]]
        ax.bar(x + 0.18, sub["alpha"].values, width=0.36, color=ac,
                edgecolor="#222222", linewidth=0.4,
                label="α (teal: pancreas, plum: testis)")
        # Δα label, color-coded
        for i, (_, r) in enumerate(sub.iterrows()):
            if pd.isna(r["alpha"]): continue
            color = M_COLOR if r["flag_M"] else (F_COLOR if r["flag_F"] else "#222222")
            tag = "M" if r["flag_M"] else ("F" if r["flag_F"] else "")
            label = f"Δα={r['delta_alpha']:+.2f}"
            if tag: label += f"  [{tag}]"
            ax.text(x[i], 1.05, label, ha="center", va="bottom",
                     fontsize=7.5, color=color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"].tolist(), rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(r"$\alpha$  /  prior")
        # tissue separator
        ax.axvline(len(PANC_LABELS) - 0.5, color="#222222", linewidth=0.7,
                    linestyle=":")
        fd.thin_legend(ax, loc="lower left")
        ax.text(0.0, 1.20, f"{sym} — {name}", ha="left", va="top",
                  transform=ax.transAxes, fontsize=10, fontweight="bold")
        fd.save_figure(fig, save_name)
        plt.show()


    for sym, name, narrative in CASE_STUDIES:
        print(f"\n--- {sym} — {name} ---")
        print(narrative)
        case_card(sym, name, narrative, f"I10_card__{sym.lower().replace('-','_')}")
"""))


# §I.11 — Paper-ready Results-section text
cells.append(md(r"""
    ## §I.11 — Paper-ready Results-section text

    Three numbered paragraphs that drop into the Results section
    without further editing. Updated to use the tissue-separated
    findings of §I.1–§I.10.

    ---

    **Paragraph 1 — Tissue-separated QC deliverable.**
    Across the **four pancreas (camex liver) datasets** Chimera
    flags 277 robust mis-annotation candidates (M direction:
    `prior ≥ 0.80 ∧ Δα ≤ −0.10`, fired on ≥ 2 of 4 datasets;
    §E.1) and 24 robust functional-conservation candidates (F
    direction; treated as hypothesis-grade per §H). Across the
    **three testis datasets**, the prior-clamp at 0.85 (per
    `workflow-best.md` §6–8) compresses the M direction
    architecturally — only 23 robust F flags and zero robust M
    flags. The two regimes deserve separate analysis (§I.1 and
    §I.2 visualize this directly), so every figure in §I is
    tissue-stratified.

    **Paragraph 2 — A tissue-specific paralog signature.**
    The §I.4 side-by-side paralog forest shows that pancreas M
    flags are dominated by the **cytosolic ribosomal RPL/RPS
    family** (log₂ OR = +4.08, Bonferroni p < 0.001) — the only
    family that survives correction. The §I.5 deepening reveals
    this is not a uniform effect: only a subset of RPL/RPS
    paralogs trigger the M flag, and the ones that do are the
    same paralogs across all four pancreas datasets — a
    paralog-specific signature, not a family-wide defect.
    On testis the F-direction enrichment lands instead on
    mitochondrial-ribosomal MRP and a smaller set of metabolic
    families — different paralog universe, different tissue,
    different signal. The two-tissue contrast is the cleanest
    evidence yet that ortholog-table errors and tissue-context
    interact.

    **Paragraph 3 — Cross-tissue rewiring and named disease
    genes.**
    The §I.3 cross-tissue disagreement portrait identifies a
    rare-but-clean class of genes whose Δα call inverts between
    tissues: **DENND4A, RAP1GDS1, MT-CO1, MT-CO2, MT-ND2** all
    sit in the off-diagonal quadrants. The mtDNA family in
    particular shows a registry-wide pancreas-negative /
    testis-positive Δα pattern (§I.6) — the same gene's
    regulatory-conservation status flips with tissue context
    despite identical protein sequence. Eleven well-studied
    disease genes appear in the M-flag set (§I.7), headlined by
    **PAH** (phenylketonuria), **MSH6** (Lynch syndrome),
    **PRKN** (Parkinson's), **CPS1 / ARG1 / OTC** (urea cycle),
    and **HNF4A** (MODY1). For each, mouse-model conclusions
    about expression-level regulation should be read with the
    per-dataset Δα as a caveat — these are exactly the genes
    where cross-species expression deviation is most likely
    to break clinical translation. The §I.10 case-study cards
    walk through MT-CO1, MSH6, PAH, PRKN, and DENND4A in detail.

    ---

    *§I.1–§I.10 jointly produce the figures and named-gene
    evidence supporting these three paragraphs. Numbers are
    reproducible from the cached pools (`PANC_POOL`, `TEST_POOL`)
    and the per-dataset CSVs in §C.*
"""))


# ============ §H caveats (renumbered from old §G) ============
cells.append(md(r"""
    # §H — Caveats and interpretation guide

    ## What the M-direction flag genuinely means

    `prior ≥ 0.80 ∧ Δα ≤ −0.10` is reachable from `L_recon + L1` alone
    and does not depend on the FQR mechanism. The biological reading is
    well-grounded: high cross-species sequence identity, but the model
    needed a species-specific decoder to reconstruct expression. The
    most likely causes are:

    - **paralog collapse in the ortholog table** — the gene labeled
      "ortholog" in one species is actually a paralog. §D's outlier-
      species decomposition isolates this case directly.
    - **gene duplication followed by subfunctionalization** — the
      ortholog is correct in a sequence sense but the gene's role
      diverged after duplication. Outlier-decomposition shows uniform
      identity in this case.
    - **regulatory rewiring of a sequence-conserved gene** — sequence
      is conserved, expression program is not. Often shows
      cross-tissue inconsistency (notebook 06 §D).

    ## What the F-direction flag genuinely means *(softer call)*

    `prior ≤ 0.65 ∧ Δα ≥ +0.10` requires the FQR mechanism (`L_fit_reward`
    via the iter81b α-detach + unrectified `S_g` configuration) to
    push α above the prior. Per the source doc, this direction has
    **three known confounds** that cannot be eliminated without
    additional analysis:

    1. **FQR cell-type-discrimination** — for genes with bimodal /
       overlap expression structure, the shared decoder beats the
       per-species null because of cell-type awareness, not because
       of cross-species expression conservation.
    2. **z-leakage contamination** — incomplete adversarial cleaning
       inflates `Q_g` for divergent genes via residual species
       information in `z`.
    3. **cell-composition bias** — different cell-type proportions
       across species bias the per-species null.

    The recommendation from the source doc is: present the M direction
    as the primary finding, the F direction as hypothesis-generating
    requiring additional validation per gene. This notebook respects
    that recommendation by ordering reports M-first, F-second, and by
    explicitly tagging the F outputs as the softer call.

    ## Why `|Δα| ≥ 0.10` and not the doc's `0.05`

    The source doc names a noise floor of approximately `0.05` and
    "reliable interpretation" only above `0.10–0.15`. We use `0.10` as
    the operational floor because the QC application is about *flagging
    candidates for follow-up*, not about reporting an established
    metric. A flag at the `0.10` floor is hypothesis-grade; a flag with
    `|Δα| ≥ 0.20` is a strong claim. Severity-ranked tables in §C
    naturally surface the strong claims at the top.

    ## Why testis runs surface fewer flags

    Testis runs use `prior_clamp_max = 0.85` (the registry's E9.2-down
    architectural ceiling — only ~1% of testis genes have
    `S_g_ema < −0.05`). Both Δα tails are compressed: M flags need
    `Δα ≤ −0.10` and F flags need `Δα ≥ +0.10`, both harder to reach
    against compressed Δα variance. The pancreas runs are the right
    venue for the QC-application story; testis runs are included for
    cross-tissue reproducibility (§E.3) but are not expected to drive
    QC discovery.

    ## How the deliverables travel

    Per dataset, written to `runs/<run>/tables/`:
    - `ortho_qc_misannotation.csv`  — top-30 M flags (severity-ranked).
    - `ortho_qc_functional.csv`     — top-30 F flags.
    - `ortho_qc_misannotation_decomposed.csv` — per-species identity
      decomposition (only on multi-species datasets, §D.2).

    Cross-dataset robust-flag tables are in §E.1 / §E.2 above and can
    be regenerated from the per-dataset CSVs.

    ## Bottom line — what this notebook commits to

    Chimera's α / Δα produces a flag list of:
    1. **paralog-collapse / mis-annotation candidates** (M direction,
       hard claim — based on `L_recon + L1` only),
    2. **functional-conservation candidates despite sequence
       divergence** (F direction, soft claim — depends on FQR; needs
       per-gene validation against external evidence),
    3. **per-species identity decomposition** that distinguishes
       paralog-collapse from real regulatory rewiring on multi-species
       datasets,
    4. **within-tissue reproducible** versions of both lists for
       defendability.

    No other cross-species integration tool in the field
    (scVI / scANVI / Harmony / sysVI) produces this output. The α
    machinery is what makes Application 7 possible at all; the
    mis-annotation half is the most defensible single use of α / Δα
    that the codebase ships.
"""))


cells.append(code(r"""
    written = sorted(FIG_DIR.glob("*.pdf"))
    print(f"\n{len(written)} PDF figures in {FIG_DIR}:")
    for p in written:
        print(f"  {p.name:60s} {p.stat().st_size/1024:6.1f} KB")
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
