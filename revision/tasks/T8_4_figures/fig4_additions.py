"""Fig 4 additions: re-rendered 4c, cell-type drivers, margin support, corrected cross-tissue panel.

  fig4c_prior_landscape    Re-render of Fig 4c from code (D12): protein-identity prior vs Delta-alpha on
                           liver 4 species, the same four named gene groups as the original panel, no
                           internal run name on any axis. A dagger marks a named tail gene (|Delta-alpha|
                           > 0.05) that lacks margin support (T2.5: count evidence S_g unmasked, of the sign
                           of Delta-alpha and at least the cohort median |S_g|), i.e. imputed from features.
  fig4_celltype_drivers    Where each named gene's count-evidence margin S_g comes from: the exact
                           per-cell-type split C_gk (posterior-mean z) of T2.5, liver 4 species.
  fig4_margin_support      Fraction of tail genes (|Delta-alpha| > 0.05) with margin support, per cohort and
                           tail, one trained run per cohort: T2.5 tables on the canonical s42 runs, and the same
                           flag (``margin.margin_flag``) on the first kept run of every other cohort in H1
                           ``runs.tsv``, so later cohorts join when H1 lists them. Checked against T2.5.
  fig4_cross_tissue        Corrected cross-tissue comparison (T2.3 primary version): Delta-alpha against
                           the raw prior, centred on the cohort median, same species pair, genes joined
                           on Ensembl ID. Replaces old Fig 7e (mean over cohorts, clamped prior).
Sources: ``results/revision/T2_5/`` (canonical s42 runs), ``results/revision/T2_3/tissue_comparison.csv``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from revision.tasks.T8_4_figures import common as cm

T25 = cm.RES / "T2_5"
T23 = cm.RES / "T2_3"
# Gene groups of the original Fig 4c (notebooks/_alpha_biology_builder.py, H_THEMES), same colours.
THEMES = {
    "urea cycle / gluconeogenesis": (["CPS1", "ARG1", "ASL", "OTC", "PCK1", "PAH"], "#0072B2"),
    "mitochondrially encoded OXPHOS": (["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4",
                                        "MT-ATP6"], "#5B3E80"),
    "liver support": (["SELP", "SELE", "VCAM1", "F3", "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC"],
                      "#D55E00"),
    "housekeeping": (["DDX4", "RPL3", "EEF1A1", "EEF2", "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A"], "#7F7F7F"),
}
DAGGER = "†"


def genes_4sp() -> pd.DataFrame:
    return pd.read_parquet(T25 / "liv_4sp__genes.parquet")


def named(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme, (genes, col) in THEMES.items():
        s = g[g.symbol.isin(genes)].copy()
        s["theme"], s["color"] = theme, col
        rows.append(s)
    out = pd.concat(rows)
    # Dagger: T2.5 flag ``lacks_margin_support`` (a tail gene, |Delta-alpha| > 0.05, whose count evidence
    # does not back its sign), as planned for the main panel. Near-zero genes outside the tail carry no
    # claim and no dagger.
    out["label"] = out.symbol + np.where(out.lacks_margin_support, DAGGER, "")
    return out


def panel_landscape() -> plt.Figure:
    g = genes_4sp()
    nm = named(g)
    # Size of the original panel c beside the original panels a, b (F1).
    fig = plt.figure(figsize=(2.76, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5.5, 1], height_ratios=[1, 6.5], hspace=0.03, wspace=0.03,
                          left=0.155, right=0.995, top=0.995, bottom=0.215)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    cm.fd.density_scatter(ax, g.prior, g.delta_alpha, s=1.6, alpha=0.25, color="#BBBBBB")
    texts, handles = [], []
    for theme, (genes, col) in THEMES.items():
        s = nm[nm.theme == theme]
        handles.append(ax.scatter(s.prior, s.delta_alpha, s=16, color=col, edgecolor="white", linewidth=0.4,
                                  zorder=4, label=theme))
        for r in s.itertuples():
            texts.append(ax.text(r.prior, r.delta_alpha, r.label, fontsize=5, ha="center", va="center",
                                 color=cm.INK, zorder=5))
    ax.axhline(0, color="#888888", lw=0.5, ls="--", zorder=2)
    ax.set_xlabel("protein-identity prior π")
    ax.set_ylabel("Δα (liver, 4 species)")
    ax.set_xlim(0.15, 1.0)
    handles.append(Line2D([], [], ls="none", label=f"{DAGGER} no margin support: Δα not backed by counts"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=5.5, frameon=False,
               handletextpad=0.1, labelspacing=0.15, borderpad=0)
    top.hist(g.prior, bins=60, color="#999999", edgecolor="none")
    right.hist(g.delta_alpha, bins=60, color="#999999", edgecolor="none", orientation="horizontal")
    for a, hide in ((top, ("top", "right", "left")), (right, ("top", "right", "bottom"))):
        for sp in hide:
            a.spines[sp].set_visible(False)
    top.set_yticks([])
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    right.set_xticks([])
    right.tick_params(axis="y", left=False, labelleft=False)
    adjust_text(texts, x=nm.prior.to_numpy(), y=nm.delta_alpha.to_numpy(), ax=ax, expand=(1.5, 2.2),
                force_text=(0.7, 1.0), force_static=(0.5, 0.9), force_explode=(0.4, 0.8),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.3), min_arrow_len=1.0)
    return fig


CT_GROUPS = [("hepatocytes", ["Hepatocytes"], "#E69F00"), ("endothelial", ["Endothelial"], "#56B4E9"),
             ("stellate / fibroblasts", ["fibroblasts"], "#009E73"),
             ("Kupffer cells and macrophages", ["Kupffer cells", "Macrophages"], "#CC79A7"),
             ("other cell types", None, "#BBBBBB")]


def panel_drivers() -> plt.Figure:
    g = genes_4sp()
    nm = named(g).sort_values("delta_alpha")
    C = pd.read_parquet(T25 / "liv_4sp__C_gk.parquet")
    C = C[C.var_name.isin(nm.var_name)]
    named_types = {t for _, ts, _ in CT_GROUPS if ts for t in ts}
    W = C.groupby("cell_type").weight_k.first()
    fig, ax = plt.subplots(figsize=(3.25, 5.05), layout="constrained")
    for y, r in enumerate(nm.itertuples()):
        c = C[C.var_name == r.var_name].set_index("cell_type").C_gk_meanz
        pos = neg = 0.0
        for name, ts, col in CT_GROUPS:
            v = c[[t for t in c.index if t not in named_types]].sum() if ts is None else c.reindex(ts).fillna(0).sum()
            if v >= 0:
                ax.barh(y, v, left=pos, color=col, height=0.72, lw=0)
                pos += v
            else:
                ax.barh(y, v, left=neg, color=col, height=0.72, lw=0)
                neg += v
        ax.plot([c.sum()], [y], marker="|", ms=5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(nm)))
    ax.set_yticklabels([f"{r.label}  ({r.delta_alpha:+.2f})" for r in nm.itertuples()], fontsize=5.3)
    ax.set_ylim(len(nm) - 0.4, -0.6)
    ax.set_xlabel("contribution to count-evidence margin $S_g$, by cell type\n(liver, 4 species; tick: total $S_g$)",
                  fontsize=6.5)
    hs = [Patch(color=col, label=f"{name} (weight {W.reindex(ts).sum():.2f})" if ts else name)
          for name, ts, col in CT_GROUPS]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=5.2, handlelength=1.0,
               columnspacing=0.8)
    ax.text(0.01, 0.01, f"(Δα in brackets)   {DAGGER} no margin support", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5, color=cm.MUTED)
    return fig


MS_COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
TISSUES = ["liver", "testis", "bone marrow", "kidney", "islet", "eye"]


def cohort_rows() -> pd.DataFrame:
    """Every cohort of H1 in tissue order: cohort, tissue, label, first kept run."""
    runs = pd.read_csv(cm.RES / "H1" / "runs.tsv", sep="\t")
    runs = runs[runs.kept.astype(str) == "True"].sort_values("seed").groupby("cohort", sort=False).first().reset_index()
    table = pd.read_csv(cm.RES / "H1" / "cohort_table.csv").set_index("cohort")
    labels = table.label
    runs = runs.set_index("cohort").reindex(table.index).reset_index()
    runs["label"] = [cm.COHORT.get(c, labels[c].replace(" H+", ", human + ").lower().replace("m. mulatta", "M. mulatta")
                                   .replace("marrow 3", "marrow, 3").replace("eye 5", "eye, 5").replace("eye primates", "eye, primates"))
                     for c in runs.cohort]
    runs["order"] = [TISSUES.index(t) for t in runs.tissue]
    return runs.sort_values("order", kind="stable").reset_index(drop=True)


def _run_tail(c: str, run: str) -> pd.DataFrame:
    """One run: Delta-alpha against the prior it anchors on, S_g_ema, margin flag of T2.5."""
    sys.path.insert(0, str(cm.REPO / "revision" / "tasks" / "T2_5_celltype_margin"))
    import margin as MG

    if c in MS_COHORTS:
        g = pd.read_parquet(T25 / f"{c}__genes.parquet")
        return g[g["tail"]][["delta_alpha", "margin_support"]]
    with np.load(cm.REPO / "training" / "runs" / run / "final_state.npz", allow_pickle=True) as z:
        da, S = z["alpha"] - z["prior"], z["S_g_ema"]
    t = pd.DataFrame(dict(delta_alpha=da, margin_support=MG.margin_flag(da, S)))
    return t[np.abs(da) > MG.TAIL]


def panel_margin_support() -> plt.Figure:
    h = pd.read_csv(T25 / "headline_margin_support.csv")
    h = h[(h.run_label == h.cohort) & (h.group_key == "cell_type")].set_index("cohort")
    rows = cohort_rows()
    # Exact size and axes, shared with the sequence-axis panel below it in Fig 4 (F1).
    pitch = 12.2 / 72
    H = pitch * len(rows) + 0.66
    fig = plt.figure(figsize=(3.4, H))
    ax = fig.add_axes([1.06 / 3.4, 0.48 / H, 2.3 / 3.4, pitch * len(rows) / H])
    source = []
    for i, r0 in enumerate(rows.itertuples()):
        c = r0.cohort
        t = _run_tail(c, r0.run)
        neg, pos = t[t.delta_alpha < 0], t[t.delta_alpha > 0]
        for k, (sub, col, name) in enumerate([(neg, "#0072B2", "negative tail (Δα < −0.05)"),
                                               (pos, "#D55E00", "positive tail (Δα > +0.05)")]):
            y = i + (k - 0.5) * 0.46
            source.append(dict(cohort=c, run=r0.run, tail=name[:8], n=len(sub),
                               frac=sub.margin_support.mean() if len(sub) else np.nan))
            if len(sub) == 0:
                ax.text(0.01, y, "no genes", va="center", fontsize=5, color=cm.MUTED)
                continue
            f = sub.margin_support.mean()
            ax.barh(y, f, height=0.42, color=col, lw=0, label=name if not ax.containers or len(ax.containers) < 2 else None)
            ax.text(f + 0.01, y, f"{f:.2f}  (n {len(sub):,})", va="center", fontsize=5, color=cm.INK)
        if c in MS_COHORTS:
            assert abs(h.loc[c, "frac_tail_with_margin_support"] - t.margin_support.mean()) < 1e-9
    pd.DataFrame(source).to_csv(cm.RES / "T8_4" / "fig4_margin_support_source.csv", index=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.label, fontsize=5.8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in np.where(np.diff(rows.order))[0]:
        ax.axhline(i + 0.5, color=cm.RULE, lw=0.4)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of tail genes whose Δα is backed by\ncount evidence of the same sign (margin support)",
                  fontsize=6, labelpad=1)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.3, handlelength=1.0, frameon=False,
               borderpad=0, columnspacing=0.8, handletextpad=0.3)
    return fig


PAIRS = [("liv_hmonkey", "test_hmac", "H+Monkey vs H+Macaque", "human + monkey (liver) vs human + macaque (testis)"),
         ("liv_hmouse", "test_hmouse", "H+Mouse vs H+Mouse", "human + mouse, liver vs testis")]


def _centred(c: str) -> pd.DataFrame:
    g = pd.read_parquet(T25 / f"{c}__genes.parquet").dropna(subset=["hsapiens_gene_id"])
    g = g.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["da"] = g.delta_alpha_raw - g.delta_alpha_raw.median()
    return g


def panel_cross_tissue() -> plt.Figure:
    ref = pd.read_csv(T23 / "tissue_comparison.csv")
    ref = ref[ref.version == "raw_centred"].set_index("pair")
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.95), layout="constrained")
    for ax, (lc, tc, key, title) in zip(axes, PAIRS):
        L, T = _centred(lc), _centred(tc)
        ix = L.index.intersection(T.index)
        x, y = L.loc[ix, "da"], T.loc[ix, "da"]
        rho = spearmanr(x, y).statistic
        r = ref.loc[key]
        assert abs(rho - r.gene_rho) < 1e-9 and len(ix) == r.n_genes, (key, rho, r.gene_rho, len(ix))
        cm.fd.density_scatter(ax, x, y, s=1.4, alpha=0.2, color="#888888")
        mt = L.loc[ix, "symbol"].astype(str).str.startswith("MT-")
        ax.scatter(x[mt], y[mt], s=14, color="#5B3E80", edgecolor="white", lw=0.4, zorder=4,
                   label="mitochondrially encoded genes")
        ax.axhline(0, color=cm.MUTED, lw=0.5)
        ax.axvline(0, color=cm.MUTED, lw=0.5)
        lim = 1.08 * max(np.nanquantile(np.abs(np.r_[x, y]), 0.995), np.abs(np.r_[x[mt], y[mt]]).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δα in liver (raw prior, centred)", fontsize=6.5)
        ax.set_title(title, fontsize=6.3)
        cm.fd.annotate_stat(ax, f"Spearman ρ = {rho:+.3f}, n = {len(ix):,}\nsign differs in {r.flip_rate_observed:.0%}"
                                f" of genes\n(expected if unrelated: {r.flip_rate_expected:.0%})", "lower right")
    axes[0].set_ylabel("Δα in testis (raw prior, centred)", fontsize=6.5)
    axes[0].legend(loc="upper left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
    return fig


def main() -> list:
    cm.setup()
    out = []
    for name, fn in [("fig4c_prior_landscape", panel_landscape), ("fig4_celltype_drivers", panel_drivers),
                     ("fig4_margin_support", panel_margin_support), ("fig4_cross_tissue", panel_cross_tissue)]:
        out.append(cm.save(fn(), name, exact=name == "fig4_margin_support"))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
