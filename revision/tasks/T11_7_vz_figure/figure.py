"""T11.7: Van Zyl (eye) figure and tidy summary table (re-runnable; datasets from ``vz.usable_datasets``).

Same panel types, names and palette as the liver and testis panels of the paper
(``notebooks/_figure_design.py``, ``revision/tasks/T8_4_figures/common.py``). Writes
``results/revision/T11/summary.csv`` (one row per reported statistic, mixing facts on every row) and
``results/revision/T11/figures/vz_figure.{pdf,png}``:

a    pi vs alpha, 5-species model, named glaucoma genes marked.
b    Delta-alpha distribution, centred within dataset, all usable datasets.
c    Gene classes: stratified AUROC of Delta-alpha and of profile correlation (B1b) against matched genes.
d-f  V2, human-macaque-pig-mouse gradient, 5-species model: profile correlation with human, Chimera
     per-species residual, Chimera per-species fit margin; glaucoma genes beside all other genes.
g    V3, cell-type drivers: named glaucoma genes, share of each gene's divergent contribution per class.
h    Cell-type drivers: share of the total divergent contribution per class, all genes.
i    V1: stratified AUROC of glaucoma genes for Delta-alpha and B1b, all usable datasets.
j    Integration composite per model and dataset (notebook 07 formula, min-max within dataset).
k    Count evidence, feature dependence and mixing per dataset, as plain values.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_7_vz_figure/figure.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from revision.tasks.T11_5_vz_genes import vz  # noqa: E402
from revision.tasks.T8_4_figures import common as cm  # noqa: E402

R = vz.OUT
FIG = R / "figures"
PRIMARY = "full"
SP_ORDER = ["macaca_fascicularis", "macaca_mulatta", "pig", "mouse"]
SP_LABEL = {"macaca_fascicularis": "M. fascicularis", "macaca_mulatta": "M. mulatta", "pig": "pig", "mouse": "mouse"}
SETS = [("C1", "housekeeping"), ("C2", "ribosome etc."), ("V2", "interferon"), ("V3", "extracellular")]
MODELS = ["Chimera", "scVI-sp", "scVI-donor", "scANVI", "sysVI", "Harmony", "PCA"]
C_DA, C_B1B, C_S = cm.STY["D"][0], cm.STY["B1b"][0], cm.STY["S"][0]  # paper's readout colours
INK, INK2, GRID, SURFACE, MID = cm.INK, cm.MUTED, "#E6E6E6", "white", "#F0F0F0"
GENE_GREY = "#BBBBBB"
DS_COLOR = {"full": "#222222", "hmouse": "#D55E00", "hmmul": "#0072B2", "hpig": "#CC79A7", "hmfas": "#56B4E9", "primates": "#009E73"}
SEQ = LinearSegmentedColormap.from_list("seq", [MID, "#BFDCEC", "#56A3CF", "#0072B2", "#00466E"])
DIV = LinearSegmentedColormap.from_list("div", ["#00466E", "#0072B2", "#8FC2DF", MID, "#F0B48A", "#D55E00", "#8A3D00"])


def ds_label(ds: str) -> str:
    return vz.DATASETS[ds]["label"]


def short(ds: str) -> str:
    return ds_label(ds).replace("human + ", "H+").replace("5 species", "5 sp").replace("M. mulatta", "Mmul").replace("M. fascicularis", "Mfas")


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=6.3, width=0.6, length=2.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, title, x=-0.02):
    ax.set_title(title, color=INK, loc="left", fontsize=7.5, pad=4)
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")


def matrix(ax, M: np.ndarray, rows, cols, norm, cmap, fmt="{:.2f}", fs=6):
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)), cols, fontsize=5.8, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=6, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                rgba = cmap(norm(M[i, j]))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs, color="white" if lum < 0.45 else INK)


def summary(iq: dict) -> pd.DataFrame:
    rows = []
    it = pd.read_csv(R / "integration" / "integration_table.csv")
    for r in it.itertuples():
        for k in ("composite", "knn_mixing_entropy", "knn_species_purity", "label_transfer_acc", "sil_celltype"):
            rows.append(dict(task="T11.4", dataset=r.dataset, readout="integration", item=r.model, statistic=k, value=getattr(r, k)))
    for ds in iq:
        m = json.loads((R / "readout" / f"meta_{ds}.json").read_text())
        rows += [dict(task="T11.5", dataset=ds, readout="seeds", item="D", statistic="n_seeds_in_D", value=len(m["seeds_D"])),
                 dict(task="T11.5", dataset=ds, readout="seeds", item="D", statistic="n_genes_saturated_flag", value=m["n_saturated_flag"]),
                 dict(task="T11.5", dataset=ds, readout="covariate", item="D", statistic="spearman_D_S_ema", value=m["rho_D_S"]),
                 dict(task="T11.5", dataset=ds, readout="covariate", item="D", statistic="spearman_D_B1raw", value=m["rho_D_B1raw"]),
                 dict(task="T11.5", dataset=ds, readout="margin", item="D", statistic="frac_tail_with_margin_support",
                      value=m["margin_support"]["mean_frac_tail_with_margin_support"])]
        for r in pd.read_csv(R / "readout" / f"covariate_audit_{ds}.csv").itertuples():
            rows.append(dict(task="T11.5", dataset=ds, readout="covariate", item=r.covariate, statistic="spearman_with_D", value=r.spearman))
        for r in pd.read_csv(R / "readout" / f"cv_r2_{ds}.csv").itertuples():
            rows.append(dict(task="T11.5", dataset=ds, readout="cv_r2", item=r.features, statistic="cv_r2_D", value=r.cv_r2))
        for r in pd.read_csv(R / "readout" / f"sets_{ds}.csv").itertuples():
            rows.append(dict(task="T11.5", dataset=ds, readout=f"set {r.set} (predicted {r.direction})", item=r.score, statistic="strat_auroc",
                             value=getattr(r, "auroc", np.nan), ci_lo=getattr(r, "auroc_ci_lo", np.nan),
                             ci_hi=getattr(r, "auroc_ci_hi", np.nan), p=getattr(r, "p_perm", np.nan), q=getattr(r, "q_bh", np.nan), n=r.n_set))
    for r in pd.read_csv(R / "glaucoma" / "v1_contrast.csv").itertuples():
        rows.append(dict(task="T11.6", dataset=r.dataset, readout=f"V1 glaucoma ({r.subset})", item=r.score, statistic="strat_auroc",
                         value=getattr(r, "auroc", np.nan), ci_lo=getattr(r, "auroc_ci_lo", np.nan), ci_hi=getattr(r, "auroc_ci_hi", np.nan),
                         p=getattr(r, "p_perm_two_sided", np.nan), n=r.n_set, note="primary" if r.primary else ""))
    for r in pd.read_csv(R / "glaucoma" / "v2_species_fidelity.csv").itertuples():
        for k in ("mean_glaucoma", "mean_background", "mean_percentile_glaucoma"):
            rows.append(dict(task="T11.6", dataset=r.dataset, readout="V2 species", item=f"{r.method}:{r.species}", statistic=k,
                             value=getattr(r, k), n=r.n_glaucoma))
    for r in pd.read_csv(R / "glaucoma" / "v2_contrast.csv").itertuples():
        rows.append(dict(task="T11.6", dataset=r.dataset, readout="V2 contrast macaques - (pig, mouse)", item=f"{r.method}:{r.scale}",
                         statistic="glaucoma", value=r.glaucoma, ci_lo=r.ci_lo, ci_hi=r.ci_hi, note=f"all other genes {r.background:.4f}"))
    for r in pd.read_csv(R / "glaucoma" / "drivers_summary.csv").itertuples():
        rows.append(dict(task="T11.5", dataset=r.dataset, readout="cell-type drivers", item=r.cell_class, statistic="share_divergent_mass",
                         value=r.share_divergent_mass, n=r.n_species_with_class, note="n = species with the class"))
    t = pd.DataFrame(rows)
    for ds, q in iq.items():
        m = t["dataset"] == ds
        t.loc[m, "mixing_entropy"] = q["knn_mixing_entropy"]
        t.loc[m, "mixing_entropy_random"] = q["entropy_random"]
        t.loc[m, "adversary_accuracy"] = q["adv_acc"]
        t.loc[m, "majority_rate"] = q["majority_rate"]
        t.loc[m, "label_transfer"] = q["label_transfer_acc"]
    return t


def named_heatmap(ax, fig):
    """V3: named glaucoma genes x classes, share of each gene's divergent contribution (5-species model)."""
    from revision.tasks.T11_6_vz_glaucoma import glaucoma as GL

    a = vz.read_h5ad(PRIMARY, backed=True)
    fwd = GL.forward_mean(PRIMARY, vz.CLASS_KEY, a.var_names)
    types = fwd["cell_types"]
    div = np.maximum(-np.stack([fwd["C_gk"][k] for k in types]), 0)
    v3 = pd.read_csv(R / "glaucoma" / f"v3_drivers_{PRIMARY}.csv")
    v3 = v3[v3["vanzyl_named"]].sort_values("D")
    sym = pd.read_parquet(R / "readout" / f"genes_{PRIMARY}.parquet")["symbol"].to_numpy()
    idx = [int(np.where(sym == s)[0][0]) for s in v3["symbol"]]
    M = div[:, idx].T
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    keep = np.where(M.max(0) >= 0.10)[0]
    order = keep[np.argsort(-M[:, keep].sum(0))]
    cols = [types[j] for j in order]
    ax.imshow(M[:, order], aspect="auto", cmap=SEQ, vmin=0, vmax=0.8)
    ax.set_yticks(range(len(idx)), v3["symbol"], fontsize=6, color=INK, style="italic")
    ax.set_xticks(range(len(cols)), [c.replace(" cell", "") for c in cols], fontsize=5.8, rotation=35, ha="right", color=INK, rotation_mode="anchor")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(idx)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1)
    ax.tick_params(which="minor", length=0)
    pos = {c: j for j, c in enumerate(cols)}
    for i, c in enumerate(v3["human_top_expr_class"]):  # human top-expressing class of the gene
        if c in pos:
            ax.scatter(pos[c], i, s=16, facecolor="none", edgecolor=C_DA, linewidth=1, zorder=3)
    a.file.close()
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 0.8))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.012, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=0, colors=INK2)
    cb.set_label("share of the gene's divergent contribution", fontsize=5.4, color=INK2, labelpad=1)


def figure(iq: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cm.fd.apply_design()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 6.5, "axes.titlesize": 7, "pdf.fonttype": 42})
    dss = list(iq)
    fig = plt.figure(figsize=(7.4, 9.9), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 6, height_ratios=[1.05, 0.8, 1.3, 0.95], hspace=0.6, wspace=1.2,
                          left=0.1, right=0.985, top=0.975, bottom=0.04)
    g = pd.read_csv(R / "genes" / f"genes_{PRIMARY}.tsv", sep="\t")
    v3 = pd.read_csv(R / "glaucoma" / f"v3_drivers_{PRIMARY}.csv")
    named = v3[v3["vanzyl_named"]]["symbol"].tolist()
    acol = [c for c in g.columns if c.startswith("alpha_s")]

    # a: pi vs alpha
    ax = fig.add_subplot(gs[0, :2])
    style(ax, grid=None)
    alpha = g[acol].mean(axis=1)
    cm.fd.density_scatter(ax, g["pi"], alpha, s=1.4, alpha=0.2, color=GENE_GREY)
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.5, ls="--")
    m = g["symbol"].isin(named)
    ax.scatter(g.loc[m, "pi"], alpha[m], s=11, color=C_DA, edgecolor="white", linewidth=0.3, zorder=3, label="named glaucoma genes")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("protein-identity prior π", color=INK)
    ax.set_ylabel("α (eye, 5 species)", color=INK)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", handletextpad=0.1)
    panel(ax, "a", "π vs α")

    # b: Delta-alpha distribution
    ax = fig.add_subplot(gs[0, 2:4])
    style(ax, grid=None)
    for ds in dss:
        t = pd.read_csv(R / "genes" / f"genes_{ds}.tsv", sep="\t")
        ax.hist(t["delta_alpha_centred"], bins=cm.fd.DELTA_ALPHA_BINS, histtype="step", linewidth=1, color=DS_COLOR.get(ds, "#666666"),
                density=True, label=short(ds))
    ax.axvline(0, color="#888888", lw=0.5, ls="--")
    ax.set_xlim(-0.3, 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Δα (centred within dataset)", color=INK)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1, borderaxespad=0)
    panel(ax, "b", "Δα distribution")

    # c: gene classes
    ax = fig.add_subplot(gs[0, 4:])
    M, rows = [], []
    for score, lab in (("D", "Δα"), ("B1b", "B1b")):
        for st, name in SETS:
            line = []
            for ds in dss:
                sdf = pd.read_csv(R / "readout" / f"sets_{ds}.csv")
                v = sdf[(sdf["score"] == score) & (sdf["set"] == st)]["auroc"]
                line.append(v.iloc[0] if len(v) else np.nan)
            M.append(line)
            rows.append(f"{lab} · {name}")
    matrix(ax, np.array(M), rows, [short(d) for d in dss], TwoSlopeNorm(vcenter=0.5, vmin=0.2, vmax=0.8), DIV, fs=5.4)
    ax.tick_params(axis="x", labelsize=5.2)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.set_title("Gene classes vs matched genes (AUROC)", color=INK, loc="left", fontsize=7, pad=4, x=-0.45)
    ax.text(-0.52, 1.04, "c", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")

    # d-f: V2 gradient
    v2 = pd.read_csv(R / "glaucoma" / "v2_species_fidelity.csv")
    v2 = v2[v2["dataset"] == PRIMARY].set_index(["method", "species"])
    for j, (method, title, xl) in enumerate((
            ("pseudobulk", "Profile correlation with human", "pseudobulk r, outflow classes"),
            ("residual", "Chimera per-species residual", "residual RMS (lower: closer to shared)"),
            ("margin", "Chimera per-species margin", "margin $S_g$ (higher: better shared fit)"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        style(ax)
        t = v2.loc[method].reindex(SP_ORDER)
        y = np.arange(len(SP_ORDER))[::-1]
        ax.hlines(y, np.minimum(t["mean_glaucoma"], t["mean_background"]), np.maximum(t["mean_glaucoma"], t["mean_background"]),
                  color=GRID, linewidth=2, zorder=1)
        ax.scatter(t["mean_background"], y, s=26, marker="|", color=INK2, linewidth=1.3, zorder=2, label="all other genes")
        ax.scatter(t["mean_glaucoma"], y, s=20, color=C_DA, edgecolor="white", linewidth=0.5, zorder=3, label="glaucoma genes")
        ax.set_yticks(y, [SP_LABEL[s] for s in SP_ORDER], color=INK, fontsize=6.3, style="italic")
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel(xl, color=INK)
        ax.margins(x=0.14)
        panel(ax, "def"[j], title)
        if j == 0:
            ax.legend(frameon=False, fontsize=5.6, loc="lower right", handletextpad=0.1, borderaxespad=0)

    # g: V3 heatmap
    ax = fig.add_subplot(gs[2, :4])
    named_heatmap(ax, fig)
    panel(ax, "g", "Cell-type drivers, named glaucoma genes, 5 species (ring: human top-expressing class)")

    # h: drivers, all genes
    ax = fig.add_subplot(gs[2, 5:])
    style(ax)
    d = pd.read_csv(R / "glaucoma" / "drivers_summary.csv")
    d = d[d["dataset"] == PRIMARY].sort_values("share_divergent_mass").tail(10)
    y = np.arange(len(d))
    ax.hlines(y, 0, d["share_divergent_mass"], color=C_S, linewidth=1.8)
    ax.scatter(d["share_divergent_mass"], y, s=16, color=C_S, marker="s", edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y, [f"{c.replace(' cell', '')} ({n})" for c, n in zip(d["cell_class"], d["n_species_with_class"])], fontsize=5.8, color=INK)
    ax.set_xlabel("share of divergent contribution", color=INK)
    ax.text(-1.32, 1.02, "All genes (n species)", transform=ax.transAxes, fontsize=7, color=INK, ha="left", va="bottom")
    ax.text(-1.36, 1.04, "h", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")

    # i: V1
    ax = fig.add_subplot(gs[3, :2])
    style(ax)
    v1 = pd.read_csv(R / "glaucoma" / "v1_contrast.csv")
    yi, labs, ys = 0, [], []
    for ds in dss[::-1]:
        for sc, col, mk, lab in (("B1b", C_B1B, "^", "B1b"), ("D", C_DA, "o", "Δα")):
            r = v1[(v1["dataset"] == ds) & (v1["subset"] == "all") & (v1["score"] == sc)]
            if r.empty or not np.isfinite(r["auroc"].iloc[0]):
                continue
            r = r.iloc[0]
            ax.hlines(yi, r.auroc_ci_lo, r.auroc_ci_hi, color=col, linewidth=1.3)
            ax.scatter([r.auroc], [yi], s=16, color=col, marker=mk, edgecolor="white", linewidth=0.4, zorder=3)
            labs.append(f"{lab} · {short(ds)}")
            ys.append(yi)
            yi += 1
        yi += 0.5
    ax.axvline(0.5, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_yticks(ys, labs, fontsize=5.6, color=INK)
    ax.set_xlabel("stratified AUROC (95% CI)", color=INK)
    panel(ax, "i", "Glaucoma vs matched genes")

    # j: integration composite
    ax = fig.add_subplot(gs[3, 2:4])
    it = pd.read_csv(R / "integration" / "integration_table.csv")
    M = np.full((len(MODELS), len(dss)), np.nan)
    for i, mdl in enumerate(MODELS):
        for k, ds in enumerate(dss):
            v = it[(it["dataset"] == ds) & (it["model"] == mdl)]["composite"]
            M[i, k] = v.iloc[0] if len(v) else np.nan
    matrix(ax, M, MODELS, [short(d) for d in dss], plt.Normalize(0, 1), SEQ, fs=5.4)
    ax.tick_params(axis="x", labelsize=5.2)
    panel(ax, "j", "Integration composite", x=-0.3)

    # k: plain values
    ax = fig.add_subplot(gs[3, 4:])
    ax.axis("off")
    rows, cols = [], [short(d) for d in dss]
    vals = {k: [] for k in ("runs kept", "ρ(Δα, $S_g$)", "margin support", "CV R², features", "mixing entropy", "random mixing", "adversary", "majority rate", "label transfer")}
    for ds in dss:
        m = json.loads((R / "readout" / f"meta_{ds}.json").read_text())
        cv = pd.read_csv(R / "readout" / f"cv_r2_{ds}.csv").set_index("features")["cv_r2"]
        q = iq[ds]
        for k, v in (("runs kept", str(len(m["seeds_D"]))), ("ρ(Δα, $S_g$)", f"{m['rho_D_S']:.2f}"),
                     ("margin support", f"{m['margin_support']['mean_frac_tail_with_margin_support']:.2f}"), ("CV R², features", f"{cv['all72']:.2f}"),
                     ("mixing entropy", f"{q['knn_mixing_entropy']:.2f}"), ("random mixing", f"{q['entropy_random']:.2f}"),
                     ("adversary", f"{q['adv_acc']:.2f}"), ("majority rate", f"{q['majority_rate']:.2f}"), ("label transfer", f"{q['label_transfer_acc']:.2f}")):
            vals[k].append(v)
    tb = ax.table(cellText=list(vals.values()), rowLabels=list(vals), colLabels=cols, cellLoc="center", rowLoc="right", bbox=[0.3, 0.02, 0.7, 0.9])
    tb.auto_set_font_size(False)
    tb.set_fontsize(4.9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.4)
        cell.set_facecolor(SURFACE)
        cell.set_text_props(color=INK)
    ax.set_title("Per dataset", color=INK, loc="left", fontsize=7, pad=4)
    ax.text(-0.01, 1.04, "k", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"vz_figure.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    iq = json.loads((R / "glaucoma" / "integration_quality.json").read_text())
    summary(iq).to_csv(R / "summary.csv", index=False)
    figure({k: v for k, v in iq.items() if not vz.DATASETS[k].get("beside")})
    print("wrote", R / "summary.csv", FIG / "vz_figure.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
