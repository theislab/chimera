"""Panels first made by notebooks 04 and 06, redrawn at their size on the page, from the same data.

supp_testis_categories_{a,b}  A1 box plot and A3 effect sizes of ``_alpha_biology_builder.py`` for testis
                              Human+Macaque: gene table from the notebook's own cells, the same statistics,
                              row pitch of 11.5 pt so all 41 categories fit one page at a readable size.
supp_hallmark_{a,b,c}         B1, B2 and D1 of ``_pathway_builder.py`` from the cached GSEA tables, with
                              the cohort names of the paper in place of the internal ``panc`` run names.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.F1_figures.notebook_panels
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from revision.tasks.T8_4_figures import common as cm

NB = cm.REPO / "notebooks"
RUNS = cm.REPO / "training" / "runs"


def notebook_namespace(path, stop_after: int, dataset_order: list) -> dict:
    """Run the notebook's setup cells only, restricted to some datasets, in a fresh namespace."""
    cells = json.loads(path.read_text())["cells"]
    ns = {"__name__": "nb"}
    for i, c in enumerate(cells[: stop_after + 1]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = src.replace("plt.show()", "pass")
        src = re.sub(r"^for k in DATASET_ORDER:\n\s+plot_\w+\(.*\)$", "", src, flags=re.M)
        if i == 10:  # coverage heatmap: only cats_admissible is needed
            src = src.split("\n")[0]
        exec(compile(src, f"{path.name}[{i}]", "exec"), ns)  # noqa: S102
        if "DATASET_ORDER" in ns and ns["DATASET_ORDER"] != dataset_order:
            ns["DATASET_ORDER"] = dataset_order
    return ns


def testis_categories() -> list:
    ns = notebook_namespace(NB / "04_delta_alpha_biology.ipynb", 13, ["test_hmac"])
    key = "test_hmac"
    sub, order = ns["_category_order_and_subset"](key)
    df = ns["GENE_TABLES"][key]
    label = ns["DATASETS"][key]["display_label"]
    pu, fd = ns["pu"], ns["fd"]
    n = len(order)
    pitch = 11.5 / 72
    H = pitch * n + 0.5

    # (a) A1 box plot, same statistics and styles; only the size changes.
    fig = plt.figure(figsize=(3.95, H))
    ax = fig.add_axes([1.3 / 3.95, 0.42 / H, 2.05 / 3.95, pitch * n / H])
    data = [sub.loc[sub["category"] == c, "delta_alpha"].values for c in order]
    ax.boxplot(data, vert=False, positions=range(n), widths=0.62, patch_artist=True,
               medianprops=dict(color="#0072B2", linewidth=1.0),
               boxprops=dict(facecolor="#dddddd", edgecolor="#222222", linewidth=0.5),
               whiskerprops=dict(color="#222222", linewidth=0.5), capprops=dict(color="#222222", linewidth=0.5),
               flierprops=dict(marker="o", markersize=1.0, markerfacecolor="#222222", markeredgecolor="none", alpha=0.4))
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xlim(*fd.DELTA_ALPHA_RANGE)
    ax.set_xticks(list(fd.DELTA_ALPHA_TICKS))
    ax.set_xlabel(f"{fd.LABEL['delta_alpha']}  ({label})", fontsize=6.5, labelpad=1)
    ax.set_ylim(-0.5, n - 0.5)  # largest median at the top, as in (b)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c}  ($n${int((sub['category'] == c).sum())})" for c in order], fontsize=5.5)
    ax.set_ylabel("Gene category", fontsize=6.5, labelpad=1)
    ax.tick_params(axis="x", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ns["_annotate_significance_per_category"](ax, sub, order, df["delta_alpha"].values)
    out = [cm.save(fig, "supp_testis_categories_a")]

    # (b) A3 effect sizes, same statistics and BH pool as plot_effect_size_heatmap_for_dataset.
    from scipy.stats import mannwhitneyu, wilcoxon

    v_all = df["delta_alpha"].values
    a_all = np.abs(v_all)
    sd_full, sd_abs, mean_abs = np.std(v_all, ddof=1), np.std(a_all, ddof=1), np.mean(a_all)
    rows = []
    for c in order:
        m = (df["category"] == c).values
        d_shift = np.mean(v_all[m]) / sd_full
        d_amp = (np.mean(a_all[m]) - mean_abs) / sd_abs * (1.0 if d_shift >= 0 else -1.0)
        p_d = wilcoxon(v_all[m], alternative="two-sided", zero_method="wilcox")[1]
        p_amp = mannwhitneyu(a_all[m], a_all[~m], alternative="greater")[1]
        rows.append(dict(category=c, d=d_shift, d_amp=d_amp, p_d=p_d, p_amp=p_amp))
    enr = pd.DataFrame(rows)
    q = pu.bh_fdr(np.r_[enr.p_d.values, enr.p_amp.values])
    enr["sig_d"], enr["sig_amp"] = q[: n] < 0.05, q[n:] < 0.05
    enr = enr.set_index("category").loc[list(reversed(order))]
    val, sig = enr[["d", "d_amp"]].values, enr[["sig_d", "sig_amp"]].values
    vmax = float(max(np.abs(val).max(), 1.0))
    fig = plt.figure(figsize=(2.72, H))
    ax = fig.add_axes([1.2 / 2.72, 0.42 / H, 1.02 / 2.72, pitch * n / H])
    im = ax.imshow(val, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(n):
        for j in range(2):
            ax.text(j, i, f"{val[i, j]:+.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(val[i, j]) > 0.6 * vmax else "#222222")
            if sig[i, j]:
                ax.text(j + 0.40, i - 0.18, "★", ha="center", va="center", fontsize=4,
                        color="white" if abs(val[i, j]) > 0.6 * vmax else "#222222")
    ax.set_xticks([0, 1], ["shift", "amplitude"], fontsize=6)
    ax.set_xlabel(r"Cohen's $d$ on $\Delta\alpha$ / signed $|\Delta\alpha|$" + f"\n({label})", fontsize=6, labelpad=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(enr.index, fontsize=5.5)
    ax.tick_params(length=0)
    cax = fig.add_axes([2.3 / 2.72, 0.42 / H + 0.33 * pitch * n / H, 0.09 / 2.72, 0.34 * pitch * n / H])
    cb = fig.colorbar(im, cax=cax, label="effect size")
    cb.ax.tick_params(labelsize=5.5)
    cb.ax.yaxis.label.set_size(6)
    out.append(cm.save(fig, "supp_testis_categories_b"))
    return out


HALL = "MSigDB_Hallmark_2020"
HRUN = {"liver H+Monkey": "iter90-150ep-s42", "liver H+Mouse": "xds-hmouse-iter90-150ep-s42",
        "liver H+Zebrafish": "xds-hzeb-iter85-s42", "liver 4 species": "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42",
        "testis H+Macaque": "testis-hmac-iter109-pclamp085-200ep-s42",
        "testis H+Mouse": "testis-hmouse-iter111-pclamp-coc-200ep-s42",
        "testis 3 species": "testis-unified-iter115-pclamp-coc-100ep-s42"}
FDR = 0.05


def _gsea(label: str, metric: str) -> pd.DataFrame:
    d = pd.read_csv(RUNS / HRUN[label] / "cache" / f"gsea__{HALL}__{metric}.csv")
    for c in ("NES", "FDR q-val"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _pathway(t: str) -> str:
    return re.sub(r"^HALLMARK[_ -]+", "", str(t).split("__")[-1], flags=re.I)


def _nice(name: str, max_len: int) -> str:
    name = name.replace("Pperoxisome", "Peroxisome")  # typo in the Enrichr library
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def hallmark() -> list:
    # (a) B1 scatter, liver Human+Mouse.
    d, p = _gsea("liver H+Mouse", "delta"), _gsea("liver H+Mouse", "prior")
    d["pathway"], p["pathway"] = d.Term.map(_pathway), p.Term.map(_pathway)
    m = d.merge(p, on="pathway", suffixes=("_delta", "_prior"))
    m["sig_delta"] = m["FDR q-val_delta"] < FDR
    m["sig_prior"] = m["FDR q-val_prior"] < FDR
    lim = 1.05 * float(np.nanmax(np.abs(m[["NES_delta", "NES_prior"]].values)))
    fig, ax = plt.subplots(figsize=(2.75, 2.5), layout="constrained")
    ax.axhline(0, color="#888888", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.5, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color="#888888", linewidth=0.5, linestyle=":", zorder=0)
    ns = m[~m.sig_delta]
    ax.scatter(ns.NES_prior, ns.NES_delta, s=11, c="#cccccc", alpha=0.7, linewidths=0, zorder=1)
    for sub, color in [(m[m.sig_delta & (m.NES_delta > 0)], "#b2182b"), (m[m.sig_delta & (m.NES_delta < 0)], "#2166ac")]:
        ax.scatter(sub.NES_prior, sub.NES_delta, s=22, c=color, alpha=0.9, zorder=2,
                   edgecolors=np.where(sub.sig_prior, "#222222", "#999999"), linewidths=np.where(sub.sig_prior, 0.6, 0.2))
    m["spec"] = (np.abs(m.NES_delta) - 0.5 * np.abs(m.NES_prior)) + 0.7 * (np.sign(m.NES_delta) != np.sign(m.NES_prior)).astype(float)
    for _, r in m.nlargest(4, "spec").iterrows():
        if r.sig_delta:
            ax.text(r.NES_prior, r.NES_delta + 0.08, _nice(r.pathway, 24), fontsize=5.3, ha="center", va="bottom", color="#222222")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("NES  ·  prior-ranked", fontsize=6.5)
    ax.set_ylabel("NES  ·  Δα-ranked  (Chimera)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    cm.fd.annotate_stat(ax, "liver Human+Mouse  ·  Hallmark", "upper left")
    out = [cm.save(fig, "supp_hallmark_a")]

    # (b) B2 stacked bars over the seven cohorts.
    q = []
    for lab in HRUN:
        d, p = _gsea(lab, "delta"), _gsea(lab, "prior")
        d["pathway"], p["pathway"] = d.Term.map(_pathway), p.Term.map(_pathway)
        j = d.merge(p, on="pathway", suffixes=("_d", "_p"))
        sd, sp = j["FDR q-val_d"] < FDR, j["FDR q-val_p"] < FDR
        same = np.sign(j.NES_d) == np.sign(j.NES_p)
        q.append(dict(label=lab, prior_only=int((~sd & sp).sum()), both_same=int((sd & sp & same).sum()),
                      delta_only=int((sd & ~sp).sum()), both_opp=int((sd & sp & ~same).sum())))
    q = pd.DataFrame(q).set_index("label")
    print(q.to_string())
    styles = {"prior_only": ("prior-only  (Δα misses)", "#bdbdbd"), "both_same": ("both  (same direction — confirms)", "#737373"),
              "delta_only": ("Δα-only  (Chimera adds)", "#b2182b"), "both_opp": ("both  (opposite — regulatory rewiring)", "#542788")}
    fig, ax = plt.subplots(figsize=(3.92, 2.5), layout="constrained")
    x = np.r_[np.arange(4), 4.7 + np.arange(3)]
    bottom = np.zeros(7)
    for k, (lab, col) in styles.items():
        ax.bar(x, q[k].values, bottom=bottom, color=col, edgecolor="#222222", linewidth=0.4, width=0.78, label=lab)
        bottom += q[k].values
    ax.set_xticks(x, [s.split(" ", 1)[1] for s in q.index], fontsize=5.8, rotation=20, ha="right",
                  rotation_mode="anchor")
    ax.set_ylabel("Hallmark pathways at FDR < 0.05", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6)
    top = bottom.max() * 1.08
    for lab, xs in (("Liver", x[:4]), ("Testis", x[4:])):
        ax.plot([xs.min() - 0.45, xs.max() + 0.45], [top, top], color="#222222", linewidth=0.7)
        ax.text(xs.mean(), top * 1.02, lab, ha="center", va="bottom", fontsize=7, fontweight="semibold", color="#222222")
    ax.set_ylim(0, top * 1.48)
    ax.set_xlim(-0.6, x.max() + 0.6)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=5.5, handlelength=1.0, handletextpad=0.3, borderaxespad=0,
              columnspacing=0.6)
    ax.set_title("Hallmark · 50 pathways", loc="right", fontsize=6, pad=1, color="#222222")
    out.append(cm.save(fig, "supp_hallmark_b"))

    # (c) D1 cross-tissue volcano.
    x = pd.read_csv(RUNS / HRUN["liver 4 species"] / "cache" / f"gsea_crosstissue__{HALL}.csv")
    x["NES"], x["FDR q-val"] = pd.to_numeric(x.NES, errors="coerce"), pd.to_numeric(x["FDR q-val"], errors="coerce")
    x["pathway"] = x.Term.map(_pathway)
    x["nl"] = -np.log10(x["FDR q-val"].clip(1e-300, 1.0))
    x["sig"] = x["FDR q-val"] < FDR
    fig, ax = plt.subplots(figsize=(6.75, 2.0), layout="constrained")
    ns = x[~x.sig]
    ax.scatter(ns.NES, ns.nl, s=11, c="#cccccc", alpha=0.7, linewidths=0, zorder=1)
    sp, sn = x[x.sig & (x.NES > 0)], x[x.sig & (x.NES < 0)]
    ax.scatter(sp.NES, sp.nl, s=24, c="#b2182b", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES > 0  (more liver-conserved)")
    ax.scatter(sn.NES, sn.nl, s=24, c="#2166ac", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES < 0  (more testis-conserved)")
    ax.axhline(-np.log10(FDR), color="#888888", linewidth=0.5, linestyle="--", zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.5, zorder=0)
    texts = [ax.text(r.NES, r.nl, _nice(r.pathway, 30), fontsize=5.3, ha="center", va="center", color="#222222")
             for s in (sp, sn) for _, r in s.nlargest(5, "nl").iterrows()]
    ax.set_xlabel(r"NES  (rank metric: $\Delta\alpha_{\rm liver}-\Delta\alpha_{\rm testis}$)", fontsize=6.5)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{FDR}\;q)$", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.margins(x=0.18)
    ax.legend(loc="center right", frameon=False, fontsize=6, handletextpad=0.1, borderaxespad=0)
    adjust_text(texts, x=x.NES.to_numpy(), y=x.nl.to_numpy(), ax=ax, expand=(1.2, 1.6), force_text=(0.4, 0.8),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.3), min_arrow_len=2.0)
    out.append(cm.save(fig, "supp_hallmark_c"))
    return out


def main() -> None:
    cm.setup()
    for p in hallmark() + testis_categories():
        print(p.name, cm.check_pdf(p))


if __name__ == "__main__":
    main()
