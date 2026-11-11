"""The Hallmark panels first made by notebook 06, redrawn at their size on the page, from the same data.

supp_hallmark_{a,b,c}         B1, B2 and D1 of ``_pathway_builder.py`` from the cached GSEA tables, with
                              the cohort names of the paper in place of the internal ``panc`` run names.

The gene-category panels of testis Human+Macaque, which this script also drew, are drawn by
``figures/redraw/categories.py`` together with those of the four-species liver model.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.F1_figures.notebook_panels
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from analysis.tasks.T8_4_figures import common as cm

RUNS = cm.REPO / "training" / "runs"


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


def _nice(name: str) -> str:
    """The name of a Hallmark set as printed: in full, under its name on the figures."""
    return cm.style.hallmark_name(name)


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
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("NES  ·  prior-ranked", fontsize=6.5)
    ax.set_ylabel("NES  ·  Δα-ranked  (Chimera)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_title(f"{cm.COHORT['liv_hmouse']} · Hallmark", loc="left", fontsize=6, pad=1, color="#222222")
    # Names in full, on a white ground that keeps the diagonal off the letters. A name stands above its point; where a
    # marker or another name is in the way, under it, then at its right, then at its left.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame, points = ax.get_window_extent(), ax.transData.transform(m[["NES_prior", "NES_delta"]].to_numpy())
    reach = 2.8 * fig.dpi / 72                                   # a marker with its edge, in pixels
    places = (((0, 3.6), "center", "bottom"), ((0, -3.6), "center", "top"), ((4.2, 0), "left", "center"), ((-4.2, 0), "right", "center"))
    names = []
    for _, r in m.nlargest(4, "spec").iterrows():                # the four most specific sets, named if significant on Δα
        if not r.sig_delta:
            continue
        for offset, ha, va in places:
            t = ax.annotate(_nice(r.pathway), (r.NES_prior, r.NES_delta), xytext=offset, textcoords="offset points", fontsize=5.3, ha=ha, va=va,
                            color="#222222", zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
            box = t.get_window_extent(renderer)
            free = (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
                    and not any(np.hypot(max(box.x0 - x, 0, x - box.x1), max(box.y0 - y, 0, y - box.y1)) < reach for x, y in points)
                    and not any(box.overlaps(n.get_window_extent(renderer)) for n in names))
            if free:
                names.append(t)
                break
            t.remove()
        else:
            raise SystemExit(f"no free place for the name '{_nice(r.pathway)}' beside its point")
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
    styles = {"prior_only": ("prior ranking only", "#bdbdbd"), "both_same": ("both, same sign", "#737373"),
              "delta_only": ("Δα ranking only", "#b2182b"), "both_opp": ("both, opposite sign", "#542788")}
    fig, ax = plt.subplots(figsize=(3.92, 2.5), layout="constrained")
    x = np.r_[np.arange(4), 4.7 + np.arange(3)]
    bottom = np.zeros(7)
    for k, (lab, col) in styles.items():
        ax.bar(x, q[k].values, bottom=bottom, color=col, edgecolor="#222222", linewidth=0.4, width=0.78, label=lab)
        bottom += q[k].values
    ax.set_xticks(x, [cm.style.cohort_name(s).split(" ", 1)[1] for s in q.index], fontsize=5.8, rotation=20, ha="right",
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
    ax.scatter(sp.NES, sp.nl, s=24, c="#b2182b", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES > 0  (higher Δα in liver)")
    ax.scatter(sn.NES, sn.nl, s=24, c="#2166ac", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES < 0  (higher Δα in testis)")
    ax.axhline(-np.log10(FDR), color="#888888", linewidth=0.5, linestyle="--", zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.5, zorder=0)
    texts = [ax.text(r.NES, r.nl, _nice(r.pathway), fontsize=5.3, ha="center", va="center", color="#222222")
             for s in (sp, sn) for _, r in s.nlargest(5, "nl").iterrows()]
    ax.set_xlabel("NES  (genes ranked by Δα in liver − Δα in testis)", fontsize=6.5)
    ax.set_ylabel("−log₁₀ (FDR $q$)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.margins(x=0.18)
    ax.legend(loc="center right", frameon=False, fontsize=6, handletextpad=0.1, borderaxespad=0)
    # iter_lim: the placement runs until it settles; without it a time limit of one second applies and a slow machine
    # stops it early, at another placement
    adjust_text(texts, x=x.NES.to_numpy(), y=x.nl.to_numpy(), ax=ax, expand=(1.2, 1.6), force_text=(0.4, 0.8), iter_lim=2000,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.3), min_arrow_len=2.0)
    out.append(cm.save(fig, "supp_hallmark_c"))
    return out


def main() -> None:
    cm.setup()
    for p in hallmark():
        print(p.name, cm.check_pdf(p))


if __name__ == "__main__":
    main()
