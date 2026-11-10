"""C1 figure: new use cases. Panels in the T8.4 style (`T8_4_figures/common.py`, `notebooks/_figure_design.py`), composed with F1's compose.py.
Reads result tables only. Output kept out of the manuscript (CHIMERA_MANUSCRIPT points to results/analysis/C1/fig_ms) for the coordinator to place.

    CHIMERA_MANUSCRIPT=$PWD/results/analysis/C1/fig_ms python analysis/tasks/C1_cohorts/figure.py
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("CHIMERA_MANUSCRIPT", str(ROOT / "results/analysis/C1/fig_ms"))
(Path(os.environ["CHIMERA_MANUSCRIPT"]) / "figures" / "revision").mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.tasks.T8_4_figures import common as cm  # noqa: E402

cm.PREVIEW = ROOT / "results/analysis/C1/fig_preview"
cm.PREVIEW.mkdir(parents=True, exist_ok=True)
R = ROOT / "results/analysis/C1"
SC = {"identity_partner": ("#999999", "v", "protein identity"), "log_dnds_mouse": ("#666666", "<", "dN/dS"),
      "D": cm.STY["D"][:2] + ("Δα (Chimera)",), "S_ema": cm.STY["S"][:2] + ("$S_g$ (Chimera)",), "B1": cm.STY["B1b"][:2] + ("profile correlation (B1)",)}
W = 2.3


def forest(ax, rows, scores, label_of):
    y = 0
    ticks = []
    for key, sub in rows:
        for j, s in enumerate(scores):
            r = sub[sub.score == s]
            if r.empty:
                continue
            r = r.iloc[0]
            col, mk, _ = SC[s]
            yy = y - j * 0.16
            ax.errorbar(r.val, yy, xerr=[[r.val - r.lo], [r.hi - r.val]], fmt=mk, color=col, ms=3.2, lw=0.8, capsize=0)
        ticks.append((y - 0.16 * (len(scores) - 1) / 2, label_of(key)))
        y -= 1
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.axvline(0, color=cm.RULE, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_a():
    t = pd.read_csv(R / "cortex/e_b2.tsv", sep="\t")
    keep = [("brain_dlpfc_hchimp", "MTG_macaque", "identity_ge_99_5", "DLPFC human+chimpanzee\nMTG gold, n = 4,770"),
            ("brain_dlpfc", "MTG_marmoset", "identity_ge_99_5", "DLPFC 4 primates\nMTG gold, n = 4,928"),
            ("brain_m1", "MTG_marmoset", "identity_ge_99_5", "M1 human, marmoset, mouse\nMTG gold, n = 816")]
    rows = []
    for c, g, st, lab in keep:
        s = t[(t.cohort == c) & (t.gold == g) & (t.stratum == st)].rename(columns={"spearman": "val", "ci_lo": "lo", "ci_hi": "hi"})
        rows.append((lab, s))
    fig, ax = plt.subplots(figsize=(W * 1.25, 2.3))
    forest(ax, rows, ["identity_partner", "log_dnds_mouse", "D", "S_ema", "B1"], lambda k: k)
    ax.set_xlabel("Spearman with held-out profile agreement\n(proteins ≥ 99.5% identical)")
    ax.set_title("Near-identical proteins: counts order them, sequence does not", loc="left")
    hs = [plt.Line2D([], [], color=SC[s][0], marker=SC[s][1], ls="", ms=3.5, label=SC[s][2]) for s in ["identity_partner", "log_dnds_mouse", "D", "S_ema", "B1"]]
    ax.legend(handles=hs, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, fontsize=7)
    return cm.save(fig, "c1a_near_identical")


def panel_b():
    t = pd.read_csv(R / "cortex/e_b1.tsv", sep="\t")
    lab = {"brain_dlpfc": "DLPFC 4 primates", "brain_dlpfc_hchimp": "DLPFC H+chimp.", "brain_dlpfc_hmarm": "DLPFC H+marmoset", "brain_m1": "M1 3 species"}
    gl = {"MTG_marmoset": "MTG", "MTG_macaque": "MTG", "M1_marmoset": "M1", "DLPFC_marmoset": "DLPFC"}
    pick = [("brain_dlpfc", "MTG_marmoset"), ("brain_dlpfc_hchimp", "MTG_macaque"), ("brain_dlpfc_hmarm", "MTG_marmoset"), ("brain_m1", "DLPFC_marmoset")]
    rows = [(f"{lab[c]}\nvs {gl[g]} ({g.split('_')[1]})", t[(t.cohort == c) & (t.gold == g)].rename(columns={"rho_p": "val", "ci_lo": "lo", "ci_hi": "hi"})) for c, g in pick]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    forest(ax, rows, ["D", "S_ema", "B1"], lambda k: k)
    ax.set_xlabel(cm.RHO_P)
    ax.set_title("Held-out cortical regions", loc="left")
    return cm.save(fig, "c1b_heldout_regions")


def panel_c():
    rows = []
    for case, name in (("ad", "Alzheimer"), ("lung", "Lung fibrosis"), ("muscle", "Muscle ageing")):
        s = pd.read_csv(R / case / "hd4/hd4_null_summary.tsv", sep="\t")
        h = s[s.kind == "human_permuted"].set_index("cell_type")
        m = s[s.kind == "mouse_permuted"].set_index("cell_type")
        for t in h.index:
            if h.loc[t, "n_human_sig_obs"] == 0:
                continue
            rows.append(dict(case=name, type=t, obs=h.loc[t, "n_transfers_obs"], q95=h.loc[t, "n_transfers_null_q95"], p=max(h.loc[t, "n_transfers_p_ge"], m.loc[t, "n_transfers_p_ge"])))
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.obs, height=0.6, color="#56B4E9")
    ax.scatter(d.q95, y, marker="|", s=60, color=cm.INK, zorder=3, label="label-null 95th percentile")
    for yy, r in zip(y, d.itertuples()):
        ax.text(max(r.obs, r.q95) + 2, yy, f"p {r.p:.2g}", va="center", fontsize=7, color=cm.MUTED)
    ax.set_yticks(y, [f"{r.case}: {r.type}" for r in d.itertuples()], fontsize=7)
    ax.set_xlabel("human disease programs that transfer to the mouse model")
    ax.set_title("Program transfer against relabelled donors", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cm.save(fig, "c1c_transfer")


def panel_d():
    t = pd.read_csv(R / "tumour_zilionis/e_t2.tsv", sep="\t")
    ch = t[t.run != "scVI"].groupby("cell_type").mean_entropy.agg(["mean", "min", "max"])
    sv = t[t.run == "scVI"].set_index("cell_type").mean_entropy
    order = ch.sort_values("mean").index
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(order))
    ax.errorbar(ch.loc[order, "mean"], y, xerr=[ch.loc[order, "mean"] - ch.loc[order, "min"], ch.loc[order, "max"] - ch.loc[order, "mean"]], fmt="o", color=cm.STY["D"][0], ms=3.5, lw=0.8, label="Chimera, 3 seeds")
    ax.scatter(sv.loc[order], y, marker="s", s=10, color="#999999", label="scVI, same cells")
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1)
    ax.set_xlabel("species mixing within class (kNN entropy)")
    ax.set_title("Tumour myeloid: macrophages stay apart", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cm.save(fig, "c1d_tumour_mixing")


def panel_e():
    import json

    lab = {"brain_m1_hmouse": "M1 H+mouse", "brain_m1": "M1 3 sp.", "brain_dlpfc": "DLPFC 4 sp.", "brain_dlpfc_hchimp": "DLPFC H+chimp.", "brain_dlpfc_hmarm": "DLPFC H+marm.",
           "brain_dlpfc_hmac_own": "DLPFC H+mac., own table", "lung_healthy": "Lung 3 sp.", "tumour_zilionis": "Tumour H+mouse", "muscle_ageing": "Muscle H+mouse"}
    runs = pd.read_csv(R / "kept_runs.tsv", sep="\t")
    rows = []
    for c in lab:
        g = pd.read_csv(R / c / "grid_report.tsv", sep="\t").set_index("run")
        k = runs[runs.cohort == c].run
        v = g.reindex(k).emb_label_transfer_acc.dropna()
        s = json.loads((R / c / "scvi_reference.json").read_text())["label_transfer_acc"]
        rows.append(dict(c=lab[c], lo=v.min(), hi=v.max(), mid=v.mean(), scvi=s))
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(d))[::-1]
    ax.errorbar(d.mid, y, xerr=[d.mid - d.lo, d.hi - d.mid], fmt="o", color=cm.STY["D"][0], ms=3.5, lw=0.8, label="Chimera, kept seeds")
    ax.scatter(d.scvi, y, marker="s", s=10, color="#999999", label="scVI, same object")
    ax.set_yticks(y, d.c)
    ax.set_xlabel("label transfer across species")
    ax.set_title("Embedding against scVI", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cm.save(fig, "c1e_embedding")


if __name__ == "__main__":
    cm.setup()
    for f in (panel_a, panel_b, panel_c, panel_d, panel_e):
        print(f())
