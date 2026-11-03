"""X8 supplementary panels, one PDF each in results/revision/X8/panels/, composed by F1_figures/compose.py with specs_x8.py into
results/revision/X8/composed/ (the manuscript is not touched). Reads results/revision/X8/ only. Design: notebooks/_figure_design.py.

    PYTHONPATH=$PWD python -m revision.tasks.X8_gate_divergence.figures
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from revision.tasks.X8_gate_divergence import rules as RU  # noqa: E402

sys.path.insert(0, str(RU.REPO / "notebooks"))
import _figure_design as fd  # noqa: E402

OUT = RU.OUT
PANELS = OUT / "panels"
UP, DOWN, GREY, INK, MUTED = "#D55E00", "#0072B2", "#BBBBBB", "#222222", "#777777"
FREE_C, CAN_C, GREEN = "#0072B2", "#D55E00", "#009E73"
TISSUE_ORDER = ["liver", "testis", "bone marrow", "kidney", "islet", "eye", "cortex", "lung", "skeletal muscle", "tumour myeloid"]
TLAB = {"liver": "Liver", "testis": "Testis", "bone marrow": "Bone marrow", "kidney": "Kidney", "islet": "Islet", "eye": "Eye", "cortex": "Cortex", "lung": "Lung",
        "skeletal muscle": "Muscle", "tumour myeloid": "Tumour myeloid"}
COV_SHORT = {"expr": "expression", "detection": "detection", "abs_S_free": "|$S_g$| per-gene", "abs_S_can": "|$S_g$| canonical", "pi_raw": "identity $\\pi$",
             "tau": "specificity $\\tau$", "det_gap": "detection gap", "ALL_R2": "joint $R^2$"}
SET_LABEL = {"class|mitochondrial": "Mitochondrial proteins", "class|TF": "Transcription factors", "class|TF_C2H2_ZF": "C2H2 zinc-finger TFs", "class|receptor": "Receptors",
             "broad|Translation & ribosome": "Translation & ribosome", "broad|Protein homeostasis": "Protein homeostasis", "hallmark|E2F Targets": "E2F targets",
             "hallmark|G2-M Checkpoint": "G2-M checkpoint", "hallmark|Myc Targets V1": "Myc targets V1", "hallmark|Myc Targets V2": "Myc targets V2"}
REF_LABEL = {"pattern": "pattern", "pattern_primates": "pattern, primates", "level": "level", "level_atlas": "level, atlas", "pattern_fine": "pattern, fine",
             "pattern_nephron": "pattern, nephron", "pattern_noacinar": "pattern, no acinar", "pattern_fine_noacinar": "pattern, fine, no acinar", "pattern_coe": "pattern (COE)"}


def rc() -> None:
    fd.apply_design()
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6})


def psave(fig, name: str) -> None:
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(PANELS / f"{name}.png", format="png", dpi=200, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def cohort_order() -> list[tuple[str, str, str]]:
    reg = RU.cohorts()
    rows = [(c, v["tissue"], v["label"]) for c, v in reg.items()]
    return sorted(rows, key=lambda r: (TISSUE_ORDER.index(r[1]), list(reg).index(r[0])))


def short(label: str) -> str:
    return label.replace("Cortex ", "").replace("Liver ", "").replace("Testis ", "").replace("Eye ", "").replace(" H+Mouse", " H+Mouse")


# ---------------------------------------------------------------- (a) one cohort
def p_scatter(c: str = "kidney", label: list[str] | None = None) -> None:
    g = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
    fig, ax = plt.subplots(figsize=(2.55, 2.35))
    o = g[g.call == 0]
    ax.scatter(o.z_can, o.z_free, s=1.2, c=GREY, lw=0, rasterized=True)
    for v, col, lab in ((1, UP, "per-gene reading higher"), (-1, DOWN, "per-gene reading lower")):
        s = g[g.call == v]
        ax.scatter(s.z_can, s.z_free, s=1.6, c=col, lw=0, rasterized=True, label=f"{lab} ({len(s):,})")
    lim = (-2.8, 2.8)
    ax.plot(lim, lim, color=INK, lw=0.5, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("canonical Δα, protein-feature gate (rank z)")
    ax.set_ylabel("per-gene Δα, free gate (rank z)")
    from adjustText import adjust_text

    texts = []
    for sym in label or []:
        r = g[g.symbol == sym]
        if len(r):
            r = r.iloc[0]
            ax.plot(r.z_can, r.z_free, "o", mfc="none", mec=INK, ms=3, mew=0.5)
            texts.append(ax.text(r.z_can, r.z_free, sym, fontsize=5.6, style="italic"))
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK), expand=(1.6, 1.8))
    rho = pd.Series(g.z_free).corr(pd.Series(g.z_can), method="spearman")
    ax.text(0.03, 0.97, f"{RU.cohorts()[c]['label']}\n$\\rho$ = {rho:.2f}, n = {len(g):,}", transform=ax.transAxes, va="top", fontsize=5.8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), markerscale=4, handletextpad=0.2, borderaxespad=0.1)
    psave(fig, "x8_a_scatter")


# ---------------------------------------------------------------- (b) reliability and calls per cohort
def p_cohorts() -> None:
    s = pd.read_csv(OUT / "cohort_gap.csv").set_index("cohort")
    cal = pd.read_csv(OUT / "calibration.csv")
    order = cohort_order()
    y = np.arange(len(order))[::-1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(4.4, 3.5), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.12))
    nul = cal.groupby(["cohort", "gate"]).reliability_split.mean()  # null: mean over splits, per calibration cohort and gate
    a1.axvspan(nul.min(), nul.max(), color="#E4E4E4", lw=0)
    for yi, (c, t, lab) in zip(y, order):
        r = s.loc[c]
        if np.isfinite(r.reliability_split):
            a1.plot(r.reliability_split, yi, "o", color=INK, ms=2.6)
        else:
            a1.text(-0.3, yi, "one canonical seed", va="center", fontsize=6.4, color=MUTED)
        a2.plot(r.rho_zf_zc, yi, "o", color=GREEN, ms=2.6)
        a3.barh(yi, (r.n_up) / r.n_U * 100, color=UP, height=0.65)
        a3.barh(yi, -(r.n_down) / r.n_U * 100, color=DOWN, height=0.65)
    a1.set_xlim(-0.35, 1.02)
    a1.set_xlabel("gap reliability\n(split-half, seeds)")
    a2.set_xlim(0.2, 0.85)
    a2.set_xlabel("$\\rho$(per-gene, canonical)\non count-supported genes")
    a3.axvline(0, color=INK, lw=0.5)
    a3.set_xlabel("genes called, %\n(lower | higher)")
    lim = max(abs(np.array(a3.get_xlim())))
    a3.set_xlim(-lim, lim)
    a3.set_xticks([-15, 0, 15])
    a3.set_xticklabels(["15", "0", "15"])
    a1.set_yticks(y)
    a1.set_yticklabels([f"{TLAB[t]}, {short(lab)}" if not lab.startswith(TLAB[t]) else lab for c, t, lab in order], fontsize=5.6)
    for a in (a1, a2, a3):
        a.tick_params(axis="y", length=0)
    a1.text(nul.mean(), y[0] + 1.0, "null", ha="center", fontsize=6.4, color=MUTED)
    psave(fig, "x8_b_cohorts")


# ---------------------------------------------------------------- (c) protein neighbourhoods
def p_neighbours() -> None:
    n = pd.read_csv(OUT / "nbr_resid.csv").set_index("cohort")
    order = cohort_order()
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(2.3, 3.5))
    for yi, (c, t, lab) in zip(y, order):
        r = n.loc[c]
        ax.plot([0, max(r.share_can, r.share_free)], [yi, yi], color="#EEEEEE", lw=0.8, zorder=0)
        ax.plot(r.share_free_null_hi, yi, "|", color=MUTED, ms=4)
        ax.plot(r.share_can, yi, "s", color=CAN_C, ms=2.4)
        ax.plot(r.share_free, yi, "o", color=FREE_C, ms=2.6)
        ax.plot(r.pull, yi, "^", color=GREEN, ms=2.4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for c, t, lab in order], fontsize=5.4)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK, lw=0.4)
    ax.set_xlabel("$\\rho$ with the mean of 15 ESM-2 neighbours\n(beyond expression, detection, $\\pi$, $\\tau$)")
    h = [Line2D([], [], marker="o", ls="", color=FREE_C, ms=3, label="per-gene reading shared by neighbours"),
         Line2D([], [], marker="s", ls="", color=CAN_C, ms=3, label="canonical reading shared by neighbours"),
         Line2D([], [], marker="^", ls="", color=GREEN, ms=3, label="canonical follows neighbours' counts,\ngiven the gene's own"),
         Line2D([], [], marker="|", ls="", color=MUTED, ms=5, label="random neighbours, 97.5%")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1, handletextpad=0.3)
    psave(fig, "x8_c_neighbours")


# ---------------------------------------------------------------- (d) covariates
def p_covariates() -> None:
    cv = pd.read_csv(OUT / "covariates.csv")
    d = cv[cv.readout == "delta"].pivot(index="cohort", columns="covariate", values="partial_rho")
    cols = ["expr", "detection", "abs_S_free", "abs_S_can", "pi_raw", "tau", "det_gap", "ALL_R2"]
    order = cohort_order()
    M = d.reindex([c for c, _, _ in order])[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(2.35, 3.5))
    im = ax.imshow(np.where(np.arange(len(cols))[None, :] < 7, M, np.nan), cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.6, 0.6), aspect="auto")
    ax.imshow(np.where(np.arange(len(cols))[None, :] == 7, M, np.nan), cmap="Greys", vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and (j == 7 or abs(M[i, j]) >= 0.25):
                ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.6, color="white" if (j == 7 and M[i, j] > 0.4) or abs(M[i, j]) > 0.45 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([COV_SHORT[c] for c in cols], rotation=55, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for c, t, lab in order], fontsize=5.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, shrink=0.5)
    cb.set_label("partial $\\rho$(gap, covariate)", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.2)
    psave(fig, "x8_d_covariates")


# ---------------------------------------------------------------- (e) across cohorts
def p_cross() -> None:
    X = pd.read_csv(OUT / "cross_cohort_rho.csv")
    icc = pd.read_csv(OUT / "icc.csv").set_index("readout").icc
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    keys = [("rho_z_free", "per-gene Δα", FREE_C), ("rho_z_can", "canonical Δα", CAN_C), ("rho_delta", "gap", INK)]
    pos = 0
    ticks = []
    rng = np.random.default_rng(0)
    for same, lab in ((True, "same tissue"), (False, "different tissues")):
        for k, name, col in keys:
            v = X[X.same_tissue == same][k].dropna().to_numpy()
            ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, medianprops=dict(color=col, lw=1.2), boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
            ax.scatter(pos + rng.uniform(-0.18, 0.18, len(v)), v, s=0.8, color=col, alpha=0.35, lw=0, rasterized=True)
            ticks.append((pos, name))
            pos += 1
        pos += 0.6
    ax.axhline(0, color=INK, lw=0.4)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], rotation=45, ha="right", fontsize=5.6)
    ax.text(1, 1.02, "same tissue", ha="center", transform=ax.get_xaxis_transform(), fontsize=5.6)
    ax.text(4.6, 1.02, "different tissues", ha="center", transform=ax.get_xaxis_transform(), fontsize=5.6)
    ax.set_ylabel("Spearman between cohorts\n(shared genes)")
    ax.text(0.99, 0.02, f"gene share (ICC, 8 tissues):\nper-gene {icc['z_free']:.2f}, canonical {icc['z_can']:.2f}, gap {icc['delta']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=MUTED)
    psave(fig, "x8_e_cross")


# ---------------------------------------------------------------- (f) sets
def p_sets() -> None:
    r = pd.read_csv(OUT / "set_recurrence.csv")
    th = list(RU.thread())
    keep = r[(r.readout == "delta") & r.recurrent].sort_values("mean_z", ascending=False).key.tolist()
    sub = r[r.readout == "delta"].set_index("key").loc[keep]
    M = sub[[f"z_{t}" for t in th]].to_numpy(float)
    side = np.c_[r[r.readout == "z_free"].set_index("key").loc[keep].mean_z, r[r.readout == "z_can"].set_index("key").loc[keep].mean_z]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.3, 2.0), gridspec_kw=dict(width_ratios=[8, 2.2], wspace=0.06))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -6, 6), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 1.96:
                ax.text(j, i, "•", ha="center", va="center", fontsize=5, color="white" if abs(M[i, j]) > 4 else INK)
    ax.set_xticks(range(len(th)))
    ax.set_xticklabels([TLAB[t] for t in th], rotation=55, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([SET_LABEL.get(k, k.split("|")[1]) for k in keep], fontsize=5.6)
    bx.imshow(side, cmap="RdBu_r", norm=TwoSlopeNorm(0, -6, 6), aspect="auto")
    for i in range(side.shape[0]):
        for j in range(2):
            bx.text(j, i, f"{side[i, j]:.1f}", ha="center", va="center", fontsize=4.6, color="white" if abs(side[i, j]) > 4 else INK)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["per-gene", "canonical"], rotation=55, ha="right", fontsize=5.6)
    bx.set_yticks([])
    for a in (ax, bx):
        a.tick_params(length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=[ax, bx], fraction=0.04, pad=0.02, shrink=0.8)
    cb.set_label("set z of the gap (• |z| ≥ 1.96);\nright: mean z of each reading", fontsize=5.4)
    cb.ax.tick_params(labelsize=5.2)
    psave(fig, "x8_f_sets")


# ---------------------------------------------------------------- (g) independent references
def p_golds() -> None:
    T = pd.read_csv(OUT / "golds_trackc.csv")
    rows = []
    for (c, kind), g in T.groupby(["cohort", "kind"], sort=False):
        f = g[(g.group == "given_can") & (g.method == "z_free")]
        k = g[(g.group == "given_free") & (g.method == "z_can")]
        if len(f) and len(k):
            rows.append(dict(cohort=c, unit=kind, f=f.value.iloc[0], flo=f.ci_lo.iloc[0], fhi=f.ci_hi.iloc[0], k=k.value.iloc[0], klo=k.ci_lo.iloc[0], khi=k.ci_hi.iloc[0]))
    cf = OUT / "golds_cortex.csv"
    if cf.exists():
        C = pd.read_csv(cf)
        for (c, gname), g in C.groupby(["cohort", "gold"], sort=False):
            f = g[(g.group == "given_can") & (g.score == "z_free")]
            k = g[(g.group == "given_free") & (g.score == "z_can")]
            rows.append(dict(cohort=c, unit=f"held-out {gname.replace('_', ' ')}", f=f.rho_p.iloc[0], flo=f.ci_lo.iloc[0], fhi=f.ci_hi.iloc[0], k=k.rho_p.iloc[0], klo=k.ci_lo.iloc[0],
                             khi=k.ci_hi.iloc[0]))
    D = pd.DataFrame(rows)
    reg = RU.cohorts()
    D["label"] = [f"{reg[c]['label']}, {REF_LABEL.get(u, u)}" for c, u in zip(D.cohort, D.unit)]
    y = np.arange(len(D))[::-1]
    fig, ax = plt.subplots(figsize=(2.6, 0.125 * len(D) + 0.5))
    ax.axvline(0, color=INK, lw=0.4)
    ax.errorbar(D.f, y + 0.17, xerr=[D.f - D.flo, D.fhi - D.f], fmt="o", color=FREE_C, ms=2.3, lw=0.6, label="per-gene reading, given canonical")
    ax.errorbar(D.k, y - 0.17, xerr=[D.k - D.klo, D.khi - D.k], fmt="s", color=CAN_C, ms=2.1, lw=0.6, label="canonical reading, given per-gene")
    ax.set_yticks(y)
    ax.set_yticklabels(D.label, fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("partial $\\rho$ with independent reference\n(given covariates and the other reading)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.06 - 0.7 / (0.125 * len(D) + 0.5)), ncol=1, handletextpad=0.3, fontsize=7.0)
    psave(fig, "x8_g_golds")


# ---------------------------------------------------------------- (h) sequence constraint
def p_sequence() -> None:
    s = pd.read_csv(OUT / "sequence.csv")
    order = cohort_order()
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(2.0, 3.5))
    ax.axvline(0, color=INK, lw=0.4)
    for yi, (c, t, lab) in zip(y, order):
        for k, (con, col, mk, off) in enumerate((("log10_dNdS", INK, "o", 0.17), ("LOEUF", GREEN, "s", -0.17))):
            r = s[(s.cohort == c) & (s.constraint == con) & (s.readout == "delta")]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r.partial_rho, yi + off, xerr=[[r.partial_rho - r.p_ci_lo], [r.p_ci_hi - r.partial_rho]], fmt=mk, color=col, ms=2.2, lw=0.5)
    ax.set_xlim(-0.3, 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for c, t, lab in order], fontsize=5.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("partial $\\rho$(gap, constraint)\ngiven $\\pi$, expression, detection")
    h = [Line2D([], [], marker="o", ls="", color=INK, ms=3, label="dN/dS, human-mouse"), Line2D([], [], marker="s", ls="", color=GREEN, ms=3, label="LOEUF")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, columnspacing=0.8)
    psave(fig, "x8_h_sequence")


# ---------------------------------------------------------------- (i) named examples
def p_named(genes: list[str]) -> None:
    reg = RU.cohorts()
    tis = [t for t in TISSUE_ORDER]
    M = np.full((len(genes), len(tis)), np.nan)
    C = np.zeros((len(genes), len(tis)))
    for c in reg:
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").set_index("symbol")
        j = tis.index(reg[c]["tissue"])
        for i, s in enumerate(genes):
            if s in g.index:
                r = g.loc[[s]].iloc[0]
                if r.call != 0:
                    C[i, j] = r.call
                M[i, j] = np.nanmean([M[i, j], r.delta]) if np.isfinite(M[i, j]) else r.delta
    nb = pd.read_csv(OUT / "named.csv").set_index("symbol")
    fig, ax = plt.subplots(figsize=(3.3, 0.16 * len(genes) + 0.55))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
    for i in range(len(genes)):
        for j in range(len(tis)):
            if C[i, j] != 0:
                ax.text(j, i, "•", ha="center", va="center", fontsize=5.5, color="white" if abs(M[i, j]) > 2 else INK)
    ax.set_xticks(range(len(tis)))
    ax.set_xticklabels([TLAB[t] for t in tis], rotation=55, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=5.8, style="italic")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, s in enumerate(genes):
        if s in nb.index:
            ax.text(len(tis) - 0.35, i, ", ".join(nb.loc[s, "nbr_symbols"].split(";")[:3]), va="center", fontsize=4.9, color=MUTED, style="italic")
    ax.text(len(tis) - 0.35, -0.9, "nearest ESM-2 neighbours", fontsize=5.0, color=MUTED)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.36, shrink=0.45, anchor=(0.0, 1.0))
    cb.set_label("gap, mean over the tissue's cohorts (• called in a cohort)", fontsize=5.4)
    cb.ax.tick_params(labelsize=5.2)
    psave(fig, "x8_i_named")


def main() -> int:
    rc()
    named = pd.read_csv(OUT / "named.csv")
    pick = (sys.argv[1].split(",") if len(sys.argv) > 1 else named[named.clean].symbol.head(16).tolist())
    p_scatter("kidney", ["MARK2", "ARF5", "TERF2IP", "FKBP5", "PPP2CB"])
    p_cohorts()
    p_neighbours()
    p_covariates()
    p_cross()
    p_sets()
    if (OUT / "golds_trackc.csv").exists():
        p_golds()
    p_sequence()
    p_named(pick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
