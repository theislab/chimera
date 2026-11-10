"""Disease figure (main): which human disease responses the mouse model reproduces.

One PDF per panel in ``figures/panels/main_disease/`` (kidney_transfer, stage_transfer, healthy_tissue_test,
transfer_vs_null, injury_genes; figure key ``main_disease``); PNG preview of the whole figure and the plotted values
(``fig8_source.tsv``) in ``results/analysis/T7_10/``. Every number is read from the result files below;
nothing is typed in. Re-runnable: the islet H-D1 slot in panel c is filled when its output exists.

Panels
  a  Kidney, human AKI vs mouse IRI: Hallmark/Reactome programs significant in human AKI, split into
     transfers / does not transfer / opposite in mouse (10,000 gene-set permutations), with the
     label-null p of the transfer count (human labels permuted; mouse labels permuted).
     Bold type labels: transfer count and program-NES correlation both at p <= 0.05 under both
     relabellings.            P13 ``hd4/hd4.tsv``, ``hd4/hd4_null_summary.tsv``
  b  Kidney time course: the same human contrast against mouse IRI 2 d only and late IRI only
     (4-20 wk); transferring programs and program-NES Spearman. Descriptive, no label null.
                              P13 ``hd4_primary_sens_acute/``, ``hd4_primary_sens_late/``
  c  H-D1 (pre-specified, healthy-tissue readout): rho_p of D (Delta-alpha, seed consensus) with gene-level transfer given the
     covariates, 95% ESM-block bootstrap CI; grey band, mean +- 2 SD of rho_p over fits with human
     disease labels permuted.  P13 ``hd1/h1.tsv``, ``hd1/h1_controls_summary.tsv``;
     T7_3 ``h1.tsv``, ``h1_controls.tsv``; islet T7_9 ``hd1/h1.tsv`` (+ controls) when present.
  d  Program transfer against the human-label null in the three tissues: observed transferring
     programs, and the null median to 95th percentile.
                              P13 ``hd4/hd4_null.tsv``; T7_3 ``h4_summary.tsv``, ``h4_null.tsv``;
                              T7_9 ``hd4/hd4_summary.tsv``, ``hd4/hd4_null.tsv``
  e  Cippa 2018 anchor: stratified AUROC, shared vs human-specific genes, D against the pseudobulk
     profile-correlation baseline B1b. The panels label D as Delta-alpha, as the main text does.   P13 ``cippa/ek3.tsv``

Run from ``chimera/``::

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T7_10_fig8.fig8_disease
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from analysis.tasks.T8_4_figures import common as cm

P13 = cm.RES / "P13"
T73 = cm.RES / "T7_3"
T79 = cm.RES / "T7_9"
OUT = cm.RES / "T7_10"
NAME = "fig8_disease"

# Islet H-D1 output, expected in the P13 hd1.py schema (cell_type, score, rho_p, ci_lo, ci_hi, q_bh;
# controls: cell_type, kind, mean, std). Either controls file layout is accepted.
ISLET_H1 = T79 / "hd1" / "h1.tsv"
ISLET_CTRL_SUMMARY = T79 / "hd1" / "h1_controls_summary.tsv"
ISLET_CTRL_RAW = T79 / "hd1" / "h1_controls.tsv"
ISLET_DONE = T79 / "hd1" / "DONE"  # written last by T7_9_islet/hd1.py; the slot waits for it

K_TYPES = ["pt", "tal", "ec", "fib", "pc", "myeloid", "dct", "ic", "lymphoid"]
K_LAB = {"pt": "PT", "tal": "TAL", "ec": "EC", "fib": "FIB", "pc": "PC", "myeloid": "myeloid",
         "dct": "DCT", "ic": "IC", "lymphoid": "lymphoid"}
I_TYPES = ["beta", "alpha", "delta", "ductal", "endothelial", "stellate"]
I_LAB = {"beta": "β", "alpha": "α", "delta": "δ", "ductal": "ductal", "endothelial": "endothelial",
         "stellate": "stellate"}

C_TRANSFER = "#0072B2"   # Okabe-Ito blue
C_OPPOSITE = "#E69F00"   # Okabe-Ito orange
C_NOT = "#C9C9C9"
C_BAND = "#E4E4E4"
C_D = cm.STY["D"][0]
C_B1B = cm.STY["B1b"][0]
ALPHA_NULL = 0.05

src_rows: list[dict] = []


def rec(panel: str, tissue: str, cell_type: str, quantity: str, value, source) -> None:
    try:
        source = source.relative_to(cm.REPO.parent)
    except ValueError:
        pass
    src_rows.append(dict(panel=panel, tissue=tissue, cell_type=cell_type, quantity=quantity, value=value,
                         source=str(source)))


def _tsv(p):
    return pd.read_csv(p, sep="\t")


# ----------------------------------------------------------------------------- data

def kidney_calls() -> pd.DataFrame:
    f = P13 / "hd4" / "hd4.tsv"
    h = _tsv(f)
    c = h.groupby(["cell_type", "call"]).size().unstack(fill_value=0)
    for k in ("transfers", "does_not_transfer", "opposite"):
        if k not in c:
            c[k] = 0
    c = c.reindex(K_TYPES)
    for t in K_TYPES:
        for k in ("transfers", "does_not_transfer", "opposite"):
            rec("a", "kidney", t, f"n_{k}", int(c.loc[t, k]), f)
    return c


def kidney_null() -> pd.DataFrame:
    f = P13 / "hd4" / "hd4_null_summary.tsv"
    s = _tsv(f)
    out = {}
    for t in K_TYPES:
        r = s[s.cell_type == t].set_index("kind")
        d = dict(p_tr_h=r.loc["human_permuted", "n_transfers_p_ge"],
                 p_tr_m=r.loc["mouse_permuted", "n_transfers_p_ge"],
                 p_nes_h=r.loc["human_permuted", "nes_spearman_p_ge"],
                 p_nes_m=r.loc["mouse_permuted", "nes_spearman_p_ge"],
                 n_h=int(r.loc["human_permuted", "n_null"]), n_m=int(r.loc["mouse_permuted", "n_null"]),
                 nes=r.loc["human_permuted", "nes_spearman_obs"])
        d["beyond"] = max(d["p_tr_h"], d["p_tr_m"], d["p_nes_h"], d["p_nes_m"]) <= ALPHA_NULL
        out[t] = d
        for k, v in d.items():
            rec("a", "kidney", t, k, v, f)
    return pd.DataFrame(out).T


def kidney_timecourse() -> dict[str, pd.DataFrame]:
    res = {}
    for stage, d in (("2d", "hd4_primary_sens_acute"), ("late", "hd4_primary_sens_late")):
        f = P13 / d / "hd4_summary.tsv"
        s = _tsv(f).set_index("cell_type")
        if stage == "late":
            # lymphoid has no acute Sham/IRI unit, so its "late only" fit is the primary fit: not shown.
            s = s.drop(index=[t for t in s.index if t == "lymphoid"])
        res[stage] = s
        for t in s.index:
            rec("b", "kidney", t, f"{stage}_n_transfer", int(s.loc[t, "n_transfer"]), f)
            rec("b", "kidney", t, f"{stage}_n_not_transfer", int(s.loc[t, "n_not_transfer"]), f)
            rec("b", "kidney", t, f"{stage}_nes_spearman", s.loc[t, "nes_spearman_all_sets"], f)
    return res


def _ctrl_spread(summary, raw, types) -> dict[str, tuple[float, float]]:
    """Mean and SD of rho_p(D) over human-label-permuted fits, per cell type."""
    out = {}
    if summary is not None and summary.exists():
        s = _tsv(summary)
        s = s[s.kind.str.contains("human")]
        for t in types:
            r = s[s.cell_type == t]
            if len(r):
                out[t] = (float(r["mean"].iloc[0]), float(r["std"].iloc[0]))
        return out
    if raw is not None and raw.exists():
        s = _tsv(raw)
        s = s[s.kind.str.contains("human")]
        col = "rho_p_D" if "rho_p_D" in s else "rho_p"
        if "cell_type" in s:
            for t in types:
                v = s.loc[s.cell_type == t, col].dropna()
                if len(v):
                    out[t] = (float(v.mean()), float(v.std(ddof=1)))
        else:
            v = s[col].dropna()
            out[types[0]] = (float(v.mean()), float(v.std(ddof=1)))
    return out


def hd1_rows() -> tuple[list[dict], bool]:
    rows = []
    f = P13 / "hd1" / "h1.tsv"
    h = _tsv(f)
    h = h[h.score == "D"].set_index("cell_type")
    fc = P13 / "hd1" / "h1_controls_summary.tsv"
    ctrl = _ctrl_spread(fc, None, K_TYPES)
    for t in K_TYPES:
        r = h.loc[t]
        rows.append(dict(tissue="kidney", cell_type=t, rho=r.rho_p, lo=r.ci_lo, hi=r.ci_hi,
                         sig=bool(r.q_bh < 0.05 and r.rho_p > 0), ctrl=ctrl.get(t)))
        for k, v in (("rho_p", r.rho_p), ("ci_lo", r.ci_lo), ("ci_hi", r.ci_hi), ("q_bh", r.q_bh)):
            rec("c", "kidney", t, k, v, f)
        rec("c", "kidney", t, "ctrl_human_mean", ctrl[t][0], fc)
        rec("c", "kidney", t, "ctrl_human_sd", ctrl[t][1], fc)
    f = T73 / "h1.tsv"
    h = _tsv(f)
    r = h[(h.score == "D") & (h.contrast == "primary")].iloc[0]
    fc = T73 / "h1_controls.tsv"
    ctrl = _ctrl_spread(None, fc, ["hepatocyte"])
    rows.append(dict(tissue="liver", cell_type="hepatocyte", rho=r.rho_p, lo=r.ci_lo, hi=r.ci_hi,
                     sig=bool(r.p_one_sided < 0.05 and r.rho_p > 0), ctrl=ctrl.get("hepatocyte")))
    for k, v in (("rho_p", r.rho_p), ("ci_lo", r.ci_lo), ("ci_hi", r.ci_hi), ("p_one_sided", r.p_one_sided)):
        rec("c", "liver", "hepatocyte", k, v, f)
    rec("c", "liver", "hepatocyte", "ctrl_human_mean", ctrl["hepatocyte"][0], fc)
    rec("c", "liver", "hepatocyte", "ctrl_human_sd", ctrl["hepatocyte"][1], fc)
    islet = ISLET_H1.exists() and ISLET_DONE.exists()
    if islet:
        h = _tsv(ISLET_H1)
        h = h[h.score == "D"].set_index("cell_type")
        ctrl = _ctrl_spread(ISLET_CTRL_SUMMARY, ISLET_CTRL_RAW, I_TYPES)
        for t in I_TYPES:
            if t not in h.index:
                continue
            r = h.loc[t]
            q = r.q_bh if "q_bh" in h else r.p_one_sided
            rows.append(dict(tissue="islet", cell_type=t, rho=r.rho_p, lo=r.ci_lo, hi=r.ci_hi,
                             sig=bool(q < 0.05 and r.rho_p > 0), ctrl=ctrl.get(t)))
            for k, v in (("rho_p", r.rho_p), ("ci_lo", r.ci_lo), ("ci_hi", r.ci_hi)):
                rec("c", "islet", t, k, v, ISLET_H1)
    return rows, islet


def transfer_vs_null() -> list[dict]:
    rows = []
    fs = P13 / "hd4" / "hd4_null_summary.tsv"
    s = _tsv(fs)
    fn = P13 / "hd4" / "hd4_null.tsv"
    n = _tsv(fn)
    n = n[n.kind == "human_permuted"]
    for t in K_TYPES:
        obs = int(s[(s.cell_type == t) & (s.kind == "human_permuted")].n_transfers_obs.iloc[0])
        v = n.loc[n.cell_type == t, "n_transfers"].to_numpy(float)
        rows.append(dict(tissue="kidney", cell_type=t, obs=obs, med=np.median(v), q95=np.quantile(v, 0.95),
                         n_null=len(v)))
    fs = T73 / "h4_summary.tsv"
    s = _tsv(fs)
    fn_l = T73 / "h4_null.tsv"
    n = _tsv(fn_l)
    v = n.loc[n.kind == "human_permuted", "n_transfers"].to_numpy(float)
    obs = int(s.loc[s.contrast == "primary", "n_transfers"].iloc[0])
    rows.append(dict(tissue="liver", cell_type="hepatocyte", obs=obs, med=np.median(v), q95=np.quantile(v, 0.95),
                     n_null=len(v)))
    fs_i = T79 / "hd4" / "hd4_summary.tsv"
    s = _tsv(fs_i).set_index("cell_type")
    fn_i = T79 / "hd4" / "hd4_null.tsv"
    n = _tsv(fn_i)
    n = n[n.kind == "human"]
    for t in I_TYPES:
        v = n.loc[n.cell_type == t, "n_transfer"].to_numpy(float)
        rows.append(dict(tissue="islet", cell_type=t, obs=int(s.loc[t, "n_transfer"]), med=np.median(v),
                         q95=np.quantile(v, 0.95), n_null=len(v)))
    obs_src = {"kidney": P13 / "hd4" / "hd4_null_summary.tsv", "liver": T73 / "h4_summary.tsv", "islet": fs_i}
    for r in rows:
        f = {"kidney": fn, "liver": fn_l, "islet": fn_i}[r["tissue"]]
        rec("d", r["tissue"], r["cell_type"], "transfer_obs", r["obs"], obs_src[r["tissue"]])
        for k in ("med", "q95", "n_null"):
            rec("d", r["tissue"], r["cell_type"], f"transfer_{k}", r[k], f)
        rec("d", r["tissue"], r["cell_type"], "transfer_obs_gt_q95", bool(r["obs"] > r["q95"]), f)
    return rows


def cippa() -> pd.DataFrame:
    f = P13 / "cippa" / "ek3.tsv"
    e = _tsv(f)
    e = e[e.score.isin(["D", "B1b_raw"])]
    for _, r in e.iterrows():
        for k in ("auroc_stratified", "p_one_sided", "n_shared", "n_human_specific"):
            rec("e", "kidney", f"{r.table}", f"{r.score}_{k}", r[k], f)
    return e


# ----------------------------------------------------------------------------- drawing

def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=6.6, pad=3)


def _fmt_p(p):
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(False)


def draw_a(ax, calls, null):
    y = np.arange(len(K_TYPES))
    left = np.zeros(len(K_TYPES))
    for k, col, lab in (("transfers", C_TRANSFER, "transfers to mouse"),
                        ("does_not_transfer", C_NOT, "does not transfer"),
                        ("opposite", C_OPPOSITE, "opposite direction")):
        w = calls[k].to_numpy(float)
        ax.barh(y, w, left=left, height=0.64, color=col, edgecolor="white", linewidth=0.5, label=lab)
        left += w
    for i, t in enumerate(K_TYPES):
        r = null.loc[t]
        ax.text(left.max() + 12, i, f"{_fmt_p(r.p_tr_h)} · {_fmt_p(r.p_tr_m)}", va="center", ha="left",
                fontsize=5.6, color=cm.INK)
    ax.text(left.max() + 12, -0.9, "label-null p\nhuman · mouse", va="bottom", ha="left", fontsize=5.4,
            color=cm.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([K_LAB[t] for t in K_TYPES])
    for lab, t in zip(ax.get_yticklabels(), K_TYPES):
        if null.loc[t, "beyond"]:
            lab.set_fontweight("bold")
    ax.set_ylim(len(K_TYPES) - 0.5, -0.6)
    ax.set_xlim(0, left.max() + 95)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("programs significant in human AKI", fontsize=6.2)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False, handlelength=1.0, borderaxespad=0.0,
              bbox_to_anchor=(0.0, -0.25), ncol=3, columnspacing=1.0)
    _clean(ax)


def _dumbbell(ax, tc, col, xlab, xlim, ticks):
    for i, t in enumerate(K_TYPES):
        a = tc["2d"][col].get(t, np.nan)
        b = tc["late"][col].get(t, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([a, b], [i, i], color=cm.RULE, lw=0.9, zorder=1)
        if np.isfinite(a):
            ax.plot(a, i, "o", ms=4, color=C_TRANSFER, mec=C_TRANSFER, zorder=3)
        if np.isfinite(b):
            ax.plot(b, i, "o", ms=4, color="white", mec=C_TRANSFER, mew=0.9, zorder=3)
        if not np.isfinite(a):
            ax.text(xlim[1], i, "no 2 d fit" if np.isfinite(b) else "no stage fit", va="center", ha="right",
                    fontsize=5.2, color=cm.MUTED)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlab, fontsize=6.2)
    _clean(ax)
    ax.tick_params(axis="y", left=False, labelleft=False)


def draw_b(ax1, ax2, tc):
    _dumbbell(ax1, tc, "n_transfer", "transferring programs", (0, 210), [0, 100, 200])
    _dumbbell(ax2, tc, "nes_spearman_all_sets", "program NES, Spearman", (0, 0.8), [0, 0.4, 0.8])
    ax1.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4, color=C_TRANSFER, label="mouse IRI 2 d"),
                        Line2D([], [], marker="o", ls="none", ms=4, color="white", mec=C_TRANSFER,
                               label="mouse IRI 4–20 wk")],
               loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=2, fontsize=5.5, frameon=False,
               handletextpad=0.2, borderaxespad=0.0, columnspacing=1.0)


def _bottom_rows(islet_types):
    """y layout shared by c and d: header rows then cell types."""
    rows, y = [], 0.0
    for tissue, head, types, lab in (
            ("kidney", "kidney: AKI vs IRI", K_TYPES, K_LAB),
            ("liver", "liver: MASH vs Western diet", ["hepatocyte"], {"hepatocyte": "hepatocyte"}),
            ("islet", "islet: T2D vs db/db, mSTZ", islet_types, I_LAB)):
        rows.append(dict(kind="head", tissue=tissue, label=head, y=y))
        y += 1
        for t in types:
            rows.append(dict(kind="type", tissue=tissue, cell_type=t, label=lab[t], y=y))
            y += 1
        y += 0.4
    return rows, y - 0.4


LEGEND_ROWS = 2.6  # empty rows under the last cell type: the legend of (d) sits there, clear of the points


def _bottom_axis(ax, rows, ymax, labels=True):
    ys = [r["y"] for r in rows]
    ax.set_yticks(ys)
    if labels:
        ax.set_yticklabels([r["label"] for r in rows])
        for lab, r in zip(ax.get_yticklabels(), rows):
            if r["kind"] == "head":
                lab.set_fontweight("bold")
                lab.set_fontsize(6.0)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ymax + LEGEND_ROWS + 0.5, -0.6)  # free rows at the bottom hold the legend of (d)
    _clean(ax)


def draw_c(ax, rows, hd1, islet_done):
    by = {(r["tissue"], r["cell_type"]): r for r in hd1}
    ymin_islet = ymax_islet = None
    for r in rows:
        if r["kind"] != "type":
            continue
        h = by.get((r["tissue"], r["cell_type"]))
        y = r["y"]
        if r["tissue"] == "islet" and not islet_done:
            ymin_islet = y if ymin_islet is None else ymin_islet
            ymax_islet = y
            continue
        if h is None:
            continue
        if h["ctrl"] is not None:
            m, sd = h["ctrl"]
            ax.add_patch(Rectangle((m - 2 * sd, y - 0.36), 4 * sd, 0.72, color=C_BAND, lw=0, zorder=0))
        ax.plot([h["lo"], h["hi"]], [y, y], color=C_D, lw=1.0, zorder=2)
        ax.plot(h["rho"], y, "o", ms=4, color=C_D, mfc=C_D if h["sig"] else "white", mec=C_D, mew=0.9, zorder=3)
    if ymin_islet is not None:
        ax.add_patch(Rectangle((-0.075, ymin_islet - 0.45), 0.15, ymax_islet - ymin_islet + 0.9, fill=False,
                               ec=cm.RULE, lw=0.7, ls=(0, (3, 2)), zorder=1))
        ax.text(0.0, (ymin_islet + ymax_islet) / 2, "islet test pending\n(seeds still training)", ha="center",
                va="center", fontsize=5.6, color=cm.MUTED)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
    lim = max([0.08] + [abs(v) + 0.005 for h in hd1 for v in (h["lo"], h["hi"])])
    lim = float(np.ceil(lim / 0.04) * 0.04)
    ax.set_xlim(-lim, lim)
    ticks = np.arange(-lim, lim + 1e-9, 0.04)
    ax.set_xticks(ticks)
    ax.set_xticklabels([("−" if t < -1e-9 else "") + (f"{abs(t):.2f}".rstrip("0").rstrip(".") if abs(t) > 1e-9 else "0")
                        for t in ticks])
    ax.set_xlabel("Δα vs gene-level transfer, $\\rho_p$ (95% CI)", fontsize=6.2)


def draw_d(ax, rows, tvn):
    by = {(r["tissue"], r["cell_type"]): r for r in tvn}
    for r in rows:
        if r["kind"] != "type":
            continue
        t = by.get((r["tissue"], r["cell_type"]))
        if t is None:
            continue
        y = r["y"]
        ax.plot([t["med"], t["q95"]], [y, y], color="#9A9A9A", lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot([t["med"]], [y], "|", ms=5, mew=1.0, color="#6A6A6A", zorder=2)
        beyond = t["obs"] > t["q95"]
        ax.plot(t["obs"], y, "o", ms=4, color=cm.INK, mfc=cm.INK if beyond else "white", mec=cm.INK, mew=0.9,
                zorder=3)
    ax.set_xlim(-4, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("transferring programs", fontsize=6.2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4, color=cm.INK, label="observed, above null 95%"),
                       Line2D([], [], marker="o", ls="none", ms=4, color="white", mec=cm.INK, label="observed, within null"),
                       Line2D([], [], color="#9A9A9A", lw=2.2, label="human-label null, median–95%")],
              loc="lower right", fontsize=5.4, frameon=False, handlelength=1.8, handletextpad=0.5, borderaxespad=0.1)


def draw_e(ax, ek):
    groups = [("acute", "acute lists\n(primary)"), ("late", "late lists")]
    for i, (tab, lab) in enumerate(groups):
        for dy, score, col, mk in ((-0.16, "D", C_D, "o"), (0.16, "B1b_raw", C_B1B, "^")):
            r = ek[(ek.table == tab) & (ek.score == score)].iloc[0]
            ax.plot(r.auroc_stratified, i + dy, mk, ms=4.2, color=col, zorder=3)
            ax.text(r.auroc_stratified + 0.012, i + dy, f"{r.auroc_stratified:.2f}", va="center", ha="left",
                    fontsize=5.4, color=cm.INK)
    ax.axvline(0.5, color=cm.MUTED, lw=0.6, zorder=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([g[1] for g in groups])
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0.4, 0.82)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("stratified AUROC,\nshared vs human-specific", fontsize=6.2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4, color=C_D, label="Chimera Δα"),
                       Line2D([], [], marker="^", ls="none", ms=4, color=C_B1B, label="profile correlation")],
              loc="center right", bbox_to_anchor=(1.02, 0.5), fontsize=5.4, frameon=False, handletextpad=0.2,
              borderaxespad=0, borderpad=0)
    _clean(ax)
    ax.tick_params(axis="y", length=0)


def main() -> int:
    cm.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    calls, null = kidney_calls(), kidney_null()
    tc = kidney_timecourse()
    hd1, islet_done = hd1_rows()
    tvn = transfer_vs_null()
    ek = cippa()

    # At its size on the page (F1): panels do not overlap, both rows fill the text width at one scale.
    fig = plt.figure(figsize=(6.9, 5.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[9.6, 17.0], hspace=0.42)
    top_all = outer[0].subgridspec(1, 2, width_ratios=[2.45, 2.15], wspace=0.04)
    top = top_all[1].subgridspec(1, 2, width_ratios=[1.2, 1.1], wspace=0.14)
    bot_all = outer[1].subgridspec(1, 3, width_ratios=[2.85, 1.0, 0.3], wspace=0.36)
    bot = bot_all[0].subgridspec(1, 2, wspace=0.08)
    ax_a = fig.add_subplot(top_all[0])
    ax_b1 = fig.add_subplot(top[0], sharey=ax_a)
    ax_b2 = fig.add_subplot(top[1], sharey=ax_a)
    ax_c = fig.add_subplot(bot[0])
    ax_d = fig.add_subplot(bot[1], sharey=ax_c)
    ax_e = fig.add_subplot(bot_all[1])

    draw_a(ax_a, calls, null)
    draw_b(ax_b1, ax_b2, tc)
    rows, ymax = _bottom_rows(I_TYPES)
    draw_c(ax_c, rows, hd1, islet_done)
    _bottom_axis(ax_c, rows, ymax, labels=True)
    draw_d(ax_d, rows, tvn)
    _bottom_axis(ax_d, rows, ymax, labels=False)
    draw_e(ax_e, ek)

    _title(ax_a, "Kidney: AKI programs in mouse IRI")
    _title(ax_b1, "IRI 2 d vs 4–20 wk")
    _title(ax_c, "Healthy-tissue test: does Δα predict transfer?")
    _title(ax_d, "Program transfer vs label null")
    _title(ax_e, "Cippà 2018 injury genes")
    fig.align_xlabels([ax_a, ax_b1, ax_b2])

    # One PDF per panel; the figure spec (figures/specs/main.py, main_disease) places them and the layout tool adds the letters.
    pdfs = [cm.save_axes(fig, f"{NAME}_{k}", axes)
            for k, axes in (("a", [ax_a]), ("b", [ax_b1, ax_b2]), ("c", [ax_c]), ("d", [ax_d]), ("e", [ax_e]))]
    png = cm.fd.save_figure(fig, NAME, formats=("png",), out_dir=OUT, close=True)[0]
    src = pd.DataFrame(src_rows)
    src.to_csv(OUT / "fig8_source.tsv", sep="\t", index=False)
    chks = [cm.check_pdf(p) for p in pdfs]
    print("wrote " + ", ".join(p.name for p in pdfs) + f" in {pdfs[0].parent}\n      {png}\n      {OUT / 'fig8_source.tsv'} ({len(src)} values)")
    print(f"islet H-D1: {'drawn from ' + str(ISLET_H1) if islet_done else 'absent (' + str(ISLET_H1) + '); slot drawn as pending'}")
    print(f"pdf checks: {chks}")
    return 0 if all(c["ok"] for c in chks) else 1


if __name__ == "__main__":
    sys.exit(main())
