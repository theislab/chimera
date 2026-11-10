"""Scoreboard figures (one panel each, PDF with embedded fonts, ``notebooks/_figure_design.py``).

- forest_H1_H4.pdf: rho_p of D with the block-bootstrap 95% CI per hypothesis, primary
  H+Mouse cohort and the canonical multi-species cohort (4-sp liver, 3-sp testis) as an
  estimate (Figs 4-6 rule); noise ceiling marked where defined.
- head_to_head.pdf: Delta_j = rho_p(D) - rho_p(B_j) on U_pair; thick = per-comparison 95% CI,
  thin = simultaneous 95% CI (max-|t|); dotted lines at +-0.03.
- set_discrimination.pdf: stratified AUROC of D per set and tissue with 95% CI; pi_raw
  (expression-stratified) as the open reference.
- coverage.pdf: genes in each universe (U_g of D, and of each baseline) per hypothesis.
- reliability.pdf: rel(x|C) of D and baselines and rel(G|C) per hypothesis.

Usage: python -m analysis.tasks.T4_3_scoreboard.figures <outdir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HYP = ["H1", "H2", "H3", "H4"]
HLAB = {"H1": "H1 liver level", "H2": "H2 testis level", "H3": "H3 liver pattern", "H4": "H4 testis pattern"}
J = ["B1b", "B1c", "B2b", "B3a", "B1z"]


def _save(fig, outdir: Path, name: str) -> Path:
    d = outdir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{name}.pdf"
    tmp = d / f"{name}.pdf.tmp"
    fig.savefig(tmp, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    os.replace(tmp, final)
    return final


def forest(sb: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = FD.make_axes((FD.SINGLE_W + 0.4, 2.6))
    y, ticks, labels = 0, [], []
    for h in HYP:
        for fam, mk, lab in (("F1", "o", "H+Mouse (primary)"), ("F3_figs4_6", "s", "multi-species (estimate)")):
            r = sb[(sb.family == fam) & (sb.hyp == h) & (sb.score == "D")]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color="#222222", lw=0.9)
            ax.plot(r.rho_p, y, mk, color="#222222", mfc="#222222" if fam == "F1" else "white", ms=4)
            c = r.get("ceiling", np.nan)
            if np.isfinite(c):
                ax.plot(c, y, "|", color="#888888", ms=6)
            ticks.append(y)
            cohort = "4-sp" if r.cohort == "liv_4sp" else ("3-sp" if r.cohort == "test_3sp" else "H+Mouse")
            labels.append(f"{HLAB[h]}, {cohort}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color="#222222", lw=0.5)
    for v in (0.03, 0.10):
        ax.axvline(v, color="#888888", lw=0.5, ls=":")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\rho_p$(D, G) with 95% block-bootstrap CI")
    return _save(fig, outdir, "forest_H1_H4")


def head_to_head(h2h: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = FD.make_axes((FD.SINGLE_W + 0.3, 3.2))
    y, ticks, labels = 0, [], []
    for h in HYP:
        d = h2h[h2h.id == f"{h}:h2h"].set_index("comparator").reindex(J)
        for j in J:
            if j not in d.index or not np.isfinite(d.loc[j, "delta"]):
                continue
            r = d.loc[j]
            ax.plot([r.sim95_lo, r.sim95_hi], [y, y], color="#888888", lw=0.6)
            ax.plot([r.pc_lo, r.pc_hi], [y, y], color="#222222", lw=1.6)
            ax.plot(r.delta, y, "o", color="#222222", ms=3)
            ticks.append(y)
            labels.append(f"{h} vs {j}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color="#222222", lw=0.5)
    for v in (-0.03, 0.03):
        ax.axvline(v, color="#888888", lw=0.5, ls=":")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(r"$\Delta_j=\rho_p(D)-\rho_p(B_j)$ on $U_{pair}$")
    return _save(fig, outdir, "head_to_head")


def set_discrimination(sets: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = FD.make_axes((FD.SINGLE_W, 2.6))
    f2 = sets[sets.family == "F2"]
    y, ticks, labels = 0, [], []
    for _, r in f2.sort_values(["tissue", "set"]).iterrows():
        ax.plot([r.auroc_ci_lo, r.auroc_ci_hi], [y, y], color="#222222", lw=0.9)
        ax.plot(r.auroc, y, "o", color="#222222", ms=3.5)
        pi = sets[(sets.id == f"set:{r.tissue}:{r.set}:pi_raw")]
        if len(pi) and np.isfinite(pi.iloc[0].get("auroc", np.nan)):
            ax.plot(pi.iloc[0].auroc, y, "o", color="#888888", mfc="white", ms=3.5)
        ticks.append(y)
        arrow = "above" if r.direction == "above" else "below"
        labels.append(f"{r.tissue} {r.set} ({arrow}; q={r.q_bh:.2g})")
        y -= 1
    ax.axvline(0.5, color="#222222", lw=0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("stratified AUROC (D), open: pi_raw")
    return _save(fig, outdir, "set_discrimination")


def coverage(sb: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = FD.make_axes((FD.SINGLE_W, 2.4))
    scores = ["D"] + J + ["B3b"]
    w = 0.11
    for k, s in enumerate(scores):
        vals = []
        for h in HYP:
            fam = "F1" if s == "D" else "scoreboard"
            r = sb[(sb.family == fam) & (sb.hyp == h) & (sb.score == s) & (sb.analysis != "control_raw_spearman_only")]
            vals.append(float(r.iloc[0].n) if len(r) else np.nan)
        shade = "#222222" if s == "D" else str(0.35 + 0.08 * k)
        ax.bar(np.arange(4) + (k - 3) * w, vals, width=w, color=shade, label=s)
    ax.set_xticks(range(4))
    ax.set_xticklabels(HYP)
    ax.set_ylabel("genes in the universe")
    FD.thin_legend(ax, loc="upper right", ncol=2)
    return _save(fig, outdir, "coverage")


def reliability(sb: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = FD.make_axes((FD.SINGLE_W, 2.4))
    scores = ["D"] + J + ["B3b"]
    for i, h in enumerate(HYP):
        for k, s in enumerate(scores):
            fam = "F1" if s == "D" else "scoreboard"
            r = sb[(sb.family == fam) & (sb.hyp == h) & (sb.score == s) & (sb.analysis != "control_raw_spearman_only")]
            if not len(r):
                continue
            v = r.iloc[0].get("rel_x", np.nan)
            if np.isfinite(v):
                ax.plot(i + (k - 3) * 0.1, v, "o" if s == "D" else ".", color="#222222" if s == "D" else "#777777",
                        ms=4 if s == "D" else 3)
        r = sb[(sb.family == "F1") & (sb.hyp == h) & (sb.score == "D")]
        g = r.iloc[0].get("rel_G", np.nan) if len(r) else np.nan
        if np.isfinite(g):
            ax.plot([i - 0.4, i + 0.4], [g, g], color="#222222", lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(HYP)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("split-half reliability given C")
    ax.text(0.01, 0.02, "filled: D (seed splits); dots: baselines (cell halves); dashed: gold", transform=ax.transAxes,
            fontsize=6)
    return _save(fig, outdir, "reliability")


def make(outdir: Path) -> list[Path]:
    FD.apply_design()
    sb = pd.read_csv(outdir / "scoreboard.csv")
    h2h = pd.read_csv(outdir / "head_to_head.csv")
    sets = pd.read_csv(outdir / "sets.csv")
    return [forest(sb, outdir), head_to_head(h2h, outdir), set_discrimination(sets, outdir), coverage(sb, outdir),
            reliability(sb, outdir)]


if __name__ == "__main__":
    for p in make(Path(sys.argv[1])):
        print(p)
