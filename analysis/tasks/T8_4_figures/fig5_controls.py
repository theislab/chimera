"""Fig 5 controls panel: feature destruction, count permutation and matched cell types, with the seed floor.

Each point is one control run; its value is the Spearman correlation of the run's readout with the
reference readout of normal training (consensus over the other seeds of the cohort). The grey band is
the seed floor: the range of the same correlation between one normal seed and the consensus of the
others (T3.4 ``floors.csv``, T3.5 ``floor.csv``, T3.6 ``floor.csv``), i.e. what an unperturbed rerun gives.
Count permutation is shown twice: the readout at the gene's own sequence ("stays with sequence") and at
the gene whose counts moved there ("follows the counts"); the thin bar is the offline projection
null for the first (T3.5 ``null_per_run.csv``, 2.5-97.5%).

Sources: T3.4 ``arms.csv``; T3.5 ``per_run.csv``; T3.6 ``matched_vs_canonical.csv`` and
``matched_three_seeds.csv``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.tasks.T8_4_figures import common as cm

FEAT = {"m": "no gene features (free per-gene offset)", "rand": "random gene features",
        "zesm": "protein-language features set to zero", "zexpr": "expression features set to zero"}
PERM = {"a": "counts shuffled across genes (all features detached)",
        "b": "counts shuffled across genes (expression features follow)",
        "c": "counts shuffled within expression bins"}
COHORTS = ["liv_hmouse", "liv_4sp", "test_hmouse"]


def rows_for(c: str) -> list[dict]:
    a = pd.read_csv(cm.RES / "T3_4" / "arms.csv")
    p = pd.read_csv(cm.RES / "T3_5" / "per_run.csv")
    nul = pd.read_csv(cm.RES / "T3_5" / "null_per_run.csv").set_index("run")
    out = []
    for arm in ["m", "rand", "zesm", "zexpr"]:
        s = a[(a.cohort == c) & (a.arm == arm)]
        if len(s):
            out.append(dict(label=FEAT[arm], D=s.rho_D_spearman.tolist(), S=s.rho_S_vs_S_ref.tolist(),
                            note="degenerate run: Δα constant" if s.rho_D_spearman.isna().all() else ""))
    for arm in ["a", "b", "c"]:
        s = p[(p.cohort == c) & (p.arm == arm)]
        if len(s):
            nl = nul.loc[s.run]
            out.append(dict(label=PERM[arm] + ":\n  readout stays with the gene's sequence",
                            D=s.D_spearman_slot.tolist(), S=s.S_spearman_slot.tolist(),
                            null=(nl.spearman_slot__p2_5.min(), nl.spearman_slot__p97_5.max())))
            out.append(dict(label="  readout follows the moved counts", D=s.D_spearman_moved.tolist(),
                            S=s.S_spearman_moved.tolist()))
    if c == "liv_4sp":
        m = pd.read_csv(cm.RES / "T3_6" / "matched_three_seeds.csv")
        m = m[m.arm.str.startswith("matchA s")]
        ok = m.dropna(subset=["D_spearman"])
        failed = sorted(set(m.arm) - set(ok.arm))
        out.append(dict(label="hepatocytes and endothelial cells only", D=ok.D_spearman.tolist(), S=ok.S.tolist(),
                        note=f"{len(failed)} of {len(m)} seeds failed training" if failed else ""))
    return out


def floor(c: str) -> dict:
    f = pd.read_csv(cm.RES / "T3_5" / "floor.csv")
    f = f[f.cohort == c]
    return {r: (f[f.readout == r].spearman.min(), f[f.readout == r].spearman.max()) for r in ("D", "S")}


def panel() -> plt.Figure:
    blocks = [(c, rows_for(c), floor(c)) for c in COHORTS]
    # cross-check: T3.4 floors agree with T3.5 floors (same seeds, same definition)
    f4 = pd.read_csv(cm.RES / "T3_4" / "floors.csv").set_index("cohort")
    for c, _, fl in blocks:
        v = [float(x) for x in f4.loc[c, "F_spearman"].split()]
        assert abs(min(v) - fl["D"][0]) < 1e-3 and abs(max(v) - fl["D"][1]) < 1e-3, c
    n = sum(len(r) for _, r, _ in blocks)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.2 * n + 1.5), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1]))
    yt, yl, y, spans = [], [], 0.0, []
    for c, rows, fl in blocks:
        y0 = y
        for r in rows:
            r["y"] = y
            yt.append(y)
            yl.append(r["label"])
            y += 1.35 if "\n" in r["label"] else 1
        spans.append((c, y0 - 0.5, y - 0.5, fl))
        y += 1.4
    for ax, key, title in zip(axes, ["D", "S"], ["Δα", "count-evidence margin $S_g$"]):
        col, mk = cm.STY[key][0], cm.STY[key][1]
        for c, lo, hi, fl in spans:
            ax.add_patch(plt.Rectangle((fl[key][0], lo), fl[key][1] - fl[key][0], hi - lo, color="#E6E6E6",
                                       lw=0, zorder=0))
            ax.text(-0.38, lo - 0.25, cm.COHORT[c], ha="left", va="bottom", fontsize=6.5, fontweight="bold",
                    color=cm.INK)
        for _, rows, _ in blocks:
            for r in rows:
                vals = [v for v in r[key] if np.isfinite(v)]
                yy = r["y"] + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.09
                if key == "D" and r.get("null"):
                    ax.plot(r["null"], [r["y"] + 0.42] * 2, color=cm.MUTED, lw=1.6, solid_capstyle="butt",
                            zorder=2)
                ax.scatter(vals, yy, s=11, marker=mk, color=col, edgecolor="white", lw=0.3, zorder=3)
                if r.get("note") and not vals:
                    ax.text(0.0, r["y"], r["note"], va="center", fontsize=5.3, color=cm.MUTED, style="italic")
                elif r.get("note") and key == "D":
                    ax.text(min(vals) - 0.04, r["y"], r["note"], va="center", ha="right", fontsize=5.3,
                            color=cm.MUTED, style="italic")
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        ax.set_xlim(-0.4, 1.02)
        ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(f"{title} of control run vs normal training\n(Spearman ρ across genes)", fontsize=6.5)
    axes[0].set_yticks(yt)
    axes[0].set_yticklabels(yl, fontsize=5.8)
    axes[0].set_ylim(y - 1.0, -1.4)
    fig.legend(handles=[Patch(color="#E6E6E6", label="seed floor: one normal seed vs the others (range)"),
                        Line2D([], [], color=cm.MUTED, lw=1.6,
                               label="projection null for 'stays with sequence' (2.5-97.5%)"),
                        Line2D([], [], marker="o", color=cm.STY["D"][0], ls="none", ms=4, label="Δα, one control run"),
                        Line2D([], [], marker="s", color=cm.STY["S"][0], ls="none", ms=4,
                               label="$S_g$, one control run")],
               loc="outside lower center", ncol=2, fontsize=5.8)
    return fig


def main() -> list:
    cm.setup()
    return [cm.save(panel(), "fig5_controls")]


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
