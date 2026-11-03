"""E1 candidate panels in the style of the harmonised panels (H1 figures.py; notebooks/_figure_design.py).

Panels in results/revision/E1/panels/; composed with F1_figures/compose.py --spec revision/tasks/E1_atlas/specs_e1.py, with
CHIMERA_MANUSCRIPT pointed at a scratch folder (E1 does not write into the manuscript); composed PDFs copied to results/revision/E1/figures/.

    PYTHONPATH=$PWD python -m revision.tasks.E1_atlas.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from revision.tasks.E1_atlas import rules as RU  # noqa: E402
from revision.tasks.H1_harmonise.figures import SET_LABEL, TISSUE_LABEL, panel_rc  # noqa: E402
from revision.tasks.T8_4_figures import common as cm  # noqa: E402

PANELS = RU.OUT / "panels"
UP, DOWN = "#D55E00", "#0072B2"
LADDER_COL = {"cortex": "#009E73", "eye": "#CC79A7", "liver": "#E69F00", "testis": "#56B4E9"}
QUAD = [("fast", "above", "Fast-evolving proteins, deployment above the prior"), ("conserved", "above", "Conserved proteins, deployment above the prior"),
        ("conserved", "below", "Conserved proteins, deployment below the prior"), ("fast", "below", "Fast-evolving proteins, deployment below the prior")]
SHORT = {"Cleavage on pair of basic residues [PTM]": "Propeptide cleavage [PTM]", "external encapsulating structure (CC)": "external encapsulating structure (CC)"}


def save(fig, name):
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(PANELS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def label(r) -> str:
    n = str(r["name"])
    if r["library"] == "class":
        n = SET_LABEL.get(n, n.replace("broad:", "")) + ("" if "(category)" in n else " (class)")
    n = SHORT.get(n, n)
    n = n.replace(" [Cellular component]", " [CC]").replace(" [Molecular function]", " [MF]").replace(" [Biological process]", " [BP]")
    if r["library"] in ("reactome", "reactome_top"):
        n += " (R)"
    return n if len(n) <= 48 else n[:46] + "…"


def protein_side(A):
    """conserved: identity recurrent above matched-expression genes with positive Stouffer z; fast: the mirror."""
    return np.where(A.pi_recurrent & (A.pi_direction == "above") & (A.pi_stouffer > 0), "conserved",
                    np.where(A.pi_recurrent & (A.pi_direction == "below") & (A.pi_stouffer < 0), "fast", "neither"))


def atlas_rows(k=6) -> tuple[pd.DataFrame, list]:
    """Rule: STRICT sets, pruned for redundancy, split by protein side (pi recurrent above = conserved, below = fast) and deployment
    side; top k per quadrant by the mean Stouffer z of the two gates."""
    A = pd.read_csv(RU.OUT / "atlas.csv", index_col=0)
    A = A[A.strict & A.redundant_with.isna()].copy()
    A["prot"] = protein_side(A)
    rows, groups = [], []
    for p, d, lab in QUAD:
        x = A[(A.prot == p) & (A.can_direction == d)].sort_values("joint_stouffer", ascending=(d == "below"))
        x = x[x.groupby("library").cumcount() < 2].head(k)  # at most two sets per library, so one vocabulary does not fill a group
        rows.append(x)
        groups.append((lab, len(x)))
    return pd.concat(rows), groups


def p_atlas():
    R, groups = atlas_rows()
    T = RU.TISSUES
    rh, wl, wc, ws = 0.118, 1.95, 0.205, 0.34
    nr = len(R)
    W = wl + 2 * 8 * wc + 0.12 + 3 * ws
    top, bot = 0.62, 0.05
    H = rh * nr + top + bot
    fig = plt.figure(figsize=(W, H))
    vmax = 0.6
    for b, (gate, pre) in enumerate((("canonical gate", "can"), ("per-gene gate", "free"))):
        ax = fig.add_axes([(wl + b * (8 * wc + 0.06)) / W, bot / H, 8 * wc / W, rh * nr / H])
        M = R[[f"{pre}_e_{t}" for t in T]].to_numpy(float)
        Q = R[[f"{pre}_q_{t}" for t in T]].to_numpy(float)
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        for i in range(nr):
            for j in range(8):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=4.6, color=cm.MUTED)
                elif Q[i, j] < 0.05:
                    ax.text(j, i, "•", ha="center", va="center", fontsize=4.6, color="#ffffff" if abs(v) > 0.6 * vmax else "#222222")
        ax.set_xticks(range(8))
        ax.set_xticklabels([TISSUE_LABEL[t] for t in T], fontsize=5.4, rotation=40, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.set_yticks(range(nr))
        ax.set_yticklabels([label(r) for _, r in R.iterrows()] if b == 0 else [], fontsize=5.3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(3.5, -3.9, gate, ha="center", va="bottom", fontsize=6.0, color=cm.INK)
        y = -0.5
        for _, n in groups[:-1]:
            y += n
            ax.axhline(y, color="#222222", lw=0.8)
    ax2 = fig.add_axes([(wl + 2 * 8 * wc + 0.12) / W, bot / H, 3 * ws / W, rh * nr / H])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(nr - 0.5, -0.5)
    for j, hd in enumerate(["π z", "dN/dS z", "B1 z"]):
        ax2.text(j, -1.0, hd, fontsize=5.6, ha="center", va="bottom", color=cm.INK)
    for i, (_, r) in enumerate(R.iterrows()):
        ax2.text(0, i, f"{r.pi_stouffer:+.0f}", fontsize=4.9, ha="center", va="center", color=cm.INK)
        ax2.text(1, i, f"{r.dNdS_stouffer:+.0f}", fontsize=4.9, ha="center", va="center", color=cm.INK)
        ax2.text(2, i, f"{r.B1_stouffer:+.0f}", fontsize=4.9, ha="center", va="center", color=cm.INK)
    ax2.axis("off")
    short = ["fast protein,\nabove prior", "conserved protein,\nabove prior", "conserved protein,\nbelow prior", "fast protein,\nbelow prior"]
    y = 0
    for (lab, n), sl in zip(groups, short):
        ax2.text(2.75, y + (n - 1) / 2, sl, fontsize=5.2, ha="left", va="center", color=cm.INK, clip_on=False)
        y += n
    cax = fig.add_axes([(wl + 2 * 8 * wc + 0.12 + 0.05) / W, (H - 0.16) / H, 1.0 / W, 0.05 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("Δα effect vs matched genes (SD)", fontsize=5.2, pad=1.5)
    save(fig, "e1_atlas")
    R[["name", "library", "can_stouffer", "free_stouffer", "pi_stouffer", "dNdS_stouffer", "B1_stouffer"]].to_csv(PANELS / "e1_atlas_rows.csv")
    return groups


def p_tissue():
    t = pd.read_csv(RU.OUT / "tissue_programmes.csv")
    s = pd.read_csv(RU.OUT / "sets" / "sets.csv").set_index("key")
    T = RU.TISSUES
    rows = []
    for own, ids in RU.TISSUE_PROGRAMMES.items():
        for k in t[(t.tissue == own) & (t.readout == "D_can")].key:
            if k not in [r[1] for r in rows if r[0] == own]:
                rows.append((own, k))
    rows = [r for r in rows if np.isfinite(t[(t.tissue == r[0]) & (t.key == r[1]) & (t.readout == "D_can")].z_own).all()]
    nr = len(rows)
    rh, wl, wc = 0.112, 1.75, 0.165
    blocks = (("D_can", "canonical gate"), ("D_free", "per-gene gate"), ("B1", "pseudobulk profile\ncorrelation (B1)"))
    W = wl + 3 * 8 * wc + 2 * 0.08
    top, bot = 0.78, 0.05
    H = rh * nr + top + bot
    fig = plt.figure(figsize=(W, H))
    vmax = 8.0
    for b, (r_, title) in enumerate(blocks):
        ax = fig.add_axes([(wl + b * (8 * wc + 0.08)) / W, bot / H, 8 * wc / W, rh * nr / H])
        M = np.array([[t[(t.readout == r_) & (t.tissue == own) & (t.key == k)][f"z_{x}"].iloc[0] for x in T] for own, k in rows], float)
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        for i, (own, k) in enumerate(rows):
            j = T.index(own)
            rr = t[(t.readout == r_) & (t.tissue == own) & (t.key == k)].iloc[0]
            hit = (rr.q_contrast < 0.05) and (rr.contrast > 0)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.0 if hit else 0.5, ec="#222222", ls="-" if hit else ":"))
        ax.set_xticks(range(8))
        ax.set_xticklabels([TISSUE_LABEL[x] for x in T], fontsize=5.3, rotation=40, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.set_yticks(range(nr))
        lab = []
        for own, k in rows:
            n = str(s.loc[k, "name"])
            n = n if len(n) <= 34 else n[:32] + "…"
            lab.append(f"{TISSUE_LABEL[own]}: {n}")
        ax.set_yticklabels(lab if b == 0 else [], fontsize=5.1)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(3.5, -4.6, title, ha="center", va="bottom", fontsize=5.8, color=cm.INK)
        prev = rows[0][0]
        for i, (own, _) in enumerate(rows):
            if own != prev:
                ax.axhline(i - 0.5, color="#ffffff", lw=1.2)
                prev = own
    cax = fig.add_axes([0.25 / W, (H - 0.2) / H, 1.0 / W, 0.05 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("set z vs matched genes", fontsize=5.2, pad=1.5)
    save(fig, "e1_tissue")


def p_distance():
    d = pd.read_csv(RU.OUT / "distance_cohorts.csv")
    d = d[d.pair & d.tissue.isin(LADDER_COL) & ~d.cohort.isin(["brain_m1_hmouse"])]
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.85), gridspec_kw=dict(wspace=0.42))
    for ax, (col, ylab) in zip(axs, (("pole", "programme pole, mean effect\n(SD units, matched genes)"), ("rho_consensus", "agreement with other\ntissues' programme order (ρ)"))):
        for lad, c in LADDER_COL.items():
            for gate, mk in (("canonical", "o"), ("per-gene", "s")):
                x = d[(d.tissue == lad) & (d.gate == gate)].sort_values("divergence_my")
                x = x.groupby("divergence_my", as_index=False).mean(numeric_only=True)
                kw = dict(color=c, marker=mk, ms=3.2, lw=0.9, mfc=c if gate == "canonical" else "white", mew=0.8, ls="-" if gate == "canonical" else "--")
                if col == "pole":
                    ax.plot(x.divergence_my, x.above_pole, **kw)
                    ax.plot(x.divergence_my, x.below_pole, **kw)
                else:
                    ax.plot(x.divergence_my, x.rho_consensus, **kw)
        ax.set_xscale("log")
        ax.set_xticks([6.4, 29, 90, 429])
        ax.set_xticklabels(["6", "29", "90", "429"])
        ax.set_xlabel("divergence from human (My)")
        ax.set_ylabel(ylab)
        cm.fd.style_axes(ax) if hasattr(cm.fd, "style_axes") else None
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].axhline(0, color=cm.MUTED, lw=0.5)
    axs[0].text(7, 0.52, "above-prior programmes", fontsize=5.2, color=cm.INK)
    axs[0].text(7, -0.47, "below-prior programmes", fontsize=5.2, color=cm.INK)
    axs[0].set_ylim(-0.55, 0.62)
    axs[1].set_ylim(0.15, 1.0)
    h = [Line2D([], [], color=c, lw=0.9, label=l) for l, c in LADDER_COL.items()]
    h += [Line2D([], [], color=cm.INK, marker="o", lw=0.9, ms=3, label="canonical"), Line2D([], [], color=cm.INK, marker="s", mfc="white", ls="--", lw=0.9, ms=3, label="per-gene")]
    axs[1].legend(handles=h, fontsize=5.0, frameon=False, loc="lower right", ncol=3, handlelength=1.6, columnspacing=0.6)
    save(fig, "e1_distance")


def p_genes(k_up=24, k_dn=14):
    R = pd.read_csv(RU.OUT / "genes" / "recurrent.csv", index_col=0)
    R = R[~R.is_mt]
    K = pd.read_parquet(RU.OUT / "genes" / "calls.parquet")
    prim = [RU.PRIMARY[t] for t in RU.TISSUES]
    up = R[R.direction == "above"].sort_values(["n_tissues", "n_cohorts_up", "int_can"], ascending=False).head(k_up)
    dn = R[R.direction == "below"].sort_values(["n_tissues", "n_cohorts_down", "int_can"], ascending=[False, False, True]).head(k_dn)
    G = pd.concat([up, dn])
    th = K[K.cohort.isin(prim) & K.hsapiens_gene_id.isin(G.index)]
    T = RU.TISSUES
    nr = len(G)
    fig = plt.figure(figsize=(2.75, 0.105 * nr + 0.6))
    ax = fig.add_axes([0.2, 0.02, 0.62, 0.84])
    for i, g in enumerate(G.index):
        for j, tt in enumerate(T):
            x = th[(th.hsapiens_gene_id == g) & (th.tissue == tt)]
            if not len(x):
                continue
            x = x.iloc[0]
            if x.call != 0:
                ax.scatter(j, i, s=16, c=UP if x.call > 0 else DOWN, edgecolors="none")
            else:
                ax.scatter(j, i, s=5, c="#cccccc", edgecolors="none")
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([TISSUE_LABEL[x] for x in T], fontsize=5.3, rotation=40, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(nr))
    ax.set_yticklabels(G.symbol, fontsize=5.2, style="italic")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(len(up) - 0.5, color="#222222", lw=0.7)
    ax2 = fig.add_axes([0.83, 0.02, 0.16, 0.84], sharey=ax)
    ax2.set_xlim(-0.5, 1.5)
    for j, hd in enumerate(["π", "cohorts"]):
        ax2.text(j, -1.2, hd, fontsize=5.3, ha="center", va="bottom", color=cm.INK)
    for i, (_, r) in enumerate(G.iterrows()):
        ax2.text(0, i, f"{r.pi_raw:.2f}", fontsize=4.8, ha="center", va="center", color=cm.INK)
        ax2.text(1, i, f"{int(r.n_cohorts_up if r.direction == 'above' else r.n_cohorts_down)}/{int(r.n_cohorts_tested)}", fontsize=4.8, ha="center", va="center", color=cm.INK)
    ax2.axis("off")
    save(fig, "e1_genes")


def main() -> int:
    cm.fd.apply_design()
    panel_rc()
    g = p_atlas()
    print(g)
    p_tissue()
    p_distance()
    p_genes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
