"""Y6 (EXPLORATORY): panels of fig_y6_uses, composed by F1_figures.compose with the output folder redirected to results/analysis/Y6 (never the manuscript).

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.figures
    CHIMERA_MANUSCRIPT=results/analysis/Y6/compose PYTHONPATH=$PWD python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/Y6_uses/specs.py fig_y6_uses
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.tasks.Y6_uses import rules as RU  # noqa: E402

sys.path.insert(0, str(RU.ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = RU.OUT / "figures" / "panels"
U1 = RU.OUT / "use1"
# Fixed categorical order, reference palette; every score also has its own marker, so identity is never colour alone.
C = {"chimera": "#2a78d6", "b1": "#eb6834", "zero_gap": "#1baf7a", "zero_s": "#eda100", "gap": "#6250d6", "identity": "#7a7a78", "delta": "#e87ba4"}
M = {"chimera": "o", "b1": "s", "zero_gap": "^", "zero_s": "v", "gap": "D", "identity": "X", "delta": "P"}
MS = 4.2


def setup():
    FD.apply_design()
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8, "pdf.fonttype": 42,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def forest_axes(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)


def panel_a():
    """AP lift on even autosomes, S_row against baselines, per row and label."""
    m = pd.read_csv(U1 / "metrics.csv")
    m = m[(m.universe == "expr") & m.score.notna() & m.note.isna()]
    ph = pd.read_csv(U1 / "posthoc_ap.csv")
    ph = ph[ph.universe == "expr"]
    rows = [("bone_marrow", "mouse_lemur", "T_later", "Lemur, marrow · later change"), ("bone_marrow", "mouse_lemur", "T_lowconf", "Lemur, marrow · low confidence"), ("vz_full", "pig", "T_later", "Pig, eye · later change"),
            ("vz_full", "pig", "T_lowconf", "Pig, eye · low confidence"), ("test_3sp", "Macaque", "T_newasm", "Macaque, testis · not in Mmul_8"), ("test_3sp", "Macaque", "T_later", "Macaque, testis · later change"), ("liv_4sp", "Monkey", "T_later", "Macaque, liver · later change"),
            ("vz_full", "mouse", "T_lowconf", "Mouse, eye · low confidence"), ("test_3sp", "Mouse", "T_later", "Mouse, testis · later change"), ("liv_4sp", "Zebrafish", "T_later", "Zebrafish, liver · later change")]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.axvline(1, color="#bbbbb8", lw=0.6, zorder=0)
    for i, (f, l, lb, _) in enumerate(rows):
        x = m[(m.fit == f) & (m.species_label == l) & (m.label == lb)].set_index("score")
        s = x.loc["S_row"]
        ax.plot([s.ap_lift_lo, s.ap_lift_hi], [i, i], color=C["chimera"], lw=0.9, zorder=2, solid_capstyle="round")
        for k, col, off in (("S_row", "chimera", 0.0), ("zero_gap", "zero_gap", 0.2), ("zero_s", "zero_s", -0.2), ("id_neg", "identity", 0.32)):
            if k == "id_neg" and lb == "T_lowconf":
                continue  # Compara's high-confidence rule uses identity, so identity is circular on this label.
            ax.plot(x.loc[k, "ap_lift"], i + off, M[col], color=C[col], ms=MS if k == "S_row" else MS - 1, mec="white", mew=0.3, zorder=3, ls="none")
        p = ph[(ph.fit == f) & (ph.species_label == l) & (ph.label == lb) & (ph.score == "gap_abs")]
        if len(p):
            ax.plot(p.ap_lift.iloc[0], i - 0.32, M["gap"], color=C["gap"], ms=MS - 1.2, mec="white", mew=0.3, zorder=3, ls="none")
    forest_axes(ax, [r[3] for r in rows])
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xlim(0.45, 80)
    ax.set_xlabel("Average precision / base rate, even autosomes")
    h = [plt.Line2D([], [], marker=M[k], color=C[k], ls="none", ms=3.6, mec="white", mew=0.3, label=t) for k, t in
         (("chimera", "Species margin"), ("zero_gap", "Zero-fraction gap"), ("zero_s", "Zero fraction"), ("gap", "CP10k gap"), ("identity", "Identity"))]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=5, frameon=False, handletextpad=0.0, columnspacing=0.55, borderaxespad=0.1)
    save(fig, "a")


def panel_b():
    z = pd.read_csv(U1 / "posthoc_zero.csv")
    rows = [("test_3sp", "Macaque", "Macaque, testis"), ("test_hmac", "Macaque", "Macaque, testis pair"), ("liv_4sp", "Monkey", "Macaque, liver"), ("vz_full", "macaca_mulatta", "M. mulatta, eye"), ("bone_marrow", "mouse_lemur", "Lemur, marrow"), ("vz_full", "pig", "Pig, eye")]
    fig, ax = plt.subplots(figsize=(2.2, 2.55))
    h = 0.34
    for i, (f, l, _) in enumerate(rows):
        r = z[(z.fit == f) & (z.species_label == l)].iloc[0]
        new = r.newasm_and_zero / max(r.newasm_total, 1)
        old = (r.exact_zero - r.newasm_and_zero) / (r.genes - r.newasm_total)
        ax.barh(i - h / 2 - 0.01, new, h, color=C["chimera"])
        ax.barh(i + h / 2 + 0.01, old, h, color=C["identity"])
        ax.text(new + 0.015, i - h / 2, f"{new:.0%}  ({int(r.newasm_and_zero):,} of {int(r.newasm_total):,})", va="center", fontsize=5.4)
        ax.text(old + 0.015, i + h / 2, f"{old:.0%}", va="center", fontsize=5.4)
    forest_axes(ax, [r[2] for r in rows])
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Genes with no counts in the species")
    h_ = [plt.Rectangle((0, 0), 1, 1, color=C["chimera"], label="Model absent from older assembly"), plt.Rectangle((0, 0), 1, 1, color=C["identity"], label="Model in older assembly")]
    ax.legend(handles=h_, loc="lower left", bbox_to_anchor=(-0.42, 1.0), ncol=1, frameon=False, borderaxespad=0.1, handlelength=0.9, handleheight=0.8, labelspacing=0.2)
    save(fig, "b")


def panel_c():
    d = pd.read_parquet(U1 / "posthoc_rows.parquet")
    x = d[(d.fit == "bone_marrow") & (d.species_label == "mouse_lemur") & (d.det_other >= RU.DET_EXPR)]
    fig, ax = plt.subplots(figsize=(2.45, 2.2))
    neg = x[(x.T_later != 1) & (x.T_lowconf != 1)]
    pos = x[(x.T_later == 1) | (x.T_lowconf == 1)]
    ax.scatter(neg.det_s, neg.S_gs, s=2, color="#c9c9c6", lw=0, rasterized=True, label="Clean labels")
    ax.scatter(pos.det_s, pos.S_gs, s=9, color=C["chimera"], lw=0.25, edgecolor="white", label="Low confidence or later change", zorder=3)
    lab = {"RPL28": (0.06, 0.0), "RPS15A": (0.06, -0.04), "RPL7": (0.04, 0.0), "HNRNPA2B1": (0.05, -0.01), "JUND": (0.04, 0.02), "GDI2": (0.04, 0.04)}
    for s, (dx, dy) in lab.items():
        r = x[x.symbol == s]
        if len(r):
            ax.annotate(s, (r.det_s.iloc[0], r.S_gs.iloc[0]), (r.det_s.iloc[0] + dx, r.S_gs.iloc[0] + dy), fontsize=5.2, va="center", arrowprops=dict(arrowstyle="-", lw=0.3, color="#555555", shrinkA=0, shrinkB=1))
    ax.axhline(0, color="#bbbbb8", lw=0.5, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Lemur cells with counts")
    ax.set_ylabel("Lemur margin $S_{gs}$")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, markerscale=1.0, handletextpad=0.0, columnspacing=0.6, borderaxespad=0.1)
    save(fig, "c")


def rho_rows(ax, items, keys):
    for i, (rows, _) in enumerate(items):
        off = np.linspace(-0.27, 0.27, len(keys))
        for o, (kk, col, name) in zip(off, keys):
            if kk not in rows:
                continue
            r = rows[kk]
            ax.plot([r.ci_lo, r.ci_hi], [i + o, i + o], color=C[col], lw=0.8, solid_capstyle="round")
            ax.plot(r.rho_p, i + o, M[col], color=C[col], ms=MS - 0.8, mec="white", mew=0.3, ls="none")
    ax.axvline(0, color="#bbbbb8", lw=0.6, zorder=0)
    forest_axes(ax, [t for _, t in items])


def panel_d():
    u = pd.read_csv(RU.OUT / "use2/rho.csv")
    u = u[u.split == "all"]

    def get(test, a, b, c):
        x = u[u.test == test].set_index("score")
        return {"chimera": x.loc[a], "b1": x.loc[b], "identity": x.loc[c]}, None

    items = [(get("testis_mac_vs_mouse", "C_chimera", "C_B1", "C_id")[0], "Testis: macaque or mouse\n(held-out primates vs outgroups)"),
             (get("liver_mac_vs_mouse", "C_chimera", "C_B1", "C_id")[0], "Liver: macaque or mouse\n(lemur vs TMS, against TS)"),
             (get("liver_mouse_row_H3", "S_mouse", "E_mouse", "id_mouse")[0], "Liver: mouse row\n(TS vs TMS)"),
             (get("marrow_mouse_row", "S_mouse", "E_mouse", "id_mouse")[0], "Marrow: mouse row\n(Shay 2013)"),
             (get("eye_mulatta_to_fas", "S_row", "E_own", "id_own")[0], "Eye: M. mulatta fit\n(held-out M. fascicularis)")]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    keys = [("chimera", "chimera", "Species margin"), ("b1", "b1", "B1"), ("identity", "identity", "Identity")]
    rho_rows(ax, items, keys)
    ax.set_xlabel(r"Partial rank correlation, $\rho_p$")
    ax.set_xlim(-0.12, 0.45)
    ax.legend(handles=[plt.Line2D([], [], marker=M[c], color=C[c], ls="none", ms=3.6, mec="white", mew=0.3, label=n) for _, c, n in keys], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, handletextpad=0.0, columnspacing=0.6, borderaxespad=0.1)
    save(fig, "d")


def panel_e():
    r = pd.read_csv(RU.OUT / "use3/ladder_same_genes.csv")
    lad = ["C0", "C1", "C2", "C3", "C4", "C5"]
    names = ["None", "+ expr.,\nlength", "+ τ", "+ identity", "+ dN/dS,\nLOEUF", "+ PubMed"]
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, y, t in zip(axs, ["y_appr", "y_gen"], ["Approved vs failed", "Genetic support"]):
        for s, col, nm in (("S", "chimera", "$S_g$"), ("D", "delta", "Δα"), ("B1", "b1", "B1")):
            x = r[(r.y == y) & (r.score == s)].set_index("ladder").loc[lad]
            ax.plot(range(6), x.rho_p, color=C[col], lw=1.1, marker=M[col], ms=MS - 1.2, mec="white", mew=0.3, label=nm)
        x = r[(r.y == y) & (r.score == "S")].set_index("ladder").loc[lad]
        ax.fill_between(range(6), x.ci_lo, x.ci_hi, color=C["chimera"], alpha=0.12, lw=0)
        ax.axhline(0, color="#bbbbb8", lw=0.6, zorder=0)
        ax.set_xticks(range(6))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=5.2, rotation_mode="anchor")
        ax.set_title(f"{t}, n = {int(x.n.iloc[0])}", fontsize=5.8)
    axs[0].set_ylabel(r"$\rho_p$, one gene set")
    axs[1].legend(loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.2)
    save(fig, "e")


def panel_f():
    a = pd.read_csv(RU.OUT / "use4/rho.csv")

    def get(fit, gold):
        x = a[(a.fit == fit) & (a.gold == gold)].set_index("score")
        return {"chimera": x.loc["S_ema"], "b1": x.loc["B1"], "gap": x.loc["gap_neg"]}

    items = [(get("liv_hmonkey", "L_HP"), "Hepatocyte progenitors,\nliver H+Monkey"), (get("vz_hmmul", "L_RPE"), "Pigment epithelium,\neye H+M. mulatta"), (get("islet", "L_PP"), "Pancreatic progenitors,\nislet H+Mouse"),
             (get("liv_hmonkey", "P_cis_depth"), "Across six types,\nliver H+Monkey"), (get("vz_hmmul", "P_cis_depth"), "Across six types,\neye H+M. mulatta"), (get("test_hmac", "P_cis_depth"), "Across six types,\ntestis H+Macaque")]
    fig, ax = plt.subplots(figsize=(2.75, 2.3))
    keys = [("chimera", "chimera", "$S_g$"), ("b1", "b1", "B1"), ("gap", "gap", "Level gap")]
    rho_rows(ax, items, keys)
    ax.set_xlabel(r"$\rho_p$ with cis agreement in human-chimp hybrids")
    ax.legend(handles=[plt.Line2D([], [], marker=M[c], color=C[c], ls="none", ms=3.6, mec="white", mew=0.3, label=n) for _, c, n in keys], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, handletextpad=0.0, columnspacing=0.6, borderaxespad=0.1)
    save(fig, "f")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    panel_e()
    panel_f()


if __name__ == "__main__":
    main()
