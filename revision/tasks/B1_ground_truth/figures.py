"""B1 figures, one per task, from the tidy tables. Written to results/revision/B1/figures (PDF, PNG); never to the manuscript.

Heatmaps use one hue from 0.5 (no detection) to 1. Readouts keep the T8.4 colours and markers.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from revision.tasks.B1_ground_truth import rules as RU

sys.path.insert(0, str(RU.REPO / "notebooks"))
import _figure_design as fd  # noqa: E402

OUT = RU.RES / "figures"
COH = {"liv_hmouse": "liver", "test_hmouse": "testis", "kidney": "kidney"}
ORDER = ["S_g", "Delta_alpha", "Delta_alpha_free", "Delta_alpha_hybrid", "B1z", "S_fwd", "Sdir", "B1", "B1b", "B1c", "B2b", "Lfc", "B3a", "S_deploy", "S_cos", "C_cos", "GeneSpectra", "CACIMAR", "SAMap", "pi", "dNdS", "LOEUF", "expr", "random"]
NAME = {"S_g": "$S_g$ (Chimera)", "Delta_alpha": "Δα (Chimera)", "Delta_alpha_free": "Δα, free gate (Chimera)", "Delta_alpha_hybrid": "Δα, gate + per-gene bias", "c_g_free": "c_g, free gate", "B1z": "B1z (Chimera latent)", "S_fwd": "$S_{fwd}$, no mask (exploratory)", "Sdir": "Sdir (labels, no model)",
        "B1": "B1 profile correlation", "B1b": "B1b residualised", "B1c": "B1c rank level", "B2b": "B2b specificity", "Lfc": "Lfc fold change", "B3a": "B3a features only",
        "S_deploy": "SATURN S_deploy", "S_cos": "SATURN S_cos", "C_cos": "CAMEX C_cos", "GeneSpectra": "GeneSpectra", "CACIMAR": "CACIMAR", "SAMap": "SAMap", "pi": "identity π", "dNdS": "dN/dS", "LOEUF": "LOEUF", "expr": "expression", "random": "random"}
STY = {"S_g": ("#0072B2", "s"), "Delta_alpha": ("#D55E00", "o"), "Delta_alpha_free": ("#882255", "h"), "SAMap": ("#44AA99", "d"), "B1z": ("#CC79A7", "P"), "Sdir": ("#009E73", "D"), "B1": ("#222222", "^"), "B1c": ("#666666", "v"), "Lfc": ("#E69F00", "X"),
        "S_fwd": ("#56B4E9", "*"), "expr": ("#999999", "<")}
EFFECT_LABEL = {"pattern": "share of swap q", "level": "fold f", "loss": "share removed e"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#a6c8e4", "#2a78d6", "#08306b"])


def setup():
    fd.apply_design()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlesize": 7, "legend.fontsize": 6, "font.size": 6, "axes.labelsize": 6,
                         "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def parse(t: pd.DataFrame) -> pd.DataFrame:
    kv = t["setting"].str.split(";", expand=True)
    for i, k in enumerate(["cohort", "effect", "band", "share", "identity"]):
        t[k] = kv[i].str.split("=").str[1]
    return t


def pick(t, **k):
    d = t
    for a, b in k.items():
        d = d[d[a] == str(b)] if a in ("cohort", "effect", "band", "share", "identity") else d[d[a] == b]
    return d


def heat_panel(ax, M: pd.DataFrame, vmin=0.5, vmax=1.0, fmt="{:.2f}", cmap=SEQ):
    im = ax.imshow(M.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=0)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=4.8, color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "#222222")
    return im


def fig_task(t: pd.DataFrame, task: str, effects: list, top_effect):
    """A: AUROC by method x cohort x band at the strongest effect, share marginal. B: dose response per cohort. C: power."""
    tt = t[t.task == task]
    methods = [m for m in ORDER if m in set(tt.method)]
    bands = RU.BANDS
    cols = [(c, b) for c in COH for b in bands]
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    axh = fig.add_subplot(gs[0, :2])
    d = tt[(tt.metric == "AUROC") & (tt.effect == str(top_effect)) & (tt.share == "all") & (tt.identity == "all")]
    M = pd.DataFrame({f"{COH[c][:3]}\n{b}": d[(d.cohort == c) & (d.band == b)].set_index("method")["value"].reindex(methods) for c, b in cols})
    M.index = [NAME[m] for m in methods]
    heat_panel(axh, M)
    axh.set_title(f"{task} {top_effect}: AUROC vs matched controls", loc="left")
    axp = fig.add_subplot(gs[0, 2])
    d = tt[(tt.metric == "power_05") & (tt.effect == str(top_effect)) & (tt.share == "all") & (tt.identity == "all") & (tt.cohort == "all")]
    P = pd.DataFrame({b: d[d.band == b].set_index("method")["value"].reindex(methods) for b in bands})
    P.index = [NAME[m] for m in methods]
    heat_panel(axp, P, vmin=0.05, vmax=1.0)
    axp.set_yticklabels([])
    axp.set_title("power at 5% FPR, 3 cohorts", loc="left")
    show = [m for m in ("S_g", "Delta_alpha", "Delta_alpha_free", "B1z", "Sdir", "B1", "B1c", "Lfc") if m in methods]
    for i, c in enumerate(COH):
        ax = fig.add_subplot(gs[1, i])
        d = tt[(tt.metric == "AUROC") & (tt.cohort == c) & (tt.band == "midhigh") & (tt.share == "all") & (tt.identity == "all") & (tt.effect != "all")]
        for m in show:
            s = d[d.method == m].assign(e=lambda x: x.effect.astype(float)).sort_values("e")
            ax.errorbar(range(len(s)), s["value"], yerr=[s["value"] - s["ci_lo"], s["ci_hi"] - s["value"]], color=STY[m][0], marker=STY[m][1], ms=3.5, lw=1, capsize=0, label=NAME[m])
        ax.set_xticks(range(len(effects)), effects)
        ax.axhline(0.5, color="#BBBBBB", lw=0.6, ls="--")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(COH[c], loc="left")
        ax.set_xlabel(EFFECT_LABEL[task])
        if i == 0:
            ax.set_ylabel("AUROC, band midhigh")
        if i == 2:
            ax.legend(frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, f"b1_task_{task}")


def fig_low(t):
    """Low-expression power: power and mask by band, for S_g, S_fwd, Δα, B1, Sdir, per alteration."""
    tasks = ["level", "pattern", "loss"]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True, constrained_layout=True)
    for j, task in enumerate(tasks):
        top = {"level": 8, "pattern": 1, "loss": 1}[task]
        d = t[(t.task == task) & (t.effect == str(top)) & (t.share == "all") & (t.identity == "all") & (t.cohort == "all")]
        for m in ("S_g", "S_fwd", "Delta_alpha", "Delta_alpha_free", "Sdir", "B1", "Lfc"):
            s = d[(d.metric == "power_05") & (d.method == m)].set_index("band").reindex(RU.BANDS)
            ax[0, j].plot(range(4), s["value"], color=STY[m][0], marker=STY[m][1], ms=3.5, lw=1, label=NAME[m])
        ax[0, j].axhline(0.05, color="#BBBBBB", lw=0.6, ls="--")
        ax[0, j].set_ylim(0, 1)
        ax[0, j].set_title(f"{task} {top}", loc="left")
        s = d[(d.metric == "frac_zero") & (d.method == "S_g")].set_index("band").reindex(RU.BANDS)
        ax[1, j].bar(range(4), s["value"], color="#0072B2", width=0.6)
        ax[1, j].set_xticks(range(4), RU.BANDS)
        ax[1, j].set_ylim(0, 1)
    ax[0, 0].set_ylabel("power at 5% FPR")
    ax[1, 0].set_ylabel("share of altered genes\nwith $S_g$ exactly 0")
    ax[0, 2].legend(frameon=False, fontsize=5)
    save(fig, "b1_low_expression")


def fig_nulls(t):
    tasks = [("placebo", "placebo sets, unaltered fits"), ("sham_swap", "sham move"), ("thin_both", "both species thinned 4-fold")]
    methods = [m for m in ORDER if m in set(t.method) and m not in ("random", )]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True, constrained_layout=True)
    for j, (task, lab) in enumerate(tasks):
        d = t[(t.task == task) & (t.metric == "reject05") & (t.band == "all") & (t.effect == "all") & (t.share == "all") & (t.identity == "all") & (t.cohort.isin(COH))]
        for i, c in enumerate(COH):
            s = d[d.cohort == c].set_index("method")["value"].reindex(methods)
            ax[j].plot(s.values, range(len(methods)), ls="", marker="soD"[i], ms=3.5, color=["#0072B2", "#D55E00", "#009E73"][i], label=COH[c])
        ax[j].axvline(0.05, color="#222222", lw=0.6, ls="--")
        ax[j].set_title(lab, loc="left")
        ax[j].set_xlim(-0.02, 1.02)
        ax[j].set_xlabel("rejection rate at α = 0.05")
        ax[j].invert_yaxis() if j == 0 else None
    ax[0].set_yticks(range(len(methods)), [NAME[m] for m in methods])
    ax[2].legend(frameon=False)
    save(fig, "b1_type1")


def fig_share(t):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, constrained_layout=True)
    for j, c in enumerate(COH):
        d = t[(t.task == "level") & (t.cohort == c) & (t.metric == "AUROC") & (t.effect == "8") & (t.band == "all") & (t.identity == "all") & (t.share != "all")]
        d2 = t[(t.task == "pattern") & (t.cohort == c) & (t.metric == "AUROC") & (t.effect == "1") & (t.band == "all") & (t.identity == "all") & (t.share != "all")]
        for m in ("S_g", "Delta_alpha", "Delta_alpha_free", "B1"):
            for dd, ls, l in ((d, "-", "level 8"), (d2, ":", "pattern 1")):
                s = dd[dd.method == m].assign(x=lambda x: x.share.astype(float)).sort_values("x")
                ax[j].plot(s["x"] * 100, s["value"], color=STY[m][0], marker=STY[m][1], ms=3, lw=1, ls=ls, label=f"{NAME[m]}, {l}")
        ax[j].set_title(COH[c], loc="left")
        ax[j].set_xlabel("share of altered genes (%)")
    ax[0].set_ylabel("AUROC")
    ax[2].legend(frameon=False, fontsize=4.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "b1_share")


def fig_identity_cells():
    """Identity tertiles from the per-fit table (subsets are scored within each fit)."""
    c = pd.read_parquet(RU.RES / "per_cell.parquet")
    c = c[(c.metric == "AUROC") & c.subset.isin(RU.IDENTITY_BANDS) & c.alt.isin(["level", "pattern", "loss"])]
    methods = ["S_g", "Delta_alpha", "Delta_alpha_free", "B1", "Sdir", "B1c"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), constrained_layout=True)
    for j, task in enumerate(("level", "pattern", "loss")):
        M = c[c.alt == task].groupby(["method", "subset"])["value"].mean().unstack().reindex(methods)[RU.IDENTITY_BANDS]
        M.index = [NAME[m] for m in methods]
        heat_panel(ax[j], M)
        ax[j].set_title(f"{task}, all effects, bands, cohorts", loc="left")
        if j:
            ax[j].set_yticklabels([])
    save(fig, "b1_identity")
    return c


def fig_identity(t):
    methods = ["S_g", "Delta_alpha", "Delta_alpha_free", "B1", "Sdir", "B1c"]
    rows = []
    for task in ("level", "pattern", "loss"):
        for m in methods:
            for ib in RU.IDENTITY_BANDS:
                d = t[(t.task == task) & (t.method == m) & (t.metric == "AUROC") & (t.identity == ib)]
                rows.append(dict(task=task, method=NAME[m], identity=ib, value=d["value"].mean()))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), constrained_layout=True)
    for j, task in enumerate(("level", "pattern", "loss")):
        M = r[r.task == task].pivot(index="method", columns="identity", values="value").loc[[NAME[m] for m in methods], RU.IDENTITY_BANDS]
        heat_panel(ax[j], M)
        ax[j].set_title(task, loc="left")
        if j:
            ax[j].set_yticklabels([])
    save(fig, "b1_identity")


def fig_examples():
    e = pd.read_parquet(RU.RES / "qc" / "example_counts.parquet")
    e = e[(e.cohort == "kidney") & (e.band == "high") & e.alt.isin(["pattern", "level", "loss"])]
    picks = e.groupby("alt")["gene"].first()
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 5.0), constrained_layout=True, sharex="col")
    for i, (alt, gene) in enumerate(picks.items()):
        d = e[(e.alt == alt) & (e.gene == gene)]
        sp = d["species_altered"].iloc[0]
        types = sorted(d.cell_type.unique())
        for j, s in enumerate((sp, "Human" if sp == "Mouse" else "Mouse")):
            bb = d[(d.state == "before") & (d.species == s)].groupby("cell_type")["count"].mean().reindex(types)
            aa = d[(d.state == "after") & (d.species == s)].groupby("cell_type")["count"].mean().reindex(types)
            x = np.arange(len(types))
            ax[i, j].bar(x - 0.2, bb, width=0.38, color="#BBBBBB", label="before")
            ax[i, j].bar(x + 0.2, aa, width=0.38, color="#0072B2", label="after")
            ax[i, j].set_title(f"{alt}, {gene}, {s} ({'altered' if j == 0 else 'untouched'})", loc="left")
            ax[i, j].set_xticks(x, types, rotation=60, ha="right")
            if j == 0:
                ax[i, j].set_ylabel("mean raw count per cell")
    ax[0, 1].legend(frameon=False)
    save(fig, "b1_spikein_raw_counts")


def fig_trackd():
    t = pd.read_csv(RU.RES / "track_d" / "runtime.csv")
    t = t[t.finished]
    meth = ["chimera", "direct", "saturn", "camex"]
    col = {"chimera": "#0072B2", "direct": "#009E73", "saturn": "#CC79A7", "camex": "#E69F00"}
    nm = {"chimera": "Chimera", "direct": "direct baselines", "saturn": "SATURN", "camex": "CAMEX"}
    full = t[t.object == "full"].groupby(["cohort", "method"]).agg(w=("wall_s", "mean"), r=("peak_rss_gb", "max")).reset_index()
    full = full[full.cohort.isin(COH)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    for i, m in enumerate(meth):
        s = full[full.method == m].set_index("cohort").reindex(list(COH))
        ax[0].bar(np.arange(3) + (i - 1.5) * 0.2, s["w"] / 3600, width=0.18, color=col[m], label=nm[m])
        ax[1].bar(np.arange(3) + (i - 1.5) * 0.2, s["r"], width=0.18, color=col[m])
    for k, l in ((0, "wall time (h), 16 CPUs"), (1, "peak memory (GB)")):
        ax[k].set_xticks(range(3), list(COH.values()))
        ax[k].set_ylabel(l)
    ax[0].legend(frameon=False, fontsize=5)
    frac = {"f25": 0.25, "f50": 0.5, "full": 1.0}
    s = t[t.cohort == "test_hmouse"].copy()
    s["f"] = s["object"].map(frac)
    for m in meth:
        q = s[s.method == m].groupby("f")["wall_s"].mean()
        ax[2].plot(q.index * 21241 / 1000, q.values / 3600, color=col[m], marker="o", ms=3, lw=1, label=nm[m])
    ax[2].set_xlabel("cells, testis (thousands)")
    ax[2].set_ylabel("wall time (h)")
    save(fig, "b1_trackD")


def main():
    setup()
    t = parse(pd.read_csv(RU.RES / "B1_tidy.csv", dtype={"setting": str}))
    fig_task(t, "pattern", [0.25, 0.5, 1.0], "1")
    fig_task(t, "level", [2, 4, 8], "8")
    fig_task(t, "loss", [0.5, 0.9, 1.0], "1")
    fig_low(t)
    fig_nulls(t)
    fig_share(t)
    fig_identity_cells()
    fig_examples()
    try:
        fig_trackd()
    except FileNotFoundError:
        print("no track D yet")


if __name__ == "__main__":
    main()
