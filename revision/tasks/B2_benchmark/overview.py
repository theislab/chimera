"""Overview of the benchmark: methods by tasks, in the scIB funky-heatmap style, merging tracks A, B, C, D.

Rule, fixed and committed before results. For each task and metric, rank every applicable method (best 1, ties share the mean rank), scaled by (rank - 1) / (n - 1); n = applicable methods with a
value in the task. Aggregate of a track = mean of a method's within-task scaled ranks over the track's tasks. Overall = mean of the track aggregates, equal weight per track. Unscaled mean ranks are written
beside. A method applicable to a task but with no value (failed fit, tool that failed there) ranks last in it. Not applicable (static score on agreement; a tool not run on that setting) is left out and drawn
hatched. A row of the tidy tables enters if rank_direction is 'higher' or 'lower'. Every task appears: page 1 groups tasks by track and family, page 2 draws every task as its own column.

Track I (integration, added by the coordinator's instruction before it was scored) is a track of its own, on its own methods (the 10 integration models, prefixed I:). Integration is the substrate of Chimera's readouts, not its aim, so the overall of the gene-level tracks A, B, C, D
keeps its committed rule and does not include I; `overall_with_I` is written beside it for the methods that appear in both (SATURN, CAMEX, and Chimera's latent shown against B1z).

Inputs: results/revision/B1/*.csv|parquet (tracks A, D: columns track, task, setting, method, metric, value, ci_lo, ci_hi, n, rank_direction), results/revision/B2/track_B/*.parquet, track_C/units and lists.
Re-runnable as B1, C1 and new cohorts add results. Outputs: results/revision/B2/overview/{ranks_long.csv, aggregate.csv, overview.pdf, overview.png}
    python -m revision.tasks.B2_benchmark.overview
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from revision.tasks.B2_benchmark import rules as RU

OUT = RU.RES / "overview"
METHOD_LABEL = {"S_g": "S_g", "D": "Δα", "B1z": "B1z (Chimera latent)", "B1": "B1 profile corr.", "B1b": "B1b resid.", "B1c": "B1c rank level", "B2b": "B2b specificity", "B3a": "B3a features",
                "Sdir": "Sdir direct margin", "Lfc": "Lfc fold change", "pi": "π identity", "dNdS": "dN/dS", "LOEUF": "LOEUF", "SATURN": "SATURN", "CAMEX": "CAMEX", "GeneSpectra": "GeneSpectra",
                "CACIMAR": "CACIMAR", "SAMap": "SAMap"}
ORDER = RU.OVERVIEW_METHODS
GROUPS = {"chimera": RU.CHIMERA, "direct": RU.DIRECT, "static": RU.STATIC, "tools": RU.TOOLS}


def read_tidy() -> pd.DataFrame:
    fr = []
    for f in glob.glob(str(RU.REPO / "results/revision/B1/tidy*.csv")) + glob.glob(str(RU.REPO / "results/revision/B1/*tidy*.parquet")) + glob.glob(str(RU.REPO / "results/revision/B1/track_*/*.csv")):
        fr.append(pd.read_parquet(f) if f.endswith("parquet") else pd.read_csv(f))
    for pat in ("track_B/*.parquet", "track_C/units/*.parquet", "track_C/lists/*.parquet", "integration/tidy.parquet"):
        for f in glob.glob(str(RU.RES / pat)):
            if not f.endswith("__h2h.parquet"):
                fr.append(pd.read_parquet(f))
    if not fr:
        raise SystemExit("no tidy tables yet")
    t = pd.concat(fr, ignore_index=True)
    return t[(t["method"].isin(ORDER) | t["method"].astype(str).str.startswith("I:")) & t["rank_direction"].isin(["higher", "lower"]) & t["value"].notna()]


def applicable(track: str, task: str, method: str) -> bool:
    if method in RU.STATIC and task.endswith("agreement"):
        return False
    return True


def rank_tasks(t: pd.DataFrame, tools_expected: dict[str, list[str]] | None = None) -> pd.DataFrame:
    rows = []
    for (track, task, metric), g in t.groupby(["track", "task", "metric"], sort=False):
        d = g.drop_duplicates("method").set_index("method")
        v = d["value"].astype(float) * (1 if (d["rank_direction"] == "higher").all() else -1)
        r = (-v).rank(method="average")
        n = len(v)
        # methods in the panel of this task with no value and not excused rank last
        ran_any_tool = any(m in v.index for m in RU.TOOLS)
        missing = [m for m in RU.CHIMERA + RU.DIRECT if m not in v.index and applicable(track, task, m)]
        if missing and track in ("B",):
            n = n + len(missing)
        for m in v.index:
            rows.append(dict(track=track, task=task, metric=metric, method=m, value=float(d.loc[m, "value"]), rank=float(r[m]), n_methods=n, scaled=(float(r[m]) - 1) / max(n - 1, 1), filled=False))
        if track == "B":
            for m in missing:
                rows.append(dict(track=track, task=task, metric=metric, method=m, value=np.nan, rank=float(n), n_methods=n, scaled=1.0, filled=True))
    return pd.DataFrame(rows)


def aggregate(R: pd.DataFrame) -> pd.DataFrame:
    a = R.groupby(["track", "method"]).agg(mean_scaled=("scaled", "mean"), mean_rank=("rank", "mean"), n_tasks=("task", "nunique")).reset_index()
    wide = a.pivot(index="method", columns="track", values="mean_scaled")
    gene = [c for c in wide.columns if c != "I"]
    wide["overall"] = wide[gene].mean(1, skipna=True)
    wide["n_tracks"] = wide[gene].notna().sum(1)
    if "I" in wide:
        i = wide["I"].copy()
        for tool in ("SATURN", "CAMEX"):
            if f"I:{tool}" in i.index and tool in wide.index:
                wide.loc[tool, "overall_with_I"] = float(np.nanmean(list(wide.loc[tool, gene].astype(float)) + [i[f"I:{tool}"]]))
    un = a.pivot(index="method", columns="track", values="mean_rank").add_prefix("mean_rank_")
    nt = a.pivot(index="method", columns="track", values="n_tasks").add_prefix("n_tasks_")
    return wide.join(un).join(nt).sort_values("overall")


def family_of(task: str) -> str:
    p = task.split("|")
    return p[1] if len(p) > 1 else "task"


def draw(R: pd.DataFrame, A: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 6, "font.family": "DejaVu Sans"})
    methods = [m for m in A.index if m in ORDER]
    R_i = R[R.track == "I"]
    R = R[R.track != "I"]
    tracks = sorted(R.track.unique())
    # page 1: tracks x families as circles, overall bars
    R = R.copy()
    R["fam"] = R["task"].map(family_of)
    R["col"] = R["track"] + "|" + R["fam"]
    if "metric" in R:
        ag = R["metric"].eq("agreement_partial")
        R.loc[ag, "col"] = R.loc[ag, "track"] + "|" + R.loc[ag, "fam"] + " agree"
    cols = [c for tr in tracks for c in sorted(R.loc[R.track == tr, "col"].unique())]
    M = R.groupby(["method", "col"]).scaled.mean().unstack().reindex(index=methods, columns=cols)
    N = R.groupby("col").task.nunique().reindex(cols)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rank", ["#1b4f8f", "#7fa8cf", "#e9e9e9"])
    with PdfPages(OUT / "overview.pdf") as pdf:
        w = max(7.0, 0.32 * len(cols) + 3.6)
        fig, ax = plt.subplots(figsize=(w, 0.27 * len(methods) + 1.6))
        x_over = len(cols) + 0.6
        for i, m in enumerate(methods):
            for j, c in enumerate(cols):
                v = M.loc[m, c]
                if np.isfinite(v):
                    s = 22 + 80 * (1 - v)
                    ax.scatter(j, i, s=s, c=[cmap(v)], edgecolors="none")
                else:
                    ax.scatter(j, i, s=30, facecolors="none", edgecolors="#bbbbbb", hatch="////", linewidths=0.3)
            ax.barh(i, (1 - A.loc[m, "overall"]) * 2.4, left=x_over, height=0.62, color="#1b4f8f")
            ax.text(x_over + 2.5, i, f"{A.loc[m, 'overall']:.2f}", va="center", ha="left", fontsize=5.5, color="#333333")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
        ax.set_ylim(len(methods) - 0.5, -2.2)
        ax.set_xlim(-0.8, x_over + 3.4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{c.split('|', 1)[1]} ({int(N[c])})" for c in cols], rotation=60, ha="left")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cuts = np.cumsum([sum(c.startswith(tr + "|") for c in cols) for tr in tracks])
        prev = 0
        for tr, cu in zip(tracks, cuts):
            ax.plot([prev - 0.4, cu - 0.6], [-1.0, -1.0], color="#333333", lw=1)
            ax.text((prev + cu - 1) / 2, -1.25, f"Track {tr}", ha="center", va="bottom", fontsize=7)
            prev = cu
        ax.text(x_over, -1.25, "overall (lower is better)", ha="left", va="bottom", fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(OUT / "overview.png", dpi=220)
        plt.close(fig)
        # page 2: every task
        tt = R.drop_duplicates(["track", "task", "metric"]).sort_values(["track", "fam", "task"])
        keys = list(zip(tt.track, tt.task, tt.metric))
        F = R.set_index(["track", "task", "metric", "method"]).scaled
        P = np.full((len(methods), len(keys)), np.nan)
        for i, m in enumerate(methods):
            for j, k in enumerate(keys):
                P[i, j] = F.get((*k, m), np.nan)
        fig, ax = plt.subplots(figsize=(min(17, max(8, 0.018 * len(keys) + 3)), 0.2 * len(methods) + 1.2))
        cm = cmap.copy()
        cm.set_bad("#ffffff")
        ax.imshow(np.ma.masked_invalid(P), aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
        ax.set_xticks([])
        prev = 0
        for tr in tracks:
            nn = sum(k[0] == tr for k in keys)
            ax.axvline(prev + nn - 0.5, color="#333333", lw=0.6)
            ax.text(prev + nn / 2, -0.8, f"Track {tr} ({nn} tasks)", ha="center", fontsize=7)
            prev += nn
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        if len(R_i):
            # page 3: Track I, integration (the substrate, not Chimera's aim), methods by cohort on composite, mixing, conservation
            met = ["composite", "mixing_score", "conservation_score"]
            imeth = R_i[R_i.metric == "composite"].groupby("method").scaled.mean().sort_values().index.tolist()
            coh = list(dict.fromkeys(R_i.task.str.split("|").str[2]))
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.24 * len(imeth) + 1.4), sharey=True)
            for ax, me in zip(axes, met):
                V = R_i[R_i.metric == me].assign(c=lambda d: d.task.str.split("|").str[2]).pivot(index="method", columns="c", values="value").reindex(index=imeth, columns=coh)
                ax.imshow(np.ma.masked_invalid(V.to_numpy(dtype=float)), aspect="auto", cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(coh)))
                ax.set_xticklabels(coh, rotation=90, fontsize=5)
                ax.set_title(me.replace("_", " "), fontsize=6)
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            axes[0].set_yticks(range(len(imeth)))
            axes[0].set_yticklabels([m[2:] for m in imeth])
            fig.suptitle("Track I: integration is the substrate of Chimera's readouts, not its aim", fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = read_tidy()
    R = rank_tasks(t)
    R.to_csv(OUT / "ranks_long.csv", index=False)
    A = aggregate(R)
    A.to_csv(OUT / "aggregate.csv")
    print(A.round(3).to_string())
    draw(R, A)


if __name__ == "__main__":
    main()
