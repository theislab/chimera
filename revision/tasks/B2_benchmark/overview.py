"""Overview of the benchmark: methods by tasks, in the scIB funky-heatmap style, merging tracks A, B, C, D.

Rule, fixed and committed before results. For each task and metric, rank every applicable method (best 1, ties share the mean rank), scaled by (rank - 1) / (n - 1); n = applicable methods with a
value in the task. Aggregate of a track = mean of a method's within-task scaled ranks over the track's tasks. Overall = mean of the track aggregates, equal weight per track. Unscaled mean ranks are written
beside. A method applicable to a task but with no value (failed fit, tool that failed there) ranks last in it. Not applicable (static score on agreement; a tool not run on that setting) is left out and drawn
hatched. A row of the tidy tables enters if rank_direction is 'higher' or 'lower'. Every task appears: page 1 groups tasks by track and family, page 2 draws every task as its own column.

Undefined cells (amendment by the coordinator after the liver 1:20 reading, before the overview was first computed). A method with no value in a task that is applicable to it (for instance a profile score when one species keeps fewer than two shared types) is not a measured
loss. The committed rule stays and is reported as `overall`: it ranks last. Beside it `overall_excl_undefined`: such cells are left out and each task ranked among the methods with a value. Both are written for every method, Chimera included; in the figures undefined cells are drawn NA,
never as a low score. `undefined_cells.csv` lists every such cell, with the groups left in the setting where known; the text states their number.

Runs tried and finished (coordinator, same reason). A failed fit or a tool that crashes or returns nothing stays out of the scores; `completion.py` counts them per method and setting, joined here as tried and finished. A cell with fewer finished than tried carries a small
"f/t"; a cell where nothing finished is a failure, drawn with a cross, not a blank, and it ranks last under the committed rule.

Identity cells (found when the overview was first read; applies to every method alike). An agreement value that compares a reading with itself (a deterministic method on
`full`, a label-blind deterministic tool on a label-only setting) is 1 by identity, not a measurement. score_b flags these rows and they are tabled, not ranked. The aggregate with
them ranked, as the committed rule first read, is written to aggregate_with_identity.csv.

Track I (integration, added by the coordinator's instruction before it was scored) is a track of its own, on its own methods (the 10 integration models, prefixed I:). Integration is the substrate of Chimera's readouts, not its aim, so the overall of the gene-level tracks A, B, C, D
keeps its committed rule and does not include I; `overall_with_I` is written beside it for the methods that appear in both (SATURN, CAMEX, and Chimera's latent shown against B1z).

Inputs: results/revision/B1/*.csv|parquet (tracks A, D: columns track, task, setting, method, metric, value, ci_lo, ci_hi, n, rank_direction), results/revision/B2/track_B/*.parquet, track_C/units and lists.
Re-runnable as B1, C1 and new cohorts add results. Outputs: results/revision/B2/overview/{ranks_long.csv, aggregate.csv, overview.pdf, overview.png}
    python -m revision.tasks.B2_benchmark.overview
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision.tasks.B2_benchmark import rules as RU

OUT = RU.RES / "overview"
METHOD_LABEL = {"S_g": "S_g", "D": "Δα", "D_free": "Δα, free gate", "B1z": "B1z (Chimera latent)", "B1": "B1 profile corr.", "B1b": "B1b resid.", "B1c": "B1c rank level", "B2b": "B2b specificity", "B3a": "B3a features",
                "Sdir": "Sdir direct margin", "Lfc": "Lfc fold change", "pi": "π identity", "dNdS": "dN/dS", "LOEUF": "LOEUF", "SATURN": "SATURN", "CAMEX": "CAMEX", "GeneSpectra": "GeneSpectra",
                "CACIMAR": "CACIMAR", "SAMap": "SAMap"}
ORDER = RU.OVERVIEW_METHODS
GROUPS = {"chimera": RU.CHIMERA, "direct": RU.DIRECT, "static": RU.STATIC, "tools": RU.TOOLS}


def read_tidy() -> pd.DataFrame:
    fr = []
    b1 = RU.REPO / "results/revision/B1/B1_overview.csv"  # B1's rows for the overview: tracks A and D
    if b1.exists():
        d = pd.read_csv(b1)
        d["method"] = d["method"].replace({"Delta_alpha": "D", "Delta_alpha_free": "D_free", "S_deploy": "SATURN", "C_cos": "CAMEX"})  # B1's names to the overview's
        a = d[d["track"] == "A"].copy()
        # one column per task x cohort x metric; the all-cohort rows repeat them and stay out, so that nothing counts twice
        coh = a["setting"].astype(str).str.extract(r"cohort=([^;]+)")[0]
        a = a[coh.notna() & (coh != "all")].copy()
        a["task"] = "A|" + a["task"].astype(str) + "|" + coh[coh.notna() & (coh != "all")] + "|" + a["metric"].astype(str)
        for tag, x in (("closer_to_0.5", 0.5), ("closer_to_0.05", 0.05)):
            m = a["rank_direction"] == tag
            a.loc[m, "value"] = -(a.loc[m, "value"] - x).abs()  # distance from the nominal value, ranked higher = closer
            a.loc[m, "rank_direction"] = "higher"
        fr.append(a)
        # Track D: B1's rows are per tool or per family of readouts. One fit gives every Chimera readout; the direct family gives every direct score.
        dd = d[d["track"] == "D"].copy()
        expand = {"Chimera": RU.CHIMERA, "Chimera (one fit)": RU.CHIMERA, "Direct (B1, B1b, B1c, B2b, Sdir, B3a)": ["B1", "B1b", "B1c", "B2b", "Sdir", "B3a"],
                  "SATURN + direct": ["SATURN"], "CAMEX + direct": ["CAMEX"], "SATURN": ["SATURN"], "CAMEX": ["CAMEX"]}
        rows = []
        for r in dd.itertuples(index=False):
            for m in expand.get(r.method, []):
                rows.append({**r._asdict(), "method": m, "task": f"D|{r.task}|{r.setting}|{r.metric}"})
        dd = pd.DataFrame(rows)
        if len(dd):
            k = dd.groupby("task")["method"].transform(lambda x: x.map(lambda m: "chimera" if m in RU.CHIMERA else m).nunique())
            fr.append(dd[k >= 2])  # a task with one competitor (scaling with species, Chimera alone) ranks nothing; it is in the table
    for pat in ("track_B/*.parquet", "track_C/units/*.parquet", "track_C/lists/*.parquet", "integration/tidy.parquet"):
        for f in glob.glob(str(RU.RES / pat)):
            if not f.endswith("__h2h.parquet"):
                fr.append(pd.read_parquet(f))
    if not fr:
        raise SystemExit("no tidy tables yet")
    t = pd.concat(fr, ignore_index=True)
    # final scope (owner): B2 tiers 0-2, so Track B main settings only, every cohort; the other liver settings ran before the tiers and stay in the tables; the depth
    # law arm is a second Chimera reading beside depth25 and is tabled, not ranked
    from revision.tasks.B2_benchmark.tiers import MAIN

    isB = t["track"] == "B"
    t = t[~isB | t["setting"].isin(MAIN)]
    return t[(t["method"].isin(ORDER) | t["method"].astype(str).str.startswith("I:")) & t["rank_direction"].isin(["higher", "lower"]) & t["value"].notna()]


def applicable(track: str, task: str, method: str) -> bool:
    if method in RU.STATIC and task.endswith("agreement"):
        return False
    return True


def tool_ran(method: str, task: str) -> bool:
    """A tool has output for this Track B setting, so a task with no value from it is undefined, not not-applicable."""
    p = task.split("|")
    return (RU.RES / "tool_scores" / method / f"{RU.run_name(p[3], p[2], 0)}.csv").exists()


def tabled_not_ranked() -> set:
    """(task, method) of Track B rows that exist but are not ranked (identity, exploratory): neither ranked nor counted as undefined."""
    fr = [pd.read_parquet(f) for f in glob.glob(str(RU.RES / "track_B/*.parquet")) if not f.endswith("__h2h.parquet")]
    if not fr:
        return set()
    b = pd.concat(fr)
    b = b[b["rank_direction"].fillna("") == ""]
    return set(zip(b["task"], b["method"]))


def rank_tasks(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skip = tabled_not_ranked()
    for (track, task, metric), g in t.groupby(["track", "task", "metric"], sort=False):
        d = g.drop_duplicates("method").set_index("method")
        v = d["value"].astype(float) * (1 if (d["rank_direction"] == "higher").all() else -1)
        r = (-v).rank(method="average")
        n_def = len(v)
        missing = []
        if track == "B":  # the committed rule acts where every method has a reading to give
            missing = [m for m in RU.CHIMERA + RU.DIRECT if m not in v.index and applicable(track, task, m) and (task, m) not in skip]
            missing += [m for m in RU.TOOLS if m not in v.index and tool_ran(m, task) and (task, m) not in skip]
        n = n_def + len(missing)
        for m in v.index:
            rows.append(dict(track=track, task=task, metric=metric, method=m, value=float(d.loc[m, "value"]), rank=float(r[m]), n_methods=n, scaled=(float(r[m]) - 1) / max(n - 1, 1),
                             scaled_excl=(float(r[m]) - 1) / (n_def - 1) if n_def > 1 else np.nan, undefined=False))
        for m in missing:
            rows.append(dict(track=track, task=task, metric=metric, method=m, value=np.nan, rank=float(n), n_methods=n, scaled=1.0, scaled_excl=np.nan, undefined=True))
    R = pd.DataFrame(rows)
    f = RU.RES / "completion.csv"
    R["tried"], R["finished"] = np.nan, np.nan
    if f.exists():
        C = pd.read_csv(f).set_index(["cohort", "setting", "method"])
        for i, r in R[R.track == "B"].iterrows():
            p = r["task"].split("|")
            m = "Chimera" if r["method"] in RU.CHIMERA else r["method"]
            if (p[3], p[2], m) in C.index:
                R.loc[i, ["tried", "finished"]] = C.loc[(p[3], p[2], m), ["tried", "finished"]].to_numpy(dtype=float)
    return R


def aggregate(R: pd.DataFrame) -> pd.DataFrame:
    a = R.groupby(["track", "method"]).agg(mean_scaled=("scaled", "mean"), mean_scaled_excl=("scaled_excl", "mean"), mean_rank=("rank", "mean"), n_tasks=("task", "nunique"), n_undefined=("undefined", "sum")).reset_index()
    wide = a.pivot(index="method", columns="track", values="mean_scaled")
    gene = [c for c in wide.columns if c != "I"]
    wide["overall"] = wide[gene].mean(1, skipna=True)
    ex = a.pivot(index="method", columns="track", values="mean_scaled_excl")
    wide["overall_excl_undefined"] = ex[[c for c in gene if c in ex]].mean(1, skipna=True)
    wide["n_undefined_cells"] = a[a.track != "I"].groupby("method").n_undefined.sum()
    wide["n_tracks"] = wide[gene].notna().sum(1)
    if "I" in wide:
        i = wide["I"].copy()
        for tool in ("SATURN", "CAMEX"):
            if f"I:{tool}" in i.index and tool in wide.index:
                wide.loc[tool, "overall_with_I"] = float(np.nanmean(list(wide.loc[tool, gene].astype(float)) + [i[f"I:{tool}"]]))
    un = a.pivot(index="method", columns="track", values="mean_rank").add_prefix("mean_rank_")
    nt = a.pivot(index="method", columns="track", values="n_tasks").add_prefix("n_tasks_")
    return wide.join(un).join(nt).join(ex.add_prefix("excl_")).sort_values("overall")


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
    M = R[~R.undefined].groupby(["method", "col"]).scaled.mean().unstack().reindex(index=methods, columns=cols)
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
                sub = R[(R.method == m) & (R.col == c) & (R.tried > R.finished)]
                if len(sub):
                    ax.text(j + 0.32, i - 0.3, f"{int(sub.finished.sum())}/{int(sub.tried.sum())}", fontsize=3.8, color="#a33", ha="left", va="center")
                    if (sub.finished == 0).any() and not np.isfinite(v):
                        ax.scatter(j, i, marker="x", s=22, c="#a33", linewidths=0.8)
            ax.barh(i, (1 - A.loc[m, "overall"]) * 2.4, left=x_over, height=0.62, color="#1b4f8f")
            ax.text(x_over + 2.5, i, f"{A.loc[m, 'overall']:.2f} / {A.loc[m, 'overall_excl_undefined']:.2f}", va="center", ha="left", fontsize=5.5, color="#333333")
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
        ax.text(x_over, -1.25, "overall, committed / undefined cells excluded (lower is better)", ha="left", va="bottom", fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(OUT / "overview.png", dpi=220)
        plt.close(fig)
        # page 2: every task
        tt = R.drop_duplicates(["track", "task", "metric"]).sort_values(["track", "fam", "task"])
        keys = list(zip(tt.track, tt.task, tt.metric))
        F = R[~R.undefined].set_index(["track", "task", "metric", "method"]).scaled
        P = np.full((len(methods), len(keys)), np.nan)
        for i, m in enumerate(methods):
            for j, k in enumerate(keys):
                P[i, j] = F.get((*k, m), np.nan)
        fig, ax = plt.subplots(figsize=(min(17, max(8, 0.018 * len(keys) + 3)), 0.2 * len(methods) + 1.2))
        cm = cmap.copy()
        cm.set_bad("#d9d9d9")  # NA: undefined or not applicable, never a low score
        ax.imshow(np.ma.masked_invalid(P), aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
        ax.set_xticks([])
        fail = R[(R.tried > 0) & (R.finished == 0)]
        kk = {k: j for j, k in enumerate(keys)}
        mm = {m: i for i, m in enumerate(methods)}
        xs = [(kk[(r.track, r.task, r.metric)], mm[r.method]) for r in fail.itertuples() if r.method in mm and (r.track, r.task, r.metric) in kk]
        if xs:
            ax.scatter(*zip(*xs), marker="x", s=8, c="#a33", linewidths=0.5)
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
    ti = t.copy()
    fr = [pd.read_parquet(f) for f in glob.glob(str(RU.RES / "track_B/*.parquet")) if not f.endswith("__h2h.parquet")]
    if fr:
        b = pd.concat(fr)
        b = b[b.get("identity", False).fillna(False).astype(bool) & b.method.isin(ORDER)].assign(rank_direction="higher")
        from revision.tasks.B2_benchmark.tiers import MAIN
        ti = pd.concat([ti, b[b.setting.isin(MAIN)]], ignore_index=True)
        aggregate(rank_tasks(ti)).to_csv(OUT / "aggregate_with_identity.csv")
    R = rank_tasks(t)
    R.to_csv(OUT / "ranks_long.csv", index=False)
    U = R[R.undefined].copy()
    if len(U):
        def groups(task):
            p = task.split("|")
            f = RU.RES / "meta" / f"{RU.run_name(p[3], p[2], 0)}.json"
            return "" if not f.exists() else ";".join(json.loads(f.read_text()).get("groups", []))
        U["shared_groups_r0"] = U["task"].map(groups)
    U.to_csv(OUT / "undefined_cells.csv", index=False)
    A = aggregate(R)
    A.to_csv(OUT / "aggregate.csv")
    print(A.round(3).to_string())
    draw(R, A)


if __name__ == "__main__":
    main()
