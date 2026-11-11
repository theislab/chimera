"""Overview of the benchmark: methods by tasks, in the scIB funky-heatmap style, merging tracks A, B, C, D.

Rule, fixed before results. For each task and metric, rank every applicable method (best 1, ties share the mean rank), scaled by (rank - 1) / (n - 1); n = applicable methods with a
value in the task. Aggregate of a track = mean of a method's within-task scaled ranks over the track's tasks. Overall = mean of the track aggregates, equal weight per track. Unscaled mean ranks are written
beside. A method applicable to a task but with no value (failed fit, tool that failed there) ranks last in it. Not applicable (static score on agreement; a tool not run on that setting) is left out and drawn
hatched. A row of the tidy tables enters if rank_direction is 'higher' or 'lower'. Every task appears: the overview panel groups tasks by track and family, the task panel draws every task as its own column.

Undefined cells (amendment after the liver 1:20 reading, before the overview was first computed). A method with no value in a task that is applicable to it (for instance a profile score when one species keeps fewer than two shared types) is not a measured
loss. The fixed rule stays and is reported as `overall`: it ranks last. Beside it `overall_excl_undefined`: such cells are left out and each task ranked among the methods with a value. Both are written for every method, Chimera included; in the figures undefined cells are drawn NA,
never as a low score. `undefined_cells.csv` lists every such cell, with the groups left in the setting where known; the text states their number.

Runs tried and finished (added for the same reason). A failed fit or a tool that crashes or returns nothing stays out of the scores; `completion.py` counts them per method and setting, joined here as tried and finished. A cell with fewer finished than tried carries a small
"f/t"; a cell where nothing finished is a failure, drawn with a cross, not a blank, and it ranks last under the fixed rule.

Identity cells (found when the overview was first read; applies to every method alike). An agreement value that compares a reading with itself (a deterministic method on
`full`, a label-blind deterministic tool on a label-only setting) is 1 by identity, not a measurement. score_b flags these rows and they are tabled, not ranked. The aggregate with
them ranked, as the fixed rule first read, is written to aggregate_with_identity.csv.

Track I (integration, added before it was scored) is a track of its own, on its own methods (the 10 integration models, prefixed I:). Integration is the substrate of Chimera's readouts, not its aim, so the overall of the gene-level tracks A, B, C, D
keeps its fixed rule and does not include I; `overall_with_I` is written beside it for the methods that appear in both (SATURN, CAMEX, and Chimera's latent shown against B1z).

Inputs: results/analysis/B1/*.csv|parquet (tracks A, D: columns track, task, setting, method, metric, value, ci_lo, ci_hi, n, rank_direction), results/analysis/B2/track_B/*.parquet, track_C/units and lists.
Re-runnable as B1_ground_truth, C1_cohorts and new cohorts add results. Outputs: results/analysis/B2/overview/{ranks_long.csv, aggregate.csv} and the two panels
figures/panels/supp_benchmark_overview/{overview, tasks}.pdf, drawn at print size (648 bp wide) and saved through figures/style.py.
    python -m analysis.tasks.B2_benchmark.overview            # rank, aggregate, draw
    python -m analysis.tasks.B2_benchmark.overview --draw     # draw only, from the two tables (draw_panels); CHIMERA_FIGURES_OUT=<folder> for a dry run
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from analysis.tasks.B1_ground_truth import figures as B1F
from analysis.tasks.B2_benchmark import rules as RU
from figures import style

OUT = RU.RES / "overview"
ORDER = RU.OVERVIEW_METHODS
GROUPS = {"chimera": RU.CHIMERA, "direct": RU.DIRECT, "static": RU.STATIC, "tools": RU.TOOLS}

# ---------------------------------------------------------------- the two panels: names, order, colors and sizes (house style: figures/style.py)
PANEL_OVERVIEW, PANEL_TASKS = "supp_benchmark_overview/overview", "supp_benchmark_overview/tasks"
# Rows carry the reader names of the ground-truth figures. Method keys of this script that differ from theirs:
B1_KEY = {"D": "Delta_alpha", "D_free": "Delta_alpha_free", "SATURN": "S_deploy", "CAMEX": "C_cos"}
METHOD_LABEL = {m: B1F.NAME[B1_KEY.get(m, m)] for m in ORDER}
# Row blocks in the order drawn; inside a block the rows are sorted by the overall scaled rank.
FAMILY = {"Chimera readouts": RU.CHIMERA + RU.CHIMERA_FREE, "direct baselines": RU.DIRECT, "other tools": RU.TOOLS, "sequence measures": RU.STATIC}
TRACK_LABEL = {"A": "A ground truth", "B": "B robustness", "C": "C independent data", "D": "D cost"}
# Task families (columns of the overview, brackets of the task panel) in the order drawn. In track B every kind of perturbation has
# two families: the partial correlation with the independent reference, and the agreement with the method's own reading of the
# unperturbed data. A family that is not listed here is drawn after the listed ones of its track, under its key.
FAMILY_LABEL = {
    "A|level": B1F.TASK["level"], "A|pattern": B1F.TASK["pattern"], "A|loss": B1F.TASK["loss"],
    "A|placebo": "null: " + B1F.TASK["placebo"], "A|sham_swap": "null: " + B1F.TASK["sham_swap"], "A|thin_both": "null: " + B1F.TASK["thin_both"],
    "B|control": "unperturbed: reference", "B|control agree": "unperturbed: agreement",
    "B|label": "label quality: reference", "B|label agree": "label quality: agreement",
    "B|composition": "composition: reference", "B|composition agree": "composition: agreement",
    "B|sampling": "sampling: reference", "B|sampling agree": "sampling: agreement",
    "B|technical": "technical: reference", "B|technical agree": "technical: agreement",
    "C|pattern": "pattern", "C|pattern_primates": "pattern: held-out primates", "C|pattern_coe": "pattern: sorted lineages",
    "C|pattern_fine": "pattern: second atlas", "C|pattern_nephron": "pattern: second atlas, nephron",
    "C|pattern_noacinar": "pattern: no acinar cells", "C|pattern_fine_noacinar": "pattern: second atlas, no acinar",
    "C|level": "level: bulk tissue", "C|level_atlas": "level: atlases", "C|list": "named gene lists",
    "D|runtime": "run time", "D|scaling_cells": "run time, fewer cells", "D|pipeline_runtime": "run time, full pipeline", "D|memory": "peak memory",
}
S_HEAD, S_LABEL, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
NA_CELL = "#F2F2F2"  # task panel: a task in which a method has no value
NA_EDGE = "#AAAAAA"  # overview: hatched circle, no value in the family
FAIL = "#AA3333"  # runs finished of runs tried where fits failed; cross where no run finished
RANK_RANGE = (0.97, 0.22)  # part of the sequential map used for scaled ranks: 0 (best) dark, 1 (worst) still visibly blue
DIAMETER = (3.4, 8.4)  # bp, circles of scaled rank 1 and 0
NA_DIAMETER = 4.6  # bp
MARGIN = 3.0  # bp of white kept at the page edges
HEAD_H = 9.0  # bp, row that holds the heading of a block of methods
ROW_H, ROW_H_TASKS = 12.6, 10.0  # bp, method rows of the overview and of the task panel
PITCH_MAX, TRACK_GAP, FAMILY_GAP = 13.5, 6.0, 1.2  # bp: column of the overview; white between tracks; between families of the task panel
BAR_W, BAR_H, NUM_W = 50.0, 6.5, 20.0  # bp: bar of overall rank 0, its height, room for the number beside it
TASKS_RIGHT = 60.0  # bp right of the task columns, for the slanted names of the last families
LEAD_H, LABEL_SEP = 14.0, 11.5  # bp: height of the leader lines of the task panel; least distance between two slanted names
FOOT = 2.5  # bp from the middle of a column to the start of the baseline of its slanted name, so that the letters stand over the column
SLANT = np.deg2rad(45)


def read_tidy() -> pd.DataFrame:
    fr = []
    b1 = RU.REPO / "results/analysis/B1/B1_overview.csv"  # rows of B1_ground_truth for the overview: tracks A and D
    if b1.exists():
        d = pd.read_csv(b1)
        d["method"] = d["method"].replace({"Delta_alpha": "D", "Delta_alpha_free": "D_free", "S_deploy": "SATURN", "C_cos": "CAMEX"})  # names of B1_ground_truth to the overview's
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
        # Track D: the rows of B1_ground_truth are per tool or per family of readouts. One fit gives every Chimera readout; the direct family gives every direct score.
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
    # final scope: tiers 0-2 of this benchmark, so Track B main settings only, every cohort; the other liver settings ran before the tiers and stay in the tables; the depth
    # law arm is a second Chimera reading beside depth25 and is tabled, not ranked
    from analysis.tasks.B2_benchmark.tiers import MAIN

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
        if track == "B":  # the fixed rule acts where every method has a reading to give
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


def family_label(col: str) -> str:
    return FAMILY_LABEL.get(col, col.split("|", 1)[1])


def columns(R: pd.DataFrame) -> tuple:
    """The gene-level tracks with the family and the overview column of every task, and the columns in the order drawn."""
    R = R[R.track != "I"].copy()
    R["fam"] = R["task"].map(family_of)
    R["col"] = R["track"] + "|" + R["fam"]
    if "metric" in R:
        ag = R["metric"].eq("agreement_partial")
        R.loc[ag, "col"] = R.loc[ag, "track"] + "|" + R.loc[ag, "fam"] + " agree"
    seen = set(R["col"])
    listed = [c for c in FAMILY_LABEL if c in seen]
    return R, [c for tr in sorted(R.track.unique()) for c in listed + sorted(seen - set(listed)) if c.startswith(tr + "|")]


def row_blocks(A: pd.DataFrame) -> list:
    """(heading, methods) in the order drawn: the blocks of FAMILY, each sorted by the overall scaled rank."""
    blocks = []
    for fam, members in FAMILY.items():
        ms = [m for m in A.index if m in members]
        if ms:
            blocks.append((fam, sorted(ms, key=lambda m: float(A.loc[m, "overall"]))))
    return blocks


def rank_cmap() -> ListedColormap:
    cm = ListedColormap(matplotlib.colormaps[style.SEQUENTIAL](np.linspace(*RANK_RANGE, 256)), name="scaled_rank")
    cm.set_bad(NA_CELL)
    return cm


def circle_area(v) -> np.ndarray:
    """Marker area of a scaled rank: the diameter falls linearly from rank 0 to rank 1."""
    return (DIAMETER[0] + (DIAMETER[1] - DIAMETER[0]) * (1 - np.asarray(v, dtype=float))) ** 2


def text_width(s: str, size: float) -> float:
    """Width of a string in bp at a font size."""
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=style.FONT)).get_extents().width


def slant_reach(s: str) -> float:
    """How far a slanted name reaches up and to the right of its foot, bp."""
    return (text_width(s, S_TICK) + 1.2 * S_TICK) * float(np.sin(SLANT))


def slanted(ax, x: float, y: float, s: str) -> None:
    ax.text(x, y, s, fontsize=S_TICK, rotation=np.rad2deg(SLANT), rotation_mode="anchor", ha="left", va="bottom")


def canvas(height: float) -> tuple:
    """A page of the landscape width with one axes over all of it, measured in bp from the top left."""
    fig = plt.figure(figsize=(style.inch(style.LANDSCAPE), style.inch(height)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, style.LANDSCAPE)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def row_layout(blocks: list, top: float, pitch: float) -> tuple:
    """Top of every block heading, center of every method row, and the bottom of the last row."""
    heads, ys, y = [], {}, top
    for fam, ms in blocks:
        heads.append((fam, y))
        y += HEAD_H
        for m in ms:
            ys[m] = y + pitch / 2
            y += pitch
    return heads, ys, y


def draw_rows(ax, heads: list, ys: dict, x_label: float, x_right: float) -> None:
    """Block headings, each under a thin rule, and the method names."""
    for fam, y in heads:
        ax.plot([MARGIN, x_right], [y + 0.5, y + 0.5], color=RULE, lw=0.5, solid_capstyle="butt")
        ax.text(MARGIN, y + HEAD_H / 2 + 0.8, fam, fontsize=S_TICK, fontweight="bold", ha="left", va="center_baseline")
    for m, y in ys.items():
        ax.text(x_label, y, METHOD_LABEL.get(m, m), fontsize=S_TICK, ha="right", va="center_baseline")


def overview_panel(R: pd.DataFrame, A: pd.DataFrame):
    """Methods by task families. Circle size and color: mean scaled rank within task. Right: the overall scaled rank."""
    R, cols = columns(R)
    blocks = row_blocks(A)
    methods = [m for _, ms in blocks for m in ms]
    tracks = sorted(R.track.unique())
    M = R[~R.undefined].groupby(["method", "col"]).scaled.mean().unstack().reindex(index=methods, columns=cols)
    N = R.groupby("col").task.nunique().reindex(cols)
    short = R[R.tried > R.finished]
    cmap = rank_cmap()
    # layout, bp
    W = style.LANDSCAPE
    x_label = MARGIN + max(text_width(METHOD_LABEL.get(m, m), S_TICK) for m in methods) + 1
    x_bar = W - MARGIN - NUM_W - BAR_W
    x0 = x_label + 5
    pitch = min(PITCH_MAX, (x_bar - 9 - x0 - TRACK_GAP * (len(tracks) - 1)) / len(cols))
    xs, span, x = {}, {}, x0
    for tr in tracks:
        mine = [c for c in cols if c.startswith(tr + "|")]
        span[tr] = (x, x + pitch * len(mine))
        for c in mine:
            xs[c] = x + pitch / 2
            x += pitch
        x += TRACK_GAP
    y_names = MARGIN + max(slant_reach(family_label(c)) for c in cols)
    y_count = y_names + 6.5
    y_track = y_count + 16
    heads, ys, y_end = row_layout(blocks, y_track + 2.5, ROW_H)
    fig, ax = canvas(y_end + MARGIN)
    draw_rows(ax, heads, ys, x_label, W - MARGIN)
    # columns: family names, number of tasks, track headings
    for c in cols:
        slanted(ax, xs[c] + FOOT, y_names, family_label(c))
        ax.text(xs[c], y_count, f"{int(N[c])}", fontsize=S_TEXT, color=MUTED, ha="center", va="center_baseline")
    ax.text(x_label, y_count, "number of tasks", fontsize=S_TEXT, color=MUTED, ha="right", va="center_baseline")
    for tr in tracks:
        ax.plot([span[tr][0] + 1, span[tr][1] - 1], [y_track, y_track], color=INK, lw=0.8, solid_capstyle="butt")
        ax.text(sum(span[tr]) / 2, y_track - 2.5, TRACK_LABEL.get(tr, tr), fontsize=S_HEAD, ha="center", va="bottom")
    # cells
    cx, cy, cv, nx, ny, fx, fy, example = [], [], [], [], [], [], [], None
    for m in methods:
        for c in cols:
            v = M.loc[m, c]
            if np.isfinite(v):
                cx.append(xs[c]), cy.append(ys[m]), cv.append(v)
            else:
                nx.append(xs[c]), ny.append(ys[m])
            sub = short[(short.method == m) & (short.col == c)]
            if len(sub):
                count = f"{int(sub.finished.sum())}/{int(sub.tried.sum())}"
                example = example or count
                r = float(np.sqrt(circle_area(v))) / 2 if np.isfinite(v) else NA_DIAMETER / 2
                ax.text(xs[c], ys[m] - r - 0.3, count, fontsize=S_TEXT, color=FAIL, ha="center", va="bottom")
                if (sub.finished == 0).any() and not np.isfinite(v):
                    fx.append(xs[c]), fy.append(ys[m])
    ax.scatter(cx, cy, s=circle_area(cv), c=cmap(np.asarray(cv)), edgecolors="none")
    ax.scatter(nx, ny, s=NA_DIAMETER ** 2, facecolors="none", edgecolors=NA_EDGE, hatch="//////", linewidths=0.4)
    if fx:
        ax.scatter(fx, fy, marker="x", s=22, c=FAIL, linewidths=0.8)
    # overall scaled rank: bar and number
    for m in methods:
        o = float(A.loc[m, "overall"])
        ax.barh(ys[m], (1 - o) * BAR_W, left=x_bar, height=BAR_H, color=cmap(0.0), linewidth=0)
        ax.text(x_bar + BAR_W + 3, ys[m], f"{o:.2f}", fontsize=S_TEXT, ha="left", va="center_baseline")
    ax.text(x_bar, y_track - 10.5, "overall scaled rank", fontsize=S_LABEL, ha="left", va="bottom")
    ax.text(x_bar, y_track - 2.5, "0 best; bar = 1 − rank", fontsize=S_TEXT, color=MUTED, ha="left", va="bottom")
    # key, in the corner above the method names
    kx = [MARGIN + 12 + 17 * k for k in range(5)]
    kv = np.linspace(0, 1, 5)
    ax.text(MARGIN, MARGIN, "mean scaled rank within task", fontsize=S_TICK, ha="left", va="top")
    ax.scatter(kx, [MARGIN + 15] * 5, s=circle_area(kv), c=cmap(kv), edgecolors="none")
    for k, lab in ((0, "0 best"), (2, "0.5"), (4, "1 worst")):
        ax.text(kx[k], MARGIN + 25, lab, fontsize=S_TEXT, ha="center", va="center_baseline")
    y = MARGIN + 36
    ax.scatter([kx[0]], [y], s=NA_DIAMETER ** 2, facecolors="none", edgecolors=NA_EDGE, hatch="//////", linewidths=0.4)
    ax.text(kx[0] + 7, y, "not applicable or not run", fontsize=S_TICK, ha="left", va="center_baseline")
    if example:
        y += 9
        ax.text(MARGIN + 2, y, example, fontsize=S_TEXT, color=FAIL, ha="left", va="center_baseline")
        ax.text(MARGIN + 5 + text_width(example, S_TEXT), y, "runs finished / runs tried", fontsize=S_TICK, ha="left", va="center_baseline")
    if fx:
        y += 9
        ax.scatter([kx[0]], [y], marker="x", s=22, c=FAIL, linewidths=0.8)
        ax.text(kx[0] + 7, y, "no run finished", fontsize=S_TICK, ha="left", va="center_baseline")
    return fig


def spread(centers: list, sep: float, highest: list) -> list:
    """Positions in the order of ``centers``, no two closer than ``sep``, none beyond its entry of ``highest``."""
    p = list(centers)
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + sep)
    p[-1] = min(p[-1], highest[-1])
    for i in range(len(p) - 2, -1, -1):
        p[i] = min(p[i], highest[i], p[i + 1] - sep)
    return p


def tasks_panel(R: pd.DataFrame, A: pd.DataFrame):
    """Methods by single tasks: one column per task, colored by the scaled rank within the task; families under brackets."""
    R, cols = columns(R)
    blocks = row_blocks(A)
    methods = [m for _, ms in blocks for m in ms]
    tracks = sorted(R.track.unique())
    tt = R.drop_duplicates(["track", "task", "metric"])
    tt = tt.assign(order=tt["col"].map({c: i for i, c in enumerate(cols)})).sort_values(["order", "task", "metric"])
    F = R[~R.undefined].set_index(["track", "task", "metric", "method"]).scaled
    fail = R[(R.tried > 0) & (R.finished == 0) & R.method.isin(methods)]
    cmap = rank_cmap()
    # layout, bp: tasks of equal width, white between families and between tracks
    W = style.LANDSCAPE
    x_label = MARGIN + max(text_width(METHOD_LABEL.get(m, m), S_TICK) for m in methods) + 1
    x0, x1 = x_label + 5, W - TASKS_RIGHT
    unit = (x1 - x0 - FAMILY_GAP * (len(cols) - len(tracks)) - TRACK_GAP * (len(tracks) - 1)) / len(tt)
    left, span, x = {}, {}, x0
    for tr in tracks:
        mine = [c for c in cols if c.startswith(tr + "|")]
        a = x
        for c in mine:
            left[c] = x
            x += unit * int((tt["col"] == c).sum()) + FAMILY_GAP
        span[tr] = (a, x - FAMILY_GAP)
        x += TRACK_GAP - FAMILY_GAP
    y_names = MARGIN + max(slant_reach(family_label(c)) for c in cols)
    y_bracket = y_names + LEAD_H
    heads, ys, y_end = row_layout(blocks, y_bracket + 2.5, ROW_H_TASKS)
    fig, ax = canvas(y_end + 22 + MARGIN)
    draw_rows(ax, heads, ys, x_label, x1)
    # cells, block of methods by family of tasks
    at = {}
    for c in cols:
        keys = [k for k in zip(tt.track, tt.task, tt.metric, tt["col"]) if k[3] == c]
        xe = left[c] + unit * np.arange(len(keys) + 1)
        at.update({k[:3]: left[c] + unit * (j + 0.5) for j, k in enumerate(keys)})
        for _, ms in blocks:
            P = np.array([[F.get((*k[:3], m), np.nan) for k in keys] for m in ms], dtype=float)
            ye = np.append(ys[ms[0]] - ROW_H_TASKS / 2, [ys[m] + ROW_H_TASKS / 2 for m in ms])
            ax.pcolormesh(xe, ye, np.ma.masked_invalid(P), cmap=cmap, vmin=0, vmax=1, edgecolors="face", linewidth=0.15)
    crosses = [(at[(r.track, r.task, r.metric)], ys[r.method]) for r in fail.itertuples() if (r.track, r.task, r.metric) in at]
    if crosses:
        ax.scatter(*zip(*crosses), marker="x", s=8, c=FAIL, linewidths=0.5)
    # families: bracket over the tasks, slanted name on a leader line
    center = [left[c] + unit * int((tt["col"] == c).sum()) / 2 for c in cols]
    foot = spread(center, LABEL_SEP, [W - MARGIN - FOOT - slant_reach(family_label(c)) for c in cols])
    for c, xc, xf in zip(cols, center, foot):
        ax.plot([left[c], 2 * xc - left[c]], [y_bracket, y_bracket], color=INK, lw=0.7, solid_capstyle="butt")
        ax.plot([xc, xf], [y_bracket, y_names + 1.5], color=RULE, lw=0.5)
        slanted(ax, xf + FOOT, y_names, family_label(c))
    # tracks, under the cells
    for tr in tracks:
        ax.plot(span[tr], [y_end + 3, y_end + 3], color=INK, lw=0.8, solid_capstyle="butt")
        ax.text(sum(span[tr]) / 2, y_end + 5, f"{TRACK_LABEL.get(tr, tr)}\n{int((tt.track == tr).sum())} tasks", fontsize=S_HEAD, ha="center", va="top", linespacing=1.15)
    # key, in the corner above the method names
    ax.text(MARGIN, MARGIN, "scaled rank within task", fontsize=S_TICK, ha="left", va="top")
    bx = MARGIN + 12 + np.linspace(0, 68, 65)
    ax.pcolormesh(bx, [MARGIN + 11.5, MARGIN + 17.5], np.linspace(0, 1, 64)[None, :], cmap=cmap, vmin=0, vmax=1, edgecolors="face", linewidth=0.15)
    for xk, lab in ((bx[0], "0 best"), (bx[32], "0.5"), (bx[-1], "1 worst")):
        ax.text(xk, MARGIN + 25, lab, fontsize=S_TEXT, ha="center", va="center_baseline")
    y = MARGIN + 36
    ax.add_patch(plt.Rectangle((bx[0] - 4.5, y - 3), 9, 6, facecolor=NA_CELL, edgecolor=RULE, linewidth=0.4))
    ax.text(bx[0] + 7, y, "not applicable or not run", fontsize=S_TICK, ha="left", va="center_baseline")
    if crosses:
        y += 9
        ax.scatter([bx[0]], [y], marker="x", s=8, c=FAIL, linewidths=0.5)
        ax.text(bx[0] + 7, y, "no run finished", fontsize=S_TICK, ha="left", va="center_baseline")
    return fig


def draw(R: pd.DataFrame, A: pd.DataFrame) -> None:
    """The two panels, from the ranked tasks and the aggregate: figures/panels/supp_benchmark_overview/{overview,tasks}.pdf."""
    style.apply()
    plt.rcParams["hatch.linewidth"] = 0.4
    style.save_panel(overview_panel(R, A), PANEL_OVERVIEW, exact=True)
    style.save_panel(tasks_panel(R, A), PANEL_TASKS, exact=True)


def draw_panels() -> None:
    """Plotting only: both panels from the tables that main() wrote last. Nothing is ranked again."""
    draw(pd.read_csv(OUT / "ranks_long.csv"), pd.read_csv(OUT / "aggregate.csv", index_col=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = read_tidy()
    ti = t.copy()
    fr = [pd.read_parquet(f) for f in glob.glob(str(RU.RES / "track_B/*.parquet")) if not f.endswith("__h2h.parquet")]
    if fr:
        b = pd.concat(fr)
        b = b[b.get("identity", False).fillna(False).astype(bool) & b.method.isin(ORDER)].assign(rank_direction="higher")
        from analysis.tasks.B2_benchmark.tiers import MAIN
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
    draw_panels() if "--draw" in sys.argv[1:] else main()
