"""Track B: score one (cohort, setting). Writes tidy rows in the benchmark schema, head-to-head, per-replicate values.

rho_p with the independent gold, and agreement with the method's full-data reading, as README.md. Run in chimera_env.
    python -m analysis.tasks.B2_benchmark.score_b --index I     (I into task_list)
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

from analysis.tasks.B2_benchmark import engine as EN
from analysis.tasks.B2_benchmark import inputs as IN
from analysis.tasks.B2_benchmark import rules as RU
from analysis.tasks.T4_3_scoreboard import stats as ST

OUT = RU.RES / "track_B"
SC = RU.RES / "scores"
HL = RU.RES / "health"
TL = RU.RES / "tool_scores"
METHODS_FIT = RU.CHIMERA + RU.CHIMERA_EXPL
METHODS_DIRECT = RU.DIRECT
FOCAL = ["S_g", "D", "B1z"]


def task_list() -> list[tuple[str, str]]:
    g = pd.DataFrame(RU.grid())
    f = RU.RES / "tiers_used.json"
    if f.exists():  # stopping rule: settings of the complete tiers only, liver whole
        keep = set(json.loads(f.read_text())["settings"])
        g = g[(g.cohort == "liv_hmouse") | g.setting.isin(keep)]
    return sorted(set(zip(g.cohort, g.setting)), key=lambda x: (x[0], list(RU.SETTINGS).index(x[1])))


def read(run: str) -> pd.DataFrame | None:
    p = SC / f"{run}.parquet"
    return pd.read_parquet(p).set_index("var_name") if p.exists() else None


def healthy(run: str) -> bool:
    p = HL / f"{run}.json"
    return p.exists() and not json.loads(p.read_text())["discard"]


def consensus(frames: list[pd.Series], index, min_reps: int = RU.MIN_REPS) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean over replicates, needing at least min_reps defined. Returns consensus and the replicate matrix."""
    if not frames:
        return np.full(len(index), np.nan), np.full((len(index), 0), np.nan)
    M = np.column_stack([f.reindex(index).to_numpy(dtype=float) for f in frames])
    n = np.isfinite(M).sum(1)
    with np.errstate(invalid="ignore"):
        c = np.nansum(M, 1) / np.where(n > 0, n, 1)
    c[n < min_reps] = np.nan
    return c, M


LABEL_BLIND_TOOLS = ("CAMEX", "SAMap")  # read no labels: on label-only settings their reading is that of `full`, marked label-independent, as Chimera on liver


def tool_scores(tool: str, cohort: str, setting: str, index, reps=None):
    reps = allowed_reps(cohort) if reps is None else reps
    if tool in LABEL_BLIND_TOOLS and setting in RU.SETTINGS and RU.SETTINGS[setting]["family"] == "label":
        setting = "full"
    fr = []
    for r in reps:
        p = TL / tool / f"{RU.run_name(cohort, setting, r)}.csv"
        if p.exists():
            fr.append(pd.read_csv(p).set_index("var_name")["score"])
    return consensus(fr, index, min_reps=1) if fr else (None, None)  # tools: SATURN on one replicate per setting, CAMEX on three, by budget; n_reps in the tool's sidecar


def allowed_reps(cohort: str) -> list[int]:
    """Stopping rule: replicates of the complete tiers only (tiers.freeze writes tiers_used.json). Liver ran every tier before the rule."""
    f = RU.RES / "tiers_used.json"
    if not f.exists() or cohort == "liv_hmouse":
        return RU.REPS
    return json.loads(f.read_text())["reps"]


def gather(cohort: str, setting: str, index) -> dict:
    """method -> dict(consensus, reps matrix, reference), for one setting."""
    grid = [r for r in RU.grid([cohort]) if r["setting"] == setting and int(r["rep"]) in allowed_reps(cohort)]
    out, n_fits, n_disc = {}, 0, 0
    d_frames, c_frames = [], {}
    for r in grid:
        f = read(r["run"])
        if f is None:
            continue
        d_frames.append(f)
        src = r["chimera_from"] if int(r["fit"]) == 0 else r["run"]
        if healthy(src):
            g = read(src)
            if g is not None:
                for k in METHODS_FIT:
                    if k in g:
                        c_frames.setdefault(k, []).append(g[k])
        elif (HL / f"{src}.json").exists():
            n_disc += 1
        n_fits += 1
    # Tiers: a setting scored on fewer than three replicates (tier 0, tier 1) reads what it has; n_reps says how many. With all three, at least MIN_REPS as committed.
    mr = RU.MIN_REPS if len(d_frames) >= 3 else 1
    for k in METHODS_DIRECT + ["random"]:
        out[k] = consensus([f[k] for f in d_frames if k in f], index, min_reps=mr)
    for k in METHODS_FIT:
        out[k] = consensus(c_frames.get(k, []), index, min_reps=mr)
    for t in RU.TOOLS:
        c, M = tool_scores(t, cohort, setting, index)
        if c is not None:
            out[t] = (c, M)
    return dict(scores=out, n_reps=len(d_frames), n_fits=n_fits, n_discarded=n_disc)


def reference(cohort: str, index) -> dict:
    """Full-data reading of each method: existing fits for Chimera (other seeds), base object for direct, reference seeds for tools."""
    out = {}
    fr = [read(r["run"]) for r in RU.ref_rows([cohort])]
    fr = [f for f in fr if f is not None]
    for k in METHODS_FIT + METHODS_DIRECT:
        cols = [f[k] for f in fr if k in f and f[k].notna().any()]
        if not cols:
            out[k] = np.full(len(index), np.nan)
        elif k in METHODS_DIRECT or len(cols) == 1:  # direct scores of the base object do not depend on the fit
            out[k] = cols[0].reindex(index).to_numpy(dtype=float)
        else:
            out[k] = consensus(cols, index)[0]
    for t in RU.TOOLS:
        c, _ = tool_scores(t, cohort, "ref", index)
        if c is None:  # deterministic given the data, or no reference seed: the full reading itself
            c, _ = tool_scores(t, cohort, "full", index)
        if c is not None:
            out[t] = c
    return out


def score_task(cohort: str, setting: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tab = IN.cohort_table(cohort)
    tab = tab[tab["var_name"].notna()].set_index("var_name")
    index = tab.index
    tab = tab.reset_index()
    fam = RU.SETTINGS[setting]["family"]
    S = gather(cohort, setting, index)
    ref = reference(cohort, index)
    if S["n_reps"] == 0:
        raise SystemExit(f"no scores for {cohort} {setting}")
    task = EN.Task(tab, RU.N_DRAWS_B, RU.useed("B", cohort, setting))
    scores = {k: v[0] for k, v in S["scores"].items()}
    core = [k for k in RU.CHIMERA + RU.DIRECT]
    groups_h2h, side_map, flags = [], {}, {}
    for kind, (gold, covs) in RU.B_GOLDS[cohort].items():
        G = tab[gold].to_numpy(dtype=float)
        ok0 = np.isfinite(G) & tab[covs].notna().all(1).to_numpy()
        in_core = [k for k in core if np.isfinite(scores[k][ok0]).mean() >= 0.5]
        U = ok0.copy()
        for k in in_core:
            U &= np.isfinite(scores[k])
        main = f"{kind}|main"
        extra = {k: scores[k] for k in ("random", "S_fwd", "D_raw") if k in scores and np.isfinite(scores[k][U]).all()}
        task.add(main, U, G, {**{k: scores[k] for k in in_core}, **extra}, covs)
        groups_h2h.append((kind, main))
        # pi: given C without pi. dN/dS, LOEUF, under-covered core scores, tools: own genes inside the gold's universe, focal readouts beside
        task.add(f"{kind}|pi", U, G, dict(pi=tab["pi_score"].to_numpy(dtype=float)), [c for c in covs if c != "prior_raw"])
        side = {"dNdS": tab["dNdS_score"].to_numpy(dtype=float), "LOEUF": tab["LOEUF_score"].to_numpy(dtype=float)}
        side.update({k: scores[k] for k in core if k not in in_core})
        side.update({k: scores[k] for k in RU.TOOLS if k in scores})
        for k, v in side.items():
            rows = ok0 & np.isfinite(v)
            foc = {f: scores[f] for f in FOCAL if f in in_core}
            rows2 = rows.copy()
            for f in foc.values():
                rows2 &= np.isfinite(f)
            if rows2.sum() < RU.STRATUM_MIN:
                flags[(kind, k)] = "fewer than 60 genes"
                continue
            task.add(f"{kind}|{k}", rows2, G, {**foc, k: v}, covs)
            side_map[(kind, k)] = f"{kind}|{k}"
    # agreement, one group per method on the genes both readings define, partial on C and raw
    agree_raw = {}
    for k in core + RU.CHIMERA_EXPL + [t for t in RU.TOOLS if t in scores]:
        if k not in ref or k not in scores:
            continue
        rows = np.isfinite(ref[k]) & np.isfinite(scores[k]) & tab[RU.COV_C].notna().all(1).to_numpy()
        if rows.sum() < RU.STRATUM_MIN:
            flags[("agreement", k)] = "fewer than 60 genes"
            continue
        task.add(f"agreement|{k}", rows, ref[k], {k: scores[k]}, RU.COV_C)
        rs = pd.Series(ref[k][rows]).corr(pd.Series(scores[k][rows]), method="spearman")
        ident = bool(np.array_equal(ref[k][rows], scores[k][rows]))  # the setting's reading is the reference itself: 1 by identity, not a measurement
        agree_raw[k] = (float(rs), float(rows.sum() / max(np.isfinite(ref[k]).sum(), 1)), ident)
    task.run()
    rows = []
    base = dict(track="B", setting=setting, family=fam, cohort=cohort, n_reps=S["n_reps"], n_fits=S["n_fits"], n_discarded=S["n_discarded"])
    label_indep = fam == "label" and not RU.uses_labels(cohort)
    for r in task.rows():
        kind, g = r["group"].split("|")
        m = r["method"]
        if kind == "agreement":
            d = dict(task=f"B|{fam}|{setting}|{cohort}|agreement", metric="agreement_partial", gold="full_data_reading", rank_direction="higher")
            d.update(raw_spearman=agree_raw[m][0], share_kept=agree_raw[m][1], identity=agree_raw[m][2])
        else:
            if g != "main" and m in FOCAL and g not in ("pi",):
                continue  # focal readouts on a side group's genes go to the head-to-head table only
            gold = RU.B_GOLDS[cohort][kind][0]
            d = dict(task=f"B|{fam}|{setting}|{cohort}|{kind}", metric="rho_p", gold=gold, rank_direction="higher")
        d.update(method=m, value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n"], n_blocks=r["n_blocks"],
                 flag=("label-independent" if (label_indep and m in METHODS_FIT) or (fam == "label" and m in LABEL_BLIND_TOOLS) else ""))
        if kind != "agreement" and m in ("S_fwd", "D_raw", "random"):
            d["rank_direction"] = ""  # exploratory or control: tabled, not ranked
        if kind == "agreement" and m in ("S_fwd", "D_raw"):
            d["rank_direction"] = ""
        if kind == "agreement" and agree_raw[m][2]:
            d["rank_direction"], d["flag"] = "", "identity: the setting's reading is the reference; tabled, not ranked"
        if kind != "agreement" and m == "pi":
            d["flag"] = "given C without pi"
        rows.append({**base, **d})
    # single-replicate mean, on the main universe of each gold
    single = []
    for kind, (gold, covs) in RU.B_GOLDS[cohort].items():
        g = next(x for x in task.groups if x["name"] == f"{kind}|main")
        Z = g["Z"]
        Gr = g["R"][:, 0]
        for m in g["names"]:
            if m not in S["scores"]:
                continue
            M = S["scores"][m][1]
            vals = []
            for j in range(M.shape[1]):
                v = M[g["idx"], j]
                okk = np.isfinite(v)
                if okk.sum() < RU.STRATUM_MIN:
                    continue
                R = np.column_stack([ST.midrank(Gr[okk]), ST.midrank(v[okk])])
                vals.append(float(ST.rho_p_matrix(R, Z[okk])[1]))
            if vals:
                single.append(dict(task=f"B|{fam}|{setting}|{cohort}|{kind}", method=m, single_mean=float(np.mean(vals)), single_min=float(np.min(vals)), single_max=float(np.max(vals)), n_single=len(vals)))
    t = pd.DataFrame(rows)
    if single:
        t = t.merge(pd.DataFrame(single), on=["task", "method"], how="left")
    t.to_parquet(OUT / f"{cohort}__{setting}.parquet")
    h = []
    for kind, main in groups_h2h:
        h += [dict(cohort=cohort, setting=setting, family=fam, kind=kind, **x) for x in task.h2h(FOCAL, [j for j in RU.J_H2H + ["B1", "Lfc"]], main)]
        h += [dict(cohort=cohort, setting=setting, family=fam, kind=kind, **x) for x in task.h2h(["S_g", "D"], ["B1z"], main)]
    for (kind, k), grp in side_map.items():
        foc = [f for f in FOCAL if f in next(x for x in task.groups if x["name"] == grp)["names"]]
        h += [dict(cohort=cohort, setting=setting, family=fam, kind=kind, **x) for x in EN.side_h2h(task, grp, k, foc)]
    if h:
        pd.DataFrame(h).to_parquet(OUT / f"{cohort}__{setting}__h2h.parquet")
    print(cohort, setting, len(t), "rows", len(h), "h2h", flags, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--cohort")
    ap.add_argument("--setting")
    a = ap.parse_args()
    if a.index is not None:
        c, s = task_list()[a.index]
    else:
        c, s = a.cohort, a.setting
    score_task(c, s)


if __name__ == "__main__":
    main()
