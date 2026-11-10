"""Gold-free: health of every free and canonical candidate seed, mixing, count dependence, relation to the prior;
selection of the lines (rules.py) and their consensus tables. Reads final_state.npz only (read-only); no gold, set or test
file is opened.

    PYTHONPATH=$PWD python -m analysis.tasks.Z3_free_gate.health
Output: results/analysis/Z3/health.csv, lines.json, consensus/<cohort>.csv
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.tasks.O1_orthologs.repair_compare import INTEGRATED
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.Z3_free_gate import rules as RU

sys.path.insert(0, str(RU.REPO / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

CACHE = RU.OUT / "health_cache"


def gate(run: str, praw: np.ndarray) -> dict:
    d = C.RUNS_DIR / run
    r = dict(run=run, finished=K.run_finished(run))
    if not r["finished"]:
        return dict(r, healthy=False)
    st = C.load_state(run)
    summ = json.loads((d / "summary.json").read_text())
    a, p, z = (np.asarray(st[k], float) for k in ("alpha", "prior", "z_mean"))
    r["no_nan"] = bool(np.isfinite(a).all() and np.isfinite(p).all() and np.isfinite(z).all()
                       and "E2.1" not in summ.get("failed_expectations", []))
    r["sat_high"], r["sat_low"] = float((a > 0.95).mean()), float((a < 0.05).mean())
    zv = z.var(0)
    r["n_active_z"] = int((zv >= RU.ZVAR_MIN).sum())
    Dr, D = a - praw, a - p
    r["tail_up_raw"], r["tail_dn_raw"] = float((Dr > RU.TAIL_DELTA).mean()), float((Dr < -RU.TAIL_DELTA).mean())
    r["tail_up"], r["tail_dn"] = float((D > RU.TAIL_DELTA).mean()), float((D < -RU.TAIL_DELTA).mean())
    base = r["no_nan"] and r["n_active_z"] == RU.N_ACTIVE and z.shape[1] == RU.N_ACTIVE
    r["healthy"] = bool(base and r["sat_high"] <= RU.SAT_MAX and r["sat_low"] <= RU.SAT_MAX
                        and r["tail_up_raw"] >= RU.TAIL_MIN and r["tail_dn_raw"] >= RU.TAIL_MIN)
    r["pass_GV_literal"] = bool(base and r["sat_high"] <= RU.SAT_MAX_GV and r["sat_low"] <= RU.SAT_MAX_GV
                                and r["tail_up"] >= RU.TAIL_MIN and r["tail_dn"] >= RU.TAIL_MIN)
    r["adversary"] = float(np.asarray(st["adversarial_accuracy_final"]))
    S = np.asarray(st["S_g_ema"], float)
    sup = np.abs(S) >= RU.EPS
    r["frac_count_supported"] = float(sup.mean())
    r["rho_D_S_supported"] = float(spearmanr(D[sup], S[sup])[0])
    r["healthy_committed_rule"] = r["healthy"]
    if st["config"].get("ablation_config") == "M":  # see rules.py
        r["healthy"] = bool(base and r["sat_high"] <= RU.SAT_MAX and r["sat_low"] <= RU.SAT_MAX
                            and r["rho_D_S_supported"] >= RU.FREE_RHO_MIN)
    r["rho_alpha_prior"] = float(spearmanr(a, p)[0])
    r["sd_D"] = float(np.std(D))
    r["sd_D_masked"] = float(np.std(D[~sup])) if (~sup).any() else np.nan
    r["neg_tail_centred"] = float((D - np.median(D) < -0.05).mean())
    r["n_failed_expectations"] = len(summ.get("failed_expectations", []))
    r["config"] = st["config"].get("ablation_config")
    return r


def integ(run: str, sp_, ct) -> dict:
    f = CACHE / f"{run}.json"
    if f.exists():
        return json.loads(f.read_text())
    z = np.load(C.RUNS_DIR / run / "final_state.npz")["z_mean"]
    if not np.isfinite(z).all():
        return {"int_note": "non-finite z: not computed"}
    m = compute_integration_metrics(z, sp_, ct, k=15, seed=42)["global"]
    out = {f"int_{k}": float(m[k]) for k in ("knn_mixing_entropy", "knn_species_purity", "label_transfer_acc", "sil_celltype")}
    f.write_text(json.dumps(out))
    return out


def main():
    RU.OUT.mkdir(parents=True, exist_ok=True)
    (RU.OUT / "consensus").mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    rows, praws, vns = [], {}, {}
    pool = {c: I.seed_runs_used(c) for c in RU.COHORTS}
    for c in RU.COHORTS:
        a = C.read_adata(c, backed=True)
        sp_, ct = a.obs["species"].to_numpy(), a.obs["cell_type"].to_numpy()
        praws[c] = C.raw_prior(c, C.CANONICAL[c], a.var["conservation_prior"].to_numpy(dtype=float))
        vns[c] = pd.Index(a.var_names.astype(str))
        cand = [("free", s, r) for s, r in RU.FREE[c].items()] + [("canonical_cpu", s, r) for s, r in RU.CAN_CPU[c].items()]
        if c == "liv_hmouse":  # the T3_3_consensus pool, for the state-matched line
            cand += [("canonical_pool", s, r) for s, r in pool[c].items() if r not in RU.CAN_CPU[c].values()]
        for kind, s, r in cand:
            g = gate(r, praws[c])
            if g["finished"]:
                g.update(integ(r, sp_, ct))
            rows.append(dict(cohort=c, kind=kind, seed=s, in_frozen_pool=r in pool[c].values(), **g))
        a.file.close()
    H = pd.DataFrame(rows)
    # liver H+Mouse training state: the four markers of O1_orthologs/repair_compare.py, midpoints between T3_3_consensus's integrated and poorly mixed seeds
    ref = pd.read_csv(RU.REPO / "results/analysis/T3_1b/seed_integration.csv")
    ref = ref[ref.cohort == "liv_hmouse"].set_index("run")
    M = []
    for s, r in pool["liv_hmouse"].items():
        x = K.read_run(r)
        M.append(dict(seed=s, adv=ref.loc[r, "adv"], ent=ref.loc[r, "knn_mixing_entropy"], sd=float(np.std(x["D"])),
                      tail=float((x["D"] - np.median(x["D"]) < -0.05).mean())))
    M = pd.DataFrame(M)
    good = M.seed.isin(INTEGRATED)
    cut = {k: float((M.loc[good, k].min() + M.loc[~good, k].max()) / 2) for k in ("ent", "sd", "tail")}
    cut["adv"] = float((M.loc[good, "adv"].max() + M.loc[~good, "adv"].min()) / 2)
    st = []
    for r in H.itertuples():
        if r.cohort != "liv_hmouse" or not r.finished:
            st.append(None)
            continue
        n = (int(r.adversary <= cut["adv"]) + int(r.int_knn_mixing_entropy >= cut["ent"]) + int(r.sd_D >= cut["sd"])
             + int(r.neg_tail_centred >= cut["tail"]))
        st.append("integrated" if n >= 3 else ("poorly mixed" if n <= 1 else "between"))
    H["state_markers"] = st
    H["state_T33"] = [("integrated" if s in INTEGRATED else "poorly mixed") if (c == "liv_hmouse" and k != "free") else None
                      for c, k, s in zip(H.cohort, H.kind, H.seed)]
    H.to_csv(RU.OUT / "health.csv", index=False)
    # lines
    lines, complete = {}, True
    for c in RU.COHORTS:
        f = H[(H.cohort == c) & (H.kind == "free") & H.healthy].sort_values("seed").head(5)
        free = {int(s): r for s, r in zip(f.seed, f.run)}
        if len(free) < 5:
            complete = False
        can_avail = dict(RU.CAN_CPU[c])
        can = {s: can_avail[s] for s in free if s in can_avail}
        for s in sorted(can_avail):
            if len(can) >= len(free):
                break
            if s not in can:
                can[s] = can_avail[s]
        lines.setdefault("free5", {})[c] = free
        lines.setdefault("can5", {})[c] = dict(sorted(can.items()))
        if c == "liv_hmouse":
            fs = H[(H.cohort == c) & H.run.isin(free.values())].set_index("run").state_markers
            need = {k: int((fs == k).sum()) for k in ("integrated", "poorly mixed", "between")}
            cp = H[(H.cohort == c) & (H.kind != "free") & H.in_frozen_pool].copy()
            cp["cpu"] = cp.kind == "canonical_cpu"
            cp = cp.sort_values(["cpu", "seed"], ascending=[False, True])
            pick = {}
            for k, n in need.items():
                want = "poorly mixed" if k == "between" else k  # 'between' has no canonical counterpart: matched to poorly mixed
                x = cp[(cp.state_T33 == want) & ~cp.run.isin(pick.values())].head(n)
                pick.update({int(s): r for s, r in zip(x.seed, x.run)})
            lines.setdefault("can_state", {})[c] = dict(sorted(pick.items()))
            lines.setdefault("free_state_counts", {})[c] = need
            fi = H[(H.cohort == c) & (H.kind == "free") & H.healthy & (H.state_markers == "integrated")].sort_values("seed")
            lines.setdefault("free_int", {})[c] = {int(s): r for s, r in zip(fi.seed, fi.run) if int(s) in free}
            lines.setdefault("can_int", {})[c] = {s: r for s, r in lines["can5"][c].items()
                                                  if H[(H.cohort == c) & (H.run == r)].state_T33.iloc[0] == "integrated"}
        else:
            lines.setdefault("can_state", {})[c] = lines["can5"][c]
            lines.setdefault("free_int", {})[c] = free
            lines.setdefault("can_int", {})[c] = lines["can5"][c]
    lines["complete"] = complete
    lines["cuts_liv_hmouse"] = cut
    (RU.OUT / "lines.json").write_text(json.dumps(lines, indent=1, default=str))
    for c in RU.COHORTS:
        out = pd.DataFrame(dict(var_name=vns[c], prior_raw=praws[c]))
        for line in ("free5", "can5", "can_state", "free_int", "can_int"):
            runs = lines[line][c]
            if not runs:
                continue
            R = [K.read_run(r) for r in runs.values()]
            A = np.column_stack([x["alpha"] for x in R]).mean(1)
            Dm = np.column_stack([x["D"] for x in R]).mean(1)
            out[f"alpha_{line}"] = A
            out[f"prior_{line}"] = R[0]["prior"]
            out[f"D_{line}"] = Dm - np.median(Dm)
            draw = A - praws[c]
            out[f"D_raw_{line}"] = draw - np.median(draw)
            out[f"S_{line}"] = np.column_stack([x["S_g_ema"] for x in R]).mean(1)
            out[f"n_seeds_{line}"] = len(R)
        out.to_csv(RU.OUT / "consensus" / f"{c}.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    cols = ["cohort", "kind", "seed", "finished", "healthy", "healthy_committed_rule", "pass_GV_literal", "sat_high", "tail_up_raw", "tail_dn_raw",
            "adversary", "int_knn_mixing_entropy", "int_label_transfer_acc", "rho_D_S_supported", "rho_alpha_prior", "sd_D",
            "sd_D_masked", "state_markers", "state_T33"]
    print(H[cols].round(3).to_string())
    print(json.dumps({k: v for k, v in lines.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
