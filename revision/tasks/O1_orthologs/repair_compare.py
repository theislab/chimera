"""O2: comparison of repaired runs with canonical seeds, gold-free. Rule in ledgers/O2_repair.md, committed before any run existed.

    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.repair_compare gate
    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.repair_compare readouts      # only after the gate
"""

from __future__ import annotations

import itertools
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import controls as K
from revision.tasks.O1_orthologs import rules as RU
from revision.tasks.T4_3_scoreboard import inputs as I

sys.path.insert(0, str(RU.REPO / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

OUT = RU.RES / "repair"
SEEDS = (42, 43, 44)
COH = {"liv_hmouse": ("liver", "human_mouse.h5ad", ("Human", "Mouse")), "liv_hmonkey": ("liver", "human_monkey.h5ad", ("Human", "Monkey")),
       "test_hmouse": ("testis", "human_mouse.h5ad", ("Human", "Mouse")), "liv_4sp": ("liver", "unified.h5ad", ("Human", "Monkey", "Mouse"))}
INTEGRATED = {42, 43, 44, 47, 48, 51}


def rho(a, b):
    return float(spearmanr(a, b)[0])


def centred(x):
    return x - np.median(x, axis=0)


def obj(c):
    t, f, _ = COH[c]
    return RU.DATA / f"repair_{t}" / f


def repaired_set(c) -> pd.DataFrame:
    t, _, sp_ = COH[c]
    L = pd.read_csv(RU.DATA / f"repair_{t}" / "repair_log.csv")
    return L[L.species.isin(sp_)]


def load_runs(c):
    """Canonical and new per-seed alpha - prior and S on the canonical gene order of each object, keyed by var_name."""
    can = {int(s): r for s, r in I.seed_runs_used(c).items()}
    a0 = C.read_adata(c, backed=True)
    v0 = pd.Index(a0.var_names.astype(str))
    a1 = ad.read_h5ad(obj(c), backed="r")
    v1 = pd.Index(a1.var_names.astype(str))
    new = {s: f"o2-{c}-s{s}" for s in SEEDS}
    ctrl = {s: f"o2-ctrl-{c}-s{s}" for s in SEEDS} if c == "liv_hmonkey" else {}
    return can, ctrl, new, a0, v0, a1, v1


def state_markers(c, can, new, a1):
    """Four markers, midpoints from canonical runs (ledger). Liver H+Mouse only."""
    ref = pd.read_csv(RU.REPO / "results/revision/T3_1b/seed_integration.csv")
    ref = ref[ref.cohort == c].set_index("run")
    rows = []
    for s, r in can.items():
        x = K.read_run(r)
        rows.append(dict(seed=s, run=r, adv=ref.loc[r, "adv"], ent=ref.loc[r, "knn_mixing_entropy"], sd=float(np.std(x["D"])), tail=float((x["D"] - np.median(x["D"]) < -0.05).mean())))
    M = pd.DataFrame(rows)
    good = M.seed.isin(INTEGRATED)
    cut = {k: float((M.loc[good, k].min() + M.loc[~good, k].max()) / 2) if k != "adv" else float((M.loc[good, k].max() + M.loc[~good, k].min()) / 2) for k in ("adv", "ent", "sd", "tail")}
    out = {}
    for s, r in new.items():
        x = K.read_run(r)
        z = np.load(C.RUNS_DIR / r / "final_state.npz")
        g = compute_integration_metrics(z["z_mean"], a1.obs["species"].to_numpy(), a1.obs["cell_type"].to_numpy(), k=15, seed=42)["global"]
        m = dict(adv=float(z["adversarial_accuracy_final"]), ent=g["knn_mixing_entropy"], sd=float(np.std(x["D"])), tail=float((x["D"] - np.median(x["D"]) < -0.05).mean()))
        n = int(m["adv"] <= cut["adv"]) + int(m["ent"] >= cut["ent"]) + int(m["sd"] >= cut["sd"]) + int(m["tail"] >= cut["tail"])
        out[s] = dict(m, n_integrated_markers=n, state="integrated" if n >= 3 else ("poorly mixed" if n <= 1 else "between"))
    return out, cut


def gate():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, verdicts, health = [], {}, []
    for c in COH:
        if not all((C.RUNS_DIR / f"o2-{c}-s{s}" / "final_state.npz").exists() for s in SEEDS):
            print("no runs for", c)
            continue
        can, ctrl, new, a0, v0, a1, v1 = load_runs(c)
        shared = v0.intersection(v1)
        p0, p1 = v0.get_indexer(shared), v1.get_indexer(shared)
        L = repaired_set(c)
        is_rep = shared.isin(set(L.var_name))
        pool = {f"can{s}": can[s] for s in can}
        pool.update({f"ctrl{s}": r for s, r in ctrl.items()})
        Dc, Sc = {}, {}
        for k, r in pool.items():
            x = K.read_run(r)
            if len(x["D"]) != len(v0):
                raise AssertionError(r)
            Dc[k], Sc[k] = x["D"][p0], x["S_g_ema"][p0]
        Dn, Sn = {}, {}
        for s, r in new.items():
            x = K.read_run(r)
            Dn[s], Sn[s] = x["D"][p1], x["S_g_ema"][p1]
            z = np.load(C.RUNS_DIR / r / "final_state.npz")["z_mean"]
            health.append(dict(cohort=c, seed=s, run=r, nan=int(np.isnan(x["alpha"]).sum()), active_z=int((np.nanstd(z, axis=0) > 0.01).sum()), sat_hi=float((x["alpha"] > 0.95).mean()), rho_D_S=rho(x["D"], x["S_g_ema"]),
                               failed=";".join(x["summary"].get("failed_expectations", []))))
        keys = sorted(pool)
        states = None
        if c == "liv_hmouse":
            states, cut = state_markers(c, can, new, a1)
            json.dump(dict(states=states, cut=cut), open(OUT / "states_liv_hmouse.json", "w"), indent=1)
        support = np.abs(np.column_stack([Sc[k] for k in keys]).mean(1)) >= RU_EPS
        for readout, A, B in (("D", Dc, Dn), ("S", Sc, Sn)):
            M = np.column_stack([A[k] for k in keys])
            for gs, mask in (("all", np.ones(len(shared), bool)), ("unrepaired", ~is_rep), ("repaired", is_rep)):
                for uni, m2 in (("all", mask), ("count_supported", mask & support)):
                    if m2.sum() < 30:
                        rows.append(dict(cohort=c, readout=readout, genes=gs, universe=uni, n=int(m2.sum()), note="under 30"))
                        continue
                    X = centred(M[m2]) if readout == "D" else M[m2]
                    Y = {s: (B[s][m2] - np.median(B[s][m2])) if readout == "D" else B[s][m2] for s in B}
                    cons = X.mean(1)
                    n = len(keys)
                    refA = {k: rho(X[:, i], np.delete(X, i, 1).mean(1)) for i, k in enumerate(keys)}
                    size = 2 if n == 5 else 3
                    combos = list(itertools.combinations(range(n), size))
                    if n > 3:
                        refB = [rho(X[:, list(p)].mean(1), np.delete(X, list(p), 1).mean(1)) for p in combos]
                    else:
                        refB = []
                    refC = [rho(X[:, i], X[:, j]) for i, j in itertools.combinations(range(n), 2)]
                    newA = {s: rho(Y[s], cons) for s in Y}
                    newB = rho(np.mean([Y[s] for s in Y], axis=0), cons)
                    r = dict(cohort=c, readout=readout, genes=gs, universe=uni, n=int(m2.sum()), A_min=min(refA.values()), A_med=float(np.median(list(refA.values()))), A_max=max(refA.values()),
                             B_min=min(refB) if refB else np.nan, B_med=float(np.median(refB)) if refB else np.nan, B_max=max(refB) if refB else np.nan, C_min=min(refC), C_med=float(np.median(refC)), C_max=max(refC),
                             B_new=newB, new_pair_med=float(np.median([rho(Y[a], Y[b]) for a, b in itertools.combinations(Y, 2)])),
                             C_new_med=float(np.median([rho(Y[s], X[:, i]) for s in Y for i in range(n)])), **{f"A_s{s}": v for s, v in newA.items()})
                    if c == "liv_hmonkey":
                        ci = [i for i, k in enumerate(keys) if k.startswith("ctrl")]
                        mi = [i for i, k in enumerate(keys) if k.startswith("can")]
                        r["ctrl_mean_vs_mps_mean"] = rho(X[:, ci].mean(1), X[:, mi].mean(1))
                        r["new_mean_vs_mps_mean"] = rho(np.mean([Y[s] for s in Y], axis=0), X[:, mi].mean(1))
                    if c == "liv_hmouse":
                        for st, keep in (("integrated", [i for i, k in enumerate(keys) if int(k[3:]) in INTEGRATED]), ("poorly mixed", [i for i, k in enumerate(keys) if int(k[3:]) not in INTEGRATED])):
                            v = [list(refA.values())[i] for i in keep]
                            r[f"A_min_{st}"], r[f"A_max_{st}"] = min(v), max(v)
                    rows.append(r)
                    if readout == "D" and uni == "all" and gs in ("all", "unrepaired"):
                        below = {}
                        for s in SEEDS:
                            st = states[s]["state"] if states else "all"
                            lo = r.get(f"A_min_{st}", r["A_min"]) if st in ("integrated", "poorly mixed") else r["A_min"]
                            below[s] = dict(state=st, rho=newA[s], ref_min=lo, below=bool(newA[s] < lo))
                        bb = bool(refB) and newB < min(refB)
                        verdicts.setdefault(c, {})[gs] = dict(seeds=below, all_seeds_below=all(x["below"] for x in below.values()), mean_rho=newB, B_min=r["B_min"], B_max=r["B_max"], mean_below=bb, beyond=all(x["below"] for x in below.values()) or bb)
        verdicts[c]["n_shared"] = int(len(shared))
        verdicts[c]["n_repaired_shared"] = int(is_rep.sum())
        verdicts[c]["genes_only_in_repaired"] = int(len(v1.difference(v0)))
        verdicts[c]["genes_only_in_canonical"] = int(len(v0.difference(v1)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(health).to_csv(OUT / "health.csv", index=False)
    verdicts["any_beyond"] = any(v[g]["beyond"] for v in verdicts.values() if isinstance(v, dict) for g in ("all", "unrepaired"))
    (OUT / "gate.json").write_text(json.dumps(verdicts, indent=1, default=float))
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    print(json.dumps(verdicts, indent=1, default=float))
    print(pd.DataFrame(health).round(3).to_string())
    print(R[(R.universe == "all")].round(3).drop(columns=["universe"]).to_string())


RU_EPS = 1e-6


def strict(alpha, pi, S):
    """H1 strict support on these seeds: rules.py. alpha, S: genes x seeds; pi: the prior the runs anchor on."""
    d = alpha.mean(1) - pi
    D = d - np.median(d)
    Sm = S.mean(1)
    sup = np.abs(Sm) >= RU_EPS
    med = np.median(np.abs(Sm[sup]))
    ds = alpha - pi[:, None]
    ds = ds - np.median(ds, axis=0)
    st = (np.sign(ds) == np.sign(D)[:, None]).all(1) & (np.sign(S) == np.sign(Sm)[:, None]).all(1)
    return sup & (np.abs(Sm) >= med) & (np.sign(Sm) == np.sign(D)) & st, D, Sm, sup


def readouts():
    gate_ = json.loads((OUT / "gate.json").read_text())
    if "any_beyond" not in gate_:
        raise SystemExit("gate first")
    rows, genes = [], []
    for c in COH:
        if c not in gate_:
            continue
        can, _, new, a0, v0, a1, v1 = load_runs(c)
        shared = v0.intersection(v1)
        p0, p1 = v0.get_indexer(shared), v1.get_indexer(shared)
        A0 = np.column_stack([K.read_run(can[s])["alpha"][p0] for s in SEEDS])
        S0 = np.column_stack([K.read_run(can[s])["S_g_ema"][p0] for s in SEEDS])
        pi0 = K.read_run(can[SEEDS[0]])["prior"][p0]
        A1 = np.column_stack([K.read_run(new[s])["alpha"][p1] for s in SEEDS])
        S1 = np.column_stack([K.read_run(new[s])["S_g_ema"][p1] for s in SEEDS])
        pi1 = K.read_run(new[SEEDS[0]])["prior"][p1]
        s0, D0, Sm0, sup0 = strict(A0, pi0, S0)
        s1, D1, Sm1, sup1 = strict(A1, pi1, S1)
        L = repaired_set(c).drop_duplicates(["var_name", "species", "kind"])
        tab = pd.read_csv(RU.TABLES[COH[c][0]]).set_index("ortholog_group_id").hsapiens_gene_name
        sym = tab.reindex([int(v.split("_")[1]) for v in shared]).to_numpy()
        d = pd.DataFrame(dict(cohort=c, var_name=shared, symbol=sym, D0=D0, D1=D1, S0=Sm0, S1=Sm1, sup0=sup0, sup1=sup1, strict0=s0, strict1=s1))
        k = L.groupby("var_name").agg(kind=("kind", lambda x: "+".join(sorted(set(x)))), species=("species", lambda x: "+".join(sorted(set(x)))), umi_after=("total_after", "sum"), umi_before=("total_before", "sum"))
        d = d.join(k, on="var_name")
        d["group"] = np.where(d.kind.isna(), "unrepaired", np.where(d.kind.fillna("").str.contains("false_zero"), "false_zero", "wrong_row"))
        q = {x: (np.quantile(d[f"{x}0"], 0.05), np.quantile(d[f"{x}1"], 0.05)) for x in ("D", "S")}
        for g, x in d.groupby("group"):
            dS, dD = x.S1 - x.S0, x.D1 - x.D0
            rows.append(dict(cohort=c, group=g, n=len(x), support0=x.sup0.mean(), support1=x.sup1.mean(), strict0=int(x.strict0.sum()), strict1=int(x.strict1.sum()), S0=x.S0.median(), S1=x.S1.median(), D0=x.D0.median(), D1=x.D1.median(),
                             dS_med=dS.median(), dD_med=dD.median(), dS_sd=dS.std(), dD_sd=dD.std(), rho_dS_dD=rho(dS, dD) if len(x) > 10 else np.nan,
                             low5_S0=float((x.S0 <= q["S"][0]).mean()), low5_S1=float((x.S1 <= q["S"][1]).mean()), low5_D0=float((x.D0 <= q["D"][0]).mean()), low5_D1=float((x.D1 <= q["D"][1]).mean()),
                             rank_S0=float(d.S0.rank(pct=True)[x.index].median()), rank_S1=float(d.S1.rank(pct=True)[x.index].median()), rank_D0=float(d.D0.rank(pct=True)[x.index].median()), rank_D1=float(d.D1.rank(pct=True)[x.index].median())))
        genes.append(d[d.group != "unrepaired"])
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "readouts_groups.csv", index=False)
    G = pd.concat(genes, ignore_index=True)
    G.to_csv(OUT / "readouts_genes.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    print(R.round(3).T.to_string())
    for c, x in G.groupby("cohort"):
        t = x[(x.group == "false_zero")].sort_values("S1", ascending=False).head(10)
        print(c, t[["symbol", "species", "umi_after", "S0", "S1", "D0", "D1", "strict0", "strict1"]].round(3).to_string())


if __name__ == "__main__":
    dict(gate=gate, readouts=readouts)[sys.argv[1]]()
