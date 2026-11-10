"""Part C: does dropping flagged pairs move D on the genes that stay? Definitions fixed before the runs existed.

Genes shared with the canonical liver H+Mouse object, joined on human Ensembl ID. Canonical = the 10 seeds of T3_3_consensus. D per run = alpha - prior, centered on the median over the shared genes.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.retrain_compare
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.T4_3_scoreboard import inputs as I

sys.path.insert(0, str(RU.REPO / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

NEW = {42: "o1-liv_hmouse-s42", 43: "o1-liv_hmouse-s43"}
INTEGRATED = {42, 43, 44, 47, 48, 51}
OBJ = RU.DATA / "rebuild_t3" / "liver_hmouse.h5ad"
TAB = RU.DATA / "rebuild_t3" / "ortholog_table.csv"
OUT = RU.RES / "retrain"


def rho(a, b):
    return float(spearmanr(a, b)[0])


def main():
    import anndata as ad

    OUT.mkdir(parents=True, exist_ok=True)
    runs = I.seed_runs_used("liv_hmouse")
    ca = ad.read_h5ad(C.gene_ids.h5ad_path("liv_hmouse"), backed="r")
    can_ids = C.gene_ids.ensembl("liv_hmouse", ca.var_names).to_numpy()
    na = ad.read_h5ad(OBJ, backed="r")
    t = pd.read_csv(TAB).set_index("ortholog_group_id")
    new_ids = t.hsapiens_gene_id.reindex([int(v.split("_")[1]) for v in na.var_names]).to_numpy()
    shared = sorted(set(can_ids) & set(new_ids))
    pc, pn = pd.Index(can_ids).get_indexer(shared), pd.Index(new_ids).get_indexer(shared)
    n = len(shared)
    Dc, Sc = {}, {}
    for s, r in sorted(runs.items()):
        x = K.read_run(r)
        Dc[s], Sc[s] = x["D"][pc] - np.median(x["D"][pc]), x["S_g_ema"][pc]
    Dn, Sn, hl = {}, {}, []
    for s, r in NEW.items():
        x = K.read_run(r)
        Dn[s], Sn[s] = x["D"][pn] - np.median(x["D"][pn]), x["S_g_ema"][pn]
        z = np.load(C.RUNS_DIR / r / "final_state.npz")["z_mean"]
        m = compute_integration_metrics(z, na.obs["species"].to_numpy(), na.obs["cell_type"].to_numpy(), k=15, seed=42)
        g = m["global"]
        hl.append(dict(seed=s, run=r, knn_mixing_entropy=g["knn_mixing_entropy"], knn_species_purity=g["knn_species_purity"], label_transfer_acc=g["label_transfer_acc"],
                       adv=float(np.load(C.RUNS_DIR / r / "final_state.npz")["adversarial_accuracy_final"]), failed_expectations=";".join(x["summary"].get("failed_expectations", [])), sd_D=float(np.std(x["D"])), frac_neg_tail=float((x["D"] < -0.05).mean()),
                       sat_hi=float((x["alpha"] > 0.95).mean()), rho_D_S=rho(x["D"], x["S_g_ema"]), nan=int(np.isnan(x["alpha"]).sum()), active_z=int((np.nanstd(z, axis=0) > 0.01).sum())))
    seeds = sorted(Dc)
    ref = pd.read_csv(RU.REPO / "results/analysis/T3_1b/seed_integration.csv")
    ref = ref[ref.cohort == "liv_hmouse"]
    state_new = {}
    ent_int = ref[ref.run.isin([runs[s] for s in seeds if s in INTEGRATED])].knn_mixing_entropy
    ent_poor = ref[ref.run.isin([runs[s] for s in seeds if s not in INTEGRATED])].knn_mixing_entropy
    for h in hl:
        h["state"] = "integrated" if h["knn_mixing_entropy"] >= (ent_int.min() + ent_poor.max()) / 2 else "poorly mixed"
        state_new[h["seed"]] = h["state"]
    pd.DataFrame(hl).to_csv(OUT / "health.csv", index=False)
    rows, res = [], {}
    for name, A, B in (("D", Dc, Dn), ("S", Sc, Sn)):
        M = np.column_stack([A[s] for s in seeds])
        cons = M.mean(1)
        for universe, mask in (("all", np.ones(n, bool)), ("count_supported", np.abs(M.mean(1)) >= 0 if name == "D" else np.abs(np.column_stack([Sc[s] for s in seeds]).mean(1)) >= 1e-6)):
            if universe == "count_supported":
                mask = np.abs(np.column_stack([Sc[s] for s in seeds]).mean(1)) >= 1e-6
            refA = {s: rho(M[mask, i], np.delete(M, i, 1).mean(1)[mask]) for i, s in enumerate(seeds)}
            refB = {p: rho(M[:, [seeds.index(q) for q in p]].mean(1)[mask], np.delete(M, [seeds.index(q) for q in p], 1).mean(1)[mask]) for p in itertools.combinations(seeds, 2)}
            refC = {p: rho(A[p[0]][mask], A[p[1]][mask]) for p in itertools.combinations(seeds, 2)}
            newA = {s: rho(B[s][mask], cons[mask]) for s in B}
            newB = rho(np.mean([B[s] for s in B], axis=0)[mask], cons[mask])
            for lab, keep in (("all seeds", seeds), ("integrated", [s for s in seeds if s in INTEGRATED]), ("poorly mixed", [s for s in seeds if s not in INTEGRATED])):
                a = [refA[s] for s in keep]
                b = [v for p, v in refB.items() if set(p) <= set(keep)]
                c = [v for p, v in refC.items() if set(p) <= set(keep)]
                rows.append(dict(readout=name, universe=universe, seeds_in_reference=lab, n_genes=int(mask.sum()), A_ref_min=min(a), A_ref_median=float(np.median(a)), A_ref_max=max(a),
                                 B_ref_min=min(b) if b else np.nan, B_ref_median=float(np.median(b)) if b else np.nan, B_ref_max=max(b) if b else np.nan,
                                 C_ref_min=min(c) if c else np.nan, C_ref_median=float(np.median(c)) if c else np.nan, C_ref_max=max(c) if c else np.nan,
                                 **{f"A_new_s{s}": v for s, v in newA.items()}, B_new=newB,
                                 **{f"C_new_s{s}_median": float(np.median([rho(B[s][mask], A[q][mask]) for q in keep])) for s in B}, new_s42_s43=rho(B[42][mask], B[43][mask])))
            res[f"{name}|{universe}"] = dict(newA=newA, newB=newB)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "comparison.csv", index=False)
    # The rule. The plan said "matching mixing state" without fixing a classifier, and entropy alone splits these two seeds against the other markers of T3_3_consensus
    # (adversary, SD of D, negative tail). So the rule is evaluated under every assignment of the two seeds to a state; the verdict is reported per assignment.
    states = ("integrated", "poorly mixed", "all seeds")
    D = R[(R.readout == "D") & (R.universe == "all")].set_index("seeds_in_reference")
    v = dict(n_shared=n, entropy_midpoint_label=state_new, assignments=[])
    allB = D.loc["all seeds"]
    v["mean_of_two"] = dict(rho=allB.B_new, ref_min=allB.B_ref_min, ref_median=allB.B_ref_median, ref_max=allB.B_ref_max, below_min=bool(allB.B_new < allB.B_ref_min))
    for a, b in itertools.product(states, states):
        x, y = D.loc[a], D.loc[b]
        below = [bool(x.A_new_s42 < x.A_ref_min), bool(y.A_new_s43 < y.A_ref_min)]
        v["assignments"].append(dict(s42=a, s43=b, s42_rho=x.A_new_s42, s42_min=x.A_ref_min, s42_max=x.A_ref_max, s43_rho=y.A_new_s43, s43_min=y.A_ref_min, s43_max=y.A_ref_max, both_below_min=all(below)))
    v["beyond_seed_range_any_assignment"] = bool(any(q["both_below_min"] for q in v["assignments"]) or v["mean_of_two"]["below_min"])
    (OUT / "verdict.json").write_text(json.dumps(v, indent=1))
    pd.set_option("display.width", 230, "display.max_columns", 40)
    print(json.dumps(v, indent=1))
    print(pd.DataFrame(hl).round(3).T.to_string())
    print(R.round(3).T.to_string())


if __name__ == "__main__":
    main()
