"""Prior-swap runs against same-seed canonical runs, against the seed floor. Control evidence
only: enters no scoreboard and no readout.

Same-seed pair: Spearman of c, Delta-alpha (each run against its own prior), S_g_ema, alpha; D also on
the residual scale of C. Floor: on the ten liv_hmouse seeds, leave-one-seed-out rho of each seed
against the mean of the others, and the cross-seed pair Spearman within each training state
(integrated 42, 43, 44, 47, 48, 51; poorly mixed 45, 46, 49, 50). Health: non-finite values, sign
profile against the run's own prior, species mixing entropy, final adversary accuracy.
With c held at the canonical same-seed value, alpha' = sigmoid(logit pi' + c).

    python analysis/tasks/T6_8_priorswap/analyze.py
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T3_3_consensus import run as T33
from analysis.tasks.T6_common import functional as FN

sys.path.insert(0, str(gene_ids.REPO_ROOT / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results/analysis/T6_8"
SEEDS = [45, 46, 47, 48]
INTEGRATED = FN.INTEGRATED_SEEDS
POOR = [45, 46, 49, 50]
ENTROPY_BAR = 0.253  # amendment log of the pre-specified analysis plan (2026-11-02)
COHORT = "liv_hmouse"
KEYS = ["correction", "D", "S_g_ema", "alpha"]


def spear(a, b) -> float:
    return float(spearmanr(a, b)[0])


def health(run: str, species, labels) -> dict:
    d = C.RUNS_DIR / run
    st = C.load_state(run)
    D = np.asarray(st["alpha"], float) - np.asarray(st["prior"], float)
    z = np.asarray(st["z_mean"], float)
    m = compute_integration_metrics(z, species, labels, k=15, seed=42)["global"]
    finite = all(np.isfinite(np.asarray(st[k], float)).all() for k in ("alpha", "prior", "correction", "S_g_ema", "z_mean"))
    return dict(run=run, finite=finite, frac_pos=float((D > 0).mean()), frac_neg=float((D < 0).mean()),
                frac_gt_005=float((D > 0.05).mean()), frac_lt_m005=float((D < -0.05).mean()),
                sd_D=float(D.std()), median_D=float(np.median(D)),
                knn_mixing_entropy=float(m["knn_mixing_entropy"]), knn_species_purity=float(m["knn_species_purity"]),
                adversary=float(np.asarray(st["adversarial_accuracy_final"])),
                state="integrated" if m["knn_mixing_entropy"] >= ENTROPY_BAR else "poorly_mixed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    adata = C.read_adata(COHORT, backed=True)
    species = adata.obs["species"].astype(str).to_numpy()
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    adata.file.close()
    canon = T33.cpu_runs(COHORT)
    all_seed_runs = {**T33.MPS[COHORT], **canon}
    swap = {s: f"rev-ctl-liv_hmouse-priorswap-s{s}" for s in SEEDS}
    H = pd.DataFrame([dict(kind="priorswap", seed=s, **health(r, species, labels)) for s, r in swap.items()]
                     + [dict(kind="canonical", seed=s, **health(canon[s], species, labels)) for s in SEEDS])
    K.write_csv(H, OUT / "health.csv")
    print(H.round(3).to_string(index=False), flush=True)
    state = dict(zip(zip(H.kind, H.seed), H.state))

    cov = K.cohort_covariates(COHORT)
    ok = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(cov.loc[ok])
    canon_d = {s: K.read_run(r) for s, r in all_seed_runs.items()}
    swap_d = {s: K.read_run(r) for s, r in swap.items()}
    rows = []
    for s in SEEDS:
        a, b = swap_d[s], canon_d[s]
        r = dict(seed=s, state_canonical=state[("canonical", s)], state_swap=state[("priorswap", s)])
        for k in KEYS:
            r[f"rho_{k}"] = spear(a[k], b[k])
        r["residual_D"] = K.rho_pair(a["D"][ok], b["D"][ok], Z)["residual"]
        ap = expit(logit(a["prior"]) + b["correction"])
        r["rho_alpha_pred_c_fixed"] = spear(ap, a["alpha"])
        r["rho_D_pred_c_fixed"] = spear(ap - a["prior"], a["D"])
        r["rho_D_pred_vs_canonical_D"] = spear(ap - a["prior"], b["D"])
        r["prior_rho"] = spear(a["prior"], b["prior"])
        rows.append(r)
    P = pd.DataFrame(rows)
    K.write_csv(P, OUT / "same_seed_pairs.csv")
    print(P.round(3).T.to_string(), flush=True)

    fl = []
    seeds = sorted(canon_d)
    for k in KEYS:
        M = pd.DataFrame({s: canon_d[s][k] for s in seeds})
        F = K.floor(M.iloc[ok.nonzero()[0]].reset_index(drop=True), Z).set_index("seed")
        for name, grp in (("integrated", INTEGRATED), ("poorly_mixed", POOR), ("all", seeds)):
            pr = [spear(M[i], M[j]) for i, j in itertools.combinations(grp, 2)]
            fl.append(dict(key=k, group=name, n_pairs=len(pr), pair_median=float(np.median(pr)), pair_min=float(np.min(pr)),
                           pair_max=float(np.max(pr)), loo_min=float(F.loc[grp, "spearman"].min()),
                           loo_median=float(F.loc[grp, "spearman"].median()), loo_residual_min=float(F.loc[grp, "residual"].min())))
    Fl = pd.DataFrame(fl)
    between = []
    for k in KEYS:
        M = pd.DataFrame({s: canon_d[s][k] for s in seeds})
        pr = [spear(M[i], M[j]) for i in INTEGRATED for j in POOR]
        between.append(dict(key=k, group="between_states", n_pairs=len(pr), pair_median=float(np.median(pr)),
                            pair_min=float(np.min(pr)), pair_max=float(np.max(pr))))
    Fl = pd.concat([Fl, pd.DataFrame(between)], ignore_index=True)
    K.write_csv(Fl, OUT / "seed_floor.csv")
    print(Fl.round(3).to_string(index=False), flush=True)

    # swap consensus over all four against seed reference; swap-vs-other-seed comparison on the same pairs
    cross = []
    for s, t in itertools.product(SEEDS, SEEDS):
        if s == t:
            continue
        for k in KEYS:
            cross.append(dict(swap_seed=s, canonical_seed=t, key=k, rho=spear(swap_d[s][k], canon_d[t][k]),
                              same_state=state[("priorswap", s)] == state[("canonical", t)]))
    Cr = pd.DataFrame(cross)
    K.write_csv(Cr, OUT / "swap_vs_other_seed.csv")
    print(Cr.groupby(["key", "same_state"]).rho.median().round(3), flush=True)

    # Same seed, same prior, different platform: the only no-swap same-seed reference on this cohort
    plat_a, plat_b = K.read_run(T33.MPS[COHORT][42]), K.read_run(T33.PLATFORM[COHORT])
    plat = dict(pair="MPS s42 vs CPU s42", **{f"rho_{k}": spear(plat_a[k], plat_b[k]) for k in KEYS},
                entropy_cpu=health(T33.PLATFORM[COHORT], species, labels)["knn_mixing_entropy"],
                entropy_mps=health(T33.MPS[COHORT][42], species, labels)["knn_mixing_entropy"])
    print(plat, flush=True)
    # Does the change in c follow the change in the prior?
    dp = []
    for s in SEEDS:
        a, b = swap_d[s], canon_d[s]
        d_pi = logit(a["prior"]) - logit(b["prior"])
        dp.append(dict(seed=s, rho_dc_dlogitpi=spear(a["correction"] - b["correction"], d_pi),
                       rho_dD_dpi=spear(a["D"] - b["D"], a["prior"] - b["prior"])))
    DP = pd.DataFrame(dp)
    print(DP.round(3).to_string(index=False), flush=True)
    # Consensus level: mean of the four swap runs against the same four canonical seeds, and the floor
    # of disjoint canonical groups of four, two of each state (the canonical composition of 45-48).
    four = []
    swapM = {k: np.mean([swap_d[s][k] for s in SEEDS], 0) for k in KEYS}
    canM = {k: np.mean([canon_d[s][k] for s in SEEDS], 0) for k in KEYS}
    for k in KEYS:
        M = pd.DataFrame({s: canon_d[s][k] for s in seeds})
        fl4 = []
        for i2 in itertools.combinations(INTEGRATED, 2):
            for p2 in itertools.combinations(POOR, 2):
                g1 = list(i2) + list(p2)
                rest_i = [x for x in INTEGRATED if x not in i2]
                rest_p = [x for x in POOR if x not in p2]
                for j2 in itertools.combinations(rest_i, 2):
                    g2 = list(j2) + rest_p
                    fl4.append(spear(M[g1].mean(1), M[g2].mean(1)))
        four.append(dict(key=k, rho_swap4_vs_canon4=spear(swapM[k], canM[k]), floor4_min=float(np.min(fl4)),
                         floor4_median=float(np.median(fl4)), floor4_max=float(np.max(fl4)), n_splits=len(fl4)))
    F4 = pd.DataFrame(four)
    K.write_csv(F4, OUT / "consensus_four_seeds.csv")
    print(F4.round(3).to_string(index=False), flush=True)
    K.write_csv(pd.DataFrame([plat]), OUT / "platform_reference.csv")
    K.write_csv(DP, OUT / "dc_vs_dpi.csv")

    calls = []
    for _, r in P.iterrows():
        grp = "integrated" if r.state_canonical == "integrated" else "poorly_mixed"
        for k in KEYS:
            f = Fl[(Fl.key == k) & (Fl.group == grp)].iloc[0]
            rho = r[f"rho_{k}"]
            calls.append(dict(seed=r.seed, key=k, rho=rho, floor_group=grp, floor_pair_min=f.pair_min, floor_pair_median=f.pair_median,
                              above_pair_median=bool(rho >= f.pair_median), above_pair_max=bool(rho >= f.pair_max)))
    Cl = pd.DataFrame(calls)
    K.write_csv(Cl, OUT / "calls.csv")
    print(Cl.round(3).to_string(index=False), flush=True)
    (OUT / "_provenance.json").write_text(json.dumps(dict(git_sha=git_sha(), runs=swap)))


if __name__ == "__main__":
    main()
