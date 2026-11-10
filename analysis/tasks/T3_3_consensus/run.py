"""T3.1 seed-pooling rule and T3.3 consensus tables and reliability.

Definitions (pre-specified analysis plan, sections 2, 3 and 9):

* Per run, D = alpha - prior (prior as anchored by the run; clamped on testis recipes).
* Seed pooling: every seed's leave-one-seed-out rho against the mean of the others, on the
  covariate-residual scale. An MPS seed stays in D if its rho >= (lowest CPU value - 0.05).
  The CPU s42 platform runs (rev-cpu-*) join the leave-one-out pool as CPU references only;
  they never enter D, so seed 42 is not counted twice.
* Consensus D = mean over pooled seeds of (alpha - prior), minus the cohort median.
* rel(D|C) = 2r / (1 + r), r = mean over all 2-vs-3 splits of the five-seed core (s42-s46) of the
  correlation of residualised split means. With more than five seeds, also all 5-vs-5 splits
  (stepped up to the full seed count by the same formula), labelled "all seeds".
* Sign consistency per gene: fraction of pooled seeds whose (alpha - prior) has the sign of
  the consensus.

Usage: python analysis/tasks/T3_3_consensus/run.py [--cohorts ...] [--outdir results/analysis/T3_3]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids

MPS = {
    "liv_hmonkey": {42: "iter90-150ep-s42", 43: "iter90-150ep-s43", 44: "iter90-150ep-s44"},
    "liv_hmouse": {42: "xds-hmouse-iter90-150ep-s42", 43: "xds-hmouse-iter90-150ep-s43",
                   44: "xds-hmouse-iter90-150ep-s44"},
    "liv_hzeb": {42: "xds-hzeb-iter85-s42"},
    "liv_4sp": {42: "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42"},
    "test_hmac": {42: "testis-hmac-iter109-pclamp085-200ep-s42"},
    "test_hmouse": {42: "testis-hmouse-iter111-pclamp-coc-200ep-s42"},
    "test_3sp": {42: "testis-unified-iter115-pclamp-coc-100ep-s42"},
}
CPU_SEEDS = {
    "liv_hmouse": list(range(45, 52)),
    "test_hmouse": list(range(43, 52)),
    "liv_4sp": [43, 44, 45, 46],
    "test_3sp": [43, 44, 45, 46],
}
PLATFORM = {"liv_hmouse": "rev-cpu-liv_hmouse-s42", "test_hmouse": "rev-cpu-test_hmouse-s42"}
CORE = [42, 43, 44, 45, 46]


def cpu_runs(cohort: str) -> dict[int, str]:
    return {s: f"rev-{cohort}-s{s}" for s in CPU_SEEDS.get(cohort, [])}


def finished(runs: dict) -> dict:
    return {s: r for s, r in runs.items() if K.run_finished(r)}


def split_rel(M: np.ndarray, Z: np.ndarray, seeds: list, k: int) -> tuple[float, float, int]:
    """Mean residual correlation of k-vs-rest split means, and its step-up 2r/(1+r)."""
    idx = list(range(len(seeds)))
    rs = []
    seen = set()
    for a in itertools.combinations(idx, k):
        b = tuple(i for i in idx if i not in a)
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        ea = K.residual_rank(M[:, list(a)].mean(1), Z)
        eb = K.residual_rank(M[:, list(b)].mean(1), Z)
        rs.append(np.corrcoef(ea, eb)[0, 1])
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r), len(rs)


def topdecile_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    qa, qb = np.quantile(a, 0.9), np.quantile(b, 0.9)
    A, B = set(np.where(a >= qa)[0]), set(np.where(b >= qb)[0])
    return len(A & B) / len(A | B)


def run_cohort(cohort: str, outdir: Path) -> dict:
    mps = finished(MPS[cohort])
    cpu = finished(cpu_runs(cohort))
    plat = {}
    if cohort in PLATFORM and K.run_finished(PLATFORM[cohort]):
        plat = {"cpu42": PLATFORM[cohort]}
    adata = C.read_adata(cohort)
    canon = C.CANONICAL[cohort]
    prior_raw = C.raw_prior(cohort, canon, adata.var["conservation_prior"].to_numpy(dtype=float))
    cov = K.cohort_covariates(cohort, adata=adata, prior_raw=prior_raw)
    ok = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(cov.loc[ok])

    all_runs = {**{f"mps{s}": r for s, r in mps.items()}, **{f"cpu{s}": r for s, r in cpu.items()}, **plat}
    data = {k: K.read_run(r) for k, r in all_runs.items()}
    Dm = pd.DataFrame({k: v["D"] for k, v in data.items()})

    # Seed pooling (section 2), only when CPU references exist.
    pool = {"cohort": cohort, "mps": sorted(mps), "cpu": sorted(cpu), "platform": bool(plat)}
    keep_mps = sorted(mps)
    cpu_keys = [k for k in Dm.columns if k.startswith("cpu")]
    if len(Dm.columns) >= 3 and cpu_keys:
        F = K.floor(Dm.iloc[ok.nonzero()[0]].reset_index(drop=True), Z)
        F = F.set_index("seed")
        lo = float(F.loc[cpu_keys, "residual"].min())
        bar = lo - 0.05
        keep_mps = [s for s in mps if F.loc[f"mps{s}", "residual"] >= bar]
        pool.update(loo=F["residual"].round(4).to_dict(), lowest_cpu=round(lo, 4), bar=round(bar, 4),
                    dropped=[s for s in mps if s not in keep_mps])
    else:
        pool.update(note="no CPU reference; all MPS seeds kept")

    seeds = {s: mps[s] for s in keep_mps}
    seeds.update(cpu)
    seed_keys = sorted(seeds)

    def col(s, key):
        k = f"cpu{s}" if s in cpu else f"mps{s}"
        return data[k][key]

    A = np.column_stack([col(s, "alpha") for s in seed_keys])
    M = np.column_stack([col(s, "D") for s in seed_keys])
    S = np.column_stack([col(s, "S_g_ema") for s in seed_keys])
    Cc = np.column_stack([col(s, "correction") for s in seed_keys])
    P = np.column_stack([col(s, "prior") for s in seed_keys])
    mean_da = M.mean(1)
    D = mean_da - np.median(mean_da)
    sgn = np.sign(mean_da)
    sign_cons = (np.sign(M) == sgn[:, None]).mean(1)

    genes = pd.DataFrame({
        "hsapiens_gene_id": gene_ids.ensembl(cohort, adata.var_names).values,
        "symbol": gene_ids.symbols(cohort, adata.var_names).values,
        "var_name": np.asarray(adata.var_names),
        "prior": P.mean(1), "prior_raw": prior_raw,
        "alpha": A.mean(1), "correction": Cc.mean(1),
        "delta_alpha": mean_da, "sd_delta_alpha": M.std(1, ddof=1) if M.shape[1] > 1 else np.nan,
        "D": D, "S_ema": S.mean(1), "sign_consistency": sign_cons, "n_seeds": M.shape[1],
    })
    K.write_csv(genes, outdir / f"{cohort}__consensus.csv")

    rows = []
    for i, j in itertools.combinations(range(len(seed_keys)), 2):
        rows.append(dict(cohort=cohort, seed_a=seed_keys[i], seed_b=seed_keys[j],
                         spearman_D=spearmanr(M[:, i], M[:, j])[0],
                         residual_D=K.rho_pair(M[ok, i], M[ok, j], Z)["residual"],
                         spearman_alpha=spearmanr(A[:, i], A[:, j])[0],
                         spearman_S=spearmanr(S[:, i], S[:, j])[0],
                         sign_agreement=float((np.sign(M[:, i]) == np.sign(M[:, j])).mean()),
                         topdecile_jaccard_absD=topdecile_jaccard(np.abs(M[:, i]), np.abs(M[:, j]))))
    pairs = pd.DataFrame(rows)
    rel = dict(cohort=cohort, n_seeds=len(seed_keys), seeds=" ".join(map(str, seed_keys)))
    if len(pairs):
        for c in ("spearman_D", "residual_D", "spearman_alpha", "spearman_S", "sign_agreement",
                  "topdecile_jaccard_absD"):
            rel[f"pair_{c}_median"] = float(pairs[c].median())
            rel[f"pair_{c}_min"] = float(pairs[c].min())
    core = [s for s in CORE if s in seed_keys]
    if len(core) == 5:
        idx = [seed_keys.index(s) for s in core]
        r, rr, n = split_rel(M[ok][:, idx], Z, core, 2)
        rel.update(rel_core_r=r, rel_core=rr, rel_core_splits=n)
        Fl = K.floor(pd.DataFrame(M[ok][:, idx], columns=core), Z)
        rel.update(floor_residual_min=float(Fl["residual"].min()),
                   floor_residual_median=float(Fl["residual"].median()),
                   floor_spearman_min=float(Fl["spearman"].min()))
    if len(seed_keys) >= 10 and len(seed_keys) % 2 == 0:
        r, rr, n = split_rel(M[ok], Z, seed_keys, len(seed_keys) // 2)
        rel.update(rel_all_r=r, rel_all=rr, rel_all_splits=n)
    rel["frac_sign_consistent_ge_0.8"] = float((sign_cons >= 0.8).mean())
    pool["used"] = seed_keys
    return dict(pool=pool, rel=rel, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=list(MPS))
    ap.add_argument("--outdir", type=Path, default=gene_ids.REPO_ROOT / "results/analysis/T3_3")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    pools, rels, pairs = [], [], []
    for c in a.cohorts:
        print(f"== {c}", flush=True)
        res = run_cohort(c, a.outdir)
        pools.append(res["pool"]); rels.append(res["rel"]); pairs.append(res["pairs"])
        print(json.dumps(res["pool"]), flush=True)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["rel"].items()}),
              flush=True)
    K.write_csv(pd.DataFrame(rels), a.outdir / "reliability.csv")
    K.write_csv(pd.concat(pairs, ignore_index=True), a.outdir / "seed_pairs.csv")
    tmp = a.outdir / "seed_pooling.json.tmp"
    tmp.write_text(json.dumps(pools, indent=2))
    tmp.replace(a.outdir / "seed_pooling.json")
    if set(a.cohorts) == set(MPS):
        (a.outdir / "DONE").write_text("all cohorts\n")


if __name__ == "__main__":
    main()
