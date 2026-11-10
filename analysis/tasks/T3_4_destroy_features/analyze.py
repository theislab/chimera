"""Analysis of the control that keeps the counts and destroys the features. Run once the arm runs and the seed runs have finished.

For each arm run ``rev-ctl-<cohort>-<arm>-s<seed>`` (``manifest.tsv``), against the cohort
consensus of D over seeds 42-46 excluding the arm's own seed (``analysis/common/controls.py``):

* rho(D_arm, D_ref) on the Spearman scale and on the covariate-residual scale;
* rho(D_arm, S_ref) and rho(S_arm, S_ref);
* tail fraction |D| > 0.05, ``expectation_summary.fail`` and ``failed_expectations``.

Calls (pre-specified analysis plan, section 9), residual scale decides: ``preserved`` rho >= min F,
``lost`` rho < 0.5 * median F, ``partial`` otherwise, where F is the five leave-one-seed-out rho of
the cohort's seeds on the same scale. Multi-seed arms (liver H+Mouse Config M and random_all) also
get the call for the arm mean and integration metrics from ``training/compare_vs_scvi.py``
(k=15, seed=42) on ``z_mean`` from ``final_state.npz``. Single-seed arms are descriptive.

Config M outputs are read from ``final_state.npz`` (``ChimeraModel.load`` does not restore
``per_gene_correction``).

Usage: python analysis/tasks/T3_4_destroy_features/analyze.py [--drop liv_hmouse:43 ...] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids

HERE = Path(__file__).resolve().parent
TAIL = 0.05


def integration(run: str, adata) -> dict:
    import sys

    sys.path.insert(0, str(gene_ids.REPO_ROOT / "training"))
    from compare_vs_scvi import compute_integration_metrics

    z = np.load(C.RUNS_DIR / run / "final_state.npz")["z_mean"]
    m = compute_integration_metrics(z, adata.obs["species"].to_numpy(), adata.obs["cell_type"].to_numpy(),
                                    k=15, seed=42)
    g = m["global"]
    return {f"int_{k}": g[k] for k in ("knn_species_purity", "knn_mixing_entropy", "sil_celltype",
                                       "knn_celltype_purity", "label_transfer_acc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", nargs="*", default=[], help="cohort:seed pairs removed by the T3.1 rule")
    ap.add_argument("--outdir", type=Path, default=gene_ids.REPO_ROOT / "results/analysis/T3_4")
    a = ap.parse_args()
    drop: dict[str, tuple] = {}
    for d in a.drop:
        c, s = d.split(":")
        drop[c] = drop.get(c, ()) + (int(s),)
    man = pd.read_csv(HERE / "manifest.tsv", sep="\t")
    man["cohort"] = man["run_name"].str.extract(r"rev-ctl-(.+?)-(?:m|rand|zesm|zexpr)-s\d+$")[0]
    man["arm"] = man["run_name"].str.extract(r"-(m|rand|zesm|zexpr)-s\d+$")[0]
    man["seed"] = man["run_name"].str.extract(r"-s(\d+)$")[0].astype(int)

    rows, means, floors = [], [], []
    for cohort, grp in man.groupby("cohort"):
        seeds = K.seed_runs(cohort, drop=drop.get(cohort, ()))
        adata = C.read_adata(cohort)
        canon = C.CANONICAL[cohort]
        prior_raw = C.raw_prior(cohort, canon, adata.var["conservation_prior"].to_numpy(dtype=float))
        cov = K.cohort_covariates(cohort, adata=adata, prior_raw=prior_raw)
        ok = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        Z = K.design(cov.loc[ok])
        Dm = K.seed_matrix(cohort, seeds, "D")
        Sm = K.seed_matrix(cohort, seeds, "S_g_ema")
        F = K.floor(Dm.iloc[np.where(ok)[0]].reset_index(drop=True), Z)
        FS = K.floor(Sm, None)
        floors.append(dict(cohort=cohort, seeds=" ".join(map(str, Dm.columns)),
                           F_residual=" ".join(f"{x:.3f}" for x in F["residual"]),
                           F_spearman=" ".join(f"{x:.3f}" for x in F["spearman"]),
                           F_S_spearman=" ".join(f"{x:.3f}" for x in FS["spearman"])))
        for _, r in grp.iterrows():
            run = r["run_name"]
            if not K.run_finished(run):
                rows.append(dict(cohort=cohort, arm=r["arm"], seed=r["seed"], run=run, status="unfinished"))
                continue
            st = K.read_run(run)
            ref_D = K.consensus_excluding(Dm, r["seed"])
            ref_S = K.consensus_excluding(Sm, r["seed"])
            pD = K.rho_pair(st["D"][ok], ref_D[ok], Z)
            summ = st["summary"].get("expectation_summary", {})
            rows.append(dict(
                cohort=cohort, arm=r["arm"], seed=r["seed"], run=run, status="finished",
                rho_D_spearman=pD["spearman"], rho_D_residual=pD["residual"],
                call_residual=K.call(pD["residual"], F["residual"].to_numpy()),
                call_spearman=K.call(pD["spearman"], F["spearman"].to_numpy()),
                rho_D_vs_S_ref=float(spearmanr(st["D"], ref_S)[0]),
                rho_S_vs_S_ref=float(spearmanr(st["S_g_ema"], ref_S)[0]),
                call_S=K.call(float(spearmanr(st["S_g_ema"], ref_S)[0]), FS["spearman"].to_numpy()),
                tail_frac=float((np.abs(st["D"]) > TAIL).mean()),
                tail_frac_ref=float((np.abs(ref_D) > TAIL).mean()),
                expect_fail=summ.get("fail"), failed=" ".join(st["summary"].get("failed_expectations", [])),
            ))
        # multi-seed arms: arm mean vs full consensus, plus integration
        for arm, ag in grp.groupby("arm"):
            fin = [x for x in ag["run_name"] if K.run_finished(x)]
            if len(ag) < 2 or len(fin) < 2:
                continue
            Dmean = np.mean([K.read_run(x)["D"] for x in fin], 0)
            ref = Dm.mean(1).to_numpy()
            p = K.rho_pair(Dmean[ok], ref[ok], Z)
            m = dict(cohort=cohort, arm=arm, n_runs=len(fin), rho_D_spearman=p["spearman"],
                     rho_D_residual=p["residual"], call_residual=K.call(p["residual"], F["residual"].to_numpy()))
            ints = [integration(x, adata) for x in fin]
            for k in ints[0]:
                m[k + "_mean"] = float(np.mean([d[k] for d in ints]))
            m.update({k + "_canonical": v for k, v in integration(canon, adata).items()})
            means.append(m)
    K.write_csv(pd.DataFrame(rows), a.outdir / "arms.csv")
    K.write_csv(pd.DataFrame(means), a.outdir / "arm_means.csv")
    K.write_csv(pd.DataFrame(floors), a.outdir / "floors.csv")
    (a.outdir / "DONE").write_text("done\n")
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(means).to_string())


if __name__ == "__main__":
    main()
