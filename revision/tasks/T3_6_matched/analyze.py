"""T3.6 analysis: matched cell types (liver 4-sp, Hepatocytes + Endothelial). Run after the run ends.

Compares ``rev-ctl-liv_4sp-matchA-s42`` with the liver 4-sp consensus of D excluding seed 42
(seeds 43-46 from T3.2; ``revision/common/controls.py``), on the Spearman scale and on the
covariate-residual scale (the matched run on covariates of its own h5ad, the consensus on the
full cohort's), for D and for S_g_ema. Calls (05 section 9), residual scale: ``preserved``
rho >= min F, ``lost`` rho < 0.5 * median F, ``partial`` otherwise, with F the five
leave-one-seed-out rho of liver 4-sp. If the call is ``partial``, two more seeds are added
before interpreting (T3.6 card). Reported beside the offline composition result S_bal
(``s_bal.py``); a two-type model has less contrast by design, so disagreement does not
automatically limit canonical claims. Also: landmark sign table, tail fractions, last
epoch and expectations.

Usage::

    python revision/tasks/T3_6_matched/analyze.py [--drop liv_4sp:43] [--allow-missing] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import cohorts as C
from revision.common import controls as K
from revision.common import gene_ids
from revision.common.outputs import TaskOutput, add_common_args

COHORT = "liv_4sp"
RUN = "rev-ctl-liv_4sp-matchA-s42"
SEED = 42
H5AD = gene_ids.REPO_ROOT / "data" / "revision" / "controls" / "liv_4sp_matchA.h5ad"
S_BAL = gene_ids.REPO_ROOT / "results" / "revision" / "T3_6" / "s_bal" / "s_bal_summary.csv"
LANDMARKS = ["CPS1", "ARG1", "ASL", "OTC", "ASS1", "NAGS", "PCK1", "PAH", "ALB", "APOA1", "HNF4A",
             "SELP", "SELE", "VCAM1", "F3", "PDGFRB", "COX4I1", "NDUFA1", "EEF1A1", "ACTB"]
TAIL = 0.05


def last_epoch(run: str):
    f = C.RUNS_DIR / run / "lightning" / "metrics.csv"
    if not f.exists():
        return None
    e = pd.read_csv(f, usecols=["epoch"])["epoch"].dropna()
    return int(e.max()) if len(e) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drop", nargs="*", default=[], help="liv_4sp:seed left out of D (T3.1 rule)")
    ap.add_argument("--allow-missing", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    outdir = args.outdir
    if args.allow_missing and not outdir:
        outdir = str(gene_ids.REPO_ROOT / "results" / "revision" / "T3_6" / "_precheck")
    out = TaskOutput("T3_6", outdir, prerun=args.prerun or args.allow_missing, force=args.force)
    drop = tuple(int(x.split(":")[1]) for x in args.drop if x.startswith(COHORT + ":"))
    if not K.run_finished(RUN):
        raise FileNotFoundError(f"{RUN} has not finished")
    runs = K.seed_runs(COHORT, drop, require_all=not args.allow_missing)
    MD = K.seed_matrix(COHORT, runs, "D")
    MS = K.seed_matrix(COHORT, runs, "S_g_ema")
    cov = K.cohort_covariates(COHORT)
    U = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(cov[U])
    a = ad.read_h5ad(H5AD)
    if not np.array_equal(a.var_names, cov.index):
        raise AssertionError("matched h5ad genes differ from the cohort")
    cov_m = K.cohort_covariates(COHORT, a)
    Um = cov_m[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    both = U & Um
    Zr, Za = K.design(cov[both]), K.design(cov_m[both])

    fl = pd.concat([K.floor(MD, Z, rows=np.where(U)[0]).assign(readout="D"),
                    K.floor(MS, Z, rows=np.where(U)[0]).assign(readout="S")], ignore_index=True)
    r = K.read_run(RUN)
    ref_seeds = [s for s in MD.columns if s != SEED]
    D_ref = MD[ref_seeds].mean(1).to_numpy()
    S_ref = MS[ref_seeds].mean(1).to_numpy()
    rec = dict(run=RUN, ref_seeds=";".join(map(str, ref_seeds)), last_epoch=last_epoch(RUN),
               expectation_fail=r["summary"]["expectation_summary"]["fail"],
               failed_expectations=";".join(r["summary"]["failed_expectations"]),
               tail_frac=float((np.abs(r["D"]) > TAIL).mean()), tail_frac_ref=float((np.abs(D_ref) > TAIL).mean()),
               n_residual=int(both.sum()))
    for key, arm, ref in (("D", r["D"], D_ref), ("S", r["S_g_ema"], S_ref)):
        res = K.rho_pair(arm[both], ref[both], Za, Zr)
        rec[f"{key}_spearman"] = float(pd.Series(arm).corr(pd.Series(ref), method="spearman"))
        rec[f"{key}_residual"] = res["residual"]
        for scale in ("spearman", "residual"):
            F = fl[fl.readout == key][scale].to_numpy()
            rec[f"floor_{key}_{scale}_min"] = float(F.min())
            rec[f"floor_{key}_{scale}_median"] = float(np.median(F))
            rec[f"call_{key}_{scale}"] = K.call(rec[f"{key}_{scale}"], F)
    rec["call"] = rec["call_D_residual"]
    rec["next_step"] = ("add seeds 43 and 44 of the matched run before interpreting" if rec["call"] == "partial"
                        else "interpret in T3.7")
    rec["canonical_D_spearman_with_s42_only"] = float(pd.Series(r["D"]).corr(pd.Series(MD[SEED]), method="spearman")) \
        if SEED in MD.columns else np.nan

    sym = gene_ids.symbols(COHORT, a.var_names).astype(str)
    lm_mask = sym.isin(LANDMARKS).to_numpy() | sym.str.startswith("MT-").to_numpy()
    lm = pd.DataFrame(dict(var_name=np.asarray(a.var_names)[lm_mask], symbol=sym.values[lm_mask],
                           D_consensus=D_ref[lm_mask], D_matched=r["D"][lm_mask],
                           S_consensus=S_ref[lm_mask], S_matched=r["S_g_ema"][lm_mask]))
    lm["sign_agree"] = np.sign(lm.D_consensus - np.median(D_ref)) == np.sign(lm.D_matched - np.median(r["D"]))
    rec["landmark_sign_agreement_centred"] = float(lm.sign_agree.mean())
    if S_BAL.exists():
        sb = pd.read_csv(S_BAL)
        sb = sb[sb.cohort == COHORT]
        if len(sb):
            for k in ("spearman_Sbal_S", "signflip_top_half", "var_share_shared", "rho_projSbal_c", "rho_projS_c"):
                rec[f"offline_{k}"] = float(sb.iloc[0][k])
    out.csv(pd.DataFrame([rec]), "matched_vs_canonical.csv")
    out.csv(fl, "floor.csv")
    out.csv(lm.sort_values("symbol"), "landmarks.csv")
    genes = pd.DataFrame(dict(var_name=np.asarray(a.var_names), symbol=sym.values, D_consensus=D_ref,
                              D_matched=r["D"], S_consensus=S_ref, S_matched=r["S_g_ema"]))
    out.csv(genes, "genes.csv")
    print(json.dumps(rec, indent=1, default=str))
    out.finish(dict(drop=args.drop, allow_missing=args.allow_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
