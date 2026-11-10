"""Controlled testis label ablation (Family 3; amendment log of the pre-specified analysis plan, 2026-11-02). Canonical testis recipes with only the
cell-type-conditional MMD removed (same MMD weight, then global), seeds 42-44, against the labeled runs.

Runs (manifest.tsv): rev-ctl-<cohort>-nolab-s{42,43,44} for test_hmac, test_hmouse, test_3sp. Labeled
comparators: canonical s42 (MPS) on every cohort; CPU seeds rev-test_hmouse-s43..s51 and rev-test_3sp-s43..s46
(T3_2_seeds); the CPU s42 platform run rev-cpu-test_hmouse-s42. test_hmac has no labeled seed beyond s42.

Sections (``--sections``, in this order; the first three read no gold):

health       per run: non-finite values; active z dims (Var over cells of z_mean >= 0.01, of 10); alpha
             saturation (> 0.95, < 0.05); tails of alpha - prior_raw (> +0.05, < -0.05; raw prior, unclamped)
             and of alpha - prior (clamped); final adversary accuracy; species mixing entropy; last epoch;
             failed expectations from summary.json; config keys that differ from the canonical recipe.
integration  compute_integration_metrics (training/compare_vs_scvi.py, unmodified; k=15, seed=42;
             obs species, obs cell_type as in the master table) on z_mean of every label-free run and every
             labeled same-seed run. Composite: T5_5_benchmark score_table (direction correction, per-row min-max,
             0.4 mixing + 0.6 biology) on the 10-model panel of master_table_ext.csv. The 10-model composite
             is rebuilt from its raw columns and must equal the stored one (unchanged). Each run is then
             added as an 11th column, one run at a time ("extra"); a second view replaces the Chimera column
             by the run ("substitute", 10 models).
readouts     per cohort: Spearman of D = alpha - prior (clamped), D_raw = alpha - prior_raw, S_g (S_g_ema),
             alpha and the MLP correction between label-free and labeled runs: same seed; 3-seed means
             (42-44) of both; label-free 3-seed mean against the labeled all-seed consensus. Residual scale:
             rank, residualize on C (controls.design), Pearson. Floors: labeled cross-seed pairs, labeled
             leave-one-seed-out, labeled split means of disjoint seed sets (3 vs 3 of 42-47 on H+Mouse;
             3 vs 2 of 42-46 on 3-sp), and the same pairs among label-free seeds. Call (controls.call) of a
             same-seed rho against the labeled cross-seed pair rhos, of the 3-seed-mean rho against the
             labeled split-mean rhos. Sign profile and Spearman with log mean expression (species-mean
             log1p CP10K, covariate log_cp10k_spmean).
validity     Family 3, cannot change any status, row or modifier. rho_p as T4_3_scoreboard (stats.py; T4_3_scoreboard cache for
             covariates, tau and ESM blocks; T4_3_scoreboard gold table) of D and S_g with H2 (EVEE level, C) and H4
             (Murat pattern, C_pat + H4_nsp; primates C_pat + H4_npri beside it) on testis H+Mouse, and H2,
             H4 on testis 3-sp (sensitivity, as the T4_3_scoreboard multi-species estimate). Scores: labeled
             consensus D and S (the scoreboard's, must reproduce T4_3_scoreboard), labeled 3-seed mean (42-44, same
             seeds), label-free 3-seed mean, D_raw for both, and each label-free seed. One block bootstrap
             (10,000 draws, the T4_3_scoreboard unit's seed) for all scores of a unit, so differences label-free minus
             labeled have percentile CIs. D is the 3-seed mean of alpha - prior minus its median.
x6           X6_adds_beyond label-degradation section (testis H+Mouse) rerun with the label-free readouts: X6_adds_beyond is first
             reproduced on its own universe; then on one universe where the X6_adds_beyond scores, the labeled 3-seed
             D and S, the label-free 3-seed D, D_raw and S, and B1z on the label-free s42 latent are all
             defined, every score on the same 2,000 draws (X6_adds_beyond joint), noise replicates averaged per draw.
             B1z_nolab: T4_2_gene_baselines B1z (15-NN, Leiden resolution 1, seed 0, pseudobulk per library, E) on z_mean
             of rev-ctl-test_hmouse-nolab-s42; the method is checked on the canonical z_mean against T4_2_gene_baselines.

Outputs: results/analysis/T5_4b/*.csv, DONE last.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python analysis/tasks/T5_4b_label_ablation/analyze.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T3_3_consensus import run as T33

REPO = gene_ids.REPO_ROOT
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T5_5_benchmark"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

OUT = REPO / "results" / "analysis" / "T5_4b"
COHORTS = ["test_hmac", "test_hmouse", "test_3sp"]
SEEDS = [42, 43, 44]
METRICS = ["knn_species_purity", "knn_mixing_entropy", "sil_species", "knn_celltype_purity", "sil_celltype",
           "label_transfer_acc"]
KEYS = ["D", "D_raw", "S_g_ema", "alpha", "correction"]
N_DRAWS = 10_000


def nolab(c: str, s: int) -> str:
    return f"rev-ctl-{c}-nolab-s{s}"


def labelled(c: str) -> dict[int, str]:
    """All labeled seed runs of the canonical recipe (s42 canonical MPS, CPU seeds of T3_2_seeds)."""
    out = {42: C.CANONICAL[c]}
    out.update(T33.cpu_runs(c))
    return {s: r for s, r in out.items() if K.run_finished(r)}


EXTRA_LAB = {"test_hmouse": {"cpu42": "rev-cpu-test_hmouse-s42"}}


def runs_table() -> pd.DataFrame:
    rows = []
    for c in COHORTS:
        for s in SEEDS:
            rows.append(dict(cohort=c, arm="label_free", seed=s, run=nolab(c, s)))
        for s, r in labelled(c).items():
            if s in SEEDS:
                rows.append(dict(cohort=c, arm="labelled", seed=s, run=r))
        for k, r in EXTRA_LAB.get(c, {}).items():
            rows.append(dict(cohort=c, arm="labelled_cpu42", seed=42, run=r))
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, name: str) -> None:
    K.write_csv(df, OUT / name)
    print(f"  wrote {name} ({len(df)} rows)", flush=True)


# ====================================================================== health and integration
def _obs(c: str):
    a = C.read_adata(c, backed=True)
    o = a.obs[["species", "cell_type"]].astype(str).copy()
    a.file.close()
    return o


def _integration(job):
    c, run = job
    o = _obs(c)
    z = np.load(C.RUNS_DIR / run / "final_state.npz")["z_mean"]
    if z.shape[0] != len(o):
        raise RuntimeError(f"{run}: z has {z.shape[0]} rows, obs {len(o)}")
    m = compute_integration_metrics(z, o["species"].to_numpy(), o["cell_type"].to_numpy(), k=15, seed=42)
    return c, run, {k: float(m["global"][k]) for k in METRICS}


def integration_metrics(rt: pd.DataFrame) -> pd.DataFrame:
    jobs = [(r.cohort, r.run) for r in rt.itertuples()]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(_integration, jobs))
    return pd.DataFrame([dict(cohort=c, run=r, **m) for c, r, m in res])


def config_diff(run: str, canon: str) -> str:
    a = json.loads((C.RUNS_DIR / run / "config.json").read_text())
    b = json.loads((C.RUNS_DIR / canon / "config.json").read_text())
    skip = {"run_name", "started_at", "seed"}
    d = []
    for k in sorted(set(a) | set(b)):
        if k in skip or a.get(k) == b.get(k):
            continue
        if k == "use_per_gene_bias" and not a.get(k) and not b.get(k):  # None and False: both off
            continue
        d.append(f"{k}={a.get(k)!r} (canonical {b.get(k)!r})")
    return "; ".join(d)


def health_row(c: str, run: str, prior_raw: np.ndarray) -> dict:
    d = C.RUNS_DIR / run
    st = C.load_state(run)
    summ = json.loads((d / "summary.json").read_text())
    alpha = np.asarray(st["alpha"], float)
    prior = np.asarray(st["prior"], float)
    z = np.asarray(st["z_mean"], float)
    finite = all(np.isfinite(np.asarray(st[k], float)).all() for k in ("alpha", "prior", "correction", "S_g_ema", "z_mean"))
    zvar = z.var(0)
    D, Dr = alpha - prior, alpha - prior_raw
    raw_file = np.load(d / "conservation_prior_raw.npy")
    r = dict(cohort=c, run=run, finite=finite, nan_failed_E2_1="E2.1" in summ.get("failed_expectations", []),
             n_active_z=int((zvar >= 0.01).sum()), n_z=int(z.shape[1]), z_var_min=float(zvar.min()),
             sat_high=float((alpha > 0.95).mean()), sat_low=float((alpha < 0.05).mean()),
             raw_tail_up=float((Dr > 0.05).mean()), raw_tail_dn=float((Dr < -0.05).mean()),
             clamped_tail_up=float((D > 0.05).mean()), clamped_tail_dn=float((D < -0.05).mean()),
             adversary=float(np.asarray(st["adversarial_accuracy_final"])),
             prior_raw_matches=bool(np.allclose(raw_file, prior_raw, atol=1e-6)),
             sg_frac_positive=float((np.asarray(st["S_g_ema"]) > 0).mean()),
             n_failed_expectations=len(summ.get("failed_expectations", [])),
             failed_expectations=",".join(summ.get("failed_expectations", [])),
             config_vs_canonical=config_diff(run, C.CANONICAL[c]))
    mets = sorted(d.glob("lightning/**/metrics.csv"))
    if mets:
        m = pd.read_csv(mets[-1])
        r["last_epoch"] = int(m["epoch"].max()) + 1 if m["epoch"].notna().any() else None
        num = m.select_dtypes("number").drop(columns=[x for x in ("epoch", "step") if x in m], errors="ignore")
        r["metrics_nonfinite_cells"] = int((~np.isfinite(num.to_numpy(float)) & num.notna().to_numpy()).sum())
    r["max_epochs"] = st["config"].get("max_epochs")
    rs = d / "run_summary.json"
    if rs.exists():
        r["elapsed_h"] = round(json.loads(rs.read_text()).get("elapsed_seconds", float("nan")) / 3600, 2)
    return r


def prior_raw_of(c: str) -> np.ndarray:
    a = C.read_adata(c, backed=True)
    p = a.var["conservation_prior"].to_numpy(dtype=float)
    a.file.close()
    return C.raw_prior(c, C.CANONICAL[c], p)


def sec_health_integration():
    from score_external import score_table  # T5_5_benchmark, unmodified

    rt = runs_table()
    write(rt, "runs.csv")
    print("integration metrics", flush=True)
    im = integration_metrics(rt)
    H = []
    for c in COHORTS:
        pr = prior_raw_of(c)
        for r in rt[rt.cohort == c].itertuples():
            H.append(dict(arm=r.arm, seed=r.seed, **health_row(c, r.run, pr)))
    H = pd.DataFrame(H).merge(im[["run", "knn_mixing_entropy"]], on="run")
    write(H, "health.csv")

    master = pd.read_csv(REPO / "notebooks/cache/baselines_master_ext/master_table_ext.csv", float_precision="round_trip")
    integ, checks = [], {}
    for c in COHORTS:
        mt = master[master.dataset == c].set_index("model")
        raw10 = mt[[f"raw__{m}" for m in METRICS]].rename(columns=lambda x: x[5:])
        sc10 = score_table(raw10)
        checks[f"{c}_10model_composite_max_abs_diff"] = float((sc10["total_score"] - mt["total_score"]).abs().max())
        canon_raw = im[im.run == C.CANONICAL[c]].iloc[0]
        checks[f"{c}_canonical_s42_metrics_vs_master_max_abs"] = float(
            max(abs(canon_raw[m] - raw10.loc["Chimera", m]) for m in METRICS))
        for r in rt[rt.cohort == c].itertuples():
            row = im[im.run == r.run].iloc[0]
            v = pd.Series({m: row[m] for m in METRICS})
            ext = pd.concat([raw10, v.to_frame("run").T])
            s11 = score_table(ext)
            sub = raw10.copy()
            sub.loc["Chimera"] = v
            s10 = score_table(sub)
            rank11 = int(s11["total_score"].rank(ascending=False, method="min")["run"])
            integ.append(dict(cohort=c, arm=r.arm, seed=r.seed, run=r.run, **{m: float(v[m]) for m in METRICS},
                              composite_extra=float(s11.loc["run", "total_score"]),
                              mixing_extra=float(s11.loc["run", "mixing"]), biology_extra=float(s11.loc["run", "biology"]),
                              rank_extra_of_11=rank11, canonical_chimera_in_11=float(s11.loc["Chimera", "total_score"]),
                              best_other_in_11=s11.drop(index="run")["total_score"].idxmax(),
                              wins_vs_pca=int(s11.loc["run", "wins_vs_pca"]),
                              composite_substitute=float(s10.loc["Chimera", "total_score"]),
                              rank_substitute_of_10=int(s10["total_score"].rank(ascending=False, method="min")["Chimera"]),
                              composite_10model_canonical=float(mt.loc["Chimera", "total_score"])))
    I = pd.DataFrame(integ)
    write(I, "integration.csv")
    # 7-row panel means: liver rows as stored, the three testis rows with Chimera replaced by one label-free
    # seed (substitute view); seed 42 canonical row = the stored table
    pm = []
    for variant in ["canonical"] + [f"label_free s{s}" for s in SEEDS]:
        tabs = []
        for ds, mt in master.groupby("dataset", sort=False):
            mt = mt.set_index("model")
            raw = mt[[f"raw__{m}" for m in METRICS]].rename(columns=lambda x: x[5:])
            if variant != "canonical" and ds in COHORTS:
                s = int(variant.split("s")[-1])
                row = im[im.run == nolab(ds, s)].iloc[0]
                raw.loc["Chimera"] = [row[m] for m in METRICS]
            sc = score_table(raw)
            tabs.append(pd.DataFrame(dict(dataset=ds, tissue=mt["tissue"], total=sc["total_score"])))
        T = pd.concat(tabs).rename_axis("model").reset_index()
        means = T.groupby("model")["total"].mean()
        tm = T[T.tissue == "testis"].groupby("model")["total"].mean()
        lm = T[T.tissue == "liver"].groupby("model")["total"].mean()
        won = T.loc[T.groupby("dataset")["total"].idxmax(), "model"].value_counts()
        for m in means.index:
            pm.append(dict(variant=variant, model=m, mean_7=float(means[m]), mean_liver=float(lm[m]),
                           mean_testis=float(tm[m]), rank_7=int(means.rank(ascending=False, method="min")[m]),
                           rows_won=int(won.get(m, 0))))
    write(pd.DataFrame(pm), "integration_panel_means.csv")
    # per cohort and arm, mean over seeds 42-44
    num = METRICS + ["composite_extra", "composite_substitute"]
    agg = I[I.arm.isin(["label_free", "labelled"])].groupby(["cohort", "arm"])[num].agg(["mean", "std", "min", "max", "count"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    write(agg.reset_index(), "integration_summary.csv")
    (OUT / "integration_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if max(v for k, v in checks.items() if "10model" in k) > 1e-12:
        raise SystemExit("10-model composite not reproduced")


# ====================================================================== readouts
def read_runs(runs: dict, prior_raw: np.ndarray) -> dict:
    out = {}
    for k, r in runs.items():
        d = K.read_run(r)
        d["D_raw"] = d["alpha"] - prior_raw
        out[k] = d
    return out


def residual(a, b, Z, ok):
    return K.rho_pair(np.asarray(a)[ok], np.asarray(b)[ok], Z)["residual"]


def sec_readouts():
    pairs, floors, signs, prior_checks = [], [], [], {}
    for c in COHORTS:
        cov = K.cohort_covariates(c, prior_raw=prior_raw_of(c))
        pr = cov["prior_raw"].to_numpy(float)
        ok = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        Z = K.design(cov.loc[ok])
        logm = cov["log_cp10k_spmean"].to_numpy(float)
        lab_runs = labelled(c)
        NL = read_runs({s: nolab(c, s) for s in SEEDS}, pr)
        LB = read_runs(lab_runs, pr)
        LB.update(read_runs({k: r for k, r in EXTRA_LAB.get(c, {}).items()}, pr))
        prior_checks[c] = float(max(np.abs(NL[s]["prior"] - LB[42]["prior"]).max() for s in SEEDS))
        lab3 = [s for s in SEEDS if s in LB]
        lab_int = sorted(s for s in LB if isinstance(s, int))
        _CACHE[c] = dict(ok=ok, Z=Z, runs={s: LB[s] for s in lab_int})

        def mean_of(d, ss, k):
            return np.mean([d[s][k] for s in ss], 0)

        # pairs
        for k in KEYS:
            for s in SEEDS:
                for ref in ([s] if s in LB else []) + (["cpu42"] if s == 42 and "cpu42" in LB else []):
                    a, b = NL[s][k], LB[ref][k]
                    pairs.append(dict(cohort=c, key=k, comparison="same_seed", label_free=f"s{s}", labelled=f"s{ref}" if ref != "cpu42" else "cpu42",
                                      spearman=float(spearmanr(a, b)[0]), residual=residual(a, b, Z, ok)))
            if len(lab3) >= 2:
                a, b = mean_of(NL, SEEDS, k), mean_of(LB, lab3, k)
                pairs.append(dict(cohort=c, key=k, comparison="mean3_vs_mean3", label_free="mean 42-44",
                                  labelled=f"mean {lab3[0]}-{lab3[-1]}", spearman=float(spearmanr(a, b)[0]),
                                  residual=residual(a, b, Z, ok)))
            if len(lab_int) >= 3:
                a, b = mean_of(NL, SEEDS, k), mean_of(LB, lab_int, k)
                pairs.append(dict(cohort=c, key=k, comparison="mean3_vs_labelled_all", label_free="mean 42-44",
                                  labelled=f"mean of {len(lab_int)} seeds", spearman=float(spearmanr(a, b)[0]),
                                  residual=residual(a, b, Z, ok)))
            # floors
            for arm, dd, ss in (("labelled", LB, lab_int), ("label_free", NL, SEEDS)):
                if len(ss) < 2:
                    continue
                pr_s = [float(spearmanr(dd[i][k], dd[j][k])[0]) for i, j in itertools.combinations(ss, 2)]
                pr_r = [residual(dd[i][k], dd[j][k], Z, ok) for i, j in itertools.combinations(ss, 2)]
                row = dict(cohort=c, key=k, arm=arm, seeds=",".join(map(str, ss)), n_pairs=len(pr_s),
                           pair_spearman_min=min(pr_s), pair_spearman_median=float(np.median(pr_s)), pair_spearman_max=max(pr_s),
                           pair_residual_min=min(pr_r), pair_residual_median=float(np.median(pr_r)), pair_residual_max=max(pr_r))
                if arm == "labelled":
                    p42 = [(i, j) for i, j in itertools.combinations(ss, 2) if i in SEEDS and j in SEEDS]
                    if p42:
                        row["pair_spearman_42_44"] = ";".join(f"{i}-{j}:{spearmanr(dd[i][k], dd[j][k])[0]:.4f}" for i, j in p42)
                    if len(ss) >= 3:
                        M = pd.DataFrame({s: dd[s][k] for s in ss})
                        F = K.floor(M.iloc[ok.nonzero()[0]].reset_index(drop=True), Z)
                        row.update(loo_spearman_min=float(F.spearman.min()), loo_residual_min=float(F.residual.min()),
                                   loo_residual_median=float(F.residual.median()))
                    pool = [s for s in ss if s <= 47][:6] if c == "test_hmouse" else [s for s in ss if s <= 46]
                    if len(pool) >= 5:
                        sp_s, sp_r = [], []
                        seen = set()
                        for A in itertools.combinations(pool, 3):
                            B = tuple(x for x in pool if x not in A)
                            if frozenset([A, B]) in seen:
                                continue
                            seen.add(frozenset([A, B]))
                            ma, mb = mean_of(dd, A, k), mean_of(dd, B, k)
                            sp_s.append(float(spearmanr(ma, mb)[0]))
                            sp_r.append(residual(ma, mb, Z, ok))
                        row.update(split_seeds=",".join(map(str, pool)), n_splits=len(sp_s),
                                   split_spearman_min=min(sp_s), split_spearman_median=float(np.median(sp_s)),
                                   split_residual_min=min(sp_r), split_residual_median=float(np.median(sp_r)))
                floors.append(row)
        # sign profile and expression
        units = {f"label_free s{s}": NL[s] for s in SEEDS}
        units.update({f"labelled s{s}": LB[s] for s in lab3})
        if "cpu42" in LB:
            units["labelled cpu42"] = LB["cpu42"]
        units["label_free mean 42-44"] = {k: mean_of(NL, SEEDS, k) for k in KEYS}
        if len(lab3) >= 2:
            units[f"labelled mean {lab3[0]}-{lab3[-1]}"] = {k: mean_of(LB, lab3, k) for k in KEYS}
        if len(lab_int) >= 3:
            units[f"labelled mean of {len(lab_int)} seeds"] = {k: mean_of(LB, lab_int, k) for k in KEYS}
        for name, d in units.items():
            D, Dr, Sg = d["D"], d["D_raw"], d["S_g_ema"]
            signs.append(dict(cohort=c, unit=name, frac_D_pos=float((D > 0).mean()), frac_Draw_pos=float((Dr > 0).mean()),
                              frac_Draw_gt_005=float((Dr > 0.05).mean()), frac_Draw_lt_m005=float((Dr < -0.05).mean()),
                              mean_Draw=float(Dr.mean()), sd_Draw=float(Dr.std()), mean_D=float(D.mean()), sd_D=float(D.std()),
                              frac_S_pos=float((Sg > 0).mean()), frac_S_zero=float((Sg == 0).mean()),
                              rho_D_logmean=float(spearmanr(D, logm)[0]), rho_Draw_logmean=float(spearmanr(Dr, logm)[0]),
                              rho_S_logmean=float(spearmanr(Sg, logm)[0]), rho_D_S=float(spearmanr(D, Sg)[0])))
    P = pd.DataFrame(pairs)
    F = pd.DataFrame(floors)
    # calls against the labeled floor (residual scale)
    calls = []
    for r in P.itertuples():
        f = F[(F.cohort == r.cohort) & (F.key == r.key) & (F.arm == "labelled")]
        if f.empty:
            continue
        f = f.iloc[0]
        if r.comparison == "same_seed":
            lab = [s for s in labelled(r.cohort)]
            ss = sorted(s for s in lab if isinstance(s, int))
            # reference: labeled cross-seed pair residual rhos
            pr = _pair_residuals(r.cohort, r.key, ss)
            calls.append(dict(cohort=r.cohort, key=r.key, comparison=r.comparison, label_free=r.label_free,
                              labelled=r.labelled, residual=r.residual, floor="labelled cross-seed pairs",
                              floor_min=float(np.min(pr)), floor_median=float(np.median(pr)), call=K.call(r.residual, pr)))
        elif r.comparison == "mean3_vs_mean3" and "split_residual_min" in f and np.isfinite(f.get("split_residual_min", np.nan)):
            sr = _split_residuals(r.cohort, r.key)
            calls.append(dict(cohort=r.cohort, key=r.key, comparison=r.comparison, label_free=r.label_free,
                              labelled=r.labelled, residual=r.residual, floor="labelled split means",
                              floor_min=float(np.min(sr)), floor_median=float(np.median(sr)), call=K.call(r.residual, sr)))
    write(P, "readout_pairs.csv")
    write(F, "readout_floor.csv")
    write(pd.DataFrame(calls), "readout_calls.csv")
    write(pd.DataFrame(signs), "readout_sign_expression.csv")
    (OUT / "readout_checks.json").write_text(json.dumps({"max_abs_prior_diff_label_free_vs_canonical": prior_checks}, indent=2))


_CACHE: dict = {}


def _cohort_ctx(c):
    return _CACHE[c]  # filled by sec_readouts


def _pair_residuals(c, k, ss):
    x = _cohort_ctx(c)
    return [residual(x["runs"][i][k], x["runs"][j][k], x["Z"], x["ok"]) for i, j in itertools.combinations(ss, 2)]


def _split_residuals(c, k):
    x = _cohort_ctx(c)
    ss = sorted(x["runs"])
    pool = [s for s in ss if s <= 47][:6] if c == "test_hmouse" else [s for s in ss if s <= 46]
    out, seen = [], set()
    for A in itertools.combinations(pool, 3):
        B = tuple(s for s in pool if s not in A)
        if frozenset([A, B]) in seen:
            continue
        seen.add(frozenset([A, B]))
        ma = np.mean([x["runs"][s][k] for s in A], 0)
        mb = np.mean([x["runs"][s][k] for s in B], 0)
        out.append(residual(ma, mb, x["Z"], x["ok"]))
    return out


# ====================================================================== validity (Family 3)
VALID_UNITS = [  # hyp, cohort, gold, covariate set, T4_3_scoreboard unit id (its bootstrap seed)
    ("H2", "test_hmouse", "G_H2", "C", "H2:D"),
    ("H4", "test_hmouse", "G_H4", "C_pat_nsp", "H4:D"),
    ("H4_primates", "test_hmouse", "G_H4_primates", "C_pat_npri", "H4:primates_only"),
    ("H2", "test_3sp", "G_H2", "C", "H2:multi:D"),
    ("H4", "test_3sp", "G_H4", "C_pat_nsp", "H4:multi:D"),
]


def centred(x):
    return x - np.median(x)


def validity_table(c: str) -> pd.DataFrame:
    """T4_3_scoreboard cache (covariates, tau, blocks) + gold table + the readouts compared here."""
    res = REPO / "results/analysis"
    tab = pd.read_parquet(res / f"T4_3/cache/{c}__inputs.parquet")
    gold = pd.read_csv(res / "T4_3/golds/gold_table.csv", index_col=0)
    cons = pd.read_csv(res / f"T3_3/{c}__consensus.csv")
    assert (cons["var_name"].to_numpy() == tab["var_name"].to_numpy()).all()
    tab["D_lab_all"] = cons["D"].to_numpy()
    tab["S_lab_all"] = cons["S_ema"].to_numpy()
    pr = tab["prior_raw"].to_numpy(float)
    LB = read_runs(labelled(c), pr)
    NL = read_runs({s: nolab(c, s) for s in SEEDS}, pr)
    pool = json.loads((res / "T3_3/seed_pooling.json").read_text())
    used = [p for p in pool if p["cohort"] == c][0]["used"]
    m = np.mean([LB[s]["D"] for s in used], 0)
    err = float(np.abs(centred(m) - tab["D_lab_all"].to_numpy()).max())
    assert err < 1e-5, f"{c}: labelled consensus D not reproduced ({err})"
    for arm, dd in (("lab3", LB), ("nolab3", NL)):
        tab[f"D_{arm}"] = centred(np.mean([dd[s]["D"] for s in SEEDS], 0))
        tab[f"Draw_{arm}"] = centred(np.mean([dd[s]["D_raw"] for s in SEEDS], 0))
        tab[f"S_{arm}"] = np.mean([dd[s]["S_g_ema"] for s in SEEDS], 0)
    for s in SEEDS:
        tab[f"D_nolab_s{s}"] = centred(NL[s]["D"])
        tab[f"S_nolab_s{s}"] = NL[s]["S_g_ema"]
        tab[f"D_lab_s{s}"] = centred(LB[s]["D"])
    tab = tab.join(gold, on="hsapiens_gene_id")
    return tab, dict(labelled_consensus_seeds=used, consensus_check_max_abs=err)


VALID_SCORES = ["D_lab_all", "D_lab3", "D_nolab3", "Draw_lab3", "Draw_nolab3", "S_lab_all", "S_lab3", "S_nolab3",
                "D_nolab_s42", "D_nolab_s43", "D_nolab_s44", "S_nolab_s42", "S_nolab_s43", "S_nolab_s44",
                "D_lab_s42", "D_lab_s43", "D_lab_s44"]
VALID_DELTAS = [("D_nolab3", "D_lab3"), ("D_nolab3", "D_lab_all"), ("Draw_nolab3", "Draw_lab3"),
                ("S_nolab3", "S_lab3"), ("S_nolab3", "S_lab_all"), ("S_nolab3", "D_nolab3"), ("S_lab3", "D_lab3")]


def _valid_unit(args):
    from analysis.tasks.T4_3_scoreboard import run as T43
    from analysis.tasks.T4_3_scoreboard import stats as S

    hyp, c, gold, cov, uid, tab, draws = args
    covs = T43.COVSETS[cov]
    rows = tab[[gold] + covs + VALID_SCORES].notna().all(1).to_numpy()
    T = tab.loc[rows]
    G = T[gold].to_numpy(float)
    R = np.column_stack([S.midrank(G)] + [S.midrank(T[s].to_numpy(float)) for s in VALID_SCORES])
    Z = S.design(T, covs)
    est = S.rho_p_matrix(R, Z)[1:]
    rpi = S.rho_p_matrix(R, S.design(T, ["prior_raw"]))[1:]
    rng = np.random.default_rng(T43.useed(uid))
    B = S.bootstrap(R, Z, draws, rng, T["block"].to_numpy())[:, 1:]
    out, dl = [], []
    for k, s in enumerate(VALID_SCORES):
        lo, hi = S.pct_ci(B[:, k])
        out.append(dict(hyp=hyp, cohort=c, gold=gold, cov=cov, score=s, n=int(rows.sum()), rho_p=float(est[k]),
                        ci_lo=lo, ci_hi=hi, rho_pi=float(rpi[k]), spearman=float(spearmanr(T[s], G)[0]),
                        bootstrap_seed_unit=uid, draws=draws))
    for a, b in VALID_DELTAS:
        i, j = VALID_SCORES.index(a), VALID_SCORES.index(b)
        lo, hi = S.pct_ci(B[:, i] - B[:, j])
        dl.append(dict(hyp=hyp, cohort=c, gold=gold, focal=a, comparator=b, delta=float(est[i] - est[j]),
                       ci_lo=lo, ci_hi=hi, n=int(rows.sum())))
    return out, dl


def sec_validity(draws: int = N_DRAWS):
    tabs, meta = {}, {}
    for c in ("test_hmouse", "test_3sp"):
        tabs[c], meta[c] = validity_table(c)
    jobs = [(h, c, g, cv, uid, tabs[c], draws) for h, c, g, cv, uid in VALID_UNITS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(_valid_unit, jobs))
    V = pd.DataFrame([x for r in res for x in r[0]])
    Dl = pd.DataFrame([x for r in res for x in r[1]])
    # reproduction of the scoreboard (labeled consensus D) and of the T4_3_scoreboard S_ema Family 3 rows
    sb = pd.concat([pd.read_csv(REPO / "results/analysis/T4_3/scoreboard.csv"),
                    pd.read_csv(REPO / "results/analysis/T4_3/family3.csv")])
    chk = []
    for h, c, g, cv, uid in VALID_UNITS:
        ref = sb[(sb["id"] == uid) & (sb["score"] == "D")]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        mine = V[(V.hyp == h) & (V.cohort == c) & (V.score == "D_lab_all")].iloc[0]
        chk.append(dict(unit=uid, n=int(mine.n), n_T43=int(ref.n), rho_p=mine.rho_p, rho_p_T43=float(ref.rho_p),
                        ci_lo=mine.ci_lo, ci_lo_T43=float(ref.ci_lo), ci_hi=mine.ci_hi, ci_hi_T43=float(ref.ci_hi)))
    for h, sid in (("H2", "H2:S_ema"), ("H4", "H4:S_ema")):
        ref = sb[(sb["id"] == sid)]
        if not ref.empty:
            mine = V[(V.hyp == h) & (V.cohort == "test_hmouse") & (V.score == "S_lab_all")].iloc[0]
            chk.append(dict(unit=sid, n=int(mine.n), n_T43=int(ref.iloc[0].n), rho_p=mine.rho_p,
                            rho_p_T43=float(ref.iloc[0].rho_p), ci_lo=mine.ci_lo, ci_lo_T43=float(ref.iloc[0].ci_lo),
                            ci_hi=mine.ci_hi, ci_hi_T43=float(ref.iloc[0].ci_hi)))
    chk = pd.DataFrame(chk)
    chk["abs_diff"] = (chk.rho_p - chk.rho_p_T43).abs()
    write(chk, "validity_repro_T4_3.csv")
    write(V, "validity_rho.csv")
    write(Dl, "validity_delta.csv")
    (OUT / "validity_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(chk.round(4).to_string(), flush=True)
    if chk.abs_diff.max() > 1e-9 or (chk.n != chk.n_T43).any():
        raise SystemExit("T4.3 not reproduced")


# ====================================================================== X6_adds_beyond rerun
def b1z_from_z(c: str, z: np.ndarray) -> np.ndarray:
    import scipy.sparse as sp

    from analysis.tasks.T4_2_gene_baselines import run as T42

    a = C.read_adata(c)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    library = a.obs[T42.LIBRARY_KEY["testis"]].astype(str).to_numpy()
    cl = T42.leiden_clusters(np.asarray(z, dtype=np.float32))
    xz, fz, _ = T42.pseudobulk(X, species, cl, library, T42.MIN_CELLS_CLUSTER)
    Ez, _, _ = T42.pooled_E(xz, fz, sorted(set(species)), a.n_vars)
    return pd.Series(Ez, index=np.asarray(a.var_names))


def sec_x6():
    from analysis.tasks.T4_3_scoreboard import stats as S
    from analysis.tasks.X6_adds_beyond import analyses as X6
    from analysis.tasks.X6_adds_beyond import rules as RU

    c = "test_hmouse"
    t42 = pd.read_csv(REPO / f"results/analysis/T4_2/baselines__{c}.csv").set_index("var_name")["B1z_raw"]
    zc = np.load(C.RUNS_DIR / C.CANONICAL[c] / "final_state.npz")["z_mean"]
    b_can = b1z_from_z(c, zc)
    d = (b_can - t42.reindex(b_can.index)).abs()
    both = b_can.notna() & t42.reindex(b_can.index).notna()
    chk = dict(b1z_canonical_zmean_vs_T4_2_max_abs=float(d[both].max()),
               b1z_defined_same=bool((b_can.notna() == t42.reindex(b_can.index).notna()).all()))
    print(chk, flush=True)
    b_nl = b1z_from_z(c, np.load(C.RUNS_DIR / nolab(c, 42) / "final_state.npz")["z_mean"])

    g = X6.genes(c, degraded=True)
    vt, _ = validity_table(c)
    add = vt.set_index("var_name")[["D_lab3", "S_lab3", "Draw_lab3", "D_nolab3", "S_nolab3", "Draw_nolab3"]]
    g = g.join(add, on="var_name")
    g["B1z_nolab"] = b_nl.reindex(g["var_name"]).to_numpy()
    conds = RU.conditions()
    deg = [f"{k}__{cd}" for cd in conds for k in RU.DEG_SCORES]
    old = pd.read_csv(REPO / "results/analysis/X6/degradation_rho.csv")
    old = old[old.cohort == c]
    est_rows, delta_rows, rep = [], [], []
    new_chim = ["D", "S", "B1z_raw", "D_raw", "D_lab3", "S_lab3", "Draw_lab3", "D_nolab3", "S_nolab3", "Draw_nolab3",
                "B1z_nolab"]
    for gold, _, covs in RU.GOLDS[c]:
        if gold not in RU.DEG_GOLDS[c]:
            continue
        # 1. reproduce X6_adds_beyond on its universe
        chim = ["D", "S", "B1z_raw", "D_raw"]
        rows = g[[gold] + covs + chim + deg].replace([np.inf, -np.inf], np.nan).notna().all(1).to_numpy()
        est, _ = X6.joint(g, gold, covs, chim + deg, rows)
        o = old[old.gold == gold]
        for s in chim + deg:
            sc, _, cd = s.partition("__")
            ref = o[(o.score == sc) & (o.condition == (cd or "model"))]
            rep.append(dict(gold=gold, score=s, rho_p=float(est[s]), rho_p_X6=float(ref.rho_p.iloc[0]),
                            n=int(rows.sum()), n_X6=int(ref.n.iloc[0])))
        # 2. common universe with the label-free readouts
        rows = g[[gold] + covs + new_chim + deg].replace([np.inf, -np.inf], np.nan).notna().all(1).to_numpy()
        est, B = X6.joint(g, gold, covs, new_chim + deg, rows)
        for k in RU.NOISE:
            for sc in RU.DEG_SCORES:
                cols = [f"{sc}__{k}_r{r}" for r in range(RU.NOISE_REPS)]
                est[f"{sc}__{k}_mean"] = est[cols].mean()
                B[f"{sc}__{k}_mean"] = B[cols].mean(1)
        base = dict(cohort=c, gold=gold, n=int(rows.sum()), universe="U_deg_with_label_free")
        for s in est.index:
            sc, _, cd = s.partition("__")
            lo, hi = S.pct_ci(B[s].to_numpy())
            est_rows.append(dict(base, condition=cd or "model", score=sc, rho_p=float(est[s]), ci_lo=lo, ci_hi=hi))
        for ref in new_chim:
            for s in est.index:
                if "__" not in s:
                    continue
                sc, _, cd = s.partition("__")
                lo, hi = S.pct_ci((B[ref] - B[s]).to_numpy())
                delta_rows.append(dict(base, ref=ref, condition=cd, score=sc, delta=float(est[ref] - est[s]),
                                       delta_lo=lo, delta_hi=hi))
        for a_, b_ in (("S_nolab3", "S_lab3"), ("D_nolab3", "D_lab3"), ("B1z_nolab", "B1z_raw"), ("S_nolab3", "S"),
                       ("D_nolab3", "D")):
            lo, hi = S.pct_ci((B[a_] - B[b_]).to_numpy())
            delta_rows.append(dict(base, ref=a_, condition="model", score=b_, delta=float(est[a_] - est[b_]),
                                   delta_lo=lo, delta_hi=hi))
    rep = pd.DataFrame(rep)
    rep["abs_diff"] = (rep.rho_p - rep.rho_p_X6).abs()
    write(rep, "x6_repro.csv")
    write(pd.DataFrame(est_rows), "x6_degradation_rho.csv")
    write(pd.DataFrame(delta_rows), "x6_degradation_delta.csv")
    chk.update(x6_repro_max_abs=float(rep.abs_diff.max()), x6_repro_n_equal=bool((rep.n == rep.n_X6).all()),
               b1z_nolab_defined=int(b_nl.notna().sum()), b1z_canonical_defined=int(b_can.notna().sum()),
               b1z_nolab_vs_canonical_spearman=float(spearmanr(b_nl, b_can, nan_policy="omit")[0]))
    (OUT / "x6_checks.json").write_text(json.dumps(chk, indent=2))
    print(json.dumps(chk, indent=2), flush=True)
    if rep.abs_diff.max() > 1e-9:
        raise SystemExit("X6 not reproduced")


SECTIONS = dict(health=sec_health_integration, readouts=sec_readouts, validity=sec_validity, x6=sec_x6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="health,readouts,validity,x6")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DONE").unlink(missing_ok=True)
    t0 = time.time()
    for s in a.sections.split(","):
        print(f"section {s}", flush=True)
        SECTIONS[s]()
    prov = dict(git_sha=git_sha(), sections=a.sections, seconds=round(time.time() - t0, 1),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    pf = OUT / "_provenance.json"
    hist = json.loads(pf.read_text()) if pf.exists() else []
    pf.write_text(json.dumps(hist + [prov], indent=2))
    if set(a.sections.split(",")) >= set(SECTIONS) or all((OUT / f).exists() for f in
                                                           ("integration.csv", "readout_pairs.csv", "validity_rho.csv",
                                                            "x6_degradation_rho.csv")):
        (OUT / "DONE").write_text(prov["finished"] + "\n")


if __name__ == "__main__":
    main()
