"""H1 step 1-2: harmonised per-gene tables, identity checks, run health, integration facts, data audit, cross-cohort table.

Re-runnable: a cohort's table is rebuilt with --force or when its kept runs change. Reads checkpoints and h5ads; writes only to
results/revision/H1/.

    PYTHONPATH=<chimera repo> python revision/tasks/H1_harmonise/tables.py [--jobs 8] [--force]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import gene_ids
from revision.common.check_cohort import check as check_cohort
from revision.tasks.H1_harmonise import rules as RU

OUT = RU.OUT
GENES = OUT / "genes"
CACHE = OUT / "cache"
RES = RU.RES

SOURCES = {
    "liver": "Guilliams 2022 (human, macaque, mouse), Wang 2023 (zebrafish), as processed by CAMEX",
    "testis": "Murat 2023, as processed by CAMEX",
    "bone marrow": "Tabula Sapiens, Tabula Microcebus, Tabula Muris Senis; bone marrow and spleen",
    "kidney": "KPMP v2.0 (Lake 2023 human; KPMP mouse), multiome nuclei",
    "islet": "Human 48-donor islet atlas; Hrovatin 2023 mouse atlas",
    "eye": "van Zyl 2020, outflow pathway",
}
RECIPE_SOURCE = {
    "liv_hmonkey": "canonical K = 2 recipe", "liv_hmouse": "canonical K = 2 recipe, adversary doubled", "liv_hzeb": "canonical K = 2 recipe, low MMD",
    "liv_4sp": "canonical K = 4 recipe", "test_hmac": "canonical testis K = 2 recipe", "test_hmouse": "canonical testis K = 2 recipe, adversary doubled",
    "test_3sp": "canonical testis K = 3 recipe", "bone_marrow": "testis K = 3 form, tuned on training health (negative-side anchor at half weight)",
    "kidney": "testis K = 2 form, adversary one step up", "islet": "testis K = 2 form, anchor doubled", "vz_hmouse": "testis K = 2 form", "vz_hmmul": "testis K = 2 form",
    "vz_primates": "canonical testis K = 3 row", "vz_full": "testis form scaled to K = 5, adversary at the testis 3-species value; chosen on count dependence", "vz_hpig": "testis K = 2 form", "vz_hmfas": "testis K = 2 form",
}


def t33_seeds() -> dict:
    j = json.loads((RES / "T3_3" / "seed_pooling.json").read_text())
    return {r["cohort"]: [int(s) for s in r["used"]] for r in j}


def o1_flags(c: str, vn: pd.Index, ens: pd.Series) -> pd.DataFrame:
    """O1 flags as columns, per var_name; empty when O1 has no row for the cohort. No gene is dropped."""
    cols = {"T2": "o1_T2", "M_wrong_row": "o1_M_wrong_row", "M_false_zero": "o1_M_false_zero", "zero_any": "o1_zero_any"}
    f = RES / "O1" / "flags_by_cohort.parquet"
    out = pd.DataFrame({v: pd.Series(pd.NA, index=vn, dtype="boolean") for v in cols.values()})
    if not f.exists():
        return out
    d = pd.read_parquet(f, columns=["cohort", "var_name", "hsapiens_gene_id", *cols])
    d = d[d.cohort == c].set_index("var_name")
    if not len(d):
        return out
    d = d.reindex(vn)
    known = d.hsapiens_gene_id.notna().to_numpy()
    if (d.hsapiens_gene_id.to_numpy()[known] != ens.to_numpy()[known]).any():
        raise AssertionError(f"{c}: O1 flags disagree with H1 on gene identity")
    return d[list(cols)].astype("boolean").rename(columns=cols)


def run_health(run: str) -> dict:
    """The health rule of the amended plan, on the run's own state. No result is read beyond these fields."""
    d = RU.RUNS / run
    r = dict(run=run, finished=(d / "final_state.npz").exists(), no_nan=False, healthy=False)
    if not r["finished"]:
        return r
    with np.load(d / "final_state.npz", allow_pickle=True) as z:
        alpha, prior, S, zm = [np.asarray(z[k], float) for k in ("alpha", "prior", "S_g_ema", "z_mean")]
    r["no_nan"] = bool(np.isfinite(alpha).all() and np.isfinite(zm).all() and np.isfinite(S).all())
    if not r["no_nan"]:
        return r
    f = d / "conservation_prior_raw.npy"
    raw = np.load(f).astype(float) if f.exists() else prior
    zv = zm.var(0)
    r.update(n_active_z=int((zv >= 0.01).sum()), sat_hi=float((alpha > RU.SAT_HI).mean()), sat_lo=float((alpha < 0.05).mean()),
             tail_up_raw=float(((alpha - raw) > RU.TAIL).mean()), tail_dn_raw=float(((alpha - raw) < -RU.TAIL).mean()),
             tail_up_anch=float(((alpha - prior) > RU.TAIL).mean()), tail_dn_anch=float(((alpha - prior) < -RU.TAIL).mean()),
             frac_support=float((np.abs(S) >= RU.MASK_EPS).mean()))
    r["healthy"] = bool(r["n_active_z"] == zm.shape[1] and r["tail_up_raw"] >= RU.TAIL_MIN and r["tail_dn_raw"] >= RU.TAIL_MIN
                        and r["sat_hi"] <= RU.SAT_MAX and r["sat_lo"] <= RU.SAT_MAX)
    return r


def runs_table() -> pd.DataFrame:
    t33 = t33_seeds()
    rows = []
    for c, d in RU.cohorts().items():
        for s, run in sorted(d["tried"].items()):
            h = run_health(run)
            keep = (s in t33.get(c, [])) if d["role"] == "reference" else h["healthy"]
            rows.append(dict(cohort=c, tissue=d["tissue"], role=d["role"], seed=s, kept=bool(keep), **h))
    return pd.DataFrame(rows)


def task_table(c: str) -> pd.DataFrame | None:
    """The owning task's gene table, for the cross-check of D and for B1. None if it has none."""
    R = RES
    p = {"bone_marrow": R / "P12/readout/bm_genes.parquet", "kidney": R / "P13/readout/kidney_genes.parquet",
         "islet": R / "T7_9/readout/islet_genes.parquet"}.get(c)
    if c.startswith("vz_"):
        p = R / f"T11/readout/genes_{c[3:]}.parquet"
    if p is not None and p.exists():
        return pd.read_parquet(p).set_index("var_name")
    return None


def b1_column(c: str, tt: pd.DataFrame | None, var_names) -> pd.Series:
    if c == "bone_marrow":
        return tt["E_human_mouse"].reindex(var_names)
    if c in ("kidney", "islet"):
        return tt["B1b_raw"].reindex(var_names)
    if c.startswith("vz_") and tt is not None:
        return tt["B1_raw"].reindex(var_names)
    f = RES / "T4_2" / f"baselines__{c}.csv"
    if f.exists():
        return pd.read_csv(f).set_index("var_name")["B1b_raw"].reindex(var_names)
    return pd.Series(np.nan, index=var_names)


def build_cohort(args) -> dict:
    c, kept = args
    d = RU.cohorts()[c]
    a = ad.read_h5ad(d["h5ad"])
    a.obs["cell_type"] = a.obs[d["ct"]].astype(str).values  # tau on the harmonised classes; expr and detection do not use it
    vn = pd.Index(a.var_names, name="var_name")
    table = gene_ids.load_table(d["table"])
    rho_table = gene_ids.assert_table_matches(a, table)
    g = pd.DataFrame(index=vn)
    g["hsapiens_gene_id"] = gene_ids.ensembl(d["table"], vn).values
    g["symbol"] = gene_ids.symbols(d["table"], vn).values
    ids = vn.str.extract(r"^ortho_(\d+)$")[0].astype(int)
    g["identity_human_mouse"] = ids.map(table["mmusculus_identity"]).to_numpy(float) / 100 if "mmusculus_identity" in table else np.nan
    adata_prior = a.var["conservation_prior"].to_numpy(float)
    A, Sm, Dm, Pc = {}, {}, {}, {}
    for s, run in kept.items():
        st = C.load_state(run)
        if len(st["alpha"]) != a.n_vars:
            raise AssertionError(f"{run}: {len(st['alpha'])} genes vs h5ad {a.n_vars}")
        f = RU.RUNS / run / "conservation_prior_raw.npy"
        raw = np.load(f).astype(float) if f.exists() else st["prior"].astype(float)
        if not np.allclose(raw, adata_prior, atol=1e-5):
            raise AssertionError(f"{run}: raw prior differs from the h5ad prior")
        A[s], Sm[s], Pc[s] = st["alpha"].astype(float), st["S_g_ema"].astype(float), st["prior"].astype(float)
        Dm[s] = A[s] - Pc[s]
    seeds = sorted(A)
    g["pi_raw"] = adata_prior
    g["pi_clamped"] = Pc[seeds[0]]
    if any(not np.allclose(Pc[s], g["pi_clamped"]) for s in seeds):
        raise AssertionError(f"{c}: seeds anchor on different priors")
    for s in seeds:
        g[f"alpha_s{s}"] = A[s]
    Am = np.stack([A[s] for s in seeds])
    g["alpha"] = Am.mean(0)
    g["delta_alpha"] = g["alpha"] - g["pi_clamped"]
    g["D"] = g["delta_alpha"] - g["delta_alpha"].median()
    r = g["alpha"] - g["pi_raw"]
    g["D_raw"] = r - r.median()
    g["S_ema"] = np.mean([Sm[s] for s in seeds], 0)
    g["count_support"] = np.abs(g["S_ema"]) >= RU.MASK_EPS
    un = g["count_support"].to_numpy()
    thr = np.median(np.abs(g["S_ema"].to_numpy()[un]))
    Ds = np.stack([Dm[s] - np.median(Dm[s]) for s in seeds])
    Ss = np.stack([Sm[s] for s in seeds])
    sign = np.sign(g["D"].to_numpy())
    g["sign_D_stable"] = (np.sign(Ds) == sign).all(0)
    g["sign_S_stable"] = (np.sign(Ss) == np.sign(g["S_ema"].to_numpy())).all(0)
    g["saturated_flag"] = (Am > RU.SAT_HI).any(0)
    g["strict_support"] = un & (np.abs(g["S_ema"].to_numpy()) >= thr) & (np.sign(g["S_ema"].to_numpy()) == sign) & g["sign_D_stable"] & g["sign_S_stable"]
    cov = C.gene_covariates(a)
    g["expr"] = cov["log_cp10k_spmean"]
    g["detection"] = cov["detection_spmean"]
    g["log_mean"] = cov["log_mean"]
    g["tau"] = cov["tau"]
    g["ppass"] = cov["ppass"]
    for k in cov.columns:
        if k.startswith("detection_") and k not in ("detection_spmean", "detection_sprange"):
            g[k] = cov[k]
    tt = task_table(c)
    g["B1"] = b1_column(c, tt, vn)
    for k, v in o1_flags(c, vn, g["hsapiens_gene_id"]).items():
        g[k] = v
    flagged = (g["o1_M_wrong_row"].fillna(False) | g["o1_M_false_zero"].fillna(False) | g["o1_zero_any"].fillna(False)).astype(bool)
    g["nameable"] = g["strict_support"] & ~flagged
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS", "LOEUF"]]
    g = g.reset_index().merge(gold, how="left", left_on="hsapiens_gene_id", right_index=True).set_index("var_name").reindex(vn)
    g.insert(0, "cohort", c)
    g.insert(1, "tissue", d["tissue"])
    g.to_parquet(GENES / f"{c}.parquet")
    g.to_csv(GENES / f"{c}.tsv.gz", sep="\t")

    # Identity checks and cross-checks against the owning task's table.
    ck = dict(cohort=c, n_genes=len(g), rho_prior_vs_table_mean_identity=rho_table, n_ensembl_missing=int(g.hsapiens_gene_id.isna().sum()),
              n_ensembl_duplicated=int(g.hsapiens_gene_id.dropna().duplicated().sum()), n_symbol_missing=int(g.symbol.isna().sum()),
              n_symbol_duplicated=int(g.symbol.dropna().duplicated().sum()), n_mt_genes=int(g.symbol.astype(str).str.startswith("MT-").sum()),
              max_pi_raw=float(g.pi_raw.max()), min_pi_raw=float(g.pi_raw.min()),
              clamp=float(g.pi_clamped.max()), clamp_consistent=bool(np.allclose(g.pi_clamped, np.minimum(g.pi_raw, g.pi_clamped.max()), atol=1e-5)),
              frac_at_clamp=float((g.pi_raw >= g.pi_clamped.max() - 1e-6).mean()), n_seeds=len(seeds))
    ck["frac_alpha_table_join"] = float(g.symbol.notna().mean())
    if tt is not None and "D" in tt:
        ck["max_abs_diff_D_task_table"] = float(np.abs(g["D"] - tt["D"].reindex(vn)).max())
    ck["n_saturated_flag"] = int(g.saturated_flag.sum())
    tidy = RES / "T11" / "genes" / f"genes_{c[3:]}.tsv"
    if c.startswith("vz_") and tidy.exists():  # the tidy table of the eye agent: same genes, same flag
        ty = pd.read_csv(tidy, sep="\t").set_index("var_name").reindex(vn)
        ck["n_saturated_flag_differs_T11"] = int((ty["saturated_flag"].astype(bool).to_numpy() != g.saturated_flag.to_numpy()).sum())
        ck["max_abs_diff_D_tidy"] = float(np.abs(g["D"] - ty["delta_alpha_centred"]).max())
    for k in ("o1_T2", "o1_M_wrong_row", "o1_M_false_zero", "o1_zero_any"):
        ck[f"n_{k}"] = int(g[k].fillna(False).sum())
    ck["n_strict"], ck["n_nameable"] = int(g.strict_support.sum()), int(g.nameable.sum())
    if c in t33_seeds():
        cs = pd.read_csv(RES / "T3_3" / f"{c}__consensus.csv").set_index("var_name").reindex(vn)
        ck["max_abs_diff_D_task_table"] = float(np.abs(g["D"] - cs["D"]).max())
        ck["max_abs_diff_S_task_table"] = float(np.abs(g["S_ema"] - cs["S_ema"]).max())
    return ck


def integration_row(args) -> dict:
    """Mixing and label transfer from the final latent: same function, k and seed in every cohort."""
    c, s, run = args
    sys.path.insert(0, str(RU.REPO / "training"))
    from compare_vs_scvi import compute_integration_metrics  # noqa: PLC0415

    d = RU.cohorts()[c]
    a = ad.read_h5ad(d["h5ad"], backed="r")
    species = a.obs["species"].astype(str).to_numpy()
    labels = a.obs[d["ct"]].astype(str).to_numpy()
    a.file.close()
    with np.load(RU.RUNS / run / "final_state.npz", allow_pickle=True) as z:
        zm = np.asarray(z["z_mean"])
        adv = float(z["adversarial_accuracy_final"]) if "adversarial_accuracy_final" in z.files else np.nan
    m = compute_integration_metrics(zm, species, labels, k=15, seed=42)["global"]
    return dict(cohort=c, seed=s, run=run, knn_mixing_entropy=m["knn_mixing_entropy"], knn_species_purity=m["knn_species_purity"],
                label_transfer_acc=m["label_transfer_acc"], adv_acc=adv)


def chance_refs(species) -> dict:
    p = pd.Series(species).value_counts(normalize=True).to_numpy()
    k, K = 15, len(p)
    draws = np.random.default_rng(0).multinomial(k, p, size=20000) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(draws > 0, draws * np.log(draws), 0).sum(1) / np.log(K)
    return dict(majority_rate=float(p.max()), entropy_random=float(H.mean()))


def z1_rows(kept_runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run count dependence from Z1's table; runs absent there through Z1's own function."""
    Z = RES / "Z1"
    t = pd.concat([pd.read_csv(f, sep="\t") for f in (Z / "diag_reference.tsv", Z / "diag_ref2.tsv") if f.exists()])
    t = t.drop_duplicates("run").set_index("run")
    rows = []
    missing = [r for r in kept_runs.run if r not in t.index]
    if missing:
        sys.path.insert(0, str(RU.REPO / "revision/tasks/Z1_retune"))
        import diagnostics as ZD  # noqa: PLC0415

        cache = CACHE / "z1_missing.tsv"
        have = pd.read_csv(cache, sep="\t").set_index("run") if cache.exists() else pd.DataFrame()
        new = [ZD.one_run(r, with_mixing=False) for r in missing if r not in have.index]
        if new:
            have = pd.concat([have, pd.DataFrame(new).set_index("run")])
            have.to_csv(cache, sep="\t")
        t = pd.concat([t, have.reindex(missing)])
    for r in kept_runs.itertuples():
        x = t.loc[r.run]
        rows.append(dict(cohort=r.cohort, seed=r.seed, run=r.run, rho_DS=x.get("rho_DS"), rho_DS_partial=x.get("rho_DS_partial"), frac_support=x.get("frac_support")))
    return pd.DataFrame(rows)


CONFIG_KEYS = ["adv_weight", "lambda_anchor", "lambda_mmd", "free_bits", "use_celltype_conditional_mmd", "celltype_key", "batch_key", "prior_clamp_max",
               "max_epochs", "lambda_anchor_neg_factor", "use_per_gene_bias", "equal_species_sampling", "n_latent", "n_hidden", "mask_species_in_encoder",
               "mask_batch_in_encoder", "use_vamp_prior", "sg_ema_beta", "lambda_fit", "lambda_res"]


def audit_cohort(args) -> dict:
    c, runs = args
    d = RU.cohorts()[c]
    a = ad.read_h5ad(d["h5ad"])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    cfgs = {r: json.loads((RU.RUNS / r / "config.json").read_text()) for r in runs}
    cfg = cfgs[runs[0]]
    bk = cfg.get("batch_key") or "batch"
    diffs = sorted({k for k in CONFIG_KEYS if len({json.dumps(c_.get(k)) for c_ in cfgs.values()}) > 1})
    t = pd.crosstab(a.obs[d["ct"]].astype(str), a.obs["species"].astype(str))
    tp = t.where(t > 0)
    tb = a.obs.groupby("species", observed=True)[bk].nunique()
    errs = check_cohort(str(d["h5ad"]), batch_key=bk, celltype_key=d["ct"])
    tab = gene_ids.load_table(d["table"])
    idc = [k for k in tab.columns if k.endswith("_identity") and k != "mean_identity"]
    return dict(cohort=c, n_cells=a.n_obs, n_genes=a.n_vars, x_integer=bool(np.allclose(X.data, np.round(X.data))), x_min=float(X.data.min()), x_max=float(X.data.max()),
                layers_empty=not len(a.layers), batch_key=bk, batch_levels=";".join(sorted(map(str, a.obs[bk].unique()))), max_batches_per_species=int(tb.max()),
                n_species=int(a.obs["species"].nunique()), species_cells=json.dumps({str(k): int(v) for k, v in a.obs["species"].value_counts().items()}),
                check_cohort_errors=";".join(errs), cell_type_key=d["ct"], n_cell_types=int(t.shape[0]), n_cell_types_all_species=int((t > 0).all(1).sum()),
                min_cells_type_species=int(np.nanmin(tp.to_numpy())), max_cells_type_species=int(t.to_numpy().max()),
                n_type_species_below_50=int((tp < 50).sum().sum()), n_type_species_below_20=int((tp < 20).sum().sum()),
                prior_species=";".join(k.replace("_identity", "") for k in idc), ortholog_table=str(gene_ids.TABLES[d["table"]]).split("/")[-1],
                config_diff_between_seeds=";".join(diffs), **{f"cfg_{k}": cfg.get(k) for k in CONFIG_KEYS})


def cross_table_identity(cohorts: dict) -> pd.DataFrame:
    """Same human Ensembl ID across cohorts: symbol agreement, mouse ortholog agreement, identity agreement, prior difference."""
    rows = []
    tabs = {c: pd.read_parquet(GENES / f"{c}.parquet") for c in cohorts}
    orth = {}
    for c, d in cohorts.items():
        t = gene_ids.load_table(d["table"])
        if "mmusculus_gene_id" in t:
            orth[c] = t.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["mmusculus_gene_id"]
    for a_, b_ in itertools.combinations(cohorts, 2):
        A, B = [tabs[x].dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id") for x in (a_, b_)]
        ix = A.index.intersection(B.index)
        r = dict(a=a_, b=b_, n_shared=len(ix), n_symbol_differs=int((A.loc[ix, "symbol"].astype(str) != B.loc[ix, "symbol"].astype(str)).sum()),
                 median_abs_diff_pi=float((A.loc[ix, "pi_raw"] - B.loc[ix, "pi_raw"]).abs().median()),
                 frac_pi_differs_gt_05=float(((A.loc[ix, "pi_raw"] - B.loc[ix, "pi_raw"]).abs() > 0.05).mean()),
                 spearman_pi=float(spearmanr(A.loc[ix, "pi_raw"], B.loc[ix, "pi_raw"])[0]),
                 median_abs_diff_identity_human_mouse=float((A.loc[ix, "identity_human_mouse"] - B.loc[ix, "identity_human_mouse"]).abs().median()))
        if a_ in orth and b_ in orth:
            oa, ob = orth[a_].reindex(ix), orth[b_].reindex(ix)
            both = oa.notna() & ob.notna()
            r["n_mouse_ortholog_differs"] = int((oa[both] != ob[both]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def within_tissue_agreement(cohorts: dict) -> pd.DataFrame:
    """Species sets of one tissue on shared genes: agreement of D and of S_ema, Spearman."""
    rows = []
    tabs = {c: pd.read_parquet(GENES / f"{c}.parquet").dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id") for c in cohorts}
    for a_, b_ in itertools.combinations(cohorts, 2):
        if cohorts[a_]["tissue"] != cohorts[b_]["tissue"]:
            continue
        A, B = tabs[a_], tabs[b_]
        ix = A.index.intersection(B.index)
        sup = A.loc[ix, "count_support"] & B.loc[ix, "count_support"]
        rows.append(dict(tissue=cohorts[a_]["tissue"], a=a_, b=b_, n=len(ix), n_support_both=int(sup.sum()), rho_D=float(spearmanr(A.loc[ix, "D"], B.loc[ix, "D"])[0]),
                         rho_D_support=float(spearmanr(A.loc[ix, "D"][sup], B.loc[ix, "D"][sup])[0]), rho_S_support=float(spearmanr(A.loc[ix, "S_ema"][sup], B.loc[ix, "S_ema"][sup])[0])))
    return pd.DataFrame(rows)


def cohort_table(runs: pd.DataFrame, integ: pd.DataFrame, z1: pd.DataFrame, audit: pd.DataFrame, cohorts: dict) -> pd.DataFrame:
    rows = []
    for c, d in cohorts.items():
        g = pd.read_parquet(GENES / f"{c}.parquet")
        rr = runs[runs.cohort == c]
        kept = rr[rr.kept]
        aud = audit.set_index("cohort").loc[c]
        a = ad.read_h5ad(d["h5ad"], backed="r")
        ref = chance_refs(a.obs["species"].astype(str).to_numpy())
        a.file.close()
        z, i = z1[z1.cohort == c], integ[integ.run.isin(kept.run)]
        un = g.count_support.to_numpy()
        tail = g[g.delta_alpha.abs() > RU.TAIL]
        ok = g.count_support & (np.sign(g.S_ema) == np.sign(g.delta_alpha)) & (g.S_ema.abs() >= np.median(g.S_ema.abs()[un]))
        rows.append(dict(
            cohort=c, tissue=d["tissue"], label=d["label"], role=d["role"], species=d["species"], n_species=int(aud.n_species), cells=int(aud.n_cells), genes=int(aud.n_genes),
            source_datasets=SOURCES[d["tissue"]], cell_types=int(aud.n_cell_types), cell_types_all_species=int(aud.n_cell_types_all_species), recipe_source=RECIPE_SOURCE[c],
            recipe=f"adv {aud.cfg_adv_weight:g}, MMD {aud.cfg_lambda_mmd if pd.notna(aud.cfg_lambda_mmd) else 'default'}, anchor {aud.cfg_lambda_anchor:g}, free bits {aud.cfg_free_bits if pd.notna(aud.cfg_free_bits) else 'default'}, "
                   f"clamp {aud.cfg_prior_clamp_max if pd.notna(aud.cfg_prior_clamp_max) else 'none'}, conditional MMD {'yes' if aud.cfg_use_celltype_conditional_mmd else 'no'}, {aud.cfg_max_epochs} epochs",
            runs_kept=int(kept.shape[0]), runs_trained=int(rr.shape[0]), seeds_kept=",".join(map(str, kept.seed)),
            count_support_share=float(z.frac_support.mean()), count_support_consensus=float(un.mean()), strict_support_n=int(g.strict_support.sum()),
            sd_delta_alpha_anchored=float(g.D.std()), sd_delta_alpha_raw=float(g.D_raw.std()),
            tail_up_D=float((g.D > RU.TAIL).mean()), tail_dn_D=float((g.D < -RU.TAIL).mean()), tail_up_Draw=float((g.D_raw > RU.TAIL).mean()), tail_dn_Draw=float((g.D_raw < -RU.TAIL).mean()),
            tail_up_raw_uncentred=float(kept.tail_up_raw.mean()), tail_dn_raw_uncentred=float(kept.tail_dn_raw.mean()),
            tail_margin_support=float(((np.sign(tail.S_ema) == np.sign(tail.delta_alpha)) & (tail.S_ema.abs() >= np.median(g.S_ema.abs()[un])) & tail.count_support).mean()),
            margin_support_up=float(ok[g.delta_alpha > RU.TAIL].mean()), margin_support_down=float(ok[g.delta_alpha < -RU.TAIL].mean()) if (g.delta_alpha < -RU.TAIL).any() else np.nan,
            n_tail_up_anchored=int((g.delta_alpha > RU.TAIL).sum()), n_tail_down_anchored=int((g.delta_alpha < -RU.TAIL).sum()), frac_S_pos=float((g.S_ema[un] > 0).mean()),
            frac_at_clamp=float((g.pi_raw >= g.pi_clamped.max() - 1e-6).mean()),
            rho_DS_z1=float(z.rho_DS.mean()), rho_DS_z1_min=float(z.rho_DS.min()), rho_DS_z1_max=float(z.rho_DS.max()), rho_DS_partial_z1=float(z.rho_DS_partial.mean()),
            rho_DS_consensus=float(spearmanr(g.D[un], g.S_ema[un])[0]),
            seed_sat_max=float(kept.sat_hi.max()), n_saturated=int(g.saturated_flag.sum()), nameable_n=int(g.nameable.sum()),
            rho_DS_no_saturated=float(spearmanr(g.D[un & ~g.saturated_flag], g.S_ema[un & ~g.saturated_flag])[0]),
            tail_up_D_no_saturated=float((g.D[~g.saturated_flag] > RU.TAIL).mean()), tail_dn_D_no_saturated=float((g.D[~g.saturated_flag] < -RU.TAIL).mean()),
            n_o1_T2=int(g.o1_T2.fillna(False).sum()), n_o1_map=int((g.o1_M_wrong_row.fillna(False) | g.o1_M_false_zero.fillna(False)).sum()),
            mixing_entropy=float(i.knn_mixing_entropy.mean()), mixing_entropy_random=ref["entropy_random"], mixing_ratio=float(i.knn_mixing_entropy.mean() / ref["entropy_random"]),
            species_purity=float(i.knn_species_purity.mean()), label_transfer=float(i.label_transfer_acc.mean()), adv_acc=float(i.adv_acc.mean()), majority_rate=ref["majority_rate"],
            sat_hi=float(kept.sat_hi.mean()),
        ))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in (OUT, GENES, CACHE):
        p.mkdir(parents=True, exist_ok=True)
    cohorts = RU.cohorts()
    runs = runs_table()
    runs.to_csv(OUT / "runs.tsv", sep="\t", index=False)
    kept = runs[runs.kept]
    bad = kept.groupby("cohort").size()
    print(bad.to_string(), flush=True)
    need = {c for c, d in cohorts.items() if d["role"] == "new" and bad.get(c, 0) < 2}
    if need:
        raise SystemExit(f"fewer than 2 healthy seeds: {need}")
    with ProcessPoolExecutor(a.jobs) as ex:
        items = [(c, {int(r.seed): r.run for r in g.itertuples()}) for c, g in kept.groupby("cohort", sort=False)]
        cks = list(ex.map(build_cohort, items))
        pd.DataFrame(cks).to_csv(OUT / "identity_checks.csv", index=False)
        audit = pd.DataFrame(list(ex.map(audit_cohort, [(c, list(g.run)) for c, g in kept.groupby("cohort", sort=False)])))
        audit.to_csv(OUT / "audit_cohorts.csv", index=False)
        f = CACHE / "integration.csv"
        done = pd.read_csv(f) if f.exists() and not a.force else pd.DataFrame(columns=["run"])
        todo = [(r.cohort, r.seed, r.run) for r in kept.itertuples() if r.run not in set(done.run)]
        new = pd.DataFrame(list(ex.map(integration_row, todo)))
        integ = pd.concat([done, new], ignore_index=True)
        integ.to_csv(f, index=False)
    z1 = z1_rows(kept)
    z1.to_csv(OUT / "z1_rows.csv", index=False)
    cross_table_identity(cohorts).to_csv(OUT / "identity_cross_cohort.csv", index=False)
    within_tissue_agreement(cohorts).to_csv(OUT / "within_tissue_models.csv", index=False)
    t = cohort_table(runs, integ, z1, audit, cohorts)
    t.to_csv(OUT / "cohort_table.csv", index=False)
    all_g = pd.concat([pd.read_parquet(GENES / f"{c}.parquet").drop(columns=[k for k in pd.read_parquet(GENES / f"{c}.parquet").columns if k.startswith(("alpha_s", "detection_"))] ) for c in cohorts])
    all_g.to_parquet(OUT / "all_genes.parquet")
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 40)
    print(t[["cohort", "cells", "genes", "cell_types", "runs_kept", "runs_trained", "count_support_share", "sd_delta_alpha_anchored", "tail_up_D", "tail_dn_D", "rho_DS_z1", "mixing_entropy", "label_transfer"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
