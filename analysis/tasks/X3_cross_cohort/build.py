"""EXPLORATORY: per-gene tables for the cross-cohort synthesis of the seven cohorts.

For each cohort, every finished seed of the canonical recipe (canonical s42 MPS run plus the
T3_2_seeds / T3_2b_extra_seeds CPU seeds; the seed-pooling rule of T3_3_consensus kept every seed, see
``results/analysis/X3/t3_3_provisional`` and ``results/analysis/T3_3``):

* ``alpha`` (mean over seeds), ``prior_raw`` (unclamped mean ortholog identity),
  ``D_raw = alpha - prior_raw - cohort median`` (primary scale), ``D_clamp`` (against the
  clamped prior the testis recipes anchor on), ``S`` (mean ``S_g_ema`` over seeds), per-seed
  ``D_raw`` in ``seeds/<cohort>.parquet``.
* Covariates from the T2_5_celltype_margin gene table (same definitions as ``analysis/common/cohorts.py``)
  and log human CDS length (T2_4_covariates cache); pair-specific protein identity from the ortholog table.
* Baselines from T4_2_gene_baselines (``B1b_raw`` cross-species cell-type profile correlation, ``B1b`` its
  residual, ``B1c_raw`` / ``B1c`` rank-matched level, ``B3b`` features-only ridge of s42 Δα).
* Cell-type decomposition of S_g from T2_5_celltype_margin (canonical s42 checkpoint; weighted ``C_gk``,
  mean-z mode; testis on ``cell_ontology_class``), as ``C_<type>`` columns, and per-species
  contributions ``Csp_<species>`` (sum over types of w_sk S_gsk).
* The 64 ESM-2 PCs of ``varm['gene_features']`` (``esm_<i>``).

Read-only on run dirs and data/processed. Writes results/analysis/X3/genes/.
"""

from __future__ import annotations

import argparse
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import add_common_args

sys.path.insert(0, str(gene_ids.REPO_ROOT / "analysis" / "tasks" / "T4_2_gene_baselines"))
import run as T42  # noqa: E402  (pseudobulk, TESTIS_COARSE, LIBRARY_KEY, MIN_CELLS_TYPE)

ROOT = gene_ids.REPO_ROOT
T25 = ROOT / "results" / "analysis" / "T2_5"
T42_DIR = ROOT / "results" / "analysis" / "T4_2"
OUT = ROOT / "results" / "analysis" / "X3"
COHORTS = C.COHORT_ORDER
LIVER = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp"]
TESTIS = ["test_hmac", "test_hmouse", "test_3sp"]
MPS = {
    "liv_hmonkey": {42: "iter90-150ep-s42", 43: "iter90-150ep-s43", 44: "iter90-150ep-s44"},
    "liv_hmouse": {42: "xds-hmouse-iter90-150ep-s42", 43: "xds-hmouse-iter90-150ep-s43",
                   44: "xds-hmouse-iter90-150ep-s44"},
}
CPU = {"liv_hmouse": range(45, 52), "test_hmouse": range(43, 52), "liv_4sp": range(43, 47),
       "test_3sp": range(43, 47)}
IDENTITY_COLS = {"Monkey": "mmulatta_identity", "Macaque": "mmulatta_identity",
                 "Mouse": "mmusculus_identity", "Zebrafish": "drerio_identity"}
COHORT_SPECIES = {"liv_hmonkey": ["Monkey"], "liv_hmouse": ["Mouse"], "liv_hzeb": ["Zebrafish"],
                  "liv_4sp": ["Monkey", "Mouse", "Zebrafish"], "test_hmac": ["Macaque"],
                  "test_hmouse": ["Mouse"], "test_3sp": ["Macaque", "Mouse"]}


def seed_runs(cohort: str) -> dict[int, str]:
    runs = dict(MPS.get(cohort, {42: C.CANONICAL[cohort]}))
    for s in CPU.get(cohort, []):
        runs[s] = f"rev-{cohort}-s{s}"
    return {s: r for s, r in sorted(runs.items()) if K.run_finished(r)}


def group_of(cohort: str, cell_type):
    """Liver: the cell type; testis: the T4_2_gene_baselines coarse stage (SG, SC, rSD, eSD, Sertoli, Other_somatic)."""
    ct = pd.Series(cell_type).astype(str)
    return ct.map(T42.TESTIS_COARSE).to_numpy() if C.is_testis(cohort) else ct.to_numpy()


def celltype_columns(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """T2_5_celltype_margin weighted contributions, testis fine types summed into coarse stages (identity kept)."""
    Cg = pd.read_parquet(T25 / f"{cohort}__C_gk.parquet")
    Cg["cell_type"] = group_of(cohort, Cg["cell_type"])
    if Cg["cell_type"].isna().any():
        raise AssertionError(f"{cohort}: unmapped cell types")
    piv = Cg.pivot_table(index="var_name", columns="cell_type", values="C_gk_meanz", aggfunc="sum")
    piv.columns = [f"C_{c}" for c in piv.columns]
    first = Cg[Cg["var_name"] == Cg["var_name"].iloc[0]]  # one row per fine type
    w = (first.groupby("cell_type").agg(weight_k=("weight_k", "sum"), shared_type=("shared_type", "max"))
         .reset_index().assign(cohort=cohort))
    S = pd.read_parquet(T25 / f"{cohort}__S_gsk.parquet")
    S["wS"] = S["weight"] * S["S_gsk_meanz"]
    spc = S.pivot_table(index="var_name", columns="species", values="wS", aggfunc="sum")
    spc.columns = [f"Csp_{c}" for c in spc.columns]
    return piv, spc, w


def type_pseudobulk(cohort: str) -> pd.DataFrame:
    """T4_2_gene_baselines pseudobulk per (species, group); species-centered difference to human per group.

    ``dc_<sp>_<t>`` = (x_sp,t - mean_t' x_sp,t') - (x_H,t - mean_t' x_H,t') over the groups shared by
    human and that species (libraries with >= 30 cells), the per-type simple baseline for S_g.
    """
    a = C.read_adata(cohort)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    species = a.obs["species"].astype(str).to_numpy()
    group = group_of(cohort, a.obs["cell_type"])
    library = a.obs[T42.LIBRARY_KEY["testis" if C.is_testis(cohort) else "liver"]].astype(str).to_numpy()
    x, f, _ = T42.pseudobulk(X, species, group, library, T42.MIN_CELLS_TYPE)
    df = {}
    for s in sorted({s for s, _ in x}):
        if s == "Human":
            continue
        st = sorted(t for (ss, t) in x if ss == s and ("Human", t) in x)
        if len(st) < 2:
            continue
        H = np.column_stack([x[("Human", t)] for t in st])
        O = np.column_stack([x[(s, t)] for t in st])
        Hc, Oc = H - H.mean(1, keepdims=True), O - O.mean(1, keepdims=True)
        for j, t in enumerate(st):
            df[f"dc_{s}_{t}"] = Oc[:, j] - Hc[:, j]
            df[f"fH_{t}"] = f[("Human", t)]
            df[f"f_{s}_{t}"] = f[(s, t)]
    return pd.DataFrame(df, index=pd.Index(a.var_names, name="var_name"))


def build(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    g = pd.read_parquet(T25 / f"{cohort}__genes.parquet").set_index("var_name")
    runs = seed_runs(cohort)
    A = pd.DataFrame({s: C.load_state(r)["alpha"].astype(float) for s, r in runs.items()}, index=g.index)
    S = pd.DataFrame({s: C.load_state(r)["S_g_ema"].astype(float) for s, r in runs.items()}, index=g.index)
    P = pd.DataFrame({s: C.load_state(r)["prior"].astype(float) for s, r in runs.items()}, index=g.index)
    if not np.allclose(P.to_numpy(), g[["prior"]].to_numpy(), atol=1e-6):
        raise AssertionError(f"{cohort}: prior differs between seeds")
    out = g[["hsapiens_gene_id", "symbol", "prior", "prior_raw", "log_mean", "detection", "tau",
             "log_cp10k_spmean", "detection_spmean", "detection_sprange", "margin_support",
             "lacks_margin_support", "tail", "S_full_meanz"]].copy()
    out["alpha"] = A.mean(1)
    out["alpha_s42"] = A[42]
    dr = out["alpha"] - out["prior_raw"]
    out["D_raw"] = dr - dr.median()
    dc = out["alpha"] - out["prior"]
    out["D_clamp"] = dc - dc.median()
    out["S"] = S.mean(1)
    out["S_s42"] = S[42]
    out["n_seeds"] = len(runs)
    Dseed = A.sub(out["prior_raw"], axis=0)
    Dseed = Dseed - Dseed.median()
    out["sign_consistency"] = (np.sign(Dseed).eq(np.sign(out["D_raw"]), axis=0)).mean(1)
    cds = K.cds_table()
    out = out.join(cds[["log_cds_length"]], on="hsapiens_gene_id")
    tab = gene_ids.load_table(cohort)
    ids = out.index.str.replace("ortho_", "").astype(int)
    for s in COHORT_SPECIES[cohort]:
        out[f"identity_{s}"] = tab[IDENTITY_COLS[s]].reindex(ids).to_numpy() / 100.0
    b = pd.read_csv(T42_DIR / f"baselines__{cohort}.csv").set_index("var_name")
    for c in ["B1b_raw", "B1b", "B1c_raw", "B1c", "B3b", "B2b"]:
        out[c] = b[c].reindex(out.index)
    piv, spc, w = celltype_columns(cohort)
    out = out.join(piv).join(spc).join(type_pseudobulk(cohort))
    a = ad.read_h5ad(gene_ids.h5ad_path(cohort), backed="r")
    if list(a.var_names) != list(out.index):
        raise AssertionError(f"{cohort}: var order differs from the T2.5 table")
    F = np.asarray(a.varm["gene_features"], dtype=float)
    a.file.close()
    esm = pd.DataFrame(F[:, :64], index=out.index, columns=[f"esm_{i}" for i in range(64)])
    out = out.join(esm)
    out.insert(0, "cohort", cohort)
    Dseed.columns = [f"D_raw_s{s}" for s in Dseed.columns]
    Dseed.insert(0, "hsapiens_gene_id", out["hsapiens_gene_id"])
    Sseed = S.copy()
    Sseed.columns = [f"S_s{s}" for s in Sseed.columns]
    Dseed = Dseed.join(Sseed)
    info = dict(cohort=cohort, n_genes=len(out), seeds=" ".join(map(str, runs)), n_seeds=len(runs),
                runs=" ".join(runs.values()))
    return out.reset_index(), Dseed.reset_index(), w, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--cohorts", nargs="+", default=COHORTS)
    a = ap.parse_args(argv)
    outdir = OUT if a.outdir is None else __import__("pathlib").Path(a.outdir)
    (outdir / "genes").mkdir(parents=True, exist_ok=True)
    (outdir / "seeds").mkdir(parents=True, exist_ok=True)
    infos, weights = [], []
    for c in a.cohorts:
        g, s, w, info = build(c)
        print(info, flush=True)
        for df, p in [(g, outdir / "genes" / f"{c}.parquet"), (s, outdir / "seeds" / f"{c}.parquet")]:
            tmp = p.with_name(p.name + ".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(p)
        infos.append(info)
        weights.append(w)
    K.write_csv(pd.DataFrame(infos), outdir / "seed_sets.csv")
    K.write_csv(pd.concat(weights, ignore_index=True), outdir / "celltype_weights.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
