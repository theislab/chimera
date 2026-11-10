"""EXPLORATORY: per-gene liver tables for the biology deep-dive.

Per liver cohort, one row per gene (``var_name``, joined on ``hsapiens_gene_id`` where other
tables are keyed by it):

* T2_5_celltype_margin gene table (canonical s42): alpha, prior, prior_raw, S_g_ema, tail, margin_support,
  dominant_cell_type, covariates, per-species detection.
* Consensus D (T3_3_consensus definition: mean over pooled seeds of alpha - prior, minus the cohort
  median), its seed SD, sign consistency and mean S_ema. liv_4sp (5 seeds) and liv_hmonkey
  (3 seeds) from results/analysis/T3_3; liv_hmouse (10 seeds) from results/analysis/X2/consensus
  (T3_3_consensus/run.py --outdir, all seeds finished at run time); liv_hzeb has one seed, so
  D = (alpha - prior) - median of s42 and the seed columns are NaN.
* Cell-type decomposition of the margin (T2_5_celltype_margin, s42): unweighted per-cell means S_gsk
  (``S__<species>__<type>``, model null) and within-type null (``W__<species>__<type>``);
  per-species contributions ``Cs__<species>`` = sum_k w_sk S_gsk (sum over species = S_g);
  per-type contributions ``Ck__<type>`` = C_gk.
* Baselines: B1b, B1b_raw, B1c (T4_2_gene_baselines); per species x type pseudobulk log2(1 + CPM)
  ``x__<species>__<type>`` (T4_2_gene_baselines definition, libraries >= 30 cells), detection ``f__``, and the
  log2 fold change to human per type ``lfc__<species>__<type>``.
* Presence: ``zero__<species>`` (no count in the cohort), ``absent__<species>`` (neither the
  native nor the human symbol is in the raw CAMEX object's feature set: a structural zero),
  ``complete`` (counts in every species).
* ``hep_enrich``: mean over species of hepatocyte pseudobulk minus the mean over that species'
  types (a hepatocyte-specificity covariate for the C_hep matching variant).
* ``D_matched_pct``: percentile of D among the 100 nearest genes in standardized rank space of
  (prior_raw, species-balanced log CP10K, species-balanced detection, log CDS length).

Writes results/analysis/X2/genes/<cohort>.parquet. Read-only elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T4_2_gene_baselines"))
import run as T42  # noqa: E402

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis"
OUT = RES / "X2"
LIVER = ["liv_4sp", "liv_hmonkey", "liv_hmouse", "liv_hzeb"]
CONSENSUS = {"liv_4sp": RES / "T3_3" / "liv_4sp__consensus.csv",
             "liv_hmonkey": RES / "T3_3" / "liv_hmonkey__consensus.csv",
             "liv_hmouse": OUT / "consensus" / "liv_hmouse__consensus.csv"}
MATCH_COLS = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
K_PCT = 100


def rank01(x: pd.Series) -> np.ndarray:
    return (x.rank(method="average") / x.notna().sum()).to_numpy()


def consensus(cohort: str, g: pd.DataFrame) -> pd.DataFrame:
    if cohort in CONSENSUS:
        c = pd.read_csv(CONSENSUS[cohort]).set_index("var_name")
        if not (c["symbol"].reindex(g.index).fillna("") == g["symbol"].fillna("")).all():
            raise AssertionError(f"{cohort}: consensus symbols differ from T2.5")
        return c[["D", "sd_delta_alpha", "sign_consistency", "n_seeds", "S_ema", "delta_alpha"]].rename(
            columns={"delta_alpha": "delta_alpha_cons", "S_ema": "S_ema_cons"})
    d = g["delta_alpha"] - g["delta_alpha"].median()
    return pd.DataFrame(dict(D=d, sd_delta_alpha=np.nan, sign_consistency=np.nan, n_seeds=1,
                             S_ema_cons=g["S_g_ema"], delta_alpha_cons=g["delta_alpha"]), index=g.index)


def decomposition(cohort: str) -> pd.DataFrame:
    s = pd.read_parquet(RES / "T2_5" / f"{cohort}__S_gsk.parquet")
    s = s[s.group_key == "cell_type"]
    out = {}
    for col, tag in (("S_gsk_meanz", "S"), ("S_gsk_within", "W")):
        p = s.pivot_table(index="var_name", columns=["species", "cell_type"], values=col)
        for sp_, t in p.columns:
            out[f"{tag}__{sp_}__{t}"] = p[(sp_, t)]
    s["wS"] = s["weight"] * s["S_gsk_meanz"]
    cs = s.groupby(["var_name", "species"])["wS"].sum().unstack()
    for sp_ in cs.columns:
        out[f"Cs__{sp_}"] = cs[sp_]
    ck = s.groupby(["var_name", "cell_type"])["wS"].sum().unstack()
    for t in ck.columns:
        out[f"Ck__{t}"] = ck[t]
    return pd.DataFrame(out)


def type_weights(cohort: str) -> pd.DataFrame:
    s = pd.read_parquet(RES / "T2_5" / f"{cohort}__S_gsk.parquet")
    s = s[s.group_key == "cell_type"].drop_duplicates(["species", "cell_type"])
    return s[["species", "cell_type", "n_cells", "weight"]].assign(cohort=cohort)


RAW_DIR = ROOT / "data" / "raw" / "camex" / "1liver" / "dataset"
RAW = {"Human": ("raw-liver-human-Martin.h5ad", "hsapiens_gene_name"),
       "Monkey": ("raw-liver-monkey-Martin.h5ad", "mmulatta_gene_name"),
       "Mouse": ("raw-liver-mouse-Martin.h5ad", "mmusculus_gene_name"),
       "Zebrafish": ("raw-liver-zebrafish-ggj5.h5ad", "drerio_gene_name")}


def presence(cohort: str, a, X) -> pd.DataFrame:
    """Per species: zero total counts in the cohort, and absent from the raw object's feature set."""
    import anndata as ad

    table = gene_ids.load_table(cohort)
    ids = pd.Index(a.var_names).str.replace("ortho_", "").astype(int)
    hum = table.loc[ids, "hsapiens_gene_name"].astype(str).to_numpy()
    out = {}
    for s in sorted(a.obs["species"].astype(str).unique()):
        m = (a.obs["species"].astype(str) == s).to_numpy()
        out[f"zero__{s}"] = np.asarray(X[m].sum(0)).ravel() == 0
        f, col = RAW[s]
        rv = set(ad.read_h5ad(RAW_DIR / f, backed="r").var_names)
        nat = table.loc[ids, col].astype(str).to_numpy()
        out[f"absent__{s}"] = np.array([(n not in rv) and (h not in rv) for n, h in zip(nat, hum)])
    df = pd.DataFrame(out, index=pd.Index(a.var_names, name="var_name"))
    df["n_zero_species"] = df[[c for c in df if c.startswith("zero__")]].sum(1)
    df["complete"] = df["n_zero_species"] == 0
    return df


def pseudobulk(cohort: str) -> pd.DataFrame:
    a = C.read_adata(cohort)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    pres = presence(cohort, a, X)
    species = a.obs["species"].astype(str).to_numpy()
    group = a.obs["cell_type"].astype(str).to_numpy()
    library = a.obs[T42.LIBRARY_KEY["liver"]].astype(str).to_numpy()
    x, f, _ = T42.pseudobulk(X, species, group, library, T42.MIN_CELLS_TYPE)
    df = {}
    for (s, t), v in x.items():
        df[f"x__{s}__{t}"] = v
        df[f"f__{s}__{t}"] = f[(s, t)]
    df = pd.DataFrame(df, index=pd.Index(a.var_names, name="var_name"))
    for (s, t) in x:
        if s != "Human" and ("Human", t) in x:
            df[f"lfc__{s}__{t}"] = df[f"x__{s}__{t}"] - df[f"x__Human__{t}"]
    return df.join(pres)


def matched_pct(g: pd.DataFrame, col: str = "D") -> pd.Series:
    ok = g[MATCH_COLS + [col]].notna().all(1).to_numpy()
    Z = np.column_stack([rank01(g.loc[ok, c]) for c in MATCH_COLS])
    nn = NearestNeighbors(n_neighbors=K_PCT + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    v = g.loc[ok, col].to_numpy()
    pct = (v[idx[:, 1:]] < v[:, None]).mean(1)
    out = pd.Series(np.nan, index=g.index)
    out.loc[ok] = pct
    return out


def build(cohort: str, cds: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(RES / "T2_5" / f"{cohort}__genes.parquet").set_index("var_name")
    g = g.join(consensus(cohort, g))
    g = g.join(cds[["log_cds_length"]], on="hsapiens_gene_id")
    bl = pd.read_csv(RES / "T4_2" / f"baselines__{cohort}.csv").set_index("hsapiens_gene_id")
    g = g.join(bl[["B1b", "B1b_raw", "B1c", "n_pairs_B1b"]], on="hsapiens_gene_id")
    g = g.join(decomposition(cohort))
    tot = g[[c for c in g.columns if c.startswith("Cs__")]].sum(1)
    err = float(np.abs(tot - g["S_full_meanz"]).max())
    if err > 1e-9:
        raise AssertionError(f"{cohort}: species identity error {err}")
    g = g.join(pseudobulk(cohort))
    # hepatocyte enrichment: mean over species of (hepatocyte pseudobulk - mean over that species' types)
    enr = []
    for sp_ in sorted({c.split("__")[1] for c in g.columns if c.startswith("x__")}):
        xs = g[[c for c in g.columns if c.startswith(f"x__{sp_}__")]]
        if f"x__{sp_}__Hepatocytes" in xs:
            enr.append(xs[f"x__{sp_}__Hepatocytes"] - xs.mean(1))
    g["hep_enrich"] = pd.concat(enr, axis=1).mean(1)
    g["D_matched_pct"] = matched_pct(g, "D")
    g["cohort"] = cohort
    print(f"{cohort}: {len(g)} genes, species identity err {err:.1e}, seeds {int(g.n_seeds.max())}", flush=True)
    return g.reset_index()


def main():
    gdir = OUT / "genes"
    gdir.mkdir(parents=True, exist_ok=True)
    cds = K.cds_table()
    ws = []
    for c in LIVER:
        build(c, cds).to_parquet(gdir / f"{c}.parquet", index=False)
        ws.append(type_weights(c))
    pd.concat(ws).to_csv(gdir / "type_weights.csv", index=False)


if __name__ == "__main__":
    main()
