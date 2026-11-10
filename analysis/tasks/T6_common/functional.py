"""Shared inputs of the functional layer (pre-specified analysis plan, section 7): residual readouts r1, r2, strata, gene
sets and interaction graphs. Gene key: human Ensembl ID, through ``gene_ids`` only.

r1 = rank-inverse-normal D, residualized on C + log variance. r2 = r1 further residualized on the
64 ESM PCs. The same two designs give the count-evidence comparators S1, S2 (from S_ema). The
prior comparator PI is rank-inverse-normal prior_raw on C without the prior, plus log variance.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import inputs as T43
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T6_1_functional.coverage import string_gene_map

REPO = gene_ids.REPO_ROOT
FUNC = REPO / "data" / "analysis" / "functional"
T33_DIR = REPO / "results" / "analysis" / "T3_3"
CACHE = REPO / "results" / "analysis" / "T6_3" / "cache"
PRIMARY = ["liv_4sp", "liv_hmouse"]
SECONDARY = ["test_3sp"]
COVS = K.COVARIATE_COLUMNS + ["log_var"]
N_PCS = 64
INTEGRATED_SEEDS = [42, 43, 44, 47, 48, 51]  # amendment log of the pre-specified analysis plan (2026-11-02)


def inverse_normal(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return norm.ppf((rankdata(np.asarray(x, dtype=float)) - 0.5) / n)


def gene_table(cohort: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Covariates, consensus readouts and 64 ESM PCs in the cohort universe (no missing covariate)."""
    f = CACHE / f"{cohort}__genes.parquet"
    fe = CACHE / f"{cohort}__esm.npy"
    if f.exists() and fe.exists():
        return pd.read_parquet(f), np.load(fe)
    a = C.read_adata(cohort)
    gene_ids.assert_table_matches(a, gene_ids.load_table(cohort))
    praw = C.raw_prior(cohort, C.CANONICAL[cohort], a.var["conservation_prior"].to_numpy(dtype=float))
    cov = K.cohort_covariates(cohort, adata=a, prior_raw=praw)
    E = np.asarray(a.varm["gene_features"], dtype=float)[:, :N_PCS]
    cons = pd.read_csv(T33_DIR / f"{cohort}__consensus.csv")
    if not (cons["var_name"].to_numpy() == cov.index.to_numpy()).all():
        raise AssertionError(f"{cohort}: consensus gene order differs from the h5ad")
    tab = cov[["hsapiens_gene_id"] + COVS].reset_index().copy()
    for c in ("symbol", "D", "S_ema", "delta_alpha", "sign_consistency"):
        tab[c] = cons[c].to_numpy()
    ok = tab[COVS].notna().all(1).to_numpy() & tab["D"].notna().to_numpy() & tab["S_ema"].notna().to_numpy()
    ok &= ~tab["hsapiens_gene_id"].duplicated(keep=False).to_numpy()
    tab, E = tab.loc[ok].reset_index(drop=True), E[ok]
    CACHE.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(str(f) + ".tmp")
    Path(str(f) + ".tmp").replace(f)
    with open(str(fe) + ".tmp", "wb") as fh:
        np.save(fh, E)
    Path(str(fe) + ".tmp").replace(fe)
    return tab, E


def designs(tab: pd.DataFrame, E: np.ndarray) -> dict:
    """Design matrices built once in the universe."""
    Z1 = K.design(tab, columns=COVS)
    Z2 = ST.design(tab, COVS, extra=E)
    Zpi = K.design(tab, columns=[c for c in COVS if c != "prior_raw"])
    return dict(Z1=Z1, Z2=Z2, Zpi=Zpi)


def residuals(tab: pd.DataFrame, Z: dict, extra: dict | None = None) -> pd.DataFrame:
    """Readouts on the residual scale. ``extra``: name -> raw score treated like D (r1, r2 versions)."""
    d = inverse_normal(tab["D"].to_numpy())
    s = inverse_normal(tab["S_ema"].to_numpy())
    pi = inverse_normal(tab["prior_raw"].to_numpy())
    out = pd.DataFrame({"r1": ST.resid(d, Z["Z1"]), "r2": ST.resid(d, Z["Z2"]),
                        "S1": ST.resid(s, Z["Z1"]), "S2": ST.resid(s, Z["Z2"]),
                        "PI": ST.resid(pi, Z["Zpi"])}, index=tab.index)
    for k, v in (extra or {}).items():
        x = inverse_normal(v)
        out[f"{k}1"] = ST.resid(x, Z["Z1"])
        out[f"{k}2"] = ST.resid(x, Z["Z2"])
    return out


def seed_runs(cohort: str) -> dict[int, str]:
    """Seeds that make up D (T3_3_consensus pooling record; CPU runs win over an MPS run of the same number)."""
    return T43.seed_runs_used(cohort)


def seed_matrix(cohort: str, tab: pd.DataFrame, key: str = "D") -> pd.DataFrame:
    """Per-seed values in the universe of ``tab``."""
    from analysis.tasks.T3_3_consensus import run as T33

    a = C.read_adata(cohort, backed=True)
    names = pd.Index(np.asarray(a.var_names))
    a.file.close()
    pos = names.get_indexer(tab["var_name"])
    if (pos < 0).any():
        raise AssertionError(f"{cohort}: universe genes missing from the h5ad")
    out = {}
    for s, r in seed_runs(cohort).items():
        out[s] = K.read_run(r)[key][pos]
    return pd.DataFrame(out, index=tab.index)


def integrated_D(tab: pd.DataFrame) -> np.ndarray:
    """liv_hmouse D from the integrated seeds only (amendment log of the pre-specified analysis plan, 2026-11-02)."""
    M = seed_matrix("liv_hmouse", tab, "D")
    v = M[INTEGRATED_SEEDS].mean(1).to_numpy()
    return v - np.median(v)


# ---------------------------------------------------------------- gene sets
def _gmt_text(text: str) -> dict[str, list[str]]:
    d = {}
    for line in text.splitlines():
        p = line.split("\t")
        if len(p) > 2:
            d[p[0]] = [g.split(",")[0] for g in p[2:] if g]
    return d


def symbol_mapper():
    import sys

    sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T4_1_gold"))
    from gold_checks import SymbolMapper, gtf_gene_names

    names, _ = gtf_gene_names(REPO / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz")
    return SymbolMapper(REPO / "data/analysis/gold/G10/hgnc_complete_set.txt", names)


def raw_libraries() -> dict[str, dict[str, set[str]]]:
    """Library -> set -> ENSG. Enrichr verbatim responses (the libraries the paper used); CollecTRI
    regulons of single-gene TFs, as sets only."""
    f = CACHE / "libraries.json"
    if f.exists():
        return {k: {t: set(v) for t, v in d.items()} for k, d in json.loads(f.read_text()).items()}
    hall = _gmt_text((FUNC / "Enrichr.MSigDB_Hallmark_2020.server_response.txt").read_text())
    reac = _gmt_text((FUNC / "Enrichr.Reactome_Pathways_2024.server_response.txt").read_text())
    col = pd.read_csv(FUNC / "collectri_human.tsv", sep="\t", dtype=str)
    col = col[~col.source_genesymbol.str.contains("_", na=True)]
    tri = col.groupby("source_genesymbol").target_genesymbol.apply(lambda s: sorted(set(s.dropna()))).to_dict()
    allsym = sorted(set().union(*map(set, hall.values()), *map(set, reac.values()), *map(set, tri.values()), tri))
    mm = symbol_mapper().map(allsym).set_index("symbol").ensg.dropna().to_dict()
    conv = lambda d: {t: {mm[g] for g in v if g in mm} for t, v in d.items()}
    out = {"Hallmark": conv(hall), "Reactome": conv(reac), "CollecTRI": conv(tri)}
    out["_tf_ensg"] = {t: {mm[t]} for t in tri if t in mm}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({k: {t: sorted(v) for t, v in d.items()} for k, d in out.items()}))
    return out


def prune_jaccard(sets: dict[str, set], thr: float = 0.5) -> dict[str, set]:
    """Greedy: largest set first, ties by name; drop a set whose Jaccard with a kept set exceeds thr."""
    kept: dict[str, set] = {}
    for t in sorted(sets, key=lambda k: (-len(sets[k]), k)):
        s = sets[t]
        if all(len(s & v) / len(s | v) <= thr for v in kept.values()):
            kept[t] = s
    return kept


def libraries(tab: pd.DataFrame) -> dict[str, dict[str, set]]:
    """Sets restricted to the universe. Hallmark >= 15 genes; Reactome 15-200, pruned at Jaccard > 0.5;
    CollecTRI: TF in the universe, >= 10 targets."""
    U = set(tab["hsapiens_gene_id"])
    L = raw_libraries()
    hall = {t: v & U for t, v in L["Hallmark"].items() if len(v & U) >= 15}
    reac = {t: v & U for t, v in L["Reactome"].items() if 15 <= len(v & U) <= 200}
    tf = L["_tf_ensg"]
    tri = {t: v & U for t, v in L["CollecTRI"].items() if len(v & U) >= 10 and t in tf and tf[t] & U}
    return {"Hallmark": hall, "Reactome": prune_jaccard(reac), "CollecTRI": tri}


def membership(tab: pd.DataFrame, sets: dict[str, set]) -> tuple[sp.csr_matrix, list[str]]:
    pos = {g: i for i, g in enumerate(tab["hsapiens_gene_id"])}
    names = list(sets)
    r, c = [], []
    for k, t in enumerate(names):
        idx = [pos[g] for g in sets[t]]
        r += [k] * len(idx)
        c += idx
    M = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(names), len(tab)))
    return M, names


# ---------------------------------------------------------------- graphs
def huri_edges() -> pd.DataFrame:
    return pd.read_csv(FUNC / "HuRI.tsv", sep="\t", header=None, names=["a", "b"], dtype=str)


def string_edges(taxon: str, thr: int) -> pd.DataFrame:
    """STRING physical links, direct ``experiments`` channel >= thr, mapped to Ensembl genes."""
    prefix = "ENSG" if taxon == "9606" else "ENSMUSG"
    gm = string_gene_map(FUNC / f"{taxon}.protein.aliases.v12.0.txt.gz", prefix)
    s = pd.read_csv(FUNC / f"{taxon}.protein.physical.links.full.v12.0.txt.gz", sep=" ",
                    usecols=["protein1", "protein2", "experiments"])
    s = s[s.experiments >= thr]
    s = pd.DataFrame({"a": s.protein1.map(gm), "b": s.protein2.map(gm)}).dropna()
    return s


def graph_arrays(edges: pd.DataFrame, ids: pd.Series) -> np.ndarray:
    """Unique undirected edges (u < v) on positions in ``ids``, no self-loops."""
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    u, v = edges.a.map(pos), edges.b.map(pos)
    m = u.notna() & v.notna()
    u, v = u[m].astype(int).to_numpy(), v[m].astype(int).to_numpy()
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    keep = lo != hi
    return np.unique(np.column_stack([lo[keep], hi[keep]]), axis=0)
