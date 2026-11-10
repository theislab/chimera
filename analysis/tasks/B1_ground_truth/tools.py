"""SATURN (1:1 arm, Adapter B) and CAMEX on altered objects, main settings.

Same adapters and parameters as the cell-level runs of the two tools (analysis/tasks/T5_2_adapters, T5_3_runs).

Subcommands, each in its own env:
  build         chimera_env   altered object into data/analysis/B1/tools/<run>/altered.h5ad
  prep_saturn   saturn_env    per-species h5ads by the Adapter B rule (prep_saturn_1to1.py)
  prep_camex    camex_env     input files of T5_2_adapters with the altered columns replaced; kidney inputs built from the object
  score_saturn  chimera_env   per-ortholog S_deploy, S_cos (T5_6_gene_level definitions) -> results/analysis/B1/tool_scores/saturn__<run>.parquet
  score_camex   chimera_env   C_cos -> tool_scores/camex__<run>.parquet
Rows: rules.TOOL_ROWS x bands x seeds at share 5%, plus unaltered objects (run b1-<cohort>-none-s<seed>); tier 3 are the main settings.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T5_2_adapters"))
import cohorts as C5  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "data" / "analysis" / "B1" / "tools"
STEM = {"liv_hmouse": "camex_human_mouse", "test_hmouse": "camex_testis_human_mouse", "kidney": "kidney_healthy_train"}
TABLE = {"liv_hmouse": REPO / "data/processed/ortholog_table.csv", "test_hmouse": REPO / "data/processed/ortholog_table_testis.csv",
         "kidney": REPO / "data/analysis/disease_kidney/ortholog_table_kidney.csv"}
SATURN_SP = {"Human": "human", "Mouse": "mouse"}
SYMBOL = {"Human": "hsapiens_gene_name", "Mouse": "mmusculus_gene_name"}
CAMEX_NAME = {"liv_hmouse": ("raw-liver-human-Martin", "raw-liver-mouse-Martin"),
              "test_hmouse": ("raw-testis-human-Murat", "raw-testis-mouse-Murat"),
              "kidney": ("raw-kidney-human-KPMP", "raw-kidney-mouse-KPMP")}


def wdir(run: str) -> Path:
    return TOOLS / os.environ["B1_TOOL"] / run


def symbol_table(cohort: str, var_names) -> pd.DataFrame:
    tab = pd.read_csv(TABLE[cohort]).set_index("ortholog_group_id")
    ids = pd.Index(var_names).astype(str).str.replace("ortho_", "", regex=False).astype(int)
    return pd.DataFrame({"var_name": pd.Index(var_names).astype(str), "Human": tab.loc[ids, SYMBOL["Human"]].values,
                         "Mouse": tab.loc[ids, SYMBOL["Mouse"]].values})


def build(row: dict):
    from analysis.tasks.B1_ground_truth import alter as A
    from analysis.tasks.B1_ground_truth import rules as RU

    d = wdir(row["run"])
    d.mkdir(parents=True, exist_ok=True)
    if row["alt"] == "none":
        a = A.read_base(row["cohort"])
        a.write_h5ad(d / "altered.h5ad")
        return
    A.build_altered(row, out_path=d / "altered.h5ad", check=False)


def prep_saturn(row: dict):
    import anndata as ad

    d = wdir(row["run"])
    out = d / "saturn_in"
    out.mkdir(parents=True, exist_ok=True)
    full = ad.read_h5ad(d / "altered.h5ad")
    syms = symbol_table(row["cohort"], full.var_names)
    rows = []
    man = {"stem": STEM[row["cohort"]], "species": {}}
    for sp_proc in ("Human", "Mouse"):
        m = (full.obs["species"].astype(str) == sp_proc).to_numpy()
        X = sp.csr_matrix(full.X[m], dtype=np.float32)
        s = pd.Series(syms[sp_proc].values, index=full.var_names.astype(str), dtype=object)
        keep = s.notna().to_numpy() & ~s.duplicated(keep="first").to_numpy()
        keep &= np.asarray((X > 0).sum(axis=0)).ravel() >= 3
        x = ad.AnnData(X=X[:, keep], obs=pd.DataFrame(index=full.obs_names[m].astype(str)), var=pd.DataFrame(index=s[keep].astype(str).values))
        x.obs["cell_type"] = full.obs["cell_type"].astype(str).values[m]
        x.obs["cell_ontology_class"] = full.obs["cell_ontology_class"].astype(str).values[m]
        x.obs["species_processed"] = sp_proc
        f = out / f"{SATURN_SP[sp_proc]}.h5ad"
        x.write_h5ad(f)
        rows.append({"path": str(f), "species": SATURN_SP[sp_proc], "embedding_path": str(C5.ESM2_DIR / f"{SATURN_SP[sp_proc]}_embedding.torch")})
        man["species"][SATURN_SP[sp_proc]] = {"n_cells": int(x.n_obs), "n_genes": int(x.n_vars)}
    pd.DataFrame(rows).sort_values("species").to_csv(out / "run.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))


def prep_camex(row: dict):
    import anndata as ad

    d = wdir(row["run"])
    out = d / "camex_in"
    out.mkdir(parents=True, exist_ok=True)
    full = ad.read_h5ad(d / "altered.h5ad")
    syms = symbol_table(row["cohort"], full.var_names)
    cohort = row["cohort"]
    names = CAMEX_NAME[cohort]
    man = {"stem": STEM[cohort], "files": {}}
    stats = {}
    if cohort == "kidney":
        for sp_proc, nm in zip(("Human", "Mouse"), names):
            m = (full.obs["species"].astype(str) == sp_proc).to_numpy()
            s = pd.Series(syms[sp_proc].values, dtype=object)
            keep = s.notna().to_numpy() & ~s.duplicated(keep="first").to_numpy()
            x = ad.AnnData(X=sp.csr_matrix(full.X[m][:, keep], dtype=np.float32), obs=full.obs.loc[m, ["cell_type", "cell_ontology_class"]].astype(str).copy(),
                           var=pd.DataFrame(index=s[keep].astype(str).values))
            x.write_h5ad(out / f"{nm}.h5ad")
            man["files"][f"{nm}.h5ad"] = {"n_cells": int(x.n_obs), "n_genes": int(x.n_vars)}
        pairs = syms.dropna(subset=["Human", "Mouse"])
        pd.DataFrame({f"{names[0]}.h5ad": pairs["Human"].values, f"{names[1]}.h5ad": pairs["Mouse"].values}).to_csv(out / "gene_matches_human2mouse.csv", index=False)
    else:
        src = C5.INPUTS / STEM[cohort] / "camex"
        base = ad.read_h5ad(C5.PROCESSED / f"{STEM[cohort]}.h5ad")
        Xb = sp.csr_matrix(base.X, dtype=np.float32)
        Xa = sp.csr_matrix(full.X, dtype=np.float32)
        Dd = (Xa - Xb).tocsc()
        Dd.eliminate_zeros()
        changed = np.flatnonzero(np.diff(Dd.indptr) > 0)
        for sp_proc, nm in zip(("Human", "Mouse"), names):
            r = ad.read_h5ad(src / f"{nm}.h5ad")
            X = sp.csc_matrix(r.X, dtype=np.float32)
            m = (base.obs["species"].astype(str) == sp_proc).to_numpy()
            pos = pd.Index(base.obs_names.astype(str)[m]).get_indexer(r.obs_names.astype(str))
            if (pos < 0).any():
                raise SystemExit("raw cells differ from the object")
            vidx = pd.Index(r.var_names.astype(str))
            rows, cols, vals = [], [], []
            n_miss = n_mismatch = 0
            for j in changed:
                sym = syms[sp_proc].iloc[j]
                if not isinstance(sym, str) or sym not in vidx:
                    n_miss += 1
                    continue
                cj = vidx.get_loc(sym)
                if not isinstance(cj, (int, np.integer)):
                    n_miss += 1
                    continue
                dv = np.asarray(Dd[:, j].todense()).ravel()[m][pos]
                b0 = np.asarray(Xb[:, j].todense()).ravel()[m][pos]
                rc = np.asarray(X[:, cj].todense()).ravel()
                if not np.array_equal(b0, rc):
                    n_mismatch += 1
                nz = np.flatnonzero(dv)
                rows.append(nz)
                cols.append(np.full(len(nz), cj))
                vals.append(dv[nz])
            if rows:
                dd = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=X.shape, dtype=np.float32)
                X = (X + dd).tocsr()
                X.data = np.maximum(X.data, 0)
                X.eliminate_zeros()
            r.X = sp.csr_matrix(X)
            r.write_h5ad(out / f"{nm}.h5ad")
            man["files"][f"{nm}.h5ad"] = {"n_cells": int(r.n_obs), "n_genes": int(r.n_vars)}
            stats[sp_proc] = dict(changed=len(changed), no_raw_symbol=n_miss, raw_differs=n_mismatch)
        rel = list(src.glob("gene_matches_*.csv"))[0]
        shutil.copy(rel, out / rel.name)
    man["stats"] = stats
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("prep_camex", row["run"], stats, flush=True)


def _cos(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def score_saturn(row: dict):
    import anndata as ad

    from analysis.tasks.B1_ground_truth import alter as A
    from analysis.tasks.B1_ground_truth import rules as RU

    d = wdir(row["run"])
    work = d / "saturn_work"
    final = pickle.load(open(glob.glob(str(work / "saturn_results/*_genes_to_macrogenes.pkl"))[0], "rb"))
    f = [p for p in glob.glob(str(work / "saturn_results/*.h5ad")) if not p.endswith("_pretrain.h5ad") and "_ep_" not in Path(p).name]
    p = ad.read_h5ad(d / "altered.h5ad")
    syms = symbol_table(row["cohort"], p.var_names)
    a = ad.read_h5ad(f[0])
    a = a[p.obs_names.astype(str).to_numpy()]
    lab = A.labels(p, row["cohort"])
    spc = p.obs["species"].astype(str).to_numpy()
    Xm = np.asarray(a.obsm["macrogenes"])
    types = A.shared_types(lab, spc)
    H = np.column_stack([Xm[(spc == "Human") & (lab == t)].mean(0) for t in types])
    M = np.column_stack([Xm[(spc == "Mouse") & (lab == t)].mean(0) for t in types])
    H0, M0 = H - H.mean(1, keepdims=True), M - M.mean(1, keepdims=True)
    den = np.sqrt((H0 ** 2).sum(1) * (M0 ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = np.where(den > 0, (H0 * M0).sum(1) / den, np.nan)
    kh, km = "human_" + syms["Human"].astype(str), "mouse_" + syms["Mouse"].astype(str)
    ok = np.array([x in final and y in final for x, y in zip(kh, km)])
    Wh = np.vstack([final[k] for k in kh[ok]]).astype(np.float64)
    Wm = np.vstack([final[k] for k in km[ok]]).astype(np.float64)
    good = np.isfinite(rm)
    wbar = (Wh + Wm) / 2
    res = pd.DataFrame({"var_name": syms["var_name"]}).set_index("var_name")
    res["S_deploy"] = np.nan
    res["S_cos"] = np.nan
    res.loc[syms["var_name"][ok], "S_deploy"] = (wbar[:, good] * rm[good]).sum(1) / wbar[:, good].sum(1)
    res.loc[syms["var_name"][ok], "S_cos"] = _cos(Wh, Wm)
    (RU.RES / "tool_scores").mkdir(parents=True, exist_ok=True)
    res.reset_index().to_parquet(RU.RES / "tool_scores" / f"saturn__{row['run']}.parquet")


def score_camex(row: dict):
    import anndata as ad

    from analysis.tasks.B1_ground_truth import rules as RU

    d = wdir(row["run"])
    a = ad.read_h5ad(d / "camex_work" / "gene_integration_hidden.h5ad")
    a.obs_names = a.obs_names.astype(str)
    a.obs["gene"] = a.obs_names
    b = a.obs["batch"].astype(str).to_numpy()
    X = np.asarray(a.X, dtype=np.float64)
    gene = a.obs["gene"].to_numpy()
    names = CAMEX_NAME[row["cohort"]]
    pos_h = {g: i for i, g in zip(np.where(b == names[0])[0], gene[b == names[0]])}
    pos_m = {g: i for i, g in zip(np.where(b == names[1])[0], gene[b == names[1]])}
    if not pos_h or not pos_m:
        raise SystemExit(f"dataset names not found in {sorted(set(b))}")
    p = ad.read_h5ad(d / "altered.h5ad", backed="r")
    syms = symbol_table(row["cohort"], p.var_names)
    ih, im = syms["Human"].map(pos_h), syms["Mouse"].map(pos_m)
    ok = (ih.notna() & im.notna()).to_numpy()
    res = pd.DataFrame({"var_name": syms["var_name"]}).set_index("var_name")
    res["C_cos"] = np.nan
    res.loc[syms["var_name"][ok], "C_cos"] = _cos(X[ih[ok].astype(int).to_numpy()], X[im[ok].astype(int).to_numpy()])
    (RU.RES / "tool_scores").mkdir(parents=True, exist_ok=True)
    res.reset_index().to_parquet(RU.RES / "tool_scores" / f"camex__{row['run']}.parquet")


def score_csv(row: dict):
    """Primary scalar of an overview-benchmark adapter; a gene without a row has no verdict and stays unscored."""
    from analysis.tasks.B1_ground_truth import rules as RU

    tool = os.environ["B1_TOOL"]
    t = pd.read_csv(wdir(row["run"]) / "out" / f"{tool}.csv")
    col = {"genespectra": "GeneSpectra", "cacimar": "CACIMAR", "samap": "SAMap"}[tool]
    (RU.RES / "tool_scores").mkdir(parents=True, exist_ok=True)
    t[["var_name", "score"]].rename(columns={"score": col}).to_parquet(RU.RES / "tool_scores" / f"{tool}__{row['run']}.parquet")


def make_manifest():
    from analysis.tasks.B1_ground_truth import rules as RU

    rows = []
    for c in RU.COHORTS:
        for (alt, e) in RU.TOOL_ROWS:
            for b in RU.BANDS:
                for s in RU.SEEDS:
                    main = (alt, e) in RU.TOOL_MAIN_ROWS and b in RU.TOOL_MAIN_BANDS and s == 42
                    rows.append(dict(cohort=c, alt=alt, effect=e, band=b, share=RU.TOOL_SHARE, seed=s, run=RU.run_name(c, alt, e, b, RU.TOOL_SHARE, s),
                                     stem=STEM[c], tier=3 if main else 6, label_key="cell_ontology_class"))
        for s in RU.TOOL_NONE_SEEDS:
            rows.append(dict(cohort=c, alt="none", effect=0, band="all", share=0.0, seed=s, run=f"b1-{c}-none-s{s}", stem=STEM[c], tier=3 if s == 42 else 6, label_key="cell_ontology_class"))
    pd.DataFrame(rows).to_csv(RU.TASK_DIR / "grid_tools_v2.tsv", sep="\t", index=False)
    print(len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "prep_saturn", "prep_camex", "score_saturn", "score_camex", "score_csv", "manifest"])
    ap.add_argument("--manifest")
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.cmd == "manifest":
        make_manifest()
        return
    row = pd.read_csv(a.manifest, sep="\t").iloc[a.index].to_dict()
    globals()[a.cmd](row)


if __name__ == "__main__":
    main()
