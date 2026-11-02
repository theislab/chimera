"""T11.6: glaucoma gene translation on Van Zyl, V1-V3 of section 11 (Family 3, descriptive).

Primary dataset fixed before any glaucoma readout (ledger T11_6): the full five-species cohort.
The other usable datasets (``vz.usable_datasets``) are reported beside it. Gene list:
``revision/tasks/T11_1_vanzyl/glaucoma_genes.tsv`` (committed at T11.1), restricted to genes in the
dataset vocabulary. Integration quality of each dataset is written beside every readout.

* V1: D of glaucoma genes against background in the same raw prior x expression decile cells
  (strata kept with >= 5 background genes per set gene; T4.3 functions). Stratified AUROC, 1,000
  gene bootstraps, two-sided permutation p (5,000 draws within strata). No direction is claimed.
  Beside it: the same for B1b (direct pseudobulk baseline, residualised) and S_ema. Subsets: all
  list genes; genes named in van Zyl 2020.
* V2: per-species fidelity of each gene across the outflow classes (``cell_class`` conventional or
  uveoscleral outflow) that species s shares with human.
  - Chimera per-species residual (the plan's readout): residual-head RMS per gene (``residuals.py``),
    mean over those classes and over the seeds in D; lower means closer to the shared program.
  - Chimera per-species fit margin S_gs (T2.5 forward pass, ``cell_ontology_class`` groups), mean
    over those classes and over the seeds in D; higher means better fit by the shared program.
  - Direct pseudobulk: Pearson r between the human and species-s pseudobulk profiles over the same
    classes (T4.2 ``pair_corr``; units ``obs['batch']`` with >= 30 cells).
  Reported per species: mean over glaucoma genes (absolute; published expectation: macaques closer
  to human than pig and mouse), the same for matched background, and the background-relative
  percentile (glaucoma genes' mean within-species percentile). Contrast macaques - (pig, mouse) with
  a 1,000-draw gene bootstrap CI.
* V3: for each glaucoma gene, the cell types that carry its divergence: Chimera C_gk (harmonised
  fine types; top divergent type = argmax of max(-C_gk, 0)) beside the pseudobulk leave-one-class-out
  driver (T12-style increase of E when a class is removed) and the human top-expressing fine type.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_6_vz_glaucoma/glaucoma.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.tasks.T11_5_vz_genes import vz
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.T4_3_scoreboard import stats as S

sys.path.insert(0, str(vz.ROOT / "revision" / "tasks" / "T2_5_celltype_margin"))
import margin as MG  # noqa: E402

PRIMARY = "full"
OUT = vz.OUT / "glaucoma"
GLAUCOMA = vz.ROOT / "revision" / "tasks" / "T11_1_vanzyl" / "glaucoma_genes.tsv"
OUTFLOW = {"conventional outflow", "uveoscleral outflow"}
MACAQUES = ["macaca_fascicularis", "macaca_mulatta"]
OTHERS = ["pig", "mouse"]
N_BOOT, N_PERM = 1000, 5000
METHODS = ("residual", "margin", "pseudobulk")


def glaucoma_list() -> pd.DataFrame:
    gl = pd.read_csv(GLAUCOMA, sep="\t")
    gl = gl[gl["hsapiens_gene_id"].notna()].drop_duplicates("hsapiens_gene_id")
    gl["vanzyl_named"] = gl["source"].str.contains("vanzyl")
    return gl


def integration_quality(ds: str) -> dict:
    t = pd.read_csv(vz.OUT / "integration" / "chimera_seeds.csv")
    t = t[t["dataset"] == ds]
    ref = json.loads((vz.OUT / "readout" / f"meta_{ds}.json").read_text())["integration_reference"]
    return dict(n_seeds=int(len(t)), knn_mixing_entropy=float(t["knn_mixing_entropy"].mean()),
                entropy_random=ref["entropy_random"],
                entropy_ratio=float(t["knn_mixing_entropy"].mean() / ref["entropy_random"]),
                adv_acc=float(t["adv_acc_final"].mean()), majority_rate=ref["majority_rate"],
                knn_species_purity=float(t["knn_species_purity"].mean()),
                label_transfer_acc=float(t["label_transfer_acc"].mean()))


def v1(g: pd.DataFrame, gl: pd.DataFrame, rng) -> list[dict]:
    rows = []
    for subset, ids in (("all", set(gl["hsapiens_gene_id"])), ("vanzyl_named", set(gl.loc[gl["vanzyl_named"], "hsapiens_gene_id"]))):
        for score in ("D", "B1b", "S_ema", "D_raw"):
            m = g[[score, "prior_raw", "log_cp10k_spmean"]].notna().all(1).to_numpy()
            T = g.loc[m]
            inset = T["hsapiens_gene_id"].isin(ids).to_numpy()
            lab = S.cells(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float))
            keep = S.kept_strata(inset, lab)
            r = dict(subset=subset, score=score, n_in_vocab=int(inset.sum()), n_set=int(inset[keep].sum()),
                     n_background=int((~inset[keep]).sum()))
            if r["n_set"] >= 3:
                sc, ins, lb = T[score].to_numpy(float)[keep], inset[keep], lab[keep]
                r["auroc"] = S.strat_auroc(sc, ins, lb)
                A, _ = S.auroc_boot(sc, ins, lb, N_BOOT, rng)
                r["auroc_ci_lo"], r["auroc_ci_hi"] = S.pct_ci(A)
                r["p_perm_two_sided"] = S.p_two_sided(r["auroc"], S.auroc_perm(sc, ins, lb, N_PERM, rng))
                r["median_score_set"] = float(np.median(sc[ins]))
                r["median_score_background"] = float(np.median(sc[~ins]))
            rows.append(r)
    return rows


def forward_mean(ds: str, key: str, var_names) -> dict:
    """Mean over the seeds in D of the T2.5 decomposition for one grouping key."""
    decs = []
    for s in vz.seeds_in_D(ds):
        fw = dict(np.load(vz.OUT / "drivers" / f"fw_{ds}_s{s}.npz", allow_pickle=False))
        if list(fw["var_names"]) != list(var_names):
            raise AssertionError(f"{ds} s{s}: var_names differ")
        decs.append(MG.decompose(fw, key))
    d0 = decs[0]
    for d in decs[1:]:
        if not (np.array_equal(d["labels"], d0["labels"]) and np.array_equal(d["species_idx"], d0["species_idx"])):
            raise AssertionError(f"{ds}: group layout differs between seeds")
    return dict(species=d0["species"], labels=d0["labels"], species_idx=d0["species_idx"], n=d0["n"],
                S_gsk=np.mean([d["S_gsk_meanz"] for d in decs], axis=0),
                C_gk={k: np.mean([d["C_gk_meanz"][k] for d in decs], axis=0) for k in d0["cell_types"]},
                cell_types=d0["cell_types"], n_seeds=len(decs))


def residual_mean(ds: str, var_names) -> dict:
    """Mean over the seeds in D of the per-species, per-class residual RMS (``residuals.py``)."""
    Rs = []
    for s in vz.seeds_in_D(ds):
        z = dict(np.load(vz.OUT / "residuals" / f"res_{ds}_s{s}.npz", allow_pickle=False))
        if list(z["var_names"]) != list(var_names):
            raise AssertionError(f"{ds} s{s}: residual var_names differ")
        Rs.append(z)
    return dict(rms=np.mean([z["rms"] for z in Rs], axis=0), species_idx=Rs[0]["species_idx"], labels=Rs[0]["labels"],
                species=[str(s) for s in Rs[0]["species"]])


def drivers_summary(ds: str, a) -> pd.DataFrame:
    """All genes: share of the total divergent contribution (max(-C_gk, 0)) carried by each class."""
    fwd = forward_mean(ds, vz.CLASS_KEY, a.var_names)
    types = fwd["cell_types"]
    div = np.maximum(-np.stack([fwd["C_gk"][k] for k in types]), 0)
    top = div.argmax(0)
    has = div.max(0) > 0
    n = pd.crosstab(a.obs[vz.CLASS_KEY].astype(str), a.obs["species"].astype(str))
    return pd.DataFrame(dict(dataset=ds, cell_class=types, share_divergent_mass=div.sum(1) / div.sum(),
                             frac_genes_top=[float((top[has] == j).mean()) for j in range(len(types))],
                             n_species_with_class=[int((n.loc[k] > 0).sum()) for k in types],
                             n_cells=[int(n.loc[k].sum()) for k in types]))


def pseudobulk(a):
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    x, f, _ = T42.pseudobulk(X, species, a.obs[vz.CLASS_KEY].astype(str).to_numpy(),
                             a.obs["batch"].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    return x, f, sorted(set(species))


def v2(ds: str, a, g: pd.DataFrame, gl: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    cls = a.obs[[vz.CLASS_KEY, "cell_class"]].astype(str).drop_duplicates()
    outflow = set(cls.loc[cls["cell_class"].isin(OUTFLOW), vz.CLASS_KEY])
    fwd = forward_mean(ds, vz.CLASS_KEY, a.var_names)
    res = residual_mean(ds, a.var_names)
    x, f, sl = pseudobulk(a)
    hi = fwd["species"].index("human")
    fid = pd.DataFrame(index=g.index)
    info = {}
    for s in [s for s in fwd["species"] if s != "human"]:
        si = fwd["species"].index(s)
        h_cls = set(fwd["labels"][fwd["species_idx"] == hi])
        s_cls = set(fwd["labels"][fwd["species_idx"] == si])
        shared = sorted(outflow & h_cls & s_cls)
        rows_s = [(i) for i, (sp_, l) in enumerate(zip(fwd["species_idx"], fwd["labels"])) if sp_ == si and l in shared]
        fid[f"margin_{s}"] = fwd["S_gsk"][rows_s].mean(0)
        res_rows = [i for i, (sp_, l) in enumerate(zip(res["species_idx"], res["labels"]))
                    if res["species"][sp_] == s and l in shared]
        fid[f"residual_{s}"] = res["rms"][res_rows].mean(0)
        grp = [k for k in T42.shared_groups(x, sl, ["human", s]) if k in outflow]
        rr, _ = T42.pair_corr(x, f, "human", s, grp, a.n_vars)
        fid[f"pseudobulk_{s}"] = rr
        info[s] = dict(margin_classes=shared, residual_classes=shared, pseudobulk_classes=grp)
    ids = set(gl["hsapiens_gene_id"])
    inset = g["hsapiens_gene_id"].isin(ids).to_numpy()
    rows = []
    for method in METHODS:
        cols = [c for c in fid.columns if c.startswith(method + "_")]
        P = fid[cols].rank(pct=True)
        for c in cols:
            s = c.split("_", 1)[1]
            v = fid[c].to_numpy()
            fin = np.isfinite(v)
            rows.append(dict(dataset=ds, method=method, species=s, classes=len(info[s][f"{method}_classes"]),
                             n_glaucoma=int((inset & fin).sum()), mean_glaucoma=float(np.nanmean(v[inset])),
                             mean_background=float(np.nanmean(v[~inset])),
                             mean_percentile_glaucoma=float(np.nanmean(P[c].to_numpy()[inset]))))
    tab = pd.DataFrame(rows)
    # contrast macaques - (pig, mouse) on glaucoma genes, absolute and percentile, gene bootstrap
    con = []
    if all(f"margin_{s}" in fid for s in MACAQUES + OTHERS):
        idx = np.where(inset)[0]
        for method in METHODS:
            V = fid[[f"{method}_{s}" for s in MACAQUES + OTHERS]].to_numpy()
            Pm = fid[[f"{method}_{s}" for s in MACAQUES + OTHERS]].rank(pct=True).to_numpy()
            for scale, M in (("absolute", V), ("percentile", Pm)):
                def stat(ii):
                    m = np.nanmean(M[ii], axis=0)
                    return float(m[:2].mean() - m[2:].mean())
                est = stat(idx)
                bg = stat(np.where(~inset)[0])
                bs = [stat(rng.choice(idx, len(idx))) for _ in range(N_BOOT)]
                lo, hi_ = np.percentile(bs, [2.5, 97.5])
                con.append(dict(dataset=ds, method=method, scale=scale, contrast="macaques minus pig and mouse",
                                glaucoma=est, ci_lo=float(lo), ci_hi=float(hi_), background=bg))
    fid.insert(0, "hsapiens_gene_id", g["hsapiens_gene_id"].values)
    fid.insert(1, "symbol", g["symbol"].values)
    fid["glaucoma"] = inset
    fid.to_csv(OUT / f"v2_fidelity_{ds}.csv")
    vz.write_json(OUT / f"v2_classes_{ds}.json", info)
    return tab, pd.DataFrame(con)


def v3(ds: str, a, g: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    fwd = forward_mean(ds, vz.LABEL_KEY, a.var_names)
    types = fwd["cell_types"]
    Cg = np.stack([fwd["C_gk"][k] for k in types])
    div = np.maximum(-Cg, 0)
    top = np.where(div.max(0) > 0, np.array(types)[div.argmax(0)], "")
    share = np.where(div.sum(0) > 0, div.max(0) / np.where(div.sum(0) > 0, div.sum(0), 1), np.nan)
    # pseudobulk leave-one-class-out driver (classes)
    x, f, sl = pseudobulk(a)
    classes = sorted({k[1] for k in x})
    E_all, _, _ = T42.pooled_E(x, f, sl, a.n_vars)
    inc = []
    for k in classes:
        Ek, _, _ = T42.pooled_E({q: v for q, v in x.items() if q[1] != k}, {q: v for q, v in f.items() if q[1] != k}, sl, a.n_vars)
        inc.append(Ek - E_all)
    inc = np.stack(inc, 1)
    fin = np.isfinite(inc).all(1)
    pos = np.where(fin[:, None], inc, -np.inf)
    b1_top = np.where(fin & (pos.max(1) > 0), np.array(classes)[np.argmax(pos, 1)], "")
    # human top-expressing fine type (mean CP10K)
    hm = (a.obs["species"].astype(str) == "human").to_numpy()
    X = a.X.tocsr()[hm].astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    cp = sp.diags(1e4 / np.maximum(lib, 1)).dot(X).tocsr()
    lab = a.obs.loc[hm, vz.LABEL_KEY].astype(str).to_numpy()
    tl = [t for t in sorted(set(lab)) if (lab == t).sum() >= 20]
    Mh = np.stack([np.asarray(cp[lab == t].mean(0)).ravel() for t in tl])
    h_top = np.array(tl)[Mh.argmax(0)]
    fine2class = a.obs[[vz.LABEL_KEY, vz.CLASS_KEY]].astype(str).drop_duplicates().set_index(vz.LABEL_KEY)[vz.CLASS_KEY].to_dict()
    t = pd.DataFrame(dict(hsapiens_gene_id=g["hsapiens_gene_id"].values, symbol=g["symbol"].values, D=g["D"].values,
                          chimera_top_div_type=top, chimera_top_div_share=share, b1_top_class=b1_top,
                          human_top_expr_type=h_top), index=g.index)
    t["chimera_top_div_class"] = t["chimera_top_div_type"].map(fine2class).fillna("")
    t["human_top_expr_class"] = t["human_top_expr_type"].map(fine2class).fillna("")
    t = t[t["hsapiens_gene_id"].isin(set(gl["hsapiens_gene_id"]))]
    t = t.merge(gl[["hsapiens_gene_id", "source", "vanzyl_named"]], on="hsapiens_gene_id", how="left")
    return t.sort_values(["vanzyl_named", "D"], ascending=[False, True])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261102)
    gl = glaucoma_list()
    v1_rows, v2_tabs, v2_cons, drv, iq = [], [], [], [], {}
    for ds in vz.usable_datasets():
        a = vz.read_h5ad(ds)
        g = pd.read_parquet(vz.OUT / "readout" / f"genes_{ds}.parquet").set_index("var_name")
        if list(g.index) != list(a.var_names):
            raise AssertionError(f"{ds}: gene table order differs from h5ad")
        iq[ds] = integration_quality(ds)
        v1_rows += [dict(dataset=ds, primary=ds == PRIMARY) | r for r in v1(g, gl, rng)]
        tab, con = v2(ds, a, g, gl, rng)
        v2_tabs.append(tab)
        v2_cons.append(con)
        v3(ds, a, g, gl).to_csv(OUT / f"v3_drivers_{ds}.csv", index=False)
        drv.append(drivers_summary(ds, a))
    pd.DataFrame(v1_rows).to_csv(OUT / "v1_contrast.csv", index=False)
    pd.concat(v2_tabs).to_csv(OUT / "v2_species_fidelity.csv", index=False)
    pd.concat(v2_cons).to_csv(OUT / "v2_contrast.csv", index=False)
    pd.concat(drv).to_csv(OUT / "drivers_summary.csv", index=False)
    vz.write_json(OUT / "integration_quality.json", iq)
    print(pd.DataFrame(v1_rows).round(3).to_string(index=False))
    print(pd.concat(v2_tabs).round(3).to_string(index=False))
    print(pd.concat(v2_cons).round(3).to_string(index=False))
    print(json.dumps(iq, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
