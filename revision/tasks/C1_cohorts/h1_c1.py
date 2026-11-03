"""C1 cohorts in H1's schema and panels: H1's own code, registry swapped for the C1 cohorts, output in results/revision/C1/h1/.

Set here, before importing H1: H1_OUT (output root) and H1_EXTRA (none). `rules.COHORTS` is replaced by the C1 cohorts that have kept runs
(`kept_runs.tsv`: cohort, seed, run). Writes genes/<cohort>.parquet, runs.tsv, set_z.csv, axis.csv, sets_membership.csv, drivers/, in the same
form as results/revision/H1/, so they join its tables by human Ensembl ID. B1 here: T4.2 pooled E on the training object, units sample_batch.
Lineage panels per new tissue are fixed in LINEAGE_C1 below, committed with this file, before any readout.

    python revision/tasks/C1_cohorts/h1_c1.py tables [--cohort X] | sets [--jobs 8] | forward --run R --cohort X --seed S | drivers | gsea | archetypes
"""
from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ["H1_OUT"] = os.environ.get("C1_H1_OUT", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "results/revision/C1/h1"))
os.environ["H1_EXTRA"] = "/nonexistent"

from revision.common import cohorts as C  # noqa: E402
from revision.common import controls as K  # noqa: E402
from revision.common import gene_ids  # noqa: E402
from revision.tasks.C1_cohorts import configs as CF  # noqa: E402
from revision.tasks.H1_harmonise import rules as RU  # noqa: E402

KEPT = __import__("pathlib").Path(os.environ.get("C1_KEPT", str(CF.RES / "kept_runs.tsv")))
LINEAGE_C1 = {
    "cortex": {"excitatory": ["SLC17A7", "SATB2", "CUX2", "RORB", "FEZF2", "THEMIS", "TLE4", "CCN2"],
               "inhibitory": ["GAD1", "GAD2", "PVALB", "SST", "VIP", "LAMP5", "SNCG", "LHX6", "ADARB2"],
               "glia": ["AQP4", "GFAP", "MOG", "MBP", "PLP1", "OLIG1", "OLIG2", "PDGFRA", "CSPG4", "CSF1R", "CX3CR1", "P2RY12"],
               "vascular": ["FLT1", "CLDN5", "PDGFRB", "DCN", "COL1A2"]},
    "lung": {"alveolar epithelium": ["SFTPC", "SFTPB", "ABCA3", "LAMP3", "AGER", "HOPX", "RTKN2"],
             "endothelium": ["PECAM1", "CA4", "CLDN5", "EDNRB", "EMCN", "GJA5", "ACKR1"],
             "stroma": ["COL1A1", "COL3A1", "PDGFRA", "PDGFRB", "ACTA2", "DCN", "ITGA8"],
             "immune": ["PTPRC", "CD3D", "CD79A", "NKG7", "CD14", "MARCO", "ITGAX", "XCR1", "CLEC4C"]},
    "skeletal muscle": {"myofibre": ["MYH1", "MYH2", "MYH7", "ACTA1", "CKM", "TTN", "DES", "MYBPC1"],
                        "satellite": ["PAX7", "MYF5", "CALCR", "NCAM1"],
                        "stroma and vessel": ["PDGFRA", "DCN", "COL1A1", "PECAM1", "CDH5", "RGS5", "MYH11"],
                        "immune": ["PTPRC", "CD163", "LYVE1", "CD14"]},
    "tumour myeloid": {"myeloid": ["CD14", "LYZ", "S100A8", "S100A9", "CSF1R", "MRC1", "C1QA", "C1QB", "APOE"],
                       "dendritic": ["XCR1", "CLEC9A", "CD1C", "FCER1A", "LAMP3", "CCR7", "IRF8", "BATF3"],
                       "neutrophil": ["CSF3R", "MMP8", "MMP9", "FCGR3B", "CXCR2"],
                       "lymphoid": ["CD3D", "CD3E", "CD79A", "MS4A1", "NKG7", "GNLY"]},
}


def registry() -> dict:
    t = pd.read_csv(KEPT, sep="\t")
    out = {}
    for c, g in t.groupby("cohort"):
        d = CF.SUBSETS.get(c) or CF.COHORTS[c]
        out[c] = dict(tissue=d["tissue"], label=d["label"], species=d.get("species"), h5ad=CF.h5ad(c), table=c, ct="cell_ontology_class", role="new",
                      tried={int(r.seed): str(r.run) for r in g.itertuples()})
    return out


def install() -> dict:
    reg = registry()
    RU.COHORTS.clear()
    RU.COHORTS.update(reg)
    RU.PENDING.clear()
    RU.LINEAGE.clear()
    RU.LINEAGE.update(LINEAGE_C1)
    return reg


def add_b1(c: str) -> None:
    from revision.tasks.T4_2_gene_baselines import run as T42

    f = RU.OUT / "genes" / f"{c}.parquet"
    g = pd.read_parquet(f)
    a = ad.read_h5ad(CF.h5ad(c))
    X = a.X.tocsr().astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, fr, units = T42.pseudobulk(X, species, a.obs["cell_ontology_class"].astype(str).to_numpy(), a.obs["sample_batch"].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E, nE, info = T42.pooled_E(x, fr, sl, a.n_vars)
    g["B1"] = pd.Series(E, index=a.var_names).reindex(g.index).to_numpy()
    for p in info:
        s1, s2 = p["pair"].split("-")
        r, _ = T42.pair_corr(x, fr, s1, s2, p["groups"], a.n_vars)
        g[f"E_{s1}_{s2}"] = pd.Series(r, index=a.var_names).reindex(g.index).to_numpy()
    tab = gene_ids.load_table(c)
    ids = pd.Index(g.index).str.replace("ortho_", "", regex=False).astype(int)
    for col in tab.columns:
        if col.endswith(("_identity", "_dn", "_ds")) or (col.endswith("_gene_name") and not col.startswith("hsapiens")):
            g[col] = tab.loc[ids, col].to_numpy()
    cds = K.cds_table()[["log_cds_length"]]
    g = g.join(cds, on="hsapiens_gene_id")
    g.to_parquet(f)
    units.to_csv(RU.OUT / "genes" / f"{c}__b1_units.csv", index=False)


def exact_table_match(adata, table) -> float:
    """Stronger than gene_ids.assert_table_matches, whose Spearman bar fails when most genes sit at the 0.95 cap (DLPFC, ties): the prior must equal
    the clamped table identity gene by gene."""
    from scipy.stats import spearmanr

    ids = pd.Index(adata.var_names).str.replace("ortho_", "", regex=False).astype(int)
    want = np.clip(table.loc[ids, "mean_identity"].to_numpy(float) / 100, 0.05, 0.95)
    got = adata.var["conservation_prior"].to_numpy(float)
    if not np.allclose(got, want, atol=1e-5):
        raise AssertionError(f"prior differs from clamped table identity: max {np.abs(got - want).max()}")
    return float(spearmanr(got, want)[0])


def tables(one: str | None) -> None:
    from revision.tasks.H1_harmonise import tables as TB

    gene_ids.assert_table_matches = exact_table_match

    reg = install()
    (RU.OUT / "genes").mkdir(parents=True, exist_ok=True)
    runs = []
    for c, d in reg.items():
        for s, run in sorted(d["tried"].items()):
            h = TB.run_health(run)
            anchored = c.startswith("brain_dlpfc")  # the ruling's scope, see grid.anchored_tails
            if anchored and h.get("no_nan"):  # coordinator ruling: both tails against the anchored prior when most genes are near pi = 1
                h["healthy"] = bool(h["n_active_z"] == 10 and h["tail_up_anch"] >= RU.TAIL_MIN and h["tail_dn_anch"] >= RU.TAIL_MIN and h["sat_hi"] <= RU.SAT_MAX and h["sat_lo"] <= RU.SAT_MAX)
            runs.append(dict(cohort=c, tissue=d["tissue"], role="new", seed=s, kept=bool(h["healthy"]), **h))
    rt = pd.DataFrame(runs)
    rt.to_csv(RU.OUT / "runs.tsv", sep="\t", index=False)
    checks = []
    for c, d in reg.items():
        if one and c != one:
            continue
        kept = {int(r.seed): r.run for r in rt[(rt.cohort == c) & rt.kept].itertuples()}
        if len(kept) < 2:
            print(f"{c}: {len(kept)} healthy seeds, not built", flush=True)
            continue
        checks.append(TB.build_cohort((c, kept)))
        add_b1(c)
        print(c, "tabled", flush=True)
    if checks:
        pd.DataFrame(checks).to_csv(RU.OUT / (f"identity_checks_{one}.csv" if one else "identity_checks.csv"), index=False)


def sets(jobs: int, n_perm: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from revision.tasks.H1_harmonise import recurrence as RC

    install()
    cohorts = {c: d for c, d in RU.COHORTS.items() if (RU.OUT / "genes" / f"{c}.parquet").exists()}
    allg = pd.concat([pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet")[["hsapiens_gene_id", "symbol"]] for c in cohorts])
    gs = RC.gene_sets(allg)
    ls = RC.lineage_sets(allg)
    gs["key"] = gs.family + "|" + gs.set
    lin = ls.assign(key="lineage|" + ls.tissue + "|" + ls.set)[["key", "hsapiens_gene_id"]]
    gs.to_csv(RU.OUT / "sets_membership.csv", index=False)
    ls.to_csv(RU.OUT / "lineage_membership.csv", index=False)
    mem = pd.concat([gs[["key", "hsapiens_gene_id"]], lin], ignore_index=True)
    args = [(c, r, mem, n_perm) for c in cohorts for r in ("D", "S", "B1", "pi", "dNdS", "D_tau", "S_tau")]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(RC.one, args))
    pd.concat([x[0] for x in res], ignore_index=True).to_csv(RU.OUT / "set_z.csv", index=False)
    RC.axis_table(cohorts).to_csv(RU.OUT / "axis.csv", index=False)


def forward(cohort: str, run: str, seed: int, threads: int, batch: int) -> None:
    gene_ids_sys = gene_ids.REPO_ROOT / "revision" / "tasks" / "T2_5_celltype_margin"
    sys.path.insert(0, str(gene_ids_sys))
    import run as T25RUN  # noqa: PLC0415

    def _read(c, backed=False):
        return ad.read_h5ad(CF.h5ad(c), backed="r" if backed else None)

    C.read_adata = _read
    T25RUN.GROUP_KEYS["liver"] = ["cell_ontology_class"]
    fw = T25RUN.forward_sums(f"{cohort}_s{seed}", cohort, run, threads, batch)
    fw["run"] = np.array(run)
    d = CF.RES / cohort / "drivers"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"fw_s{seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(d / f"fw_s{seed}.npz")


def drivers() -> None:
    from revision.tasks.T2_5_celltype_margin import margin as MG

    reg = install()
    runs = pd.read_csv(RU.OUT / "runs.tsv", sep="\t")
    out = RU.OUT / "drivers"
    out.mkdir(exist_ok=True)
    meta = []
    for c in reg:
        fp = RU.OUT / "genes" / f"{c}.parquet"
        if not fp.exists():
            continue
        seeds = sorted(runs[(runs.cohort == c) & runs.kept].seed)
        fr, rows, used, w = [], [], [], None
        for s in seeds:
            f = CF.RES / c / "drivers" / f"fw_s{s}.npz"
            if not f.exists():
                continue
            fw = dict(np.load(f, allow_pickle=False))
            k = "cell_ontology_class"
            dec = MG.decompose(fw, k)
            vn = fw["var_names"].astype(str)
            fr.append(pd.DataFrame(dec["C_gk_meanz"], index=vn))
            w = pd.Series(dec["w"]).groupby(dec["labels"]).sum()
            rows.append(dec)
            used.append(s)
        if not fr:
            continue
        m = sum(fr) / len(fr)
        m.to_parquet(out / f"{c}.parquet")
        # species-row contributions, seed mean: C_gsk = w_sk S_gsk
        dec0 = rows[0]
        long = []
        for dec in rows:
            long.append(np.asarray(dec["C_gsk_meanz"]) if "C_gsk_meanz" in dec else None)
        g = pd.read_parquet(fp).reindex(m.index)
        un = g.count_support.to_numpy()
        closure = float(np.corrcoef(m.sum(1)[un], g.S_ema[un])[0, 1])
        A = m.to_numpy()
        ty = np.array(m.columns)
        dom = ty[np.abs(A).argmax(1)]
        strict = g.strict_support.to_numpy()
        neg, pos = np.abs(np.where(A < 0, A, 0)).sum(0), np.where(A > 0, A, 0).sum(0)
        pd.DataFrame(dict(cohort=c, cell_type=ty, weight=w.reindex(ty).to_numpy(), share_abs_S=np.abs(A[un]).sum(0) / np.abs(A[un]).sum(), share_negative=neg / neg.sum(), share_positive=pos / pos.sum(),
                          dominant_in_strict_up=[(dom[strict & (g.D.to_numpy() > 0)] == k).mean() for k in ty], dominant_in_strict_down=[(dom[strict & (g.D.to_numpy() < 0)] == k).mean() for k in ty])).to_csv(out / f"{c}__shares.csv", index=False)
        meta.append(dict(cohort=c, seeds_with_forward=",".join(map(str, used)), n_types=len(ty), closure_C_sum_vs_S_ema_pearson=closure))
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    print(pd.DataFrame(meta).round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tables", "sets", "forward", "drivers", "gsea", "archetypes"])
    ap.add_argument("--cohort")
    ap.add_argument("--run")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=RU.N_PERM)
    a = ap.parse_args()
    if a.cmd == "tables":
        tables(a.cohort)
    elif a.cmd == "sets":
        sets(a.jobs, a.n_perm)
    elif a.cmd == "forward":
        forward(a.cohort, a.run, a.seed, a.threads, a.batch_size)
    elif a.cmd == "drivers":
        drivers()
    elif a.cmd == "gsea":
        install()
        sys.argv = ["gsea", "--jobs", str(a.jobs)]
        from revision.tasks.H1_harmonise import gsea as G

        G.main()
    else:
        install()
        sys.argv = ["archetypes", "--jobs", str(a.jobs)]
        from revision.tasks.H1_harmonise import archetypes as AR

        AR.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
