"""P13: pseudobulk differential expression for the kidney AKI case (plan section 8c).

Runs in ``revision_de_env`` (PyDESeq2 0.5.2), never ``chimera_env``. No Chimera readout is read.

Per DE-able type t (``results/revision/P13/gd1k_types.csv``), on units carrying >= 20 nuclei of
t (the G-D1k unit filter), with roles from ``unit_roles.tsv``:

* human, primary: multiome units, ``~ sex + collection + condition`` (AKI vs healthy DE half);
* mouse, primary: KPMP multiome units, ``~ stage + condition`` (IRI vs Sham; stage acute 2 d /
  late 4-24 wk);
* sensitivity (Family 3): human biopsies only (``~ sex + condition``); human 10x 3' v3 units
  (``~ sex + collection + condition``); mouse acute only and late only (``~ condition``);
  Kirita 10x 3' v2 IRI pooled vs Sham 14 d, and its 2 d and 6 wk arms (``~ condition``).

A covariate constant over a fit's units is dropped and logged. Wald tests; apeglm shrinkage of
the condition coefficient (``lfc_shrink``). Genes with < 10 counts summed over the units are not
tested. ``det_frac`` = nuclei with count > 0 / nuclei, over the units used (section 8c's 5% rule).

    PYTHONPATH=<chimera repo> python revision/tasks/P13_kidney/de.py [--types pt,...] [--n-cpus 16]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from revision.common import gene_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("P13_de")

ROOT = gene_ids.REPO_ROOT
DATA = ROOT / "data" / "revision" / "disease_kidney"
RES = ROOT / "results" / "revision" / "P13"
OUT = RES / "de"
MIN_CELLS_UNIT = 20
MIN_COUNTS = 10
MULTIOME = "10x multiome"


def select(pb: ad.AnnData, t: str, mask: pd.Series) -> ad.AnnData:
    o = pb.obs
    keep = (o["cell_ontology_class"] == t) & mask & (o["n_nuclei"].astype(int) >= MIN_CELLS_UNIT)
    return pb[keep.to_numpy()].copy()


def fit(sub: ad.AnnData, covariates: list[str], n_cpus: int) -> tuple[pd.DataFrame, str]:
    meta = sub.obs.copy()
    for c in meta.columns:
        if meta[c].dtype.name == "category":
            meta[c] = meta[c].astype(str)
    covs = [c for c in covariates if meta[c].nunique() > 1]
    design = "~ " + " + ".join(covs + ["condition"])
    counts = pd.DataFrame(np.asarray(sub.X.toarray(), dtype=np.int64), index=sub.obs_names,
                          columns=sub.var_names)
    counts = counts.loc[:, counts.sum(axis=0) >= MIN_COUNTS]
    inf = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True,
                           inference=inf, quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["condition", "disease", "control"], inference=inf, quiet=True)
        ds.summary()
        mle = ds.results_df.copy()
        coeff = [c for c in ds.LFC.columns if c.startswith("condition") and "disease" in c]
        if len(coeff) != 1:
            raise RuntimeError(f"condition coefficient not unique: {list(ds.LFC.columns)}")
        ds.lfc_shrink(coeff=coeff[0])
    res = ds.results_df.rename(columns={"log2FoldChange": "log2fc_shrunk", "lfcSE": "lfcse_shrunk"})
    res["log2fc_mle"] = mle["log2FoldChange"]
    res["lfcse_mle"] = mle["lfcSE"]
    res["stat"] = mle["stat"]
    res["pvalue"] = mle["pvalue"]
    res["padj"] = mle["padj"]
    det = np.asarray(sub[:, counts.columns].layers["n_detected"].sum(axis=0)).ravel()
    res["det_frac"] = det / float(sub.obs["n_nuclei"].astype(int).sum())
    res.index.name = "gene_symbol"
    res["gene_id"] = sub.var.loc[counts.columns, "gene_ids"].to_numpy()
    return res.reset_index(), design


def jobs_for(pb_h: ad.AnnData, pb_m: ad.AnnData, t: str) -> list:
    h, m = pb_h.obs, pb_m.obs
    hde = h["role"].isin(["de_control", "de_disease"])
    mde = m["role"].isin(["de_control", "de_disease"])
    msens = m["role"].isin(["sens_control", "sens_disease"])
    sham_k = m["role"] == "sens_control"
    return [
        ("human", "primary", select(pb_h, t, hde & (h["assay"] == MULTIOME)), ["sex", "collection"]),
        ("mouse", "primary", select(pb_m, t, mde), ["stage"]),
        ("human", "sens_biopsy", select(pb_h, t, hde & (h["assay"] == MULTIOME) & (h["collection"] == "biopsy")), ["sex"]),
        ("human", "sens_v3", select(pb_h, t, hde & (h["assay"] != MULTIOME)), ["sex", "collection"]),
        ("mouse", "sens_acute", select(pb_m, t, mde & (m["stage"] == "acute")), []),
        ("mouse", "sens_late", select(pb_m, t, mde & (m["stage"] == "late")), []),
        ("mouse", "sens_kirita", select(pb_m, t, msens), []),
        ("mouse", "sens_kirita_2d", select(pb_m, t, msens & (sham_k | (m["time"] == "2d"))), []),
        ("mouse", "sens_kirita_6wk", select(pb_m, t, msens & (sham_k | (m["time"] == "4-6wk"))), []),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--types", default=None)
    p.add_argument("--n-cpus", type=int, default=16)
    p.add_argument("--primary-only", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    types = pd.read_csv(RES / "gd1k_types.csv")
    de_types = types.loc[types["de_able"], "cell_type"].tolist()
    if a.types:
        de_types = [t for t in de_types if t in a.types.split(",")]
    pb_h = ad.read_h5ad(DATA / "kidney_pseudobulk_human.h5ad")
    pb_m = ad.read_h5ad(DATA / "kidney_pseudobulk_mouse.h5ad")
    summary = []
    for t in de_types:
        for species, contrast, sub, covs in jobs_for(pb_h, pb_m, t):
            if a.primary_only and contrast != "primary":
                continue
            n = sub.obs.groupby("condition", observed=True).size().to_dict()
            if n.get("control", 0) < 2 or n.get("disease", 0) < 2:
                log.info("%s %s %s: skipped, units %s", t, species, contrast, n)
                summary.append(dict(cell_type=t, species=species, contrast=contrast, design="skipped",
                                    n_units=int(sub.n_obs), n_control=n.get("control", 0),
                                    n_disease=n.get("disease", 0)))
                continue
            r, design = fit(sub, covs, a.n_cpus)
            log.info("%s %s %s: %s, units %s, %d genes", t, species, contrast, design, n, len(r))
            r.to_csv(OUT / f"{species}_{contrast}_{t}.tsv", sep="\t", index=False)
            summary.append(dict(cell_type=t, species=species, contrast=contrast, design=design,
                                n_units=int(sub.n_obs), n_control=n.get("control", 0), n_disease=n.get("disease", 0),
                                units=",".join(sub.obs["unit"].astype(str)), n_genes_tested=int(len(r)),
                                n_padj_lt_0_1=int((r["padj"] < 0.1).sum()),
                                n_padj_lt_0_05=int((r["padj"] < 0.05).sum())))
            pd.DataFrame(summary).to_csv(OUT / "de_summary.tsv", sep="\t", index=False)
    skipped = types.loc[~types["de_able"], "cell_type"].tolist()
    (OUT / "de_meta.json").write_text(json.dumps(dict(de_types=de_types, filtered_types=skipped,
                                                      min_cells_unit=MIN_CELLS_UNIT, min_counts=MIN_COUNTS,
                                                      pydeseq2="0.5.2"), indent=2))
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
