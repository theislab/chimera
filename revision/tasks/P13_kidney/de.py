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

A covariate constant over a fit's units is dropped and logged. Wald tests; PyDESeq2's apeGLM
shrinkage of the condition coefficient (``lfc_shrink``), twice: ``log2fc_shrunk`` with the adaptive
prior (the default, which collapses thin fits to zero, see T7.3) and ``log2fc_fixed`` with prior
scale 1 (``adapt=False``). H-D1 uses ``log2fc_mle`` for l^h and ``log2fc_fixed`` for m; H-D4 ranks on
``log2fc_fixed`` (plan amendment). Genes with < 10 counts summed over the units are not tested.
``det_frac`` = nuclei with count > 0 / nuclei, over the units used (section 8c's 5% rule).

``--perm <type index>`` fits the primary models of one DE-able type on permuted condition labels
(human within sex x collection, mouse within stage, ``N_PERM`` = 200 draws each, or all distinct
relabellings where fewer exist, never the observed
labels or their swap), with the fixed-scale prior only, for the H-D4 and H-D1 label null. Output
``de/perm/{human,mouse}_perm<ii>_<type>.tsv``.

    PYTHONPATH=<chimera repo> python revision/tasks/P13_kidney/de.py [--types pt,...] [--n-cpus 16]
    PYTHONPATH=<chimera repo> python revision/tasks/P13_kidney/de.py --perm 0 [--n-cpus 16]
    PYTHONPATH=<chimera repo> python revision/tasks/P13_kidney/de.py --perm-task <0..9k-1> --n-chunks k
    PYTHONPATH=<chimera repo> python revision/tasks/P13_kidney/de.py --loo <type index>
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
N_PERM, SEED = 200, 20261101  # 20 at first; raised to 200 for precision only (same design)
DE_ORDER = ["pt", "tal", "dct", "pc", "ic", "ec", "fib", "myeloid", "lymphoid"]


def select(pb: ad.AnnData, t: str, mask: pd.Series) -> ad.AnnData:
    o = pb.obs
    keep = (o["cell_ontology_class"] == t) & mask & (o["n_nuclei"].astype(int) >= MIN_CELLS_UNIT)
    return pb[keep.to_numpy()].copy()


def fit(sub: ad.AnnData, covariates: list[str], n_cpus: int, adaptive: bool = True) -> tuple[pd.DataFrame, str]:
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
    contrast = ["condition", "disease", "control"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True,
                           inference=inf, quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=contrast, inference=inf, quiet=True)
        ds.summary()
        mle = ds.results_df.copy()
        coeff = [c for c in ds.LFC.columns if c.startswith("condition") and "disease" in c]
        if len(coeff) != 1:
            raise RuntimeError(f"condition coefficient not unique: {list(ds.LFC.columns)}")
        if adaptive:
            ds.lfc_shrink(coeff=coeff[0])
            shrunk = ds.results_df.copy()
            # lfc_shrink edits the LFC table in place, so the fixed-scale prior needs a fresh DeseqStats
            ds = DeseqStats(dds, contrast=contrast, inference=inf, quiet=True)
            ds.summary()
        ds.lfc_shrink(coeff=coeff[0], adapt=False)
    res = pd.DataFrame({"baseMean": mle["baseMean"], "log2fc_fixed": ds.results_df["log2FoldChange"],
                        "lfcse_fixed": ds.results_df["lfcSE"], "log2fc_mle": mle["log2FoldChange"],
                        "lfcse_mle": mle["lfcSE"], "stat": mle["stat"], "pvalue": mle["pvalue"], "padj": mle["padj"]})
    if adaptive:
        res["log2fc_shrunk"] = shrunk["log2FoldChange"]
        res["lfcse_shrunk"] = shrunk["lfcSE"]
    det = np.asarray(sub[:, counts.columns].layers["n_detected"].sum(axis=0)).ravel()
    res["det_frac"] = det / float(sub.obs["n_nuclei"].astype(int).sum())
    res.index.name = "gene_symbol"
    res["gene_id"] = sub.var.loc[counts.columns, "gene_ids"].to_numpy()
    return res.reset_index(), design


def permuted_labels(obs: pd.DataFrame, strata: list[str], rng, n: int) -> list:
    """n distinct relabellings within strata, never the observed labels or their exact swap."""
    real = obs["condition"].astype(str).to_numpy()
    swap = np.where(real == "control", "disease", "control")
    groups = list(obs.reset_index(drop=True).groupby([obs[c].astype(str).to_numpy() for c in strata]).indices.values())
    seen, out = set(), []
    for _ in range(200 * n):
        lab = real.copy()
        for idx in groups:
            lab[idx] = rng.permutation(real[idx])
        key = tuple(lab)
        if key in seen or (lab == real).all() or (lab == swap).all():
            continue
        seen.add(key)
        out.append(lab)
        if len(out) == n:
            break
    return out


def run_perm(t: str, pb_h: ad.AnnData, pb_m: ad.AnnData, n_cpus: int, chunk: int = 0, n_chunks: int = 1) -> None:
    """Permuted-label fits for type t, draws i with i % n_chunks == chunk.

    The relabellings come from the same generator and rule as the first 20-draw run, so draws
    0-19 are identical to it; an existing fit of the same draw (same controls) is kept.
    """
    out = OUT / "perm"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {j[0]: j for j in jobs_for(pb_h, pb_m, t) if j[1] == "primary"}
    old = pd.read_csv(out / f"labels_{t}.tsv", sep="\t") if (out / f"labels_{t}.tsv").exists() else None
    rows = []
    for k, (species, strata) in enumerate((("human", ["sex", "collection"]), ("mouse", ["stage"]))):
        _, _, sub, covs = jobs[species]
        rng = np.random.default_rng([SEED, DE_ORDER.index(t), k])
        labels = permuted_labels(sub.obs, strata, rng, N_PERM)
        for i, lab in enumerate(labels):
            if i % n_chunks != chunk:
                continue
            controls = ",".join(sub.obs["unit"].astype(str)[lab == "control"])
            f = out / f"{species}_perm{i:02d}_{t}.tsv"
            prev = None if old is None else old[(old.species == species) & (old.perm == i)]
            if f.exists() and prev is not None and len(prev) == 1 and prev["controls"].iloc[0] == controls:
                rows.append(dict(cell_type=t, species=species, perm=i, design=prev["design"].iloc[0],
                                 controls=controls, n_draws_available=len(labels), reused=True))
                continue
            s = sub.copy()
            s.obs["condition"] = lab
            r, design = fit(s, covs, n_cpus, adaptive=False)
            r.to_csv(f, sep="\t", index=False)
            rows.append(dict(cell_type=t, species=species, perm=i, design=design, controls=controls,
                             n_draws_available=len(labels), reused=False))
            pd.DataFrame(rows).to_csv(out / f"labels_{t}_c{chunk}.tsv", sep="\t", index=False)
            log.info("%s %s perm %d done", t, species, i)
    pd.DataFrame(rows).to_csv(out / f"labels_{t}_c{chunk}.tsv", sep="\t", index=False)
    (out / f"DONE_{t}_c{chunk}").write_text("ok\n")


def run_loo(t: str, pb_h: ad.AnnData, pb_m: ad.AnnData, n_cpus: int) -> None:
    """Leave-one-out primary fits (section 8c): each healthy DE-half donor (human), each mouse.

    Fixed-scale prior only. Output ``de/loo/{human,mouse}_loo_<unit>_<type>.tsv`` and ``loo_<type>.tsv``.
    """
    out = OUT / "loo"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {j[0]: j for j in jobs_for(pb_h, pb_m, t) if j[1] == "primary"}
    rows = []
    for species in ("human", "mouse"):
        _, _, sub, covs = jobs[species]
        units = sub.obs.loc[sub.obs["condition"] == "control", "unit"] if species == "human" else sub.obs["unit"]
        for u in sorted(units.astype(str)):
            s = sub[(sub.obs["unit"].astype(str) != u).to_numpy()].copy()
            n = s.obs["condition"].value_counts()
            if n.get("control", 0) < 2 or n.get("disease", 0) < 2:
                continue
            safe = u.replace("|", "_").replace(" ", "").replace("'", "")
            r, design = fit(s, covs, n_cpus, adaptive=False)
            r.to_csv(out / f"{species}_loo_{safe}_{t}.tsv", sep="\t", index=False)
            rows.append(dict(cell_type=t, species=species, left_out=u, file=f"{species}_loo_{safe}_{t}.tsv",
                             design=design, n_units=int(s.n_obs)))
            pd.DataFrame(rows).to_csv(out / f"loo_{t}.tsv", sep="\t", index=False)
            log.info("%s %s leave out %s done", t, species, u)
    (out / f"DONE_{t}").write_text("ok\n")


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
    p.add_argument("--perm", type=int, default=None, help="index into DE_ORDER: permuted-label fits of that type")
    p.add_argument("--perm-task", type=int, default=None,
                   help="array task: type = DE_ORDER[task // n_chunks], chunk = task % n_chunks")
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--loo", type=int, default=None, help="index into DE_ORDER: leave-one-out primary fits")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    types = pd.read_csv(RES / "gd1k_types.csv")
    de_types = types.loc[types["de_able"], "cell_type"].tolist()
    if a.types:
        de_types = [t for t in de_types if t in a.types.split(",")]
    pb_h = ad.read_h5ad(DATA / "kidney_pseudobulk_human.h5ad")
    pb_m = ad.read_h5ad(DATA / "kidney_pseudobulk_mouse.h5ad")
    if a.perm is not None:
        run_perm(DE_ORDER[a.perm], pb_h, pb_m, a.n_cpus)
        return 0
    if a.loo is not None:
        run_loo(DE_ORDER[a.loo], pb_h, pb_m, a.n_cpus)
        return 0
    if a.perm_task is not None:
        run_perm(DE_ORDER[a.perm_task // a.n_chunks], pb_h, pb_m, a.n_cpus, a.perm_task % a.n_chunks, a.n_chunks)
        return 0
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
