"""T7.3: hepatocyte pseudobulk DE for liver MASH (plan section 8, amendment of 2026-11-01).

Runs in ``revision_de_env`` (PyDESeq2 0.5.2), never ``chimera_env``. No Chimera readout is read.
G-D1 failed, so only hepatocytes are fitted, and H-D1 and H-D4 are Family 3 descriptive.

Primary fits
  human: ~ sex + condition, NASH without cirrhosis (27 donors) vs healthy (30, HL1 right lobe, HL2)
  mouse: ~ time + diet, WD vs SD, all 12 mice (training plus hold-out, as H-D1 specifies)
Wald tests; PyDESeq2's apeGLM shrinkage of the tested coefficient (``lfc_shrink``), twice:
``log2fc_shrunk`` with the adaptive prior (the default) and ``log2fc_fixed`` with prior scale 1.
With 3 healthy donors the adaptive prior collapses the human fit (96% of detected genes at
|log2FC| < 0.001, and 3.4% change sign), so the human logFC carries no usable size or sign there.
See hd1.py and hd4.py for what uses which column.

Groups (``--group``)
  primary  the two primary fits
  sens     leave-one-healthy-donor-out; leave-one-mouse-per-arm-out (6 x 6 pairs); P98 included;
           NAFLD vs healthy; training mice only
  perm     condition labels permuted within sex (human) or within time (mouse), 50 draws each
  bulk     GSE135251 bulk biopsies: NASH F0-F3 vs control (no cirrhosis, as the primary), all NASH
           vs control, NAFL vs control. GEO gives no sex or batch, so ~ condition
  control  Gribben healthy vs Guilliams lean hepatocytes (``guilliams.py`` first)
  source   tissue source. Gribben's healthy tissue comes from deceased transplant donors and explants
           (HL1, HL2), the MASH tissue from needle biopsies, so the primary contrast carries a
           source effect. NASH vs NAFLD (biopsy against biopsy, no healthy unit), NASH vs the
           biopsy-type controls 30 and P98, and those two controls vs HL1 and HL2 (2 vs 2, ~ condition,
           positive = higher in biopsy-type tissue). Added after the primary DE was seen, before any
           Chimera readout was joined to a logFC.

A unit enters only with >= 20 hepatocyte nuclei (G-D1 reading). Genes with < 10 counts over the
units are not tested. ``det_frac`` = nuclei with count > 0 / nuclei, over the units used.

    PYTHONPATH=<chimera repo> python revision/tasks/T7_3_mash/de.py --group primary --workers 2
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import logging
import sys
import tarfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from revision.common import gene_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("T7_3_de")

ROOT = gene_ids.REPO_ROOT
DATA = ROOT / "data" / "revision" / "disease"
RES = ROOT / "results" / "revision" / "T7_3"
OUT = RES / "de"
TYPE = "hepatocyte"
MIN_NUCLEI = 20
MIN_COUNTS = 10
N_PERM = 50
SEED = 20261102
P98 = "98"


def load_hep(species: str) -> ad.AnnData:
    a = ad.read_h5ad(DATA / f"disease_pseudobulk_{species}.h5ad")
    a = a[(a.obs["cell_ontology_class"].astype(str) == TYPE).to_numpy()
          & (a.obs["n_nuclei"].astype(int) >= MIN_NUCLEI).to_numpy()].copy()
    if species == "human":
        a.var["gene_id"] = a.var["gene_id_ens110"].astype(object)
    a.obs["unit"] = a.obs["donor"].astype(str)
    return a


def fit(counts: pd.DataFrame, meta: pd.DataFrame, design: str, factor: str, test: str, ref: str,
        n_cpus: int) -> pd.DataFrame:
    keep = counts.sum(axis=0) >= MIN_COUNTS
    counts = counts.loc[:, keep]
    inf = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True, inference=inf,
                           quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=[factor, test, ref], inference=inf, quiet=True)
        ds.summary()
        mle = ds.results_df.copy()
        coeff = [c for c in ds.LFC.columns if c.startswith(factor) and test in c]
        if len(coeff) != 1:
            raise RuntimeError(f"coefficient for {factor}={test} not unique: {list(ds.LFC.columns)}")
        ds.lfc_shrink(coeff=coeff[0])
        # lfc_shrink edits the LFC table in place, so the fixed-scale prior needs a fresh DeseqStats
        ds2 = DeseqStats(dds, contrast=[factor, test, ref], inference=inf, quiet=True)
        ds2.summary()
        ds2.lfc_shrink(coeff=coeff[0], adapt=False)
    res = ds.results_df.rename(columns={"log2FoldChange": "log2fc_shrunk", "lfcSE": "lfcse_shrunk"})
    res["log2fc_fixed"] = ds2.results_df["log2FoldChange"]
    res["lfcse_fixed"] = ds2.results_df["lfcSE"]
    res["log2fc_mle"] = mle["log2FoldChange"]
    res["lfcse_mle"] = mle["lfcSE"]
    res["stat"] = mle["stat"]
    res["pvalue"] = mle["pvalue"]
    res["padj"] = mle["padj"]
    return res


def fit_units(sub: ad.AnnData, design: str, factor: str, test: str, ref: str, n_cpus: int) -> pd.DataFrame:
    counts = pd.DataFrame(np.asarray(sub.X.toarray(), dtype=np.int64), index=sub.obs_names, columns=sub.var_names)
    meta = sub.obs[["unit", "n_nuclei"] + [c for c in ("sex", "time", "diet", "condition") if c in sub.obs]].copy()
    for c in meta.columns:
        if meta[c].dtype.name == "category":
            meta[c] = meta[c].astype(str)
    res = fit(counts, meta, design, factor, test, ref, n_cpus)
    det = np.asarray(sub[:, res.index].layers["n_detected"].sum(axis=0)).ravel()
    res["det_frac"] = det / float(sub.obs["n_nuclei"].astype(int).sum())
    res.index.name = "gene_symbol"
    res["gene_id"] = sub.var.loc[res.index, "gene_id"].to_numpy()
    return res.reset_index()


def human_units(h: ad.AnnData, drop=(), with_p98=False, arm="NASH w/o cirrhosis") -> ad.AnnData:
    o = h.obs
    healthy = o["primary"].astype(bool) & (o["condition"].astype(str) == "control")
    if with_p98:
        healthy = healthy | (o["unit"] == P98)
    case = (o["condition_label"].astype(str) == arm) & (o["lobe"].astype(str) == "Right")
    if arm == "NASH w/o cirrhosis":
        case &= o["primary"].astype(bool)
    keep = (healthy | case) & ~o["unit"].isin(drop)
    s = h[keep.to_numpy()].copy()
    s.obs["condition"] = np.where(s.obs["condition"].astype(str) == "control", "control", "disease")
    return s


def mouse_units(m: ad.AnnData, drop=(), training_only=False) -> ad.AnnData:
    keep = ~m.obs["unit"].isin(drop)
    if training_only:
        keep &= m.obs["role"].astype(str) == "training"
    s = m[keep.to_numpy()].copy()
    s.obs["time"] = "w" + s.obs["time_weeks"].astype(int).astype(str)
    s.obs["diet"] = s.obs["diet"].astype(str)
    return s


HUMAN = dict(design="~ sex + condition", factor="condition", test="disease", ref="control")
MOUSE = dict(design="~ time + diet", factor="diet", test="WD", ref="SD")


def permuted_labels(obs: pd.DataFrame, col: str, strata: str, rng, n: int) -> list:
    """n distinct relabellings within strata, never the observed labels or their exact swap."""
    real = obs[col].astype(str).to_numpy()
    levels = sorted(set(real))
    swap = np.where(real == levels[0], levels[1], levels[0])
    seen, out = set(), []
    for _ in range(100 * n):
        lab = real.copy()
        for _, idx in obs.groupby(strata, observed=True).indices.items():
            lab[idx] = rng.permutation(real[idx])
        key = tuple(lab)
        if key in seen or (lab == real).all() or (lab == swap).all():
            continue
        seen.add(key)
        out.append(lab)
        if len(out) == n:
            break
    return out


def jobs_for(group: str, h: ad.AnnData, m: ad.AnnData) -> list:
    jobs = []
    if group == "primary":
        jobs.append(("human_primary", human_units(h), HUMAN))
        jobs.append(("mouse_primary", mouse_units(m), MOUSE))
    elif group == "sens":
        hp = human_units(h)
        for d in sorted(hp.obs.loc[hp.obs["condition"] == "control", "unit"]):
            jobs.append((f"human_lohd_{d}", human_units(h, drop=(d,)), HUMAN))
        jobs.append(("human_p98", human_units(h, with_p98=True), HUMAN))
        jobs.append(("human_nafld", human_units(h, arm="NAFLD"), HUMAN))
        mp = mouse_units(m)
        sd = sorted(mp.obs.loc[mp.obs["diet"] == "SD", "unit"])
        wd = sorted(mp.obs.loc[mp.obs["diet"] == "WD", "unit"])
        for a, b in itertools.product(sd, wd):
            jobs.append((f"mouse_lomo_{a}_{b}", mouse_units(m, drop=(a, b)), MOUSE))
        jobs.append(("mouse_training", mouse_units(m, training_only=True), MOUSE))
    elif group == "source":
        o = h.obs
        nafld = (o["condition_label"].astype(str) == "NAFLD") & (o["lobe"].astype(str) == "Right")
        nash = o["primary"].astype(bool) & (o["condition"].astype(str) == "disease")
        s = h[(nafld | nash).to_numpy()].copy()
        s.obs["condition"] = np.where(nafld[(nafld | nash)], "control", "disease")
        jobs.append(("human_nash_vs_nafld", s, HUMAN))
        hp = human_units(h, with_p98=True)
        s = hp[~hp.obs["unit"].isin(["HL1", "HL2"]).to_numpy()].copy()
        jobs.append(("human_biopsy_controls", s, HUMAN))
        s = hp[hp.obs["unit"].isin(["HL1", "HL2", "30", P98]).to_numpy()].copy()
        s.obs["condition"] = np.where(s.obs["unit"].isin(["30", P98]), "disease", "control")
        jobs.append(("source_biopsy_vs_donor", s, dict(HUMAN, design="~ condition")))
    elif group == "perm":
        rng = np.random.default_rng(SEED)
        hp, mp = human_units(h), mouse_units(m)
        tab = []
        for i, lab in enumerate(permuted_labels(hp.obs, "condition", "sex", rng, N_PERM)):
            s = hp.copy()
            s.obs["condition"] = lab
            jobs.append((f"perm_human_{i:02d}", s, HUMAN))
            tab.append(dict(name=f"perm_human_{i:02d}", permuted_controls=",".join(s.obs_names[lab == "control"]),
                            n_true_controls_kept=int(((lab == "control") & (hp.obs["condition"].to_numpy() == "control")).sum())))
        for i, lab in enumerate(permuted_labels(mp.obs, "diet", "time", rng, N_PERM)):
            s = mp.copy()
            s.obs["diet"] = lab
            jobs.append((f"perm_mouse_{i:02d}", s, MOUSE))
            tab.append(dict(name=f"perm_mouse_{i:02d}", permuted_controls=",".join(s.obs_names[lab == "SD"]),
                            n_true_controls_kept=int(((lab == "SD") & (mp.obs["diet"].to_numpy() == "SD")).sum())))
        pd.DataFrame(tab).to_csv(OUT / "permutations.tsv", sep="\t", index=False)
    return jobs


def run_job(job) -> dict:
    name, sub, spec = job
    log.info("%s: %d units", name, sub.n_obs)
    r = fit_units(sub, spec["design"], spec["factor"], spec["test"], spec["ref"], 1)
    r.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    return dict(name=name, design=spec["design"], n_units=int(sub.n_obs), units=",".join(sub.obs["unit"].astype(str)),
                n_genes_tested=int(len(r)), n_padj_lt_0_1=int((r["padj"] < 0.1).sum()),
                n_padj_lt_0_05=int((r["padj"] < 0.05).sum()))


# ------------------------------------------------------------------------------ bulk
def bulk_groups() -> pd.Series:
    with gzip.open(DATA / "raw" / "GSE135251" / "GSE135251_series_matrix.txt.gz", "rt") as f:
        rows = {}
        for line in f:
            p = line.rstrip("\n").split("\t")
            key = p[0]
            if key == "!Sample_geo_accession":
                rows["gsm"] = [x.strip('"') for x in p[1:]]
            if key == "!Sample_characteristics_ch1" and p[1].startswith('"group in paper'):
                rows["group"] = [x.strip('"').split(": ")[1] for x in p[1:]]
    return pd.Series(rows["group"], index=rows["gsm"], name="group")


def bulk_counts() -> pd.DataFrame:
    tar = tarfile.open(DATA / "raw" / "GSE135251" / "GSE135251_RAW.tar")
    cols = {}
    for mem in tar.getmembers():
        gsm = mem.name.split("_")[0]
        with gzip.open(tar.extractfile(mem), "rt") as f:
            s = pd.read_csv(f, sep="\t", header=None, index_col=0).iloc[:, 0]
        cols[gsm] = s
    c = pd.DataFrame(cols).fillna(0).astype(np.int64).T
    return c.loc[:, c.columns.str.startswith("ENSG")]


def run_bulk(n_cpus: int) -> list:
    group = bulk_groups()
    counts = bulk_counts()
    group = group.reindex(counts.index)
    log.info("bulk: %s", group.value_counts().to_dict())
    arms = {"bulk_nash_f0f3": ["NASH_F0-F1", "NASH_F2", "NASH_F3"],
            "bulk_nash_all": ["NASH_F0-F1", "NASH_F2", "NASH_F3", "NASH_F4"],
            "bulk_nafl": ["NAFL"]}
    out = []
    for name, arm in arms.items():
        keep = group.isin(arm + ["control"]).to_numpy()
        meta = pd.DataFrame({"condition": np.where(group[keep] == "control", "control", "disease")}, index=counts.index[keep])
        r = fit(counts.loc[keep], meta, "~ condition", "condition", "disease", "control", n_cpus)
        r.index.name = "gene_id"
        r = r.reset_index()
        r["gene_id"] = r["gene_id"].str.replace(r"\.\d+$", "", regex=True)
        r.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
        out.append(dict(name=name, design="~ condition", n_units=int(keep.sum()),
                        units=f"{(meta['condition'] == 'disease').sum()} disease, {(meta['condition'] == 'control').sum()} control",
                        n_genes_tested=int(len(r)), n_padj_lt_0_1=int((r["padj"] < 0.1).sum()),
                        n_padj_lt_0_05=int((r["padj"] < 0.05).sum())))
    return out


# ------------------------------------------------------------------------------ batch control
def run_control(n_cpus: int) -> list:
    """Gribben healthy (primary 3 donors) vs Guilliams lean donor H38; ``guilliams.py`` wrote the counts."""
    g = pd.read_csv(RES / "guilliams_hepatocytes.tsv", sep="\t", index_col=0)
    h = human_units(load_hep("human"))
    h = h[(h.obs["condition"] == "control").to_numpy()].copy()
    hc = pd.DataFrame(np.asarray(h.X.toarray(), dtype=np.int64), index=h.obs["unit"].astype(str), columns=h.var["gene_id"].astype(object))
    hc = hc.loc[:, hc.columns.notna()]
    shared = hc.columns.intersection(g.index)
    out = []
    for name, cols in {"ctrl_gribben_vs_guilliams_cs171": ["CS171"],
                       "ctrl_gribben_vs_guilliams_cs170_cs171": ["CS170", "CS171"]}.items():
        counts = pd.concat([hc[shared], g.loc[shared, cols].T.astype(np.int64)])
        # "a_guilliams" sorts first, so Guilliams is the reference level
        meta = pd.DataFrame({"study": ["gribben"] * len(hc) + ["a_guilliams"] * len(cols)}, index=counts.index)
        r = fit(counts, meta, "~ study", "study", "gribben", "a_guilliams", n_cpus)
        r.index.name = "gene_id"
        r = r.reset_index()
        r.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
        out.append(dict(name=name, design="~ study", n_units=len(counts), units=",".join(counts.index),
                        n_genes_tested=int(len(r)), n_padj_lt_0_1=int((r["padj"] < 0.1).sum()),
                        n_padj_lt_0_05=int((r["padj"] < 0.05).sum())))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=["primary", "sens", "source", "perm", "bulk", "control"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n-cpus", type=int, default=8, help="threads per fit for bulk and control")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.group == "bulk":
        rows = run_bulk(a.n_cpus)
    elif a.group == "control":
        rows = run_control(a.n_cpus)
    else:
        h, m = load_hep("human"), load_hep("mouse")
        jobs = jobs_for(a.group, h, m)
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
                rows.append(fut.result())
    pd.DataFrame(rows).sort_values("name").to_csv(OUT / f"summary_{a.group}.tsv", sep="\t", index=False)
    (OUT / f"meta_{a.group}.json").write_text(json.dumps(
        dict(type=TYPE, min_nuclei=MIN_NUCLEI, min_counts=MIN_COUNTS, n_perm=N_PERM, seed=SEED, pydeseq2="0.5.2",
             shrinkage="pydeseq2 lfc_shrink (apeGLM prior)"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
