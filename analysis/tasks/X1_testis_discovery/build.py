"""X1 (EXPLORATORY): per-gene tables for the testis discovery analysis.

For each testis cohort (canonical s42 run, and any extra seed passed with ``--seed-run``):

* Chimera readouts from the T2.5 gene table: alpha, prior (clamped), prior_raw, S_g_ema,
  margin-support flag; D_raw = (alpha - prior_raw) - cohort median (primary here) and
  D_clamp = (alpha - prior) - cohort median.
* Stage decomposition of S_g: the T2.5 per-(species, fine type) margins S_gsk are summed
  into the six coarse stages of T4.2 (SG, SC, rSD, eSD, Sertoli, Other_somatic) with the
  T2.5 weights w_sk = n_sk / (n_s K), so sum over stages of C_g,stage = S_g exactly. Also
  per species (C_g,s,stage, which sums over species to C_g,stage) and per-cell means
  (Sbar_g,s,stage), for the per-species null (meanz) and the within-type null.
* Simple baselines: per species x stage pseudobulk log2(1 + CPM) (T4.2 definition, libraries
  with >= 30 cells), the species-centred difference to human per stage, the raw per-stage
  log fold change to human, and B1b / B1b_raw from T4.2.
* Chromosome and biotype of the human gene from the Ensembl 110 GTF.

Liver cohorts get the readouts and chromosome only (tissue control).

Read-only on run dirs and data/processed. Writes results/analysis/X1/genes/.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.common import gene_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T4_2_gene_baselines"))
import run as T42  # noqa: E402  (pseudobulk, TESTIS_COARSE, LIBRARY_KEY)

ROOT = gene_ids.REPO_ROOT
T25 = ROOT / "results" / "analysis" / "T2_5"
T42_DIR = ROOT / "results" / "analysis" / "T4_2"
OUT = ROOT / "results" / "analysis" / "X1"
GTF = ROOT / "data" / "analysis" / "ensembl110" / "Homo_sapiens.GRCh38.110.gtf.gz"
STAGES = ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"]
GERM = ["SG", "SC", "rSD", "eSD"]
TESTIS = ["test_hmac", "test_hmouse", "test_3sp"]
LIVER = ["liv_hmonkey", "liv_hmouse", "liv_4sp"]


def chromosome_table() -> pd.DataFrame:
    rows = []
    pg = re.compile(r'gene_id "([^"]+)"')
    pb = re.compile(r'gene_biotype "([^"]+)"')
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t", 8)
            if p[2] != "gene":
                continue
            rows.append((pg.search(p[8]).group(1), p[0], int(p[3]), pb.search(p[8]).group(1)))
    return pd.DataFrame(rows, columns=["hsapiens_gene_id", "chrom", "start", "biotype"]).set_index("hsapiens_gene_id")


def readouts(genes: pd.DataFrame) -> pd.DataFrame:
    g = genes.copy()
    g["D_clamp"] = g["delta_alpha"] - g["delta_alpha"].median()
    g["D_raw"] = g["delta_alpha_raw"] - g["delta_alpha_raw"].median()
    return g


def stage_decomposition(S_long: pd.DataFrame) -> pd.DataFrame:
    """Coarse-stage contributions from the fine-type T2.5 table (cell_type key)."""
    s = S_long.copy()
    unk = set(s["cell_type"]) - set(T42.TESTIS_COARSE)
    if unk:
        raise AssertionError(f"unmapped labels {unk}")
    s["stage"] = s["cell_type"].map(T42.TESTIS_COARSE)
    out = {}
    for mode, tag in (("meanz", ""), ("within", "wt")):
        s[f"wS_{mode}"] = s["weight"] * s[f"S_gsk_{mode}"]
        s[f"nS_{mode}"] = s["n_cells"] * s[f"S_gsk_{mode}"]
        c = s.groupby(["var_name", "stage"])[f"wS_{mode}"].sum().unstack("stage")
        for k in c.columns:
            out[f"C{tag}_{k}"] = c[k]
        cs = s.groupby(["var_name", "species", "stage"])[f"wS_{mode}"].sum().unstack(["species", "stage"])
        for sp_, k in cs.columns:
            out[f"C{tag}_{sp_}_{k}"] = cs[(sp_, k)]
        num = s.groupby(["var_name", "species", "stage"])[f"nS_{mode}"].sum()
        den = s.groupby(["var_name", "species", "stage"])["n_cells"].sum()
        sb = (num / den).unstack(["species", "stage"])
        for sp_, k in sb.columns:
            out[f"Sbar{tag}_{sp_}_{k}"] = sb[(sp_, k)]
    df = pd.DataFrame(out)
    w = s.drop_duplicates(["species", "cell_type"])["weight"].sum()
    if abs(w - 1) > 1e-9:
        raise AssertionError(f"weights sum {w}")
    return df


def stage_weights(S_long: pd.DataFrame) -> pd.DataFrame:
    s = S_long.drop_duplicates(["species", "cell_type"]).copy()
    s["stage"] = s["cell_type"].map(T42.TESTIS_COARSE)
    return s.groupby(["species", "stage"]).agg(n_cells=("n_cells", "sum"), weight=("weight", "sum")).reset_index()


def stage_pseudobulk(cohort: str) -> pd.DataFrame:
    a = C.read_adata(cohort)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    species = a.obs["species"].astype(str).to_numpy()
    group = a.obs["cell_type"].astype(str).map(T42.TESTIS_COARSE).to_numpy()
    library = a.obs["sample_batch"].astype(str).to_numpy()
    x, f, units = T42.pseudobulk(X, species, group, library, T42.MIN_CELLS_TYPE)
    vn = pd.Index(a.var_names, name="var_name")
    out = {}
    for (s, t), v in x.items():
        out[f"x_{s}_{t}"] = v
        out[f"f_{s}_{t}"] = f[(s, t)]
    df = pd.DataFrame(out, index=vn)
    spp = sorted({s for s, _ in x})
    for s in spp:
        if s == "Human":
            continue
        st = [t for t in STAGES if (s, t) in x and ("Human", t) in x]
        H = df[[f"x_Human_{t}" for t in st]].to_numpy()
        O = df[[f"x_{s}_{t}" for t in st]].to_numpy()
        Hc, Oc = H - H.mean(1, keepdims=True), O - O.mean(1, keepdims=True)
        for j, t in enumerate(st):
            df[f"lfc_{s}_{t}"] = O[:, j] - H[:, j]
            df[f"dc_{s}_{t}"] = Oc[:, j] - Hc[:, j]
    return df


def build_testis(cohort: str, label: str, chrom: pd.DataFrame, pb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = T25 if (T25 / f"{label}__S_gsk.parquet").exists() else OUT / "forward"
    g = pd.read_parquet(src / f"{label}__genes.parquet").set_index("var_name")
    S_long = pd.read_parquet(src / f"{label}__S_gsk.parquet")
    g = readouts(g)
    dec = stage_decomposition(S_long)
    tot = dec[[f"C_{k}" for k in STAGES]].sum(1)
    err = float(np.abs(tot.loc[g.index] - g["S_full_meanz"]).max())
    if err > 1e-9:
        raise AssertionError(f"{label}: stage identity error {err}")
    g = g.join(dec).join(pb)
    bl = pd.read_csv(T42_DIR / f"baselines__{cohort}.csv").set_index("hsapiens_gene_id")[["B1b", "B1b_raw", "B1c"]]
    g = g.join(bl, on="hsapiens_gene_id")
    g = g.join(chrom, on="hsapiens_gene_id")
    g["cohort"] = cohort
    g["run_label"] = label
    w = stage_weights(S_long)
    w["cohort"], w["run_label"] = cohort, label
    print(f"{label}: {len(g)} genes, stage identity err {err:.1e}", flush=True)
    return g.reset_index(), w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-run", nargs="*", default=[],
                    help="extra run labels <cohort>_s<seed> with forward tables in results/analysis/X1/forward")
    args = ap.parse_args(argv)
    gdir = OUT / "genes"
    gdir.mkdir(parents=True, exist_ok=True)
    chrom = chromosome_table()
    chrom.to_csv(gdir / "chromosome_ensembl110.csv")
    weights = []
    pbs = {}
    labels = [(c, c) for c in TESTIS] + [(lab.rsplit("_s", 1)[0], lab) for lab in args.seed_run]
    for cohort, label in labels:
        if cohort not in pbs:
            pbs[cohort] = stage_pseudobulk(cohort)
        g, w = build_testis(cohort, label, chrom, pbs[cohort])
        g.to_parquet(gdir / f"{label}.parquet", index=False)
        weights.append(w)
    pd.concat(weights).to_csv(gdir / ("stage_weights.csv" if not args.seed_run else "stage_weights_seeds.csv"),
                              index=False)
    if not args.seed_run:
        for cohort in LIVER:
            g = readouts(pd.read_parquet(T25 / f"{cohort}__genes.parquet"))
            g = g.join(chrom, on="hsapiens_gene_id")
            g["cohort"] = cohort
            g.to_parquet(gdir / f"{cohort}.parquet", index=False)
            print(f"{cohort}: {len(g)} genes", flush=True)


if __name__ == "__main__":
    main()
