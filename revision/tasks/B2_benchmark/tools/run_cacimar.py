"""CACIMAR adapter (jiang-junyao/CACIMAR, R). Run in cacimar_env.

PRIMARY SCALAR, declared before any comparison:
    score = (power_human + power_mouse) / 2 if the gene pair is an evolutionarily conserved marker,
    meaning a marker in both species of a cell-type pair that CACIMAR calls conserved
    (identify_conserved_pair on the CSCT matrix, mutual best match above the third quartile), else 0.
    power = marker power 2 * |AUC - 0.5| from Identify_Markers. Range 0-1, higher = more conserved.
    Reason: CACIMAR has no per-gene conservation score; marker power is its only gene-level number and
    CSCT sums the powers of shared markers, so this is the gene's own term of that conservation score.

A gene is covered if it is a marker in at least one species. A gene that is a marker in neither
species gets no verdict from the tool and has no row.

Workflow, as in the tutorial: NormalizeData, Identify_Markers(PowerCutoff=0.35, DifferenceCutoff=0.1)
per species, Identify_ConservedCellTypes, identify_conserved_pair (quantile 0.75),
identify_conserved_marker. The ortholog database is built from the cohort's 1:1 table (Type hs_mm_1T1)
in the tool's column layout. Input for the tool:
  - Genes enter under Ensembl gene ids: the tool matches genes by regular expression, which fixed-width
    ids survive and symbols do not (grep("A1BG") also hits A1BG-AS1).
  - Labels enter as fixed-width codes (A01, A02, ...): the tool splits cluster names on "," and "-" and
    matches them by regular expression. Pairs of cell types come from CSCT, never from label names,
    so a label may exist in one species only.
  - Labels with fewer than 3 cells in a species are dropped in that species: Seurat and the tool's own
    per-cluster ROC cannot run on them. Recorded in the sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2t_common as C  # noqa: E402

C.set_threads(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOL_DIR = C.TOOLS / "CACIMAR"
PREFIX = {"Human": "hs", "Mouse": "mm"}
MIN_CELLS = 3
PARAMS = dict(normalization="Seurat NormalizeData default", PowerCutoff=0.35, DifferenceCutoff=0.1, PvalueCutoff=0.05,
              quantile_threshold=0.75, min_cells_per_label=MIN_CELLS)


def export_species(work, sid, X, gene_ids, keep_genes, codes):
    m = pd.notna(codes)
    mat = X[m][:, keep_genes].T.tocsc()
    mat.sort_indices()
    mat.indices.astype(np.int32).tofile(work / f"{sid}_i.bin")
    mat.indptr.astype(np.int32).tofile(work / f"{sid}_p.bin")
    mat.data.astype(np.float64).tofile(work / f"{sid}_x.bin")
    (work / f"{sid}_genes.txt").write_text("\n".join(gene_ids[keep_genes]) + "\n")
    (work / f"{sid}_cells.txt").write_text("\n".join(f"c{i}" for i in range(int(m.sum()))) + "\n")
    (work / f"{sid}_labels.txt").write_text("\n".join(codes[m].astype(str)) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None, help="scratch folder")
    a = p.parse_args(argv)
    t0 = time.time()

    X, var_names, species, labels = C.read_h5ad(a.h5ad, a.label_key)
    genes = C.ortholog_map(a.cohort, var_names)
    usable = (genes["id_human"].notna() & genes["id_mouse"].notna()
              & ~genes["id_human"].duplicated(keep="first") & ~genes["id_mouse"].duplicated(keep="first")).to_numpy()
    gid_h = genes["id_human"].astype(str).to_numpy()
    gid_m = genes["id_mouse"].astype(str).to_numpy()
    uniq = sorted({x for x in labels if x is not None})
    code_of = {x: f"A{i + 1:02d}" for i, x in enumerate(uniq)}
    label_of = {v: k for k, v in code_of.items()}

    tmp_root = Path(a.work) if a.work else Path(a.out).resolve().parent / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cacimar_", dir=tmp_root))
    dropped = {}
    try:
        for s, sid in PREFIX.items():
            m = species == s
            lab = pd.Series(labels[m], dtype=object)
            counts = lab.value_counts()
            small = counts[counts < MIN_CELLS]
            dropped[s] = {k: int(v) for k, v in small.items()}
            kept = counts[counts >= MIN_CELLS].index
            if len(kept) < 2:
                raise SystemExit(f"{s}: CACIMAR needs at least two labels with {MIN_CELLS}+ cells")
            codes = lab.map(lambda v: code_of[v] if v in kept else None).to_numpy(dtype=object)
            export_species(work, sid, X[m], gid_h if sid == "hs" else gid_m, usable, codes)
        orth = pd.DataFrame({"Type": "hs_mm_1T1", "hs_ID": gid_h[usable], "hs_Symbol": gid_h[usable],
                             "mm_ID": gid_m[usable], "mm_Symbol": gid_m[usable]})
        orth.to_csv(work / "orthg.tsv", sep="\t", index=False)

        script = str(HERE / "cacimar_steps.R")
        procs = {sid: subprocess.Popen(["Rscript", "--vanilla", script, "markers", str(work), sid, str(a.seed)])
                 for sid in PREFIX.values()}
        codes_rc = {sid: pr.wait() for sid, pr in procs.items()}
        if any(codes_rc.values()):
            raise SystemExit(f"Identify_Markers failed: {codes_rc}")
        subprocess.run(["Rscript", "--vanilla", script, "conserved", str(work)], check=True)

        mk = {sid: pd.read_csv(work / f"{sid}_markers.tsv", sep="\t", dtype={"Allcluster": str, "Allpower": str, "cluster": str})
              .set_index("gene") for sid in PREFIX.values()}
        cons = pd.read_csv(work / "conserved_markers.tsv", sep="\t", dtype=str).dropna(subset=["hs_conserved_marker", "mm_conserved_marker"])
        cons_pairs = set(zip(cons["hs_conserved_marker"], cons["mm_conserved_marker"]))
        sub = genes[usable].copy()
        pw_h = sub["id_human"].map(mk["hs"]["power"])
        pw_m = sub["id_mouse"].map(mk["mm"]["power"])
        cover = (pw_h.notna() | pw_m.notna()).to_numpy()
        is_cons = np.array([(h, m) in cons_pairs for h, m in zip(sub["id_human"], sub["id_mouse"])])
        score = np.where(is_cons, (pw_h.fillna(0).to_numpy() + pw_m.fillna(0).to_numpy()) / 2, 0.0)

        def decode(values):
            """A01,A02,Ctr0 -> original labels; Ctr0 is the tool's control entry."""
            def one(v):
                if not isinstance(v, str):
                    return ""
                return ";".join(label_of[x] for x in v.split(",") if x in label_of)
            return values.map(one)

        out = pd.DataFrame({
            "var_name": sub["var_name"].values,
            "score": score,
            "is_conserved_marker": is_cons.astype(int),
            "is_marker_in_both": (pw_h.notna() & pw_m.notna()).astype(int).values,
            "power_human": pw_h.values,
            "power_mouse": pw_m.values,
            "clusters_human": decode(sub["id_human"].map(mk["hs"]["Allcluster"])).values,
            "clusters_mouse": decode(sub["id_mouse"].map(mk["mm"]["Allcluster"])).values,
        })[cover].reset_index(drop=True)

        pairs_txt = (work / "conserved_pairs.txt").read_text().split()
        pairs = []
        for pr in pairs_txt:
            h, m = pr.split("-")
            pairs.append([label_of[h[2:]], label_of[m[2:]]])
        failed = [x for x in (work / "conserved_pairs_failed.txt").read_text().splitlines() if x]
        csct = pd.read_csv(work / "csct.tsv", sep="\t", index_col=0)
        csct.index = [label_of[x[2:]] for x in csct.index]
        csct.columns = [label_of[x[2:]] for x in csct.columns]
        C.write_outputs(
            a.out, out, tool="CACIMAR", tool_info=dict(C.tool_state(TOOL_DIR), repo="jiang-junyao/CACIMAR", version="1.0.0"),
            primary="score = (power_human + power_mouse) / 2 for conserved markers, 0 for other genes that are a marker in a species",
            params=PARAMS, args=a, t0=t0, n_input_genes=var_names.size,
            extra={"label_cells": C.label_table(species, labels), "labels_dropped_too_few_cells": dropped,
                   "n_markers": {k: int(len(v)) for k, v in mk.items()}, "n_conserved_markers": int(is_cons.sum()),
                   "conserved_cell_type_pairs": pairs, "conserved_pairs_failed": failed,
                   "csct": json.loads(csct.to_json()), "n_genes_without_1to1_id": int((~usable).sum()), "seed_used": "set.seed in R; the workflow has no random step"})
    finally:
        if not os.environ.get("B2T_KEEP_WORK"):
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
