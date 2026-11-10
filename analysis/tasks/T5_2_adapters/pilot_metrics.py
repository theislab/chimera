"""Pilot sanity metrics: ARI and NMI of Leiden clusters vs cell_ontology_class.

Run in revision_de_env. Mirrors the scib procedure behind the published CAMEX
numbers (scib.me.nmi / ari after cluster_optimal_resolution): kNN graph on the
embedding (k=15), Leiden at resolutions 0.2, 0.4, ..., 2.0, keep the
clustering with the highest NMI (sklearn, arithmetic mean), report ARI
(sklearn) of that clustering. Also reports resolution 1.0. Fixed before the
pilot ran: primary = optimal-resolution values; SATURN primary latent = raw
`.X` (z_raw.npy), L2-normalised z_mean.npy reported beside it.

Published anchors (CAMEX Source Data, Fig. S1d, 4-species liver):
CAMEX ARI 0.919, NMI 0.834; SATURN ARI 0.269, NMI 0.627.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PUBLISHED = {"camex": {"ARI": 0.919, "NMI": 0.834, "tol": 0.10},
             "saturn": {"ARI": 0.269, "NMI": 0.627, "tol": 0.15}}


def score(z: np.ndarray, labels: np.ndarray) -> dict:
    a = ad.AnnData(X=np.zeros((len(z), 1), dtype=np.float32))
    a.obsm["emb"] = z.astype(np.float32)
    sc.pp.neighbors(a, use_rep="emb", n_neighbors=15, random_state=0)
    rows = []
    for res in [round(2 * x / 10, 1) for x in range(1, 11)]:
        sc.tl.leiden(a, resolution=res, random_state=0, key_added="cl")
        cl = a.obs["cl"].astype(str).values
        rows.append({"res": res, "n_clusters": len(set(cl)),
                     "NMI": normalized_mutual_info_score(labels, cl, average_method="arithmetic"),
                     "ARI": adjusted_rand_score(labels, cl)})
    df = pd.DataFrame(rows)
    best = df.loc[df["NMI"].idxmax()]
    r1 = df.loc[df["res"] == 1.0].iloc[0]
    return {"opt_res": float(best["res"]), "NMI": float(best["NMI"]), "ARI": float(best["ARI"]),
            "n_clusters_opt": int(best["n_clusters"]),
            "NMI_res1": float(r1["NMI"]), "ARI_res1": float(r1["ARI"]), "sweep": rows}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="dir with z_mean.npy, config.json (z_raw.npy for SATURN)")
    p.add_argument("--tool", choices=["camex", "saturn"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = Path(a.run_dir)
    cfg = json.loads((d / "config.json").read_text())
    obs = ad.read_h5ad(cfg["adata"], backed="r").obs
    labels = obs["cell_ontology_class"].astype(str).values
    if len(labels) != cfg["n_obs"]:
        sys.exit("subsampled run; metrics need the full cohort")
    out = {"run_dir": str(d), "tool": a.tool, "stem": cfg["adata_stem"], "seed": cfg["seed"],
           "procedure": "kNN k=15 on embedding; Leiden res 0.2..2.0; max-NMI clustering; sklearn NMI/ARI",
           "label_key": "cell_ontology_class", "published": PUBLISHED[a.tool]}
    if a.tool == "saturn":
        out["raw_X"] = score(np.load(d / "z_raw.npy"), labels)
        out["l2"] = score(np.load(d / "z_mean.npy"), labels)
        prim = out["raw_X"]
    else:
        out["latent"] = score(np.load(d / "z_mean.npy"), labels)
        prim = out["latent"]
    pub = PUBLISHED[a.tool]
    out["primary"] = {"ARI": prim["ARI"], "NMI": prim["NMI"],
                      "dARI": prim["ARI"] - pub["ARI"], "dNMI": prim["NMI"] - pub["NMI"],
                      "pass": bool(abs(prim["ARI"] - pub["ARI"]) <= pub["tol"] and abs(prim["NMI"] - pub["NMI"]) <= pub["tol"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("stem", "tool", "primary")}, indent=2))


if __name__ == "__main__":
    main()
