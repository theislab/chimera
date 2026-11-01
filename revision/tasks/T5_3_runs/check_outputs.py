"""Check T5.3 latents: shape (n_obs, d), finite, config fields, h5ad order. Run in revision_de_env."""
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np

REPO = Path(__file__).resolve().parents[3]
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
bad = 0
for tool, stem, labels, seed, work, dest in rows:
    d = REPO / dest
    if not (d / "z_mean.npy").exists():
        print("MISSING", dest); bad += 1; continue
    z = np.load(d / "z_mean.npy"); c = json.loads((d / "config.json").read_text())
    a = ad.read_h5ad(REPO / "data/processed" / f"{stem}.h5ad", backed="r"); n = a.n_obs; a.file.close()
    ok = (z.shape[0] == n and np.isfinite(z).all() and c["seed"] == int(seed) and c["adata_stem"] == stem
          and c["label_agreement_after_alignment"] == 1.0 and c.get("git_sha") and c.get("env", {}).get("env_name"))
    bad += not ok
    print("OK " if ok else "BAD", dest, z.shape, c["git_sha"][:7], c["env"]["env_name"], c["labels_key"],
          c["wall_time_s"], c["peak_rss_gb"])
print("bad", bad)
