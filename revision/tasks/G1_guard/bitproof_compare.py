"""G1 step 2: exact comparison of the short fits. Every tensor of the state dict with torch.equal, every array of final_state.npz by bytes, the end RNG states, per group.
    python bitproof_compare.py <before tree> <after tree> <out.csv>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

before, after, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def flat(sd, pre=""):
    """Tensors of the state dict; nested entries (scvi adds an empty pyro_param_store) flattened, non-tensors kept as their repr in a byte tensor."""
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[pre + k] = v
        elif isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = torch.tensor(list(repr(v).encode()), dtype=torch.uint8)
    return out


def load(tree, run):
    d = tree / "training" / "runs" / run
    sd = flat(torch.load(d / "model" / "model.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    with np.load(d / "final_state.npz", allow_pickle=False) as f:
        fs = {k: f[k].copy() for k in f.files}
    return sd, fs, json.loads((d / "bitproof_rng.json").read_text())


def same(a, b):
    sa, fa, ra = a
    sb, fb, rb = b
    keys = list(sa) == list(sb)
    n_eq = sum(torch.equal(sa[k], sb[k]) for k in sa if k in sb)
    f_eq = sum(np.array_equal(fa[k], fb[k], equal_nan=fa[k].dtype.kind == "f") for k in fa if k in fb)
    rng = all(ra[k] == rb[k] for k in ("torch_rng", "numpy_rng", "python_rng"))
    return dict(state_keys_equal=keys, n_tensors=len(sa), n_tensors_equal=n_eq, n_final_arrays=len(fa), n_final_arrays_equal=f_eq, rng_equal=rng)


def digest(sd):
    h = hashlib.sha256()
    for k in sd:
        h.update(k.encode())
        h.update(sd[k].cpu().numpy().tobytes())
    return h.hexdigest()


rows = []
groups = sorted({p.name.rsplit("-", 1)[0] for p in (before / "training/runs").glob("g1bit-*-a")})
for g in groups:
    a, a2, c = load(before, g + "-a"), load(before, g + "-b"), load(after, g + "-a")
    for label, x, y in (("before vs before (control)", a, a2), ("before vs after, flag off", a, c)):
        rows.append(dict(group=g, compare=label, threads=x[2]["threads"], sha256_x=digest(x[0]), sha256_y=digest(y[0]), **same(x, y)))
d = pd.DataFrame(rows)
d["identical"] = (d.n_tensors == d.n_tensors_equal) & (d.n_final_arrays == d.n_final_arrays_equal) & d.state_keys_equal & d.rng_equal
d.to_csv(out, index=False)
print(d.drop(columns=["sha256_x", "sha256_y"]).to_string())
print(d[["group", "compare", "sha256_x", "sha256_y"]].to_string())
