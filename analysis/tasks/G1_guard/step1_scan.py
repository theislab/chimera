"""G1 step 1: posterior log-variance of every canonical and kept run, read from the saved encoder.

No training and no change to any run. For each run: rebuild the encoder and z_logvar head from model/model.pt, pass all cells of the run's own object through it, and record per latent dimension
the maximum, the minimum and the counts of cell x dim above +10 and below -10 (the clamp of the VAMP KL). Two passes: eval mode (running batch norm, no dropout; what inference sees) and train mode
(batch statistics of shuffled 256-cell batches, dropout on; what the sampler saw at the final state). The eval pass is checked against z_mean in final_state.npz, so a wrong object or a wrong
rebuild fails loudly.

Training-time excursions: model.pt holds the end state only. lightning/metrics.csv has z/mean_learned_var, the batch mean of exp(z_logvar) over cells and dims, per logged training step. With B
cells and d dims, one cell x dim has exp(z_logvar) <= B d v and, for the largest, >= v. So v > exp(10) proves a value above 10 in that batch; v < exp(10) / (B d) proves none; between, not known.
This reads health quantities only; no alpha, S_g or readout.

    python -m analysis.tasks.G1_guard.step1_scan --runs analysis/tasks/G1_guard/step1_runs.tsv --out results/analysis/G1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BOUND = 10.0
BATCH_TRAIN = 256
REPO = Path(__file__).resolve().parents[3]


def rebuild(sd, dropout):
    import torch.nn as nn

    n_in, h = sd["encoder.0.weight"].shape[1], sd["encoder.0.weight"].shape[0]
    enc = nn.Sequential(nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout))
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    mu = nn.Linear(h, sd["z_mean_linear.weight"].shape[0])
    mu.load_state_dict({"weight": sd["z_mean_linear.weight"], "bias": sd["z_mean_linear.bias"]})
    lv = nn.Linear(h, sd["z_logvar_linear.weight"].shape[0])
    lv.load_state_dict({"weight": sd["z_logvar_linear.weight"], "bias": sd["z_logvar_linear.bias"]})
    return enc, mu, lv


class Tally:
    def __init__(self, d):
        self.max = np.full(d, -np.inf)
        self.min = np.full(d, np.inf)
        self.above = np.zeros(d, dtype=np.int64)
        self.below = np.zeros(d, dtype=np.int64)
        self.cells_above = 0
        self.cells_below = 0
        self.n = 0

    def add(self, zl):
        self.max = np.maximum(self.max, zl.max(0))
        self.min = np.minimum(self.min, zl.min(0))
        self.above += (zl > BOUND).sum(0)
        self.below += (zl < -BOUND).sum(0)
        self.cells_above += int((zl > BOUND).any(1).sum())
        self.cells_below += int((zl < -BOUND).any(1).sum())
        self.n += zl.shape[0]


def scan_run(run, X, dropout):
    import torch

    d = REPO / "training" / "runs" / run
    ck = torch.load(d / "model" / "model.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    cfg = json.loads((d / "config.json").read_text())
    if not (cfg.get("mask_species_in_encoder", True) and cfg.get("mask_batch_in_encoder", True)):
        raise NotImplementedError(f"{run}: encoder input not masked")
    enc, mu, lv = rebuild(sd, dropout)
    N, G = X.shape
    extra = sd["encoder.0.weight"].shape[1] - G
    if extra < 2:
        raise AssertionError(f"{run}: encoder input {sd['encoder.0.weight'].shape[1]} against {G} genes")
    n_lat = lv.weight.shape[0]
    pad = torch.zeros(1, extra)

    def inp(idx):
        x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
        return torch.cat([x, pad.expand(len(idx), -1)], 1)

    ev, tr = Tally(n_lat), Tally(n_lat)
    zm = np.zeros((N, n_lat), dtype=np.float32)
    enc.eval()
    with torch.no_grad():
        for st in range(0, N, 512):
            idx = np.arange(st, min(st + 512, N))
            h = enc(inp(idx))
            zm[idx] = mu(h).numpy()
            ev.add(lv(h).numpy())
        enc.train()
        torch.manual_seed(0)
        order = np.random.default_rng(0).permutation(N)
        for st in range(0, N - 1, BATCH_TRAIN):
            idx = order[st:st + BATCH_TRAIN]
            if len(idx) < 2:
                continue
            tr.add(lv(enc(inp(idx))).numpy())
    with np.load(d / "final_state.npz", allow_pickle=False) as f:
        ref = f["z_mean"]
    match = float(np.nanmax(np.abs(ref - zm))) if ref.shape == zm.shape else float("nan")
    return ev, tr, match, cfg


def training_trace(run):
    p = REPO / "training" / "runs" / run / "lightning" / "metrics.csv"
    out = dict(trace_steps=0)
    if not p.exists():
        out["trace_note"] = "no metrics.csv"
        return out
    head = pd.read_csv(p, nrows=0).columns
    if "z/mean_learned_var" not in head:
        out["trace_note"] = "no z/mean_learned_var"
        return out
    m = pd.read_csv(p, usecols=["epoch", "step", "z/mean_learned_var"]).dropna(subset=["z/mean_learned_var"])
    v = m["z/mean_learned_var"]
    fin = v[np.isfinite(v)]
    out.update(trace_steps=len(m), trace_nonfinite=int((~np.isfinite(v)).sum()), trace_last_step=int(m["step"].max()) if len(m) else -1)
    if len(fin):
        i = fin.idxmax()
        out.update(var_max=float(fin.max()), var_max_epoch=int(m.loc[i, "epoch"]))
    return out


def trace_flags(tr, n_lat):
    v = tr.get("var_max")
    if v is None:
        return "unknown"
    if v > math.exp(BOUND) or tr.get("trace_nonfinite", 0) > 0:
        return "above"  # at least one cell x dim above 10 at a logged step
    if v < math.exp(BOUND) / (BATCH_TRAIN * n_lat):
        return "none"  # no cell x dim above 10 at any logged step
    return "indeterminate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import anndata as ad
    import scipy.sparse as sp
    import torch

    from chimera._constants import DROPOUT_RATE

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    runs = pd.read_csv(a.runs, sep="\t", keep_default_na=False)
    by_obj = defaultdict(list)
    for r in runs.itertuples():
        cfg = json.loads((REPO / "training" / "runs" / r.run / "config.json").read_text())
        by_obj[cfg["adata_path"]].append(r)
    rows, dims = [], []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path, rs in by_obj.items():
        o = ad.read_h5ad(REPO / path[path.index("data/"):])  # older configs: chimera/data/..., /Users/...
        X = o.X.tocsr() if sp.issparse(o.X) else sp.csr_matrix(o.X)
        del o
        for r in rs:
            row = dict(cohort=r.cohort, role=r.role, seed=r.seed, run=r.run, adata=path)
            try:
                ev, tr, match, cfg = scan_run(r.run, X, DROPOUT_RATE)
            except Exception as exc:  # noqa: BLE001 - one unreadable run must not stop the table; it stays in it
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                print(r.run, row["error"], flush=True)
                continue
            d = len(ev.max)
            cd = ev.n * d
            row.update(n_cells=ev.n, n_latent=d, zmean_max_abs_diff=match,
                       eval_max=float(ev.max.max()), eval_min=float(ev.min.min()),
                       eval_share_above=float(ev.above.sum() / cd), eval_share_below=float(ev.below.sum() / cd),
                       eval_cells_above=ev.cells_above, eval_cells_below=ev.cells_below,
                       eval_n_above=int(ev.above.sum()), eval_n_below=int(ev.below.sum()),
                       eval_max_per_dim=";".join(f"{v:.3f}" for v in ev.max),
                       train_max=float(tr.max.max()), train_min=float(tr.min.min()),
                       train_share_above=float(tr.above.sum() / (tr.n * d)), train_share_below=float(tr.below.sum() / (tr.n * d)),
                       train_cells_above=tr.cells_above, train_cells_below=tr.cells_below)
            t = training_trace(r.run)
            t["trace_above_10"] = trace_flags(t, d)
            row.update(t)
            rows.append(row)
            for i in range(d):
                dims.append(dict(cohort=r.cohort, seed=r.seed, run=r.run, dim=i, eval_max=ev.max[i], eval_min=ev.min[i],
                                 eval_share_above=ev.above[i] / ev.n, eval_share_below=ev.below[i] / ev.n,
                                 train_max=tr.max[i], train_min=tr.min[i],
                                 train_share_above=tr.above[i] / tr.n, train_share_below=tr.below[i] / tr.n))
            print(r.run, f"eval {row['eval_min']:.2f} {row['eval_max']:.2f} train {row['train_min']:.2f} {row['train_max']:.2f} match {match:.2e} trace {t['trace_above_10']} {t.get('var_max')}", flush=True)
        pd.DataFrame(rows).to_csv(out / "step1_runs.csv.tmp", index=False)
        os.replace(out / "step1_runs.csv.tmp", out / "step1_runs.csv")
    pd.DataFrame(dims).to_csv(out / "step1_per_dim.csv", index=False)
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
