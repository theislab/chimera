"""Standalone precompute for ``10_gin_attribution_protein_structure.ipynb``.

Memory-heavy steps (PCA fit on 4-species 5120-d ESM-2 consensus, bridge ridge
regression on 200 random ortholog groups) are expensive enough that running
them inside an interactive Jupyter kernel can push a 17 GB system over its
limit when VS Code + a browser are also resident.

This script does the heavy work as a one-shot CLI, writes lightweight caches
to ``notebooks/cache/gin_attribution/``, and exits — freeing all working
memory before the notebook is even opened.  After running this script, the
notebook's §D and §G.5 cells just load from cache and skip the heavy paths.

Run from ``chimera_env``::

    mamba run -n chimera_env python chimera/notebooks/_gin_attribution_precompute.py

Idempotent: re-runs are no-ops (cache hits).  Use ``--force`` to invalidate
caches and recompute.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

NB_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("CHIMERA_ROOT") or NB_DIR.parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import _paper_utils as pu
from chimera._data_utils import download_and_process_esm2

CACHE_DIR = NB_DIR / "cache" / "gin_attribution"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_CACHE_DIR = CACHE_DIR / "bridge"
BRIDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_EMB35M_DIR = BRIDGE_CACHE_DIR / "esm2_35m_consensus_human"
BRIDGE_EMB35M_DIR.mkdir(parents=True, exist_ok=True)

PCA_4SP_SPECIES = ["hsapiens", "mmulatta", "mmusculus", "drerio"]
DATASETS = ["panc_hmonkey", "panc_4sp"]


# ---------------------------------------------------------------------------
# §D — PCA basis (4-species)
# ---------------------------------------------------------------------------
def step_pca(force: bool = False) -> dict:
    cache = CACHE_DIR / "pca_basis_v3__4sp.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        print(f"[pca] cache hit  {cache.name}  (V={d['V'].shape})")
        return {"V": d["V"], "mean": d["mean"], "eig": d["eig"], "evr": d["evr"]}

    print(f"[pca] fitting 4-species PCA (5120 -> 64)…")
    from sklearn.decomposition import PCA
    ortho = pu.load_ortholog_table("liver")  # DATASETS are liver cohorts
    t0 = time.time()
    consensus, _ = download_and_process_esm2(ortho, PCA_4SP_SPECIES)
    print(f"[pca] consensus shape {consensus.shape}  in {time.time()-t0:.1f}s")
    t0 = time.time()
    pca = PCA(n_components=64, random_state=42).fit(consensus)
    V = pca.components_.T.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    eig = pca.explained_variance_.astype(np.float32)
    evr = pca.explained_variance_ratio_.astype(np.float32)
    np.savez(cache, V=V, mean=mean, eig=eig, evr=evr)
    print(f"[pca] fit + saved in {time.time()-t0:.1f}s  "
          f"explained var = {evr.sum():.3f}  cache size = {cache.stat().st_size/1024:.1f} KB")
    del pca, consensus
    gc.collect()
    return {"V": V, "mean": mean, "eig": eig, "evr": evr}


# ---------------------------------------------------------------------------
# §G.5 — Bridge fits (one per anchor dataset)
# ---------------------------------------------------------------------------
def step_bridges(pca_data: dict, force: bool = False, sample_n: int = 200,
                 alpha: float = 1.0) -> None:
    from sklearn.linear_model import Ridge

    ortho = pu.load_ortholog_table("liver")  # DATASETS are liver cohorts
    V = pca_data["V"]; mu = pca_data["mean"]

    # Decide which datasets need a refit
    targets = []
    for ds in DATASETS:
        cp = BRIDGE_CACHE_DIR / f"W_pc_v2_{ds}.npz"
        if cp.exists() and not force:
            print(f"[bridge:{ds}] cache hit  {cp.name}")
        else:
            targets.append((ds, cp))
    if not targets:
        return

    # Heavy load: 4-species consensus matrix once, project to PC space, free.
    print(f"[bridge] loading 4-species consensus for bridge target…")
    t0 = time.time()
    consensus_5120, _ = download_and_process_esm2(ortho, PCA_4SP_SPECIES)
    consensus_pc = (consensus_5120 - mu) @ V                 # (G, 64)
    valid_idx = []
    for og_id in range(consensus_5120.shape[0]):
        if og_id not in ortho.index:
            continue
        gid = ortho.loc[og_id].get("hsapiens_gene_id")
        if not (isinstance(gid, str) and gid):
            continue
        if np.all(consensus_5120[og_id] == 0):
            continue
        valid_idx.append(og_id)
    del consensus_5120
    gc.collect()
    print(f"[bridge] {len(valid_idx)} valid ortho groups (with human gene + non-zero "
          f"5120 consensus)  in {time.time()-t0:.1f}s")

    # Lazy-load ESM-2 35M
    print(f"[bridge] loading ESM-2 35M…")
    import esm
    import torch
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()

    # Reuse notebook 09's protein-sequence cache + extend if needed.
    PROT_CACHE = NB_DIR / "cache" / "protein_structure"
    SEQ_CACHE_PATH = PROT_CACHE / "sequences.json"
    seqs = json.loads(SEQ_CACHE_PATH.read_text()) if SEQ_CACHE_PATH.exists() else {}

    def _consensus_35m(seq: str) -> np.ndarray:
        if not seq:
            return np.zeros(480, dtype=np.float32)
        seq_use = seq[:1022]
        labels, strs, tokens = bc([("seq", seq_use)])
        with torch.no_grad():
            res = model(tokens, repr_layers=[12], return_contacts=False)
        emb = res["representations"][12][0, 1:1+len(seq_use)].numpy()
        return emb.mean(axis=0).astype(np.float32)

    def _get_or_build_consensus_35m(og_id: int, gid: str) -> np.ndarray:
        cp_emb = BRIDGE_EMB35M_DIR / f"og_{og_id}.npy"
        if cp_emb.exists():
            return np.load(cp_emb).astype(np.float32)
        seq = seqs.get(gid, "")
        if not seq:
            # fetch from ENSEMBL on demand
            try:
                import requests
                time.sleep(0.05)
                r = requests.get(f"https://rest.ensembl.org/lookup/id/{gid}?expand=1",
                                  headers={"Content-Type":"application/json"}, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    transcripts = data.get("Transcript", [])
                    canon = next((t for t in transcripts if t.get("is_canonical")), None)
                    if canon is None and transcripts:
                        canon = max(transcripts, key=lambda t: t.get("length", 0))
                    if canon is not None:
                        time.sleep(0.05)
                        r2 = requests.get(f"https://rest.ensembl.org/sequence/id/{canon['id']}?type=protein",
                                           headers={"Content-Type":"text/x-fasta"}, timeout=30)
                        if r2.status_code == 200:
                            seq = "".join(l for l in r2.text.splitlines()
                                            if not l.startswith(">")).replace("*","").upper()
                            seqs[gid] = seq
                            SEQ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                            SEQ_CACHE_PATH.write_text(json.dumps(seqs, indent=2))
            except Exception:
                pass
        if not seq:
            v = np.zeros(480, dtype=np.float32)
            np.save(cp_emb, v)
            return v
        v = _consensus_35m(seq)
        np.save(cp_emb, v)
        return v

    # Same random subsample seed across datasets so the only difference is which
    # rows survive each one's HVG / consensus filter.
    rng = np.random.default_rng(42)
    sample = rng.choice(np.asarray(valid_idx),
                         size=min(sample_n, len(valid_idx)),
                         replace=False)
    print(f"[bridge] running ESM-2 35M on {len(sample)} canonical human proteins…")
    X_480, Y_pc, og_keep = [], [], []
    t0 = time.time()
    for n_done, og_id in enumerate(sample):
        gid = ortho.loc[og_id]["hsapiens_gene_id"]
        v480 = _get_or_build_consensus_35m(int(og_id), gid)
        if np.all(v480 == 0): continue
        X_480.append(v480)
        Y_pc.append(consensus_pc[og_id])
        og_keep.append(og_id)
        if (n_done + 1) % 50 == 0:
            print(f"[bridge]   {n_done+1}/{len(sample)}  ({time.time()-t0:.0f}s)")
    X_480 = np.vstack(X_480); Y_pc = np.vstack(Y_pc)
    print(f"[bridge] running ridge on {X_480.shape[0]} pairs  ({time.time()-t0:.0f}s)")

    # Free ESM-2 35M model
    del model, alphabet, bc
    gc.collect()

    # 80/20 split for held-out R² per PC
    n = X_480.shape[0]
    rng2 = np.random.default_rng(0)
    ho_size = max(int(0.2 * n), 20)
    ho = rng2.choice(n, size=min(ho_size, n - 30), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[ho] = False
    reg = Ridge(alpha=alpha).fit(X_480[train_mask], Y_pc[train_mask])
    W_pc = reg.coef_.T.astype(np.float32)         # (480, 64)
    intercept = reg.intercept_.astype(np.float32) # (64,)
    mean_480 = X_480[train_mask].mean(axis=0).astype(np.float32)
    pred_ho = X_480[ho] @ W_pc + intercept
    ss_res = ((Y_pc[ho] - pred_ho) ** 2).sum(axis=0)
    ss_tot = ((Y_pc[ho] - Y_pc[ho].mean(axis=0)) ** 2).sum(axis=0)
    r2 = (1.0 - ss_res / np.maximum(ss_tot, 1e-9)).astype(np.float32)
    print(f"[bridge] PC R² distribution: median {np.median(r2):.3f}  "
          f"min {r2.min():.3f}  max {r2.max():.3f}")

    # Same fit applies to both anchor datasets (the bridge target is dataset-
    # independent — see §D audit).  Save the same payload to both cache files.
    for ds, cp in targets:
        np.savez(cp, W_pc=W_pc, intercept=intercept, mean_480=mean_480,
                  r2_top_pc=r2)
        print(f"[bridge:{ds}] saved → {cp.name}  ({cp.stat().st_size/1024:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="invalidate caches and recompute")
    ap.add_argument("--bridge-sample", type=int, default=200,
                     help="ortholog groups to fit the bridge on (default 200)")
    args = ap.parse_args()

    pca = step_pca(force=args.force)
    step_bridges(pca, force=args.force, sample_n=args.bridge_sample)
    print("[done] precompute finished — open the notebook now.")


if __name__ == "__main__":
    main()
