"""Cell-type decomposition of the fit-quality margin S_g (the count evidence) from checkpoints.

A checkpoint-only forward pass (no training, nothing written to run dirs) recomputes,
for every cell, the per-gene log-likelihood ratio the model uses for S_g
(``chimera/_module.py::_compute_S_g``): shared decoder on the encoder's z against the
per-species null rate, NB with the trained per-batch dispersion. Sums per
(species, cell type) give S_gsk, C_gk and the exact identity S_g = sum_k C_gk (see
``margin.py``). Two z modes: the posterior mean (deterministic; the decomposition) and one
posterior sample per cell with a fixed seed (as in training; the closure check against
the trained buffer S_g_ema). A within-type null is computed alongside.

Needs ``chimera_env`` (torch, scvi-tools, the chimera package). The model is loaded with
``ChimeraModel.load(..., accelerator="cpu")``; if that fails the script rebuilds the
encoder and shared decoder from the checkpoint state dict.

Acceptance (checked and written to ``acceptance.csv``): Pearson(pass-rate x S, S_g_ema)
>= 0.97 per run; identity sums within 1e-6; major-type (weight >= 0.10) cross-seed
Spearman >= 0.9 on cohorts with seeds.

Usage::

    python analysis/tasks/T2_5_celltype_margin/run.py [--runs all] [--threads 4] [--n-boot 200]
        [--outdir DIR] [--reuse-forward]

The closure panel of the paper, figures/panels/supp_celltype_closure/closure.pdf, is drawn from the cached gene tables alone by

    python -c "from analysis.tasks.T2_5_celltype_margin import run; run.draw_panels()"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args
from figures import style

sys.path.insert(0, str(Path(__file__).resolve().parent))
import margin as MG  # noqa: E402

# run label -> (cohort, run name)
RUNS = {c: (c, r) for c, r in C.CANONICAL.items()}
for _c, _runs in C.SEED_RUNS.items():
    for _r in _runs[1:]:
        RUNS[f"{_c}_{_r.rsplit('-', 1)[-1]}"] = (_c, _r)
GROUP_KEYS = {"liver": ["cell_type"], "testis": ["cell_type", "cell_ontology_class"]}
LANDMARKS = sorted({"CPS1", "ARG1", "ASL", "OTC", "ASS1", "NAGS", "PCK1", "PAH", "SELP", "SELE", "VCAM1", "F3",
                    "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC", "DDX4", "RPL3", "EEF1A1", "EEF2",
                    "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A", "ALB", "APOA1", "PRM2", "TNP1"})
MAIN_PANEL_RUN = "liv_4sp"
SEED = 0


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def _components_from_model(run_dir: Path, adata, cfg: dict):
    import torch.nn.functional as F
    from scvi import REGISTRY_KEYS

    from chimera import ChimeraModel

    model = ChimeraModel.load(str(run_dir / "model"), adata=adata, accelerator="cpu")
    m = model.module.eval()
    mgr = model.adata_manager
    sp_codes = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).ravel().astype(int)
    b_codes = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel().astype(int)
    sp_names = [str(x) for x in mgr.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping]
    n_s, n_b = int(m.n_species), int(m.n_batch)

    def encode(x, s_idx, b_idx):
        s_oh = F.one_hot(s_idx, n_s).float()
        b_oh = F.one_hot(b_idx, n_b).float()
        h = m.encoder(m._encoder_input(x, s_oh, b_oh))
        return m.z_mean_linear(h), m.z_logvar_linear(h)

    def shared_rate(z):
        return m.output_shared(m.decoder_hidden(z))

    return dict(source="ChimeraModel.load", encode=encode, shared_rate=shared_rate,
                theta=F.softplus(m.dispersion.detach()).double(), p_null=m.p_null.detach().double(),
                S_ema=m.S_g_ema.detach().cpu().numpy().astype(float), species_codes=sp_codes,
                batch_codes=b_codes, species_names=sp_names)


def _components_from_state_dict(run_dir: Path, adata, cfg: dict):
    """Fallback: rebuild encoder and shared decoder from the state dict (architecture of
    ``chimera/_module.py``; species and batch codes are the sorted category strings)."""
    import importlib

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    path = run_dir / "model" / "model.pt"
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:  # checkpoints pickled with numpy >= 2 read under numpy < 2
        import numpy.core as nc
        sys.modules["numpy._core"] = nc
        for s in ["multiarray", "numeric", "_multiarray_umath", "overrides", "fromnumeric", "umath"]:
            sys.modules["numpy._core." + s] = importlib.import_module("numpy.core." + s)
        d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    if list(d["var_names"]) != list(adata.var_names):
        raise AssertionError("checkpoint var_names differ from the h5ad")
    sub = lambda pre: {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}  # noqa: E731
    G, nh = sd["output_shared.weight"].shape
    nl = sd["z_mean_linear.weight"].shape[0]
    n_in = sd["encoder.0.weight"].shape[1]
    species = sorted(adata.obs["species"].astype(str).unique())
    bkey = cfg.get("batch_key") or "batch"
    batches = sorted(adata.obs[bkey].astype(str).unique())
    n_s, n_b = len(species), len(batches)
    if n_in != G + n_s + n_b or sd["p_null"].shape != (n_s, G) or sd["dispersion"].shape != (n_b, G):
        raise AssertionError("state dict shapes do not match species/batch counts")
    enc = nn.Sequential(nn.Linear(n_in, nh), nn.BatchNorm1d(nh), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.ReLU(), nn.Dropout(0.1))
    zm, zl = nn.Linear(nh, nl), nn.Linear(nh, nl)
    dec = nn.Sequential(nn.Linear(nl, nh), nn.BatchNorm1d(nh), nn.ReLU(), nn.Linear(nh, nh),
                        nn.BatchNorm1d(nh), nn.ReLU())
    sh = nn.Linear(nh, G)
    for mod, pre in ((enc, "encoder."), (zm, "z_mean_linear."), (zl, "z_logvar_linear."),
                     (dec, "decoder_hidden."), (sh, "output_shared.")):
        mod.load_state_dict(sub(pre))
        mod.eval()
    mask_s = cfg.get("mask_species_in_encoder", True)
    mask_b = cfg.get("mask_batch_in_encoder", True)

    def encode(x, s_idx, b_idx):
        s_oh = F.one_hot(s_idx, n_s).float()
        b_oh = F.one_hot(b_idx, n_b).float()
        if mask_s:
            s_oh = torch.zeros_like(s_oh)
        if mask_b:
            b_oh = torch.zeros_like(b_oh)
        h = enc(torch.cat([x, s_oh, b_oh], 1))
        return zm(h), zl(h)

    def shared_rate(z):
        return sh(dec(z))

    sp_lab = adata.obs["species"].astype(str).to_numpy()
    b_lab = adata.obs[bkey].astype(str).to_numpy()
    return dict(source="state_dict", encode=encode, shared_rate=shared_rate,
                theta=F.softplus(sd["dispersion"].float()).double(), p_null=sd["p_null"].double(),
                S_ema=sd["S_g_ema"].numpy().astype(float),
                species_codes=np.array([species.index(s) for s in sp_lab]),
                batch_codes=np.array([batches.index(b) for b in b_lab]), species_names=species)


def load_components(run_dir: Path, adata, cfg: dict):
    try:
        return _components_from_model(run_dir, adata, cfg)
    except Exception as exc:  # noqa: BLE001 - documented fallback
        print(f"  ChimeraModel.load failed ({type(exc).__name__}: {exc}); using the state dict", flush=True)
        return _components_from_state_dict(run_dir, adata, cfg)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def nb_llr(x, mu1, mu0, th):
    """log NB(x | mu1, th) - log NB(x | mu0, th); the x- and theta-only terms cancel."""
    import torch

    e = 1e-8
    return (th * (torch.log(th + mu0 + e) - torch.log(th + mu1 + e))
            + x * (torch.log(mu1 + e) - torch.log(th + mu1 + e) - torch.log(mu0 + e) + torch.log(th + mu0 + e)))


def forward_sums(label: str, cohort: str, run: str, threads: int, batch_size: int) -> dict:
    import scipy.sparse as sp
    import torch

    torch.set_num_threads(threads)
    t0 = time.time()
    run_dir = C.RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    if not cfg.get("use_observed_lib_size", True):
        raise NotImplementedError("forward pass assumes use_observed_lib_size=True")
    adata = C.read_adata(cohort)
    comp = load_components(run_dir, adata, cfg)
    fs = C.load_state(run)
    if not np.allclose(comp["S_ema"], fs["S_g_ema"], atol=1e-6):
        raise AssertionError(f"{run}: checkpoint S_g_ema differs from final_state.npz")
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    N, G = X.shape
    s_idx_all = comp["species_codes"]
    b_idx_all = comp["batch_codes"]
    species = comp["species_names"]
    S = len(species)
    keys = GROUP_KEYS["testis" if C.is_testis(cohort) else "liver"]
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)

    groups = {}
    for key in keys:
        lab = adata.obs[key].astype(str).to_numpy()
        pairs = sorted({(int(s), l) for s, l in zip(s_idx_all, lab)})
        gi = {p: i for i, p in enumerate(pairs)}
        cell_group = np.array([gi[(int(s), l)] for s, l in zip(s_idx_all, lab)])
        # within-type null: Laplace-smoothed rate of each (species, type), as precompute_p_null
        p_wt = np.zeros((len(pairs), G))
        for j, (s, l) in enumerate(pairs):
            m = cell_group == j
            p_wt[j] = (np.asarray(X[m].sum(0)).ravel() + 1.0) / (lib[m].sum() + G)
        groups[key] = dict(pairs=pairs, cell_group=cell_group, p_wt=torch.tensor(p_wt),
                           sum_meanz=np.zeros((len(pairs), G)), sum_samp=np.zeros((len(pairs), G)),
                           sum_within=np.zeros((len(pairs), G)))
    sp_sum = {m: np.zeros((S, G)) for m in ("meanz", "samp")}
    sym = gene_ids.symbols(cohort, adata.var_names)
    lm_idx = np.where(sym.isin(LANDMARKS).to_numpy() | sym.astype(str).str.startswith("MT-").to_numpy())[0]
    lm_l = np.zeros((N, len(lm_idx)))
    gen = torch.Generator().manual_seed(SEED)
    libratio = []
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_t = torch.tensor(s_idx_all[idx], dtype=torch.long)
            b_t = torch.tensor(b_idx_all[idx], dtype=torch.long)
            L = x.sum(1, keepdim=True).double()
            zmu, zlv = comp["encode"](x, s_t, b_t)
            eps = torch.randn(zmu.shape, generator=gen)
            xd = x.double()
            th = comp["theta"][b_t]
            for mode, z in (("meanz", zmu), ("samp", zmu + eps * torch.exp(0.5 * zlv))):
                mu1 = torch.exp(comp["shared_rate"](z).double()) * L
                Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
                mu0 = (comp["p_null"][s_t] * Lt).clamp(min=1e-8)
                ll = nb_llr(xd, mu1, mu0, th)
                lnp = ll.numpy()
                np.add.at(sp_sum[mode], s_idx_all[idx], lnp)
                for key, gr in groups.items():
                    np.add.at(gr[f"sum_{mode}"], gr["cell_group"][idx], lnp)
                    if mode == "samp":
                        cg = torch.tensor(gr["cell_group"][idx], dtype=torch.long)
                        mu0w = (gr["p_wt"][cg] * Lt).clamp(min=1e-8)
                        np.add.at(gr["sum_within"], gr["cell_group"][idx], nb_llr(xd, mu1, mu0w, th).numpy())
                if mode == "meanz":
                    lm_l[idx] = lnp[:, lm_idx]
                    libratio.append((Lt.squeeze(1) / L.squeeze(1).clamp(min=1)).numpy())
    fw = dict(var_names=np.array(adata.var_names, dtype=str), species=np.array(species, dtype=str),
              n_species=np.bincount(s_idx_all, minlength=S), S_ema=comp["S_ema"],
              sp_sum_meanz=sp_sum["meanz"], sp_sum_samp=sp_sum["samp"], landmark_idx=lm_idx, landmark_l=lm_l,
              libratio=np.concatenate(libratio), source=np.array(comp["source"]), keys=np.array(keys))
    for key, gr in groups.items():
        n = np.bincount(gr["cell_group"], minlength=len(gr["pairs"]))
        fw[f"{key}__groups"] = np.array([(s, l, c) for (s, l), c in zip(gr["pairs"], n)],
                                         dtype=[("species_idx", "i4"), ("label", "U64"), ("n", "i8")])
        fw[f"{key}__cell_group"] = gr["cell_group"]
        for mode in ("meanz", "samp", "within"):
            fw[f"{key}__sum_{mode}"] = gr[f"sum_{mode}"]
    print(f"  {label}: {comp['source']}, {N} cells, {G} genes, {time.time() - t0:.0f}s", flush=True)
    return fw


def save_npz(out: TaskOutput, fw: dict, name: str) -> None:
    final = out.dir / "forward" / name
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.stem + ".tmp.npz")
    np.savez_compressed(tmp, **fw)
    tmp.replace(final)
    out.files.append(str(final.relative_to(out.dir)))


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _design():
    sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
    try:
        import _figure_design as fd
        fd.apply_design()
    except Exception as exc:  # noqa: BLE001
        print("figure design module unavailable, plain matplotlib:", exc)
    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    return plt


def fig_main(C_long: pd.DataFrame, genes: pd.DataFrame, title_run: str):
    """Landmark genes x cell types, weighted contribution C_gk; dagger marks lacking support."""
    plt = _design()
    lm = genes[genes["symbol"].isin(LANDMARKS) | genes["symbol"].astype(str).str.startswith("MT-")]
    lm = lm.sort_values("delta_alpha")
    sub = C_long[C_long["var_name"].isin(lm["var_name"])]
    M = sub.pivot(index="var_name", columns="cell_type", values="C_gk_meanz").loc[lm["var_name"]]
    order = (sub.groupby("cell_type")["weight_k"].first().sort_values(ascending=False).index)
    M = M[order]
    v = float(np.nanmax(np.abs(M.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(0.32 * M.shape[1] + 1.8, 0.16 * M.shape[0] + 1.0))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    flag = lm.set_index("var_name")["lacks_margin_support"]
    ylabels = [f"{s} ({d:+.2f}){' †' if flag[g] else ''}" for g, s, d in
               zip(lm["var_name"], lm["symbol"], lm["delta_alpha"])]
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=5, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("contribution C_gk to S_g", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def fig_closure(gene_tables: dict):
    """One plot per cohort at its size on the page, four per row at the text width: S_g rebuilt from one pass over all
    cells (x) against S_g at the end of training (y), every gene drawn. Each plot has its own range, the same on both
    axes; the two axes are named once for the figure, and the free place after the last plot holds the key."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.apply()
    mark, ref_line = style.READOUT["count_evidence"], "#888888"  # the genes in the color of the count evidence; the diagonal
    labs = [k for k in gene_tables if k in C.CANONICAL]
    ncol = 4
    nrow = int(np.ceil(len(labs) / ncol))
    # geometry, bp
    W = style.TEXTWIDTH
    left, right, top, bottom, side, ticks_w, title_h, ticks_h, gap_y = 12.0, 3.0, 1.0, 12.0, 92.0, 19.0, 10.0, 11.0, 4.0
    gap_x = (W - left - right - ncol * (ticks_w + side)) / (ncol - 1)
    row_h = title_h + side + ticks_h
    H = top + nrow * row_h + (nrow - 1) * gap_y + bottom

    def place(k):  # lower left corner of plot k
        i, j = divmod(k, ncol)
        return left + j * (ticks_w + side + gap_x) + ticks_w, H - top - i * (row_h + gap_y) - title_h - side

    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    for k, lab in enumerate(labs):
        t = gene_tables[lab]
        x, y = t["ppass_S_samp"].to_numpy(), t["S_g_ema"].to_numpy()
        x0, y0 = place(k)
        ax = fig.add_axes([x0 / W, y0 / H, side / W, side / H])
        ax.scatter(x, y, s=1.4, color=mark.color, alpha=0.5, linewidths=0, rasterized=True)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())  # every gene inside the axes
        lo, hi = lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)
        ax.plot([lo, hi], [lo, hi], color=ref_line, linewidth=style.LW_REF, linestyle="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.locator_params(axis="both", nbins=4)
        ax.set_title(style.cohort_name(lab), pad=2.5)
        ax.text(0.04, 0.96, f"r = {np.corrcoef(x, y)[0, 1]:.3f}", transform=ax.transAxes, fontsize=style.SIZE_TICK, va="top")
        ax.tick_params(length=2, pad=1.5)
    fig.text(5.0 / W, (H - top + bottom) / 2 / H, "$S_g$ at the end of training", rotation=90, ha="center", va="center", fontsize=style.SIZE_LABEL)
    fig.text((left + ticks_w + W - right) / 2 / W, 1.5 / H, "$S_g$ rebuilt from one pass over all cells", ha="center", va="bottom",
             fontsize=style.SIZE_LABEL)
    if len(labs) < nrow * ncol:  # the key, in the first free place
        x0, y0 = place(len(labs))
        fig.legend([Line2D([], [], marker="o", color=mark.color, ls="", ms=2.2, mew=0), Line2D([], [], color=ref_line, lw=style.LW_REF, ls="--"),
                    Line2D([], [], ls="")], ["one gene", "equal values", "r, Pearson correlation\nover all genes"], loc="center left",
                   bbox_to_anchor=(x0 / W, (y0 + side / 2) / H), handlelength=1.6, labelspacing=0.6, borderaxespad=0)
    return fig


def draw_panels() -> None:
    """The closure panel of the paper from the cached gene tables (<cohort>__genes.parquet); nothing is computed."""
    import matplotlib.pyplot as plt

    res = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_5"
    fig = fig_closure({c: pd.read_parquet(res / f"{c}__genes.parquet") for c in C.CANONICAL})
    with plt.rc_context({"savefig.dpi": 360}):  # rasterized gene clouds: above 300 ppi on the page
        style.save_panel(fig, "supp_celltype_closure/closure", exact=True)


def fig_seed(seed_df: pd.DataFrame):
    plt = _design()
    d = seed_df[(seed_df.cell_type != "(S_g total)")]
    d = d.groupby(["cohort", "cell_type"]).agg(rho_min=("rho", "min"), rho_max=("rho", "max"),
                                               weight=("weight", "first")).reset_index()
    d = d.sort_values(["cohort", "weight"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(3.3, 0.18 * len(d) + 0.6))
    y = np.arange(len(d))
    ax.hlines(y, d.rho_min, d.rho_max, color="#555555", linewidth=1)
    ax.scatter(d.rho_min, y, s=8, color="#555555", zorder=3)
    ax.axvline(MG.ACCEPT_SEED, color="#999999", linewidth=0.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{C.LABELS[c]}: {k} (w={w:.2f})" for c, k, w in zip(d.cohort, d.cell_type, d.weight)],
                       fontsize=5)
    ax.set_xlabel("cross-seed Spearman of C_gk (range over seed pairs)", fontsize=6)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="all", help="comma-separated run labels or 'all': " + ",".join(RUNS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--reuse-forward", action="store_true",
                    help="reuse forward/<run>.npz in the output folder instead of recomputing")
    add_common_args(ap)
    args = ap.parse_args()
    labels = list(RUNS) if args.runs == "all" else args.runs.split(",")
    out = TaskOutput("T2_5", args.outdir, prerun=args.prerun, force=args.force)

    acc, heads, share_rows, seed_rows = [], [], [], []
    gene_tables, decs = {}, {}
    for lab in labels:
        cohort, run = RUNS[lab]
        print("==", lab, run, flush=True)
        f = out.dir / "forward" / f"{lab}.npz"
        if args.reuse_forward and f.exists():
            fw = load_npz(f)
        else:
            fw = forward_sums(lab, cohort, run, args.threads, args.batch_size)
            save_npz(out, fw, f"{lab}.npz")
        co = C.load_cohort(cohort, run=run)
        genes = co.genes.reset_index()
        if list(genes["var_name"]) != [str(v) for v in fw["var_names"]]:
            raise AssertionError(f"{lab}: gene order differs between forward pass and h5ad")
        for key in [str(k) for k in fw["keys"]]:
            dec = MG.decompose(fw, key)
            t = MG.gene_table(fw, dec, genes)
            boot = MG.bootstrap(fw, dec, key, args.n_boot, SEED) if args.n_boot > 0 else None
            S_long, C_long = MG.long_tables(dec, t, key, boot)
            tag = "" if key == "cell_type" else f"__{key}"
            out.parquet(S_long, f"{lab}{tag}__S_gsk.parquet")
            out.parquet(C_long, f"{lab}{tag}__C_gk.parquet")
            out.parquet(t, f"{lab}{tag}__genes.parquet")
            acc += [dict(run_label=lab, run=run, **r) for r in MG.acceptance(dec, t, cohort, key)]
            acc.append(dict(run_label=lab, run=run, cohort=cohort, group_key=key,
                            check="median library ratio L_shared / L_observed",
                            value=float(np.median(fw["libratio"])), threshold=np.nan, passed=np.nan))
            heads.append(dict(run_label=lab, run=run, forward_source=str(fw["source"]), **MG.headline(t, cohort, key)))
            share_rows += [dict(run_label=lab, **r) for r in MG.shares(dec, t, cohort, key)]
            if key == "cell_type":
                gene_tables[lab] = t
                decs.setdefault(cohort, {})[lab] = dec
                if lab == MAIN_PANEL_RUN or (MAIN_PANEL_RUN not in labels and lab == labels[0]):
                    fig = fig_main(C_long, t, lab)
                    out.figure(fig, f"fig_main_celltype_margin__{lab}.pdf")
                    out.figure(fig, f"fig_main_celltype_margin__{lab}.png")
    for cohort, d in decs.items():
        if len(d) > 1:
            seed_rows += MG.seed_stability(d, cohort, "cell_type")
    seed_df = pd.DataFrame(seed_rows)
    for cohort in seed_df.cohort.unique() if len(seed_df) else []:
        s = seed_df[(seed_df.cohort == cohort) & seed_df.major]
        v = float(s.rho.min())
        acc.append(dict(run_label=cohort, run="seeds", cohort=cohort, group_key="cell_type",
                        check="min cross-seed Spearman of C_gk, major types", value=v,
                        threshold=MG.ACCEPT_SEED, passed=bool(v >= MG.ACCEPT_SEED)))
    acc = pd.DataFrame(acc)
    out.csv(acc, "acceptance.csv")
    out.csv(pd.DataFrame(heads), "headline_margin_support.csv")
    out.csv(pd.DataFrame(share_rows), "celltype_shares.csv")
    if len(seed_df):
        out.csv(seed_df, "seed_stability.csv")
        fig = fig_seed(seed_df)
        out.figure(fig, "fig_supp_seed_stability.pdf")
    if gene_tables:
        fig = fig_closure(gene_tables)
        out.figure(fig, "fig_supp_closure.pdf")
        out.figure(fig, "fig_supp_closure.png")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(acc.to_string())
    print(pd.DataFrame(heads).round(3).to_string())
    judged = acc[acc.passed.notna()]
    out.finish(dict(acceptance_all_passed=bool(judged.passed.astype(bool).all()), runs=labels,
                    n_boot=args.n_boot, z_sample_seed=SEED, margin_support_rule=MG.__doc__.split("Margin support")[1].strip()))


if __name__ == "__main__":
    main()
