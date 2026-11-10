"""Panels of the two gate figures of the Supplement that follow one training run, redrawn at print size.

    supp_alpha_training/drift.pdf            mean absolute change of α between consecutive stored epochs
    supp_alpha_training/tails.pdf            fraction of genes with Δα above +0.05 and below −0.05, per stored epoch
    supp_alpha_training/trajectories.pdf     α of 120 genes over training, the mean of all genes, the mean of the prior
    supp_alpha_training/distribution.pdf     density of α at five stored epochs
    supp_alpha_training/loss_terms.pdf       anchor term and fit-quality reward of the loss, mean per epoch
    supp_alpha_decorrelation/detection_rate.pdf   Δα of every gene against its detection rate
    supp_alpha_decorrelation/mean_count.pdf       Δα against log10 of the mean count
    supp_alpha_decorrelation/count_variance.pdf   Δα against log10 of the count variance

Inputs (liver Human+Monkey run): ``final_state.npz`` (α and prior), ``alpha_snapshots/alpha_epoch_*.npz`` (α and prior
every five epochs), ``lightning/metrics.csv`` (loss terms per step) and the counts of
``data/processed/camex_human_monkey.h5ad`` (mean, variance and detection rate of every gene over all cells).
Every panel has the same page size and the same axes box, so that panels in a row line up (``Panel(..., ink=False)``).

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.alpha [panel ...]
"""
from __future__ import annotations

import sys
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from figures.redraw import _common as C

COHORT = "liver_hmonkey"
TRAINING, DECORRELATION = "supp_alpha_training", "supp_alpha_decorrelation"
WIDTH = 1311 * 72 / 600                # bp (157.32): three panels and two 7 bp gaps fill the 486 bp text width
N_TRACES, TRACE_SEED = 120, 0          # genes drawn in the trajectory panel
TAIL = 0.05                            # |Δα| above which a gene counts as moved off the prior
STABILITY_FLOOR = 0.01
OFF_AXIS = 0.03                        # genes beyond the Δα axis are marked at this axes fraction from its edge
LABEL_WITH_SUBSCRIPT = 7.4             # pt: a subscript is 0.7 of its label and must stay above 5 pt


def minus(text: str) -> str:
    return text.replace("-", "−")


def frame(height: float, left: float, right: float, bottom: float, top: float):
    """A figure of WIDTH x height bp with one axes box at fixed margins (all in bp; a whole number of dots at 600 dpi)."""
    fd = C.design()
    mpl.rcParams.update({"savefig.bbox": None, "savefig.dpi": 600})   # exact page size; point clouds at 600 dpi
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    ax = fig.add_axes([left / WIDTH, bottom / height, 1 - (left + right) / WIDTH, 1 - (bottom + top) / height])
    return fd, fig, ax


def legend_above(ax, handles=None, ncol: int = 1, fontsize: float | None = None) -> None:
    """Legend in the band above the axes, flush right: never over data."""
    kw = dict(handles=handles) if handles is not None else {}
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False, fontsize=fontsize or C.SIZE["legend"],
              handlelength=1.3, handletextpad=0.4, columnspacing=0.9, labelspacing=0.25, borderpad=0, borderaxespad=0.45, **kw)


# ---------------------------------------------------------------------------------------------------------------
# Training dynamics (five panels)
# ---------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def snapshots() -> tuple:
    """Stored epochs, and α and the prior of every gene at each of them (epochs x genes)."""
    d = C.run_dir(COHORT)
    files = sorted((d / "alpha_snapshots").glob("alpha_epoch_*.npz"))
    if not files:
        raise SystemExit(f"no snapshots in {d / 'alpha_snapshots'}")
    snaps = [np.load(p) for p in files]
    epochs = np.array([int(s["epoch"]) for s in snaps])
    order = np.argsort(epochs)
    return epochs[order], np.stack([snaps[i]["alpha"] for i in order]), np.stack([snaps[i]["prior"] for i in order])


def drift_series() -> tuple:
    """Mean over genes of |α(t+1) − α(t)| for consecutive stored epochs, placed at the later epoch."""
    epochs, alpha, _ = snapshots()
    return epochs[1:], np.mean(np.abs(np.diff(alpha, axis=0)), axis=1)


def tail_series() -> tuple:
    epochs, alpha, prior = snapshots()
    delta = alpha - prior
    return epochs, (delta > TAIL).mean(axis=1), (delta < -TAIL).mean(axis=1)


def trajectory_series() -> tuple:
    """Stored epochs, α of the drawn genes (epochs x 120), the mean of all genes, and the mean of the final prior."""
    epochs, alpha, _ = snapshots()
    pick = np.random.default_rng(TRACE_SEED).choice(alpha.shape[1], size=N_TRACES, replace=False)
    prior = np.load(C.run_dir(COHORT) / "final_state.npz")["prior"]
    return epochs, alpha[:, pick], alpha.mean(axis=1), float(prior.mean()), alpha.shape[1]


def distribution_series() -> tuple:
    """Five evenly spaced stored epochs and α of every gene at each."""
    epochs, alpha, _ = snapshots()
    pick = np.linspace(0, len(epochs) - 1, 5).astype(int)
    return epochs[pick], alpha[pick]


def loss_series() -> pd.DataFrame:
    """Anchor term and fit-quality reward: mean of the logged steps of every epoch."""
    p = next((C.run_dir(COHORT) / "lightning").rglob("metrics.csv"), None)
    if p is None:
        raise SystemExit(f"no metrics.csv under {C.run_dir(COHORT) / 'lightning'}")
    return pd.read_csv(p).groupby("epoch")[["train/L_alpha_anchor", "train/L_fit_reward"]].mean()


def training_frame():
    return frame(height=150.0, left=39.0, right=9.0, bottom=25.0, top=24.0)


def epoch_axis(ax) -> None:
    ax.set_xlim(-6, 156)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel("epoch")


def drift() -> list:
    x, y = drift_series()
    _, fig, ax = training_frame()
    ax.plot(x, y, marker="o", markersize=2.0, linewidth=0.9, color=C.INK)
    ax.axhline(STABILITY_FLOOR, color="#888888", linewidth=0.6, linestyle="--", label=f"stability floor ({STABILITY_FLOOR:g})")
    epoch_axis(ax)
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_ylabel("mean |α(t+1) − α(t)|")
    legend_above(ax)
    print(f"drift: largest {y.max():.4f} at epoch {x[y.argmax()]}; last stored epoch at or above the floor: {x[y >= STABILITY_FLOOR].max()}")
    return [C.save_panel(fig, TRAINING, "drift", exact=True)]


def tails() -> list:
    x, up, down = tail_series()
    _, fig, ax = training_frame()
    ax.plot(x, up, color=C.INK, linewidth=1.0, marker="o", markersize=2.0, label=minus(f"Δα > +{TAIL:g}"))
    ax.plot(x, down, color="#888888", linewidth=1.0, marker="s", markersize=2.0, label=minus(f"Δα < -{TAIL:g}"))
    epoch_axis(ax)
    ax.set_ylim(0, max(up.max(), down.max()) * 1.15 + 1e-6)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylabel("fraction of genes")
    legend_above(ax, ncol=2)
    print(f"tails at the last stored epoch: above {up[-1]:.3f}, below {down[-1]:.3f}")
    return [C.save_panel(fig, TRAINING, "tails", exact=True)]


def trajectories() -> list:
    x, traces, mean, prior_mean, n_genes = trajectory_series()
    fd, fig, ax = training_frame()
    ax.plot(x, traces, color="#bbbbbb", linewidth=0.3, alpha=0.6)
    ax.plot(x, mean, color=C.INK, linewidth=1.1)
    ax.axhline(prior_mean, color=C.DOWN, linewidth=0.6, linestyle="--")
    epoch_axis(ax)
    fd.style_alpha_axis(ax, "y")
    ax.set_ylabel("α")
    legend_above(ax, ncol=2, handles=[
        Line2D([], [], color="#aaaaaa", linewidth=0.5, label=f"{traces.shape[1]} of {n_genes:,} genes"),
        Line2D([], [], color=C.INK, linewidth=1.1, label="mean of all genes"),
        Line2D([], [], color=C.DOWN, linewidth=0.6, linestyle="--", label="prior mean")])
    print(f"trajectories: mean α at the last stored epoch {mean[-1]:.3f}, prior mean {prior_mean:.3f}")
    return [C.save_panel(fig, TRAINING, "trajectories", exact=True)]


def distribution() -> list:
    epochs, alpha = distribution_series()
    fd, fig, ax = training_frame()
    grays = [str(g) for g in np.linspace(0.85, 0.05, len(epochs))]
    for e, a, g in zip(epochs, alpha, grays):
        ax.hist(a, bins=fd.ALPHA_BINS, density=True, histtype="step", linewidth=0.8, color=g)
    fd.style_alpha_axis(ax, "x")
    ax.set_xlabel("α")
    ax.set_ylabel("density")
    legend_above(ax, ncol=3, handles=[Line2D([], [], color=g, linewidth=1.0, label=f"epoch {int(e)}") for e, g in zip(epochs, grays)])
    return [C.save_panel(fig, TRAINING, "distribution", exact=True)]


def loss_terms() -> list:
    g = loss_series()
    _, fig, ax = training_frame()
    ax.plot(g.index, g["train/L_alpha_anchor"], color=C.INK, linewidth=1.0, label=r"$L_\mathrm{anchor}$ (anchor term)")
    ax.plot(g.index, g["train/L_fit_reward"], color="#888888", linewidth=1.0, label=r"$L_\mathrm{fit‐reward}$ (fit-quality reward)")
    epoch_axis(ax)
    ax.set_ylabel("loss component\n(per-step mean)", linespacing=1.05)
    legend_above(ax, fontsize=LABEL_WITH_SUBSCRIPT)
    print("loss terms at the last epoch:", g.iloc[-1].round(4).to_dict())
    return [C.save_panel(fig, TRAINING, "loss_terms", exact=True)]


# ---------------------------------------------------------------------------------------------------------------
# Δα against expression statistics (three panels)
# ---------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def expression_statistics() -> dict:
    """Per gene, over all cells of both species: detection rate, log10 mean count, log10 count variance; and Δα."""
    import anndata as ad
    from scipy import sparse

    path = C.DATA / C.ADATA[COHORT]
    C.need(path)
    final = np.load(C.run_dir(COHORT) / "final_state.npz")
    x = ad.read_h5ad(path).X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    if x.shape[1] != len(final["alpha"]):
        raise SystemExit(f"{path.name} has {x.shape[1]} genes, the run has {len(final['alpha'])}")
    mean = np.asarray(x.mean(axis=0)).ravel()
    variance = np.asarray(x.multiply(x).mean(axis=0)).ravel() - mean ** 2          # population variance
    return dict(delta=final["alpha"] - final["prior"],
                detection_rate=np.asarray((x > 0).sum(axis=0)).ravel() / float(x.shape[0]),
                mean_count=np.log10(np.clip(mean, 1e-4, None)),                    # floors as in the first version
                count_variance=np.log10(np.clip(variance, 1e-6, None)))


def cloud(name: str, xlabel: str, xlim: tuple | None = None) -> list:
    from scipy import stats

    s = expression_statistics()
    x, delta = s[name], s["delta"]
    fd, fig, ax = frame(height=140.0, left=37.0, right=8.0, bottom=27.0, top=5.0)
    fd.density_scatter(ax, x, delta, s=0.7, alpha=0.18)       # dot area in proportion to the panel, as in the first version
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=5)
    ax.set_xlim(*(xlim or (x.min(), x.max())))
    fd.style_delta_alpha_axis(ax, "y")
    lo, hi = fd.DELTA_ALPHA_RANGE
    for off, marker, edge in ((delta < lo, "v", OFF_AXIS), (delta > hi, "^", 1 - OFF_AXIS)):   # genes beyond the Δα axis: at its edge
        if off.any():
            ax.plot(x[off], np.full(int(off.sum()), edge), marker, transform=ax.get_xaxis_transform(), markersize=1.9,
                    markerfacecolor="none", markeredgecolor="#999999", markeredgewidth=0.3, linestyle="none", zorder=4, clip_on=False)
    ax.set_xlabel(xlabel, fontsize=LABEL_WITH_SUBSCRIPT)
    ax.set_ylabel("Δα", fontsize=LABEL_WITH_SUBSCRIPT)
    rho = stats.spearmanr(x, delta).correlation
    ax.text(0.985, 0.985, minus(f"Spearman ρ = {rho:.3f}"), transform=ax.transAxes, ha="right", va="top", fontsize=C.SIZE["tick"],
            color=C.INK, zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    print(f"{name}: Spearman {rho:.3f}; {len(x):,} genes, {int((delta < lo).sum())} below {lo}, {int((delta > hi).sum())} above {hi}")
    return [C.save_panel(fig, DECORRELATION, name, exact=True)]


def detection_rate() -> list:
    return cloud("detection_rate", "detection rate (fraction of cells)", xlim=(0, 1))


def mean_count() -> list:
    return cloud("mean_count", r"$\log_{10}$(mean count)")


def count_variance() -> list:
    return cloud("count_variance", r"$\log_{10}$(count variance)")


PANELS = {"drift": drift, "tails": tails, "trajectories": trajectories, "distribution": distribution, "loss_terms": loss_terms,
          "detection_rate": detection_rate, "mean_count": mean_count, "count_variance": count_variance}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
