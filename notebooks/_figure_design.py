"""Visual identity for Chimera paper notebooks.

Single source of truth for figure aesthetics. Every paper notebook should
import this module and apply ``apply_design()`` once before any plotting.
The contract this module enforces:

  * No grids. No top/right spines. No titles on axes (titles live in captions).
  * One panel per figure (this module *makes* one-panel figures; multi-panel
    layout is the LaTeX layout's job, never the notebook's).
  * Grayscale by default. Color is opt-in for genuinely categorical axes
    (``SPECIES_PALETTE``, ``TISSUE_PALETTE``, ``SEED_MARKERS``).
  * Vector output for every figure via ``save_figure``. PDF preferred, with
    fonts embedded as text (``pdf.fonttype = 42``).
  * Print-legible at single-column width (~3.3 inches).
  * Single typeface for every label, tick, legend entry, annotation.

Cross-figure consistency contract — every paper-notebook figure family that
uses one of the following dimensions MUST pull constants from this module
rather than redefining them locally::

  ALPHA_RANGE, ALPHA_TICKS                  α axis on every histogram/scatter
  ALPHA_BINS                                identical bin edges across α-evolution panels
  DELTA_ALPHA_RANGE, DELTA_ALPHA_TICKS      symmetric Δα axis everywhere
  DELTA_ALPHA_BINS                          identical bin edges across Δα panels
  CORRECTION_RANGE                          GIN correction axis (logit space)
  CORRECTION_BINS                           |correction| histogram bin edges
  TRAINING_AXIS_LABEL                       always 'epoch'
  SPECIES_PALETTE                           one color per species, project-wide
  TISSUE_LINESTYLES                         one linestyle per tissue
  SEED_MARKERS                              one marker per training seed
  LABEL                                     canonical text for every concept
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Typeface and rcParams
# ---------------------------------------------------------------------------

# Single typeface across every paper notebook. Helvetica/Arial via DejaVu Sans
# fallback — ships with matplotlib, no font-install step required, and renders
# legibly at print sizes.
_BASE_FONT = "DejaVu Sans"


def apply_design() -> None:
    """Apply the paper visual identity to matplotlib's global rcParams.

    Idempotent: safe to call from every notebook cell that wants to be defensive.
    Must be called before the first ``plt.subplots``.
    """
    mpl.rcParams.update({
        # Figure / save
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        # Vector / journal-grade font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # Single typeface
        "font.family": _BASE_FONT,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,        # titles are off by default — see below
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 8,

        # Spines: left + bottom only
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",

        # Ticks: outward, modest density, thin
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",

        # No grids. Anywhere. Ever.
        "axes.grid": False,
        "axes.grid.which": "major",
        "grid.linewidth": 0.0,

        # Legend: no frame, no shadow
        "legend.frameon": False,
        "legend.borderpad": 0.2,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,

        # Lines / markers — thin / small by default
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.4,

        # Histogram bars — no edge by default; we'll add edge per-call when needed
        "patch.edgecolor": "#222222",
    })


# ---------------------------------------------------------------------------
# Figure-size constants (single-column / two-column print widths)
# ---------------------------------------------------------------------------

# Single-column journal width is ~3.3 inches; two-column is ~6.8 inches.
# Square panel by default; tall variant for hist-evolution; wide for trajectories.
SINGLE_W = 3.3
DOUBLE_W = 6.8

FIGSIZE_SQUARE = (3.3, 2.7)        # default panel (α/Δα hist, scatter)
FIGSIZE_TRAJECTORY = (3.6, 2.4)    # x=epoch trajectories
FIGSIZE_TALL = (3.0, 2.8)          # for hist evolutions where vertical space helps
FIGSIZE_WIDE = (4.6, 2.6)          # for log-y mass plots when horizontal range matters


# ---------------------------------------------------------------------------
# Cross-figure axis / bin contract
# ---------------------------------------------------------------------------

# α ∈ [0, 1] — canonical axis ticks across every α plot in the notebook.
ALPHA_RANGE = (0.0, 1.0)
ALPHA_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_BINS = np.linspace(0.0, 1.0, 51)         # 50 equal bins; reuse across α-evolution

# Δα — symmetric range covering observed empirical mass (typical ±0.25 incl tails).
# Pinned wide enough to fit testis tails; identical bins across all Δα panels.
DELTA_ALPHA_RANGE = (-0.4, 0.4)
DELTA_ALPHA_TICKS = (-0.4, -0.2, 0.0, 0.2, 0.4)
DELTA_ALPHA_BINS = np.linspace(-0.4, 0.4, 81)  # 80 equal bins, 0.01 wide

# Correction (logit-space output of GIN). Wider range — observed up to ~±13.
CORRECTION_RANGE = (-4.0, 4.0)
CORRECTION_TICKS = (-4.0, -2.0, 0.0, 2.0, 4.0)
CORRECTION_BINS = np.linspace(-4.0, 4.0, 81)
# |correction| histogram (one-sided), with semilog-y; thresholds at 0.01 and 0.10.
ABS_CORRECTION_BINS = np.linspace(0.0, 2.0, 81)
ABS_CORRECTION_THRESH_LOW = 0.01    # E3.5 floor (workspace docs)
ABS_CORRECTION_THRESH_HIGH = 0.10   # caveat-28 warning band

# Training-axis convention: ALWAYS epoch.
TRAINING_AXIS_LABEL = "epoch"


# ---------------------------------------------------------------------------
# Categorical encodings
# ---------------------------------------------------------------------------

# Species color map. Used wherever a species is encoded categorically. Desaturated,
# colorblind-aware (Okabe-Ito-derived). NOTE: pancreas H+Mo dataset labels its
# Old-World-primate as "Monkey"; testis H+Mac dataset labels it as "Macaque".
# Both colorize identically — treat them as the same biological organism for
# cross-tissue comparability. Document this in any cross-tissue figure caption.
SPECIES_PALETTE: dict[str, str] = {
    "Human":     "#222222",   # near-black (always present)
    "Monkey":    "#0072B2",   # blue (Old-World primate, pancreas naming)
    "Macaque":   "#0072B2",   # blue (same organism, testis naming)
    "Mouse":     "#D55E00",   # vermilion
    "Zebrafish": "#009E73",   # green
}


def species_color(name: str) -> str:
    """Look up a species color, falling back to mid-gray."""
    return SPECIES_PALETTE.get(name, "#666666")


# Tissue encoding — solid for pancreas (the default), dashed for testis. Used only
# in cross-tissue panels.
TISSUE_LINESTYLES: dict[str, str] = {
    "pancreas": "-",
    "testis": "--",
}


def tissue_linestyle(name: str) -> str:
    return TISSUE_LINESTYLES.get(name, "-")


# Seed encoding — one marker per seed. Used in every cross-seed scatter.
SEED_MARKERS: dict[int, str] = {
    42: "o",
    43: "s",
    44: "^",
}

# Grayscale shades for seeds when shape isn't enough (e.g. cross-seed line plots).
SEED_SHADES: dict[int, str] = {
    42: "#222222",
    43: "#666666",
    44: "#aaaaaa",
}

# Canonical species sort order: Human first, then by evolutionary distance from
# Human (close primate -> mouse -> zebrafish). Used wherever a categorical axis
# carries species names (legends, side-by-side panels, cross-species sort orders).
SPECIES_EVOLUTIONARY_ORDER: tuple[str, ...] = (
    "Human", "Monkey", "Macaque", "Mouse", "Zebrafish",
)


def species_sort_key(name: str) -> int:
    """Index of a species in the canonical evolutionary-order list (unknown -> 99)."""
    try:
        return SPECIES_EVOLUTIONARY_ORDER.index(name)
    except ValueError:
        return 99


# ---------------------------------------------------------------------------
# Canonical label vocabulary — pick one phrasing per concept, use everywhere
# ---------------------------------------------------------------------------

LABEL = {
    "alpha":              r"$\alpha$",
    "alpha_long":         r"per-gene $\alpha$",
    "alpha_mean":         r"mean $\alpha$",
    "alpha_std":          r"std $\alpha$",
    "delta_alpha":        r"$\Delta\alpha$",
    "delta_alpha_long":   r"$\alpha - $ prior",
    "prior":              "conservation prior",
    "correction":         "GIN correction (logit space)",
    "abs_correction":     r"$|$GIN correction$|$",
    "S_g":                r"$S_g$",
    "S_g_ema":            r"$S_g$ EMA",
    "epoch":              "epoch",
    "n_genes":            "number of genes",
    "density":            "density",
    "log_mean_expr":      r"$\log_{10}($ mean count $)$",
    "detection_rate":     "detection rate (frac. cells $> 0$)",
    "log_variance":       r"$\log_{10}($ variance $)$",
    "residual_rms":       r"residual RMS",
    "knn_sp":             "kNN species purity",
    "lt_acc":             "label-transfer accuracy",
    "spearman":           r"Spearman $\rho$",
    "frac_pushed_up":     r"frac $\Delta\alpha > +0.05$",
    "frac_pushed_down":   r"frac $\Delta\alpha < -0.05$",
    "frac_sat_high":      r"frac $\alpha > 0.95$",
    "frac_sat_low":       r"frac $\alpha < 0.05$",
}


# ---------------------------------------------------------------------------
# Helpers that ENFORCE the design system on every figure
# ---------------------------------------------------------------------------

def make_axes(figsize: tuple[float, float] = FIGSIZE_SQUARE) -> tuple[plt.Figure, plt.Axes]:
    """Create a single-panel figure with the design system applied.

    Always returns (fig, ax) — never multi-panel. If a section needs multiple
    panels, call this once per panel and save each separately.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    # Re-assert spine + grid invariants in case rcParams have drifted.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)
    return fig, ax


def style_alpha_axis(ax: plt.Axes, which: str = "x") -> None:
    """Pin α axis range and ticks. Apply on every plot whose x or y is α."""
    setter_lim = ax.set_xlim if which == "x" else ax.set_ylim
    setter_ticks = ax.set_xticks if which == "x" else ax.set_yticks
    setter_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    setter_lim(*ALPHA_RANGE)
    setter_ticks(list(ALPHA_TICKS))
    setter_label(LABEL["alpha"])


def style_delta_alpha_axis(ax: plt.Axes, which: str = "x") -> None:
    """Pin Δα axis range and ticks. Apply on every plot whose x or y is Δα."""
    setter_lim = ax.set_xlim if which == "x" else ax.set_ylim
    setter_ticks = ax.set_xticks if which == "x" else ax.set_yticks
    setter_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    setter_lim(*DELTA_ALPHA_RANGE)
    setter_ticks(list(DELTA_ALPHA_TICKS))
    setter_label(LABEL["delta_alpha"])


def thin_legend(ax: plt.Axes, loc: str = "best", **kw) -> None:
    """A consistent, frameless legend with our visual weight."""
    leg = ax.legend(loc=loc, frameon=False, fontsize=7, handlelength=1.2,
                    handletextpad=0.4, borderpad=0.2, **kw)
    return leg


def density_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    s: float = 1.5,
    alpha: float = 0.18,
    color: str = "#222222",
) -> None:
    """The canonical per-gene scatter for 5K-15K-point panels.

    Hexbins were tried first and rejected on legibility — small dots with
    low alpha preserve mode structure that hex aggregation washes out, while
    `rasterized=True` keeps PDF file size sane (axes / text remain vector).

    All gene-level scatter panels in the paper notebooks should call this
    helper, not raw ``ax.scatter``, so the marker size / alpha contract is
    enforced once.
    """
    ax.scatter(np.asarray(x), np.asarray(y), s=s, alpha=alpha, color=color,
               linewidths=0, rasterized=True)


def add_identity_line(
    ax: plt.Axes,
    *,
    lim: tuple[float, float] | None = None,
    color: str = "#888888",
    linewidth: float = 0.6,
    linestyle: str = "--",
    zorder: float = 5,
) -> None:
    """Draw an identity line on a scatter panel. Defaults to current x-limits."""
    if lim is None:
        lim = ax.get_xlim()
    ax.plot(lim, lim, color=color, linewidth=linewidth, linestyle=linestyle,
            zorder=zorder)


def annotate_stat(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    """Stat-annotation in the corner of an axis (Spearman ρ, n, etc.).

    Uses a consistent style — no box, mid-gray, fixed offsets — so every panel's
    annotation sits in the same visual register.
    """
    locs = {
        "upper left":  (0.03, 0.97, "left",  "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left":  (0.03, 0.03, "left",  "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=7, color="#222222")


# ---------------------------------------------------------------------------
# save_figure: the single chokepoint for vector output
# ---------------------------------------------------------------------------

# Default output directory — every figure in a notebook goes through here so
# files don't drift across cells.
DEFAULT_FIG_DIR = Path(__file__).resolve().parent / "figures" / "alpha_master"


def set_figure_dir(path: Path | str) -> Path:
    """Override the default figure output directory for the current notebook."""
    global DEFAULT_FIG_DIR
    DEFAULT_FIG_DIR = Path(path)
    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_FIG_DIR


def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    out_dir: Path | None = None,
    close: bool = False,
) -> list[Path]:
    """Save a figure under a stable, design-system-enforced path.

    PDF is the lift-and-drop format for LaTeX (vector, fonts embedded as text).
    PNG is included for in-notebook preview and quick visual diff. Names are
    treated as filenames without extension; whitespace becomes ``_``; only
    [a-z0-9_-] survives.
    """
    out = (out_dir or DEFAULT_FIG_DIR)
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name.strip().lower())
    written = []
    for fmt in formats:
        p = out / f"{safe}.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight", pad_inches=0.02)
        written.append(p)
    if close:
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Self-test (run with ``python _figure_design.py`` to sanity-check the module)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_design()
    rng = np.random.default_rng(0)
    fig, ax = make_axes()
    ax.hist(rng.beta(8, 2, size=2000), bins=ALPHA_BINS, color="#444444",
            edgecolor="none")
    style_alpha_axis(ax, "x")
    ax.set_ylabel(LABEL["n_genes"])
    annotate_stat(ax, r"self-test: $n=2000$", "upper left")
    paths = save_figure(fig, "self_test", out_dir=Path("/tmp/chimera_design_test"))
    print("wrote:", paths)
