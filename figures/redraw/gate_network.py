"""Panels of the five supplementary figures on what the gate network reads, redrawn at print size.

The gate network (Linear 72-64, tanh, Linear 64-32, tanh, Linear 32-1) turns 72 inputs per gene into one correction
of the gene's prior. The trained network of the four-species liver run is loaded, and the linear baselines and the
four attributions (weight path, gradient x input, integrated gradients, ablation of an input group) are computed
as in sections A, B, C.1, C.2 and F of ``notebooks/_gin_attention_overall_builder.py``. Nothing is trained.

    supp_gin_linear_baseline/      r2_by_input_set, spearman_single_inputs
    supp_gin_attribution_suite/    ablation_change_by_group, gradient_input_by_group, gradient_input_signed_single,
                                   ablation_correlation_by_group, weight_path_by_group, weight_path_top20,
                                   ig_completeness, ig_by_group
    supp_gin_method_consensus/     consensus_bars, consensus_heatmap
    supp_gin_orthogonality/        r2_components_from_expression, delta_alpha_inputs_zeroed
    supp_gin_per_pc/               ig_by_component, spearman_by_component, component_rank1, component_rank2, component_rank3

``ig_completeness`` states a bound on the largest completeness error of integrated gradients (below 1e-5). The error
itself is a rounding-level quantity of single-precision arithmetic (about 3e-06) whose last digits depend on the
machine, so the panel prints the bound, which the script checks, and not the digits.

Inputs (four-species liver run): ``model/model.pt`` (weights of the gate network), ``final_state.npz`` (correction,
prior, alpha), and the 72 standardized inputs per gene in ``varm["gene_features"]`` of the data object.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.gate_network [group ...]
"""
from __future__ import annotations

import functools
import sys
from types import SimpleNamespace

import numpy as np

from figures.redraw import _common as C

COHORT = "liver_4sp"
# The eight single-number inputs, in the order of input columns 64 to 71; columns 0 to 63 are the ESM-2 components.
SINGLE = ["esm2_intersp_var", "mean_identity", "log_mean_expr", "log_expr_var", "frac_nonzero", "std_identity", "max_identity",
          "min_identity"]
INPUT = [f"PC{i:02d}" for i in range(64)] + SINGLE
GROUP = {"ESM-2 PCs": np.arange(0, 64), "esm2_intersp_var": np.array([64]), "mean_identity": np.array([65]),
         "expression": np.array([66, 67, 68]), "identity_spread": np.array([69, 70, 71])}
METHOD = ("weight path", "gradient × input", "integrated gradients", "ablation")
METHOD_COLOR = ("#CCCCCC", "#56B4E9", "#0072B2", "#D55E00")      # as in the first version of the figure
ZEROED_COLOR = {"ESM-2 PCs": "#0072B2", "expression": "#D55E00"}   # Δα with this input group set to zero, as in the first version
THRESHOLD_COLOR = "#D55E00"    # the line at R2 = 0.05
INPUT_SET = {"expression-only": "expression only", "scalars-only": "8 single-number inputs", "ESM-2-only": "ESM-2 components only",
             "ESM-2 + expression": "ESM-2 components + expression", "all-72": "all 72 inputs"}
IG_STEPS = 200                 # integrated gradients: baseline 0 (the mean of the standardized inputs), trapezoidal rule
TOP_COMPONENTS = (13, 5, 53)   # the three components with the largest integrated-gradients importance (checked)
LEFT = 112.0                   # bp, room for the longest input name left of a bar chart
BOTTOM, TOP, HEAD = 24.0, 3.5, 12.5   # bp, margins under and over the axes (HEAD: with a heading)
COL_L, COL_R = 247.0, 232.0    # bp, the two columns of the attribution figure (486 = 247 + 7 + 232)


# ---------------------------------------------------------------------------------------------------------------------
# Stored model and the quantities derived from it
# ---------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def gate() -> SimpleNamespace:
    """Inputs, stored outputs and the trained gate network of the four-species liver run."""
    import anndata as ad
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    run = C.run_dir(COHORT)
    data = C.DATA / C.ADATA[COHORT]
    C.need(run / "model" / "model.pt", data)
    a = ad.read_h5ad(data, backed="r")
    x = np.asarray(a.varm["gene_features"], dtype=np.float32)
    a.file.close()
    final = np.load(run / "final_state.npz", allow_pickle=False)
    correction, prior, alpha = (final[k].astype(np.float32) for k in ("correction", "prior", "alpha"))
    state = torch.load(run / "model" / "model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    w = {k.removeprefix("gin.correction_mlp."): v for k, v in state.items() if k.startswith("gin.correction_mlp.")}
    l0, l1, l2 = torch.nn.Linear(72, 64), torch.nn.Linear(64, 32), torch.nn.Linear(32, 1)
    with torch.no_grad():
        for layer, k in ((l0, "0"), (l1, "2"), (l2, "4")):
            layer.weight.copy_(w[f"{k}.weight"])
            layer.bias.copy_(w[f"{k}.bias"])
    for layer in (l0, l1, l2):
        for p in layer.parameters():
            p.requires_grad_(False)

    def net(t):
        return l2(torch.tanh(l1(torch.tanh(l0(t))))).squeeze(-1)

    with torch.no_grad():
        err = float(np.max(np.abs(net(torch.from_numpy(x)).numpy() - correction)))
    if x.shape != (len(correction), 72) or err > 1e-3:
        raise SystemExit(f"the stored network does not reproduce the stored correction (largest difference {err:.2e})")
    return SimpleNamespace(x=x, correction=correction, prior=prior, delta_alpha=(alpha - prior).astype(np.float32), net=net,
                           weights=[layer.weight.detach().abs().numpy() for layer in (l0, l1, l2)])


def _by_group(per_input: np.ndarray) -> dict:
    return {g: float(per_input[idx].sum()) for g, idx in GROUP.items()}


def _percent(d: dict) -> dict:
    s = sum(d.values()) or 1.0
    return {k: 100 * v / s for k, v in d.items()}


@functools.lru_cache(maxsize=1)
def attributions() -> SimpleNamespace:
    """The four attributions of the correction to its 72 inputs, per input and summed within the five input groups."""
    import torch

    g = gate()
    xt = torch.from_numpy(g.x)
    w0, w1, w2 = g.weights
    weight_path = (w2 @ w1 @ w0).ravel()                       # |W2| |W1| |W0|, one value per input

    def gradient(t):
        t = t.clone().detach().requires_grad_(True)
        g.net(t).sum().backward()
        return t.grad.detach()

    gxi = gradient(xt).numpy() * g.x                           # gradient x input, genes x inputs
    acc, wsum = torch.zeros_like(xt), 0.0
    for m in range(IG_STEPS + 1):                              # integrated gradients from the baseline 0
        wm = 0.5 if m in (0, IG_STEPS) else 1.0
        acc += wm * gradient((m / IG_STEPS) * xt)
        wsum += wm
    ig = g.x * (acc / wsum).numpy()
    with torch.no_grad():
        change = g.net(xt).numpy() - g.net(torch.zeros_like(xt)).numpy()
        ablated = {}
        for name, idx in GROUP.items():                        # one input group set to zero
            xa = g.x.copy()
            xa[:, idx] = 0.0
            ablated[name] = g.net(torch.from_numpy(xa)).numpy()
    ig_sum = ig.sum(axis=1)
    by_group = {"weight path": _by_group(weight_path), "gradient × input": _by_group(np.abs(gxi).mean(axis=0)),
                "integrated gradients": _by_group(np.abs(ig).mean(axis=0)),
                "ablation": {k: float(np.mean(np.abs(v - g.correction))) for k, v in ablated.items()}}
    return SimpleNamespace(
        weight_path=weight_path, gxi_signed=gxi.mean(axis=0), ig_abs=np.abs(ig).mean(axis=0), ig_sum=ig_sum, change=change,
        completeness_error=float(np.max(np.abs(ig_sum - change))), ablated=ablated, by_group=by_group,
        ablation_r={k: float(np.corrcoef(v, g.correction)[0, 1]) for k, v in ablated.items()},
        percent={m: _percent(by_group[m]) for m in METHOD})


def linear_r2() -> dict:
    """In-sample R2 of a linear regression of the correction on five sets of inputs."""
    from sklearn.linear_model import LinearRegression

    g = gate()
    sets = {"expression-only": g.x[:, GROUP["expression"]], "scalars-only": g.x[:, 64:72], "ESM-2-only": g.x[:, :64],
            "ESM-2 + expression": g.x[:, list(range(64)) + [66, 67, 68]], "all-72": g.x}
    return {k: float(LinearRegression().fit(v, g.correction).score(v, g.correction)) for k, v in sets.items()}


def spearman_single() -> dict:
    """Spearman correlation of each single-number input with the correction."""
    from scipy import stats

    g = gate()
    return {name: float(stats.spearmanr(g.x[:, i], g.correction).correlation) for i, name in zip(range(64, 72), SINGLE)}


def component_r2_from_expression() -> np.ndarray:
    """R2 of each ESM-2 component in a linear regression on the three expression inputs."""
    from sklearn.linear_model import LinearRegression

    g = gate()
    e = g.x[:, GROUP["expression"]]
    return np.array([LinearRegression().fit(e, g.x[:, j]).score(e, g.x[:, j]) for j in range(64)])


def delta_alpha_ablated(group: str) -> np.ndarray:
    """Δα when one input group is set to zero: alpha = sigmoid(logit(prior) + ablated correction)."""
    g = gate()
    logit = np.log(np.clip(g.prior, 1e-6, 1 - 1e-6) / np.clip(1 - g.prior, 1e-6, 1 - 1e-6))
    return 1.0 / (1.0 + np.exp(-(logit + attributions().ablated[group]))) - g.prior


def component_spearman() -> np.ndarray:
    """Spearman correlation of each ESM-2 component with Δα."""
    from scipy import stats

    g = gate()
    return np.array([stats.spearmanr(g.x[:, j], g.delta_alpha).correlation for j in range(64)])


# ---------------------------------------------------------------------------------------------------------------------
# Drawing helpers: every panel is a page of exact size with its axes at fixed margins, so the axes of a row line up
# ---------------------------------------------------------------------------------------------------------------------
def _style():
    import matplotlib as mpl

    fd = C.design()
    mpl.rcParams.update({"savefig.bbox": "standard", "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
                         "axes.titlepad": 3.0, "axes.titlelocation": "left", "axes.titlecolor": C.INK})
    return fd


def _page(width: float, height: float, left: float, right: float = 3.0, top: float = TOP, bottom: float = BOTTOM):
    """A page of width x height bp with one axes at the given margins (bp)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 72, height / 72), dpi=600)   # fine grid: text is measured as it is set in the PDF
    ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, (height - top - bottom) / height])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.page = SimpleNamespace(width=width, height=height, axes_width=width - left - right, axes_height=height - top - bottom)
    return fig, ax


def _text_width(text: str, size: float) -> float:
    """Width in bp of the widest line of a label at a font size."""
    from matplotlib.textpath import TextPath

    return max(TextPath((0, 0), line, size=size).get_extents().width for line in text.split("\n"))


def _xlabel(ax, text: str) -> None:
    """x label centered under the axes, or ending at its right edge when it is wider than the axes."""
    wide = _text_width(text, C.SIZE["label"]) > ax.page.axes_width
    ax.set_xlabel(text, loc="right" if wide else "center")


def _save(fig, family: str, name: str):
    """Save at the exact page size after checking that nothing reaches the page edge."""
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    edge = 0.8 / 72
    if box.x0 < edge or box.y0 < edge or box.x1 > w - edge or box.y1 > h - edge:
        raise SystemExit(f"{family}/{name}: content leaves the page ({box.x0 * 72:.1f}, {box.y0 * 72:.1f}, "
                         f"{(w - box.x1) * 72:.1f}, {(h - box.y1) * 72:.1f} bp left at left, bottom, right, top)")
    return C.save_panel(fig, family, name, exact=True)


def _signed(v: float, digits: int = 3) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def _barh(ax, values, labels, label_size: float | None = None, thickness: float = 0.72):
    """Horizontal gray bars, first value at the bottom."""
    y = np.arange(len(values))
    pitch = ax.page.axes_height / len(values)
    ax.barh(y, values, height=min(thickness, 10.5 / pitch), color=C.GREY_BAR, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_size or C.SIZE["tick"])
    ax.set_ylim(-0.5, len(values) - 0.5)
    ax.tick_params(axis="y", length=0)
    return y


def _group_bars(values: dict, xlabel: str, heading: str, width: float, height: float, name: str):
    """One bar per input group (top to bottom in the order of the inputs), with its share of the total at the bar."""
    fig, ax = _page(width, height, LEFT, top=HEAD)
    names = list(GROUP)[::-1]
    v = [values[g] for g in names]
    total = sum(v) or 1.0
    y = _barh(ax, v, [C.GATE_GROUP[g] for g in names])
    room = _text_width("88.8%", C.SIZE["text"]) + 4.0
    ax.set_xlim(0, max(v) * ax.page.axes_width / (ax.page.axes_width - room))
    for yi, vi in zip(y, v):
        ax.annotate(f"{100 * vi / total:.1f}%", (vi, yi), xytext=(2.5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=C.SIZE["text"], color=C.INK)
    _xlabel(ax, xlabel)
    ax.set_title(heading)
    return _save(fig, "supp_gin_attribution_suite", name)


# ---------------------------------------------------------------------------------------------------------------------
# Linear models of the correction and correlations with the single-number inputs
# ---------------------------------------------------------------------------------------------------------------------
def linear_baseline() -> list:
    _style()
    family, height = "supp_gin_linear_baseline", 8 * 11.0 + BOTTOM + TOP
    r2 = linear_r2()
    left = _text_width(max(INPUT_SET.values(), key=len), C.SIZE["tick"]) + 6.5
    fig, ax = _page(266, height, left)
    names = list(r2)[::-1]
    v = [r2[k] for k in names]
    y = _barh(ax, v, [INPUT_SET[k] for k in names], thickness=0.78)
    for yi, vi in zip(y, v):
        ax.annotate(f"{vi:.3f}", (vi, yi), xytext=(2.5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=C.SIZE["text"], color=C.INK)
    ax.set_xlim(0, max(v) * 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    _xlabel(ax, r"$R^2$ of a linear regression of the correction")
    out = [_save(fig, family, "r2_by_input_set")]

    rho = spearman_single()
    order = sorted(rho, key=lambda k: abs(rho[k]))             # smallest |rho| at the bottom
    fig, ax = _page(213, height, LEFT)
    _barh(ax, [rho[k] for k in order], [C.GATE_INPUT[k] for k in order])
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_xlim(-0.8, 0.3)
    ax.set_xticks([-0.8, -0.4, 0.0])
    _xlabel(ax, "Spearman ρ with the correction")
    out.append(_save(fig, family, "spearman_single_inputs"))
    print("R2:", {k: round(x, 3) for k, x in r2.items()})
    print("Spearman, single-number inputs:", {k: round(x, 3) for k, x in rho.items()})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Four attribution methods
# ---------------------------------------------------------------------------------------------------------------------
# Heights in bp. Rows (a, b) and (c, d); then (e) over (g) at the left and (f) over (h) at the right, both columns equally
# high. "completeness" is the height of panel (g), the completeness check, at the width of the left column.
ROW = dict(groups=5 * 11.0 + BOTTOM + HEAD, single=8 * 9.0 + BOTTOM + HEAD, top20=20 * 6.6 + BOTTOM + HEAD, completeness=171.3)
ROW["last"] = ROW["groups"] + ROW["completeness"] - ROW["top20"]


COMPLETENESS_BOUND = 1e-5   # printed on the completeness panel; the measured error is about 3e-06


def attribution_suite() -> list:
    fd = _style()
    family = "supp_gin_attribution_suite"
    g, a = gate(), attributions()
    paths = "$|W_2|\\,|W_1|\\,|W_0|$"
    out = [
        _group_bars(a.by_group["ablation"], "mean |Δ correction| over genes", "ablation (group set to zero)", COL_L, ROW["groups"],
                    "ablation_change_by_group"),
        _group_bars(a.by_group["gradient × input"], "mean $|x\\cdot\\nabla_x f|$ over genes, group sum", "gradient × input",
                    COL_R, ROW["groups"], "gradient_input_by_group"),
        _group_bars(a.by_group["weight path"], f"{paths}, group sum", "weight path", COL_L, ROW["groups"], "weight_path_by_group"),
        _group_bars(a.by_group["integrated gradients"], "mean |IG| over genes, group sum", "integrated gradients", COL_R,
                    ROW["last"], "ig_by_group"),
    ]

    # signed gradient x input of the eight single-number inputs, smallest |value| at the bottom
    signed = a.gxi_signed[64:72]
    order = np.argsort(np.abs(signed))
    fig, ax = _page(COL_L, ROW["single"], LEFT, top=HEAD)
    _barh(ax, signed[order], [C.GATE_INPUT[SINGLE[i]] for i in order])
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_xticks([0, 0.01, 0.02])
    _xlabel(ax, "mean $x\\cdot\\nabla_x f$ over genes (signed)")
    ax.set_title("gradient × input")
    out.append(_save(fig, family, "gradient_input_signed_single"))

    # correlation of the correction with one group set to zero and the full correction
    names = list(GROUP)[::-1]
    r = [a.ablation_r[k] for k in names]
    fig, ax = _page(COL_R, ROW["single"], LEFT, top=HEAD)
    y = _barh(ax, r, [C.GATE_GROUP[k] for k in names])
    lo = min(r) - 0.05                                         # the axis starts here, not at 0 (marked on the axis)
    label = _text_width("r = +0.000", C.SIZE["text"]) + 4.0
    ax.set_xlim(lo, lo + (1.0 - lo) * ax.page.axes_width / (ax.page.axes_width - label))
    ax.axvline(1.0, color="#888888", linewidth=0.6, linestyle="--")
    for yi, ri in zip(y, r):                                   # values in a column right of the line at 1, clear of the bars
        ax.annotate(f"r = {_signed(ri)}", (1.0, yi), xytext=(3.0, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=C.SIZE["text"], color=C.INK)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["bottom"].set_bounds(lo, 1.0)
    for dx in (4.5, 7.5):                                      # two strokes across the axis: it does not start at zero
        ax.annotate("", xy=(dx + 1.4, 3.0), xytext=(dx - 1.4, -3.0), xycoords="axes points", textcoords="axes points",
                    arrowprops=dict(arrowstyle="-", color=C.INK, linewidth=0.6, shrinkA=0, shrinkB=0), annotation_clip=False)
    _xlabel(ax, "Pearson r with the full correction")
    ax.set_title("ablation (group set to zero)")
    out.append(_save(fig, family, "ablation_correlation_by_group"))

    # the 20 inputs with the largest weight path, largest at the top
    order = np.argsort(a.weight_path)[::-1][:20][::-1]
    fig, ax = _page(COL_R, ROW["top20"], LEFT, top=HEAD)
    _barh(ax, a.weight_path[order], [C.gate_input(INPUT[i]) for i in order], label_size=C.SIZE["text"])
    _xlabel(ax, f"{paths} of the input (20 largest)")
    ax.set_title("weight path")
    out.append(_save(fig, family, "weight_path_top20"))

    # completeness of integrated gradients: the attributions of a gene add up to the change of the output
    fig, ax = _page(COL_L, ROW["completeness"], 34, top=HEAD, bottom=29.5)   # room for the sum sign
    fd.density_scatter(ax, a.ig_sum, a.change, s=1.5)
    lim = (min(a.ig_sum.min(), a.change.min()), max(a.ig_sum.max(), a.change.max()))
    fd.add_identity_line(ax, lim=lim)
    ax.set_xlabel("$\\sum_i \\mathrm{IG}_i(\\mathbf{x})$, sum over the 72 inputs")
    ax.set_ylabel("$f(\\mathbf{x}) - f(\\mathbf{0})$, change of the correction")
    if not a.completeness_error < COMPLETENESS_BOUND:
        raise SystemExit(f"integrated gradients: max |error| = {a.completeness_error:.2e} is not below {COMPLETENESS_BOUND:g}")
    fd.annotate_stat(ax, "max |error| < 1e-5", "lower right")
    ax.texts[-1].set_fontsize(C.SIZE["text"])
    ax.set_title("integrated gradients")
    out.append(_save(fig, family, "ig_completeness"))
    for m in METHOD:
        print(f"{m:>22}:", "  ".join(f"{C.GATE_GROUP[k]} {v:.1f}%" for k, v in a.percent[m].items()))
    print("Pearson r after ablation:", {k: round(v, 3) for k, v in a.ablation_r.items()})
    print(f"integrated gradients, max |error| = {a.completeness_error:.2e}; correction reproduced, n = {len(g.correction):,} genes")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The four methods side by side
# ---------------------------------------------------------------------------------------------------------------------
def method_consensus() -> list:
    import matplotlib.pyplot as plt

    _style()
    family = "supp_gin_method_consensus"
    width, left, right = 420.0, 76.0, 40.0                     # the same for both panels: bar groups stand over the columns
    pct = attributions().percent
    labels = [C.GATE_GROUP[k].replace(" between", "\nbetween").replace("ESM-2 comp", "ESM-2\ncomp").replace("ortholog ", "ortholog\n")
              .replace("identity spread", "identity\nspread") for k in GROUP]
    table = np.array([[pct[m][k] for k in GROUP] for m in METHOD])

    fig, ax = _page(width, 138, left, right=right, bottom=22)
    x = np.arange(len(GROUP))
    bar = 0.18
    for i, m in enumerate(METHOD):
        ax.bar(x + (i - 1.5) * bar, table[i], width=bar, color=METHOD_COLOR[i], edgecolor="none", label=m)
    ax.set_xlim(-0.5, len(GROUP) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("% of total importance")
    ax.legend(loc="upper right", frameon=False, handlelength=1.1, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
    out = [_save(fig, family, "consensus_bars")]

    fig, ax = _page(width, 4 * 21.0 + TOP + 22, left, right=right, bottom=22)
    top = float(np.ceil(table.max() / 10.0) * 10.0)
    im = ax.imshow(table, aspect="auto", cmap="Blues", vmin=0, vmax=top)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(METHOD)))
    ax.set_yticklabels(METHOD)
    ax.tick_params(length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table[i, j]:.1f}", ha="center", va="center", fontsize=C.SIZE["tick"],
                    color="white" if table[i, j] >= 0.55 * top else C.INK)
    ax.set_xticks(x[:-1] + 0.5, minor=True)
    ax.set_yticks(np.arange(len(METHOD) - 1) + 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    p = ax.get_position()
    cax = fig.add_axes([p.x1 + 5.0 / width, p.y0, 5.0 / width, p.height])
    cb = fig.colorbar(im, cax=cax, ticks=np.arange(0, top + 1, 30))
    cb.set_label("% of total importance")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2.0, pad=1.5)
    out.append(_save(fig, family, "consensus_heatmap"))
    plt.close("all")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Protein-language inputs and expression inputs
# ---------------------------------------------------------------------------------------------------------------------
def orthogonality() -> list:
    fd = _style()
    family, width, height = "supp_gin_orthogonality", 239.5, 150.0
    g = gate()
    r2 = component_r2_from_expression()
    fig, ax = _page(width, height, 27, right=7)
    ax.hist(r2, bins=30, color=C.GREY_BAR, edgecolor="none")
    ax.axvline(0.05, color=THRESHOLD_COLOR, linewidth=0.6, linestyle="--")
    ax.set_xlabel("$R^2$ (expression inputs → ESM-2 component)")
    ax.set_ylabel("number of components")
    top = ax.get_ylim()[1]
    ax.text(0.0492, top, f"mean = {r2.mean():.3f}\nmaximum = {r2.max():.3f}\nabove 0.05: {int((r2 > 0.05).sum())} of 64 components",
            ha="right", va="top", fontsize=C.SIZE["text"], color=C.INK, linespacing=1.25)
    ax.text(0.0508, top, "$R^2$ = 0.05", ha="left", va="top", fontsize=C.SIZE["text"], color=THRESHOLD_COLOR)
    out = [_save(fig, family, "r2_components_from_expression")]

    # Δα of the full gate network and with one input group set to zero; bins and range as everywhere for Δα
    shown = {"full gate network": g.delta_alpha, "ESM-2 components\nset to zero": delta_alpha_ablated("ESM-2 PCs"),
             "expression inputs\nset to zero": delta_alpha_ablated("expression")}
    color = (C.GREY_BAR, ZEROED_COLOR["ESM-2 PCs"], ZEROED_COLOR["expression"])
    fig, ax = _page(width, height, 35, right=7)
    for i, (name, v) in enumerate(shown.items()):
        if i == 0:
            ax.hist(v, bins=fd.DELTA_ALPHA_BINS, color=color[i], edgecolor="none", label=name, zorder=1)
        else:
            ax.hist(v, bins=fd.DELTA_ALPHA_BINS, histtype="step", edgecolor=color[i], linewidth=0.9, label=name, zorder=4 - i)
    ax.set_xlim(*fd.DELTA_ALPHA_RANGE)
    ax.set_xticks(list(fd.DELTA_ALPHA_TICKS))
    ax.set_xlabel("Δα")
    ax.set_ylabel("number of genes")
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.legend(loc="upper right", frameon=False, handlelength=1.1, handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0)
    below = [int((v < fd.DELTA_ALPHA_RANGE[0]).sum()) for v in shown.values()]
    above = [int((v > fd.DELTA_ALPHA_RANGE[1]).sum()) for v in shown.values()]
    if any(above) or below[2]:
        raise SystemExit(f"genes outside the Δα range that the note on the panel does not cover: below {below}, above {above}")
    ax.text(0.02, 0.985, f"below −0.4, not shown:\n{below[0]} genes (full gate network),\n{below[1]} (ESM-2 components\nset to zero)",
            transform=ax.transAxes, ha="left", va="top", fontsize=C.SIZE["text"], color=C.MUTED, linespacing=1.25)
    out.append(_save(fig, family, "delta_alpha_inputs_zeroed"))
    print(f"R2 of the components from expression: mean {r2.mean():.3f}, max {r2.max():.3f}, above 0.05: {int((r2 > 0.05).sum())}/64")
    for name, v in shown.items():
        print(f"{name.replace(chr(10), ' '):>32}: Δα > +0.05 {100 * (v > 0.05).mean():.1f}%, Δα < -0.05 {100 * (v < -0.05).mean():.1f}%, "
              f"below -0.4 (not shown) {int((v < -0.4).sum())}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The 64 ESM-2 components one by one
# ---------------------------------------------------------------------------------------------------------------------
def per_component() -> list:
    fd = _style()
    family = "supp_gin_per_pc"
    g = gate()
    importance = attributions().ig_abs[:64]
    rho = component_spearman()
    top3 = tuple(int(j) for j in np.argsort(importance)[::-1][:3])
    if top3 != TOP_COMPONENTS:
        raise SystemExit(f"the three most important components are {top3}, not {TOP_COMPONENTS} as in the figure and its caption")
    out = []
    x = np.arange(64)
    for name, values, ylabel in (("ig_by_component", importance, "integrated gradients\n(mean |IG| over genes)"),
                                 ("spearman_by_component", rho, "Spearman ρ with Δα")):
        fig, ax = _page(239.5, 118, 40)
        ax.bar(x, values, color=C.GREY_BAR, edgecolor="none", width=0.8)
        if name.startswith("spearman"):
            ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--")
        ax.set_xlim(-1.5, 64.5)
        ax.set_xticks(np.arange(0, 64, 8))
        ax.set_xlabel("ESM-2 component (index)")
        ax.set_ylabel(ylabel)
        out.append(_save(fig, family, name))

    hidden = int((g.delta_alpha < fd.DELTA_ALPHA_RANGE[0]).sum())
    if (g.delta_alpha > fd.DELTA_ALPHA_RANGE[1]).any():
        raise SystemExit("genes above the Δα range of the axis: the note on the panels does not cover them")
    width = (486.0 - 2 * 7.0) / 3
    for rank, j in enumerate(top3, start=1):
        fig, ax = _page(width, 132, 32, top=5.0)
        fd.density_scatter(ax, g.x[:, j], g.delta_alpha, s=1.0)
        ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--")
        ax.set_ylim(*fd.DELTA_ALPHA_RANGE)
        ax.set_yticks(list(fd.DELTA_ALPHA_TICKS))
        ax.set_ylabel("Δα")
        _xlabel(ax, f"{C.gate_input(INPUT[j])} (standardized)")
        ax.text(0.985, 0.985, f"Spearman ρ = {_signed(rho[j])}", transform=ax.transAxes, ha="right", va="top",
                fontsize=C.SIZE["text"], color=C.INK)
        if hidden:                                             # genes beyond the Δα range of the axis, all below it here
            ax.annotate(f"{hidden} genes below −0.4 not shown", (0.985, 0.985), xycoords="axes fraction", xytext=(0, -7.6),
                        textcoords="offset points", ha="right", va="top", fontsize=C.SIZE["text"], color=C.MUTED)
        out.append(_save(fig, family, f"component_rank{rank}"))
    print("largest integrated-gradients importance:", [(j, round(float(importance[j]), 3), round(float(rho[j]), 3)) for j in top3])
    print("largest |Spearman ρ|:", [(int(j), round(float(rho[j]), 3)) for j in np.argsort(np.abs(rho))[::-1][:3]])
    return out


PANELS = {"linear_baseline": linear_baseline, "attribution_suite": attribution_suite, "method_consensus": method_consensus,
          "orthogonality": orthogonality, "per_component": per_component}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
