"""Residue-level figures of the Supplement, redrawn at print size from the stored tables of the residue analysis.

    supp_residue_mixed_effects/forest.pdf                  fixed effects of the residue-level mixed model (β, 95% interval)
    supp_residue_cliffs_delta/forest.pdf                   the 20 residue contrasts: Cliff's δ with interval, pooled and per-gene p
    supp_residue_attention_by_feature/distributions.pdf    per-residue |attribution| in four groupings of residues (a to d)
    supp_residue_card_<gene>/card.pdf                      ETV1, HPRT1, RUNX1: five tracks along the human protein and the
                                                           share of |attribution| by input group

Inputs (all stored; nothing is fitted, embedded or attributed here):
``notebooks/cache/gin_attribution/L_analysis/`` ``mixedlm_coef.pkl``, ``mixedlm_meta.json``, ``summary_contrasts.pkl``,
``residue_table.pkl``; ``notebooks/cache/gin_attribution/`` ``captum_attr.pkl``, ``per_residue_signals.pkl``,
``real_gin_attn.pkl``, ``selected_genes.json``; ``notebooks/cache/protein_structure/sequences.json``;
``data/processed/ortholog_table.csv``; and, for the header of a card, ``final_state.npz`` and the gene order of the data
object of the liver Human+Monkey run. The cards derive what the original drawing derived from these files: the
alignment of each ortholog to the human protein, the 11-residue moving averages and the z-scored relevance.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.residue [panel ...]
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

from figures.redraw import _common as C

L_CACHE = C.NB_CACHE / "gin_attribution" / "L_analysis"
GIN_CACHE = C.NB_CACHE / "gin_attribution"
SEQUENCES = C.NB_CACHE / "protein_structure" / "sequences.json"
ORTHOLOGS = C.DATA / "ortholog_table.csv"   # the table of the liver cohorts

FILL, LITE, DARK = "#5c6770", "#cfd2d4", "#555555"       # p < 0.05, p >= 0.05, secondary text
RAMP3 = ("#cfd2d4", "#9aa1a7", FILL)                     # three ordered classes, light to dark
RAMP4 = ("#dadcdd", "#b6bbbf", "#828a90", FILL)          # four ordered classes
GRAY_DARK, GRAY_MID, GRAY_LITE, GRAY_BG = "#222222", "#666666", "#aaaaaa", "#f0f0f0"   # tracks of the cards
WHISKERS = 1.5   # box plots: most extreme value within 1.5 x IQR of the box, as in the first version of the panel

TERM = {   # terms of the mixed model
    "C(ss)[T.H]": "secondary structure: helix", "C(ss)[T.E]": "secondary structure: strand",
    "C(sasa_bin)[T.buried]": "solvent accessibility: buried", "C(sasa_bin)[T.exposed]": "solvent accessibility: exposed",
    "C(plddt_bin)[T.very low (<50)]": "pLDDT: very low (<50)", "C(plddt_bin)[T.low (50–70)]": "pLDDT: low (50–70)",
    "C(plddt_bin)[T.very high (>90)]": "pLDDT: very high (>90)", "hydropathy_z": "hydropathy (z)",
    "C(chem)[T.hydrophobic]": "chemistry: hydrophobic", "C(chem)[T.positive]": "chemistry: positive",
    "C(chem)[T.negative]": "chemistry: negative", "entropy_z": "conservation entropy (z)",
    "esm2_pc_frac_z": "ESM-2 component fraction (z)", "rel_pos_z": "relative position N→C (z)",
}
FEATURE = {   # families of the residue contrasts
    "SS": "secondary structure", "SASA": "solvent accessibility", "pLDDT": "pLDDT", "Chem": "side-chain chemistry",
    "Contact": "contact density", "Disorder": "disorder", "Cons.": "conservation", "Special": "special residues",
}
SPECIES = {"mmulatta": "Macaque", "mmusculus": "Mouse", "drerio": "Zebrafish"}   # rows of the identity track, top to bottom
INPUT_GROUP_OF = ["esm2_pc"] * 64 + ["esm2_var", "seq_id"] + ["expr"] * 3 + ["seq_id"] * 3   # the 72 gate inputs, in order
INPUT_GROUP = {   # key: (name, bar color, color of the percentage on the bar); the colors of the attribution shares
    "esm2_pc": ("ESM-2 components", C.GATE_GROUP_COLOR["ESM-2 PCs"], "#222222"),
    "esm2_var": ("ESM-2 distance between species", C.GATE_GROUP_COLOR["esm2_intersp_var"], "#222222"),
    "seq_id": ("ortholog identity statistics", C.GATE_GROUP_COLOR["mean_identity"], "#ffffff"),   # mean, SD, minimum, maximum
    "expr": ("expression", C.GATE_GROUP_COLOR["expression"], "#ffffff"),
}
CURATED_CLASS = {"M": "curated M class", "F": "curated F class", "ref": "curated reference class"}
SMOOTH = 11      # residues in the moving average of the tracks
N_MARKED = 10    # residues marked by vertical lines in the two lower tracks


# ----------------------------------------------------------------------------- small helpers

def stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_p(p: float) -> str:
    """A p-value as printed: three decimals; below 0.001 as a power of ten ("9.6×10⁻⁵⁷"); four decimals where three would
    round a value below a star threshold up to it (0.0496 is not printed as 0.050)."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT)}"
    three = f"{p:.3f}"
    return f"{p:.4f}" if float(three) in (0.05, 0.01) and p < float(three) else three


def sheet(width_bp: float, height_bp: float):
    """A figure of the given size in bp (1 bp = 1/72 in), to be filled with ``axes_at`` and ``put``."""
    import matplotlib.pyplot as plt

    C.design()
    return plt.figure(figsize=(width_bp / 72, height_bp / 72))


def axes_at(fig, x: float, top: float, w: float, h: float, **kw):
    """Axes with the left edge at ``x`` and the upper edge at ``top`` (bp from the figure's top left corner)."""
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)


def put(fig, x: float, y: float, s: str, **kw):
    """Text with its baseline at ``y`` bp below the figure's top edge."""
    W, H = fig.get_size_inches() * 72
    kw.setdefault("va", "baseline")
    kw.setdefault("fontsize", C.SIZE["tick"])
    return fig.text(x / W, 1 - y / H, s, **kw)


def width_of(fig, s: str, size: float, **kw) -> float:
    """Width of a string in bp."""
    t = fig.text(0, 0, s, fontsize=size, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    t.remove()
    return w


def key_row(fig, x: float, y: float, entries: list, size: float | None = None, right: bool = False, gap: float = 9.0) -> float:
    """A one-line key on the baseline ``y``: entries are ("patch", face, edge, text) or ("line", style dict, text).

    Starts at ``x``, or ends there with ``right=True``. Returns the width used, in bp.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    size = size or C.SIZE["legend"]
    W, H = fig.get_size_inches() * 72
    hw, pad = 9.0, 3.0
    widths = [hw + pad + width_of(fig, e[-1], size) for e in entries]
    total = sum(widths) + gap * (len(entries) - 1)
    cur = x - total if right else x
    for e, w in zip(entries, widths):
        yc = 1 - (y - 0.36 * size) / H
        if e[0] == "patch":
            fig.add_artist(Rectangle((cur / W, yc - 2.6 / H), hw / W, 5.2 / H, transform=fig.transFigure,
                                     facecolor=e[1], edgecolor=e[2], linewidth=0.4))
        else:
            fig.add_artist(Line2D([cur / W, (cur + hw) / W], [yc, yc], transform=fig.transFigure, **e[1]))
        put(fig, cur + hw + pad, y, e[-1], fontsize=size)
        cur += w + gap
    return total


# ----------------------------------------------------------------------------- mixed model (one panel)

def mixed_effects() -> list:
    coef_file, meta_file = L_CACHE / "mixedlm_coef.pkl", L_CACHE / "mixedlm_meta.json"
    C.need(coef_file, meta_file)
    cdf = pd.read_pickle(coef_file)
    meta = json.loads(meta_file.read_text())
    if "term" in cdf.columns:
        cdf = cdf.set_index("term")
    cdf = cdf.drop(index=[i for i in cdf.index if str(i).startswith("Intercept")])
    cdf["ci_lo"] = cdf["coef"] - 1.959963984540054 * cdf["stderr"]
    cdf["ci_hi"] = cdf["coef"] + 1.959963984540054 * cdf["stderr"]
    cdf["name"] = [TERM[t] for t in cdf.index]
    cdf = cdf.sort_values("coef", ascending=True).reset_index()
    n = len(cdf)

    W, pitch, top, bottom, right = 300.0, 11.5, 14.0, 27.0, 17.0
    fig = sheet(W, top + n * pitch + bottom)
    left = max(width_of(fig, s, C.SIZE["tick"]) for s in cdf["name"]) + 7.0
    ax = axes_at(fig, left, top, W - left - right, n * pitch)
    y = np.arange(n)
    ax.barh(y, cdf["coef"], color=[FILL if (np.isfinite(p) and p < 0.05) else LITE for p in cdf["p"]], edgecolor=C.INK,
            linewidth=0.4, height=0.55, zorder=2)
    for i, (lo, hi) in enumerate(zip(cdf["ci_lo"], cdf["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=C.INK, linewidth=0.7, zorder=3)
            for v in (lo, hi):
                ax.plot([v, v], [i - 0.13, i + 0.13], color=C.INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    for i, p in enumerate(cdf["p"]):
        if stars(float(p)) != "ns":
            ax.text(1.025, i, stars(float(p)), ha="left", va="center", fontsize=C.SIZE["label"], transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(cdf["name"])
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("β on log |attribution| (other terms held fixed)")
    put(fig, W - 1, 8.5, f"$n$ = {meta['n_residues']:,} residues, {meta['n_genes']} genes  ·  ICC = {meta['icc'] * 100:.1f}%  ·  "
                         f"$R^2_{{\\mathrm{{marg}}}}$ = {meta['r2_marginal']:.3f}  ·  $R^2_{{\\mathrm{{cond}}}}$ = {meta['r2_conditional']:.3f}",
        fontsize=7.3, color=DARK, ha="right")   # 7.3 pt: the subscripts print at 5.1 pt
    print(cdf[["name", "coef", "ci_lo", "ci_hi", "p"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return [C.save_panel(fig, "supp_residue_mixed_effects", "forest")]


# ----------------------------------------------------------------------------- residue contrasts (one panel)

def contrast_name(label: str) -> str:
    """The name of a contrast as it is printed. The stored tables say "sheet" for the secondary-structure class that the
    paper calls "strand" (class E); the stored label keeps its value, the printed name follows the paper."""
    return label.replace("sheet", "strand")


def cliffs_delta() -> list:
    src = L_CACHE / "summary_contrasts.pkl"
    C.need(src)
    df = pd.read_pickle(src).sort_values("cliffs_d", ascending=True).reset_index(drop=True)
    supported = (df["pg_p"].fillna(1.0) < 0.05).to_numpy()
    n = len(df)

    W, pitch, top, bottom = 486.0, 11.0, 15.0, 41.0
    small, tick = 6.0, C.SIZE["tick"]
    fig = sheet(W, top + n * pitch + bottom)
    feature = [FEATURE[p] for p in df["panel"]]
    w_feature = max(width_of(fig, s, tick) for s in feature)
    w_contrast = max(width_of(fig, contrast_name(s), tick) for s in df["label"])
    left = 1 + w_feature + 9 + w_contrast + 5
    x_pool, x_gene, x_n = W - 108.0, W - 49.0, W - 3.0     # right edges of the three number columns
    ax = axes_at(fig, left, top, x_pool - 34 - left, n * pitch)
    y = np.arange(n)
    ax.barh(y, df["cliffs_d"], color=[FILL if s else "white" for s in supported], edgecolor=C.INK, linewidth=0.55, height=0.6, zorder=2)
    for i, (lo, hi) in enumerate(zip(df["ci_lo"], df["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=C.INK, linewidth=0.7, zorder=3)
            for v in (lo, hi):
                ax.plot([v, v], [i - 0.13, i + 0.13], color=C.INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    lo, hi = float(df[["cliffs_d", "ci_lo"]].min().min()), float(df[["cliffs_d", "ci_hi"]].max().max())
    ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("Cliff's δ (positive: first level has larger |attribution|)")

    head = top - 5.0
    put(fig, 1, head, "residue feature", fontsize=C.SIZE["title"])
    put(fig, left - 5, head, "contrast", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_pool, head, "pooled $p$", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_gene, head, "per-gene $p$", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_n, head, "$n_g$", fontsize=C.SIZE["title"], ha="right")
    for i, r in df.iterrows():
        b = top + (i + 0.5) * pitch + 0.36 * tick   # baseline of the row
        put(fig, 1, b, feature[i], fontsize=tick)
        put(fig, left - 5, b, contrast_name(r["label"]), fontsize=tick, ha="right")
        for x, p in ((x_pool, r["mwu_p"]), (x_gene, r["pg_p"])):
            put(fig, x, b, fmt_p(p), fontsize=small, ha="right")
            put(fig, x + 3, b, stars(p), fontsize=small, color=C.INK if stars(p) != "ns" else C.MUTED)
        put(fig, x_n, b, f"{int(r['n_pg'])}", fontsize=small, ha="right")
    key_row(fig, left, top + n * pitch + bottom - 3.5,
            [("patch", FILL, C.INK, "per-gene $p$ < 0.05 (within-protein test)"), ("patch", "white", C.INK, "per-gene $p$ ≥ 0.05")])
    print(df[["panel", "label", "cliffs_d", "ci_lo", "ci_hi", "mwu_p", "pg_p", "n_pg"]].to_string(index=False))
    return [C.save_panel(fig, "supp_residue_cliffs_delta", "forest")]


# ----------------------------------------------------------------------------- distributions (four sub-panels, a to d)

GROUPINGS = [   # heading, column, levels, names of the levels, colors, width of one box slot in bp
    ("secondary structure", "ss", ["C", "H", "E"], ["coil", "helix", "strand"], RAMP3, 26.0),
    ("relative solvent accessibility", "sasa_bin", ["buried", "intermediate", "exposed"], ["buried", "intermediate", "exposed"], RAMP3, 35.5),
    ("pLDDT (AlphaFold)", "plddt_bin", ["very low (<50)", "low (50–70)", "high (70–90)", "very high (>90)"],
     ["<50", "50–70", "70–90", ">90"], RAMP4, 26.0),
    ("side-chain chemistry", "chem", ["polar", "positive", "negative", "hydrophobic"],
     ["polar", "positive", "negative", "hydrophobic"], RAMP4, 35.5),
]
DIST = dict(width=486.0, left=26.0, gap=9.0, top=13.0, height=112.0, bottom=24.0, box=14.5)
PLDDT_RESIDUES = (6446, 2484, 6029, 21873)   # residues per pLDDT bin, from very low to very high, as the paper states them


def distribution_axes() -> list:
    """Left edge and width in bp of the four axes on the page (also the place of the letters a to d in the spec)."""
    out, x = [], DIST["left"]
    for *_, levels, _names, _colors, slot in GROUPINGS:
        out.append((x, slot * len(levels)))
        x += slot * len(levels) + DIST["gap"]
    return out


def attention_by_feature() -> list:
    import matplotlib as mpl

    src = L_CACHE / "residue_table.pkl"
    C.need(src)
    rt = pd.read_pickle(src)
    rt["sasa_bin"] = np.where(rt["rsasa"] < 0.20, "buried", np.where(rt["rsasa"] > 0.50, "exposed", "intermediate"))
    # the bins of the analysis (notebooks/_gin_attribution_builder.py): above 90, above 70, above 50, else very low, so a
    # residue exactly at an edge belongs to the lower bin
    rt["plddt_bin"] = np.where(rt["plddt"] > 90, "very high (>90)", np.where(rt["plddt"] > 70, "high (70–90)",
                               np.where(rt["plddt"] > 50, "low (50–70)", "very low (<50)")))
    counted = tuple(int((rt["plddt_bin"] == level).sum()) for level in GROUPINGS[2][2])
    if counted != PLDDT_RESIDUES:
        raise SystemExit(f"residues per pLDDT bin: {counted}; the paper states {PLDDT_RESIDUES}")
    y_top = float(np.quantile(rt["abs_attn"].to_numpy(), 0.99))

    d = DIST
    fig = sheet(d["width"], d["top"] + d["height"] + d["bottom"])
    tick, count = 6.0, C.SIZE["text"]   # class names at 6 pt so that all stand side by side
    base_tick = d["top"] + d["height"] + 8.5
    base_count = base_tick + 9.0
    for k, ((x, w), (title, col, levels, names, colors, slot)) in enumerate(zip(distribution_axes(), GROUPINGS)):
        ax = axes_at(fig, x, d["top"], w, d["height"])
        data = [rt.loc[rt[col] == lv, "abs_attn"].to_numpy() for lv in levels]
        bp = ax.boxplot(data, positions=np.arange(len(levels)), widths=d["box"] / slot, whis=WHISKERS, showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.0), whiskerprops=dict(color=C.INK, linewidth=0.6),
                        capprops=dict(color=C.INK, linewidth=0.6), boxprops=dict(linewidth=0.0))
        for patch, fc in zip(bp["boxes"], colors):
            patch.set(facecolor=fc, edgecolor=C.INK, linewidth=0.4)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_ylim(0, y_top)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelleft=(k == 0))
        put(fig, x, d["top"] - 4.5, title, fontsize=C.SIZE["title"])
        ends = []
        for i, (name, v) in enumerate(zip(names, data)):
            xc = x + (i + 0.5) * slot
            put(fig, xc, base_tick, name, fontsize=tick, ha="center")
            put(fig, xc, base_count, f"{len(v):,}", fontsize=count, ha="center", color=DARK)
            ends.append((xc - width_of(fig, name, tick) / 2, xc + width_of(fig, name, tick) / 2))
            print(f"{title:32s} {name:13s} n = {len(v):6,d}  median {np.median(v):.3f}  box {np.percentile(v, 25):.3f} to {np.percentile(v, 75):.3f}")
        if any(b[0] - a[1] < 1.5 for a, b in zip(ends, ends[1:])):
            raise SystemExit(f"class names of '{title}' do not fit side by side")
        if k == 0:
            ax.set_ylabel("per-residue |attribution|")
            put(fig, x - 4, base_count, "$n$ =", fontsize=count, ha="right", color=DARK)
    with mpl.rc_context({"savefig.bbox": "standard"}):   # the page stays the figure: the letters are placed by page position
        return [C.save_panel(fig, "supp_residue_attention_by_feature", "distributions", exact=True)]


# ----------------------------------------------------------------------------- protein cards (one card per figure)

def card_inputs() -> dict:
    files = dict(attr=GIN_CACHE / "captum_attr.pkl", signals=GIN_CACHE / "per_residue_signals.pkl", real=GIN_CACHE / "real_gin_attn.pkl",
                 selected=GIN_CACHE / "selected_genes.json", sequences=SEQUENCES, orthologs=ORTHOLOGS)
    C.need(*files.values())
    out = {}
    for k in ("attr", "signals", "real"):
        with open(files[k], "rb") as f:
            out[k] = pickle.load(f)
    out["selected"] = {r["gene"]: r for r in json.loads(files["selected"].read_text())}
    out["sequences"] = json.loads(files["sequences"].read_text())
    out["orthologs"] = pd.read_csv(files["orthologs"]).set_index("ortholog_group_id")
    return out


def gate_values(gene: str, inp: dict) -> dict:
    """α and the protein prior π of one gene in the run its attribution was computed on, and its row of the ortholog table."""
    import h5py

    entry = inp["selected"][gene]
    cohort = C.NOTEBOOK_KEY[entry["dataset"]]
    hit = inp["orthologs"][inp["orthologs"]["hsapiens_gene_name"] == gene]
    if not len(hit):
        raise SystemExit(f"{gene} is not in {ORTHOLOGS}")
    group = int(hit.index[0])
    adata = C.DATA / C.ADATA[cohort]
    C.need(adata)
    with h5py.File(adata, "r") as f:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"][f["var"].attrs["_index"]][:]]
    i = names.index(f"ortho_{group}")
    state = np.load(C.run_dir(cohort) / "final_state.npz")
    alpha, prior = float(state["alpha"][i]), float(state["prior"][i])
    return dict(curated=entry["direction"], cohort=cohort, alpha=alpha, prior=prior, delta=alpha - prior, row=inp["orthologs"].loc[group])


def identity_rows(human: str, others: list) -> np.ndarray:
    """Per human residue and ortholog: 1 same amino acid, 0 different, -1 not aligned (global alignment, BLOSUM62 scoring)."""
    from Bio import Align

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    L, rows = len(human), []
    for seq in others:
        row = np.full(L, -1, dtype=int)
        if seq:
            aln = aligner.align(human, seq)[0]
            for (hs, he), (xs, _xe) in zip(*aln.aligned):
                for k in range(he - hs):
                    if hs + k < L and xs + k < len(seq):
                        row[hs + k] = 1 if human[hs + k] == seq[xs + k] else 0
        rows.append(row)
    return np.vstack(rows)


def moving_average(v: np.ndarray) -> np.ndarray:
    return np.convolve(v, np.ones(SMOOTH) / SMOOTH, mode="same") if len(v) >= SMOOTH else v


def relevance(signal: dict) -> np.ndarray:
    """Structural-proxy relevance: z(divergence) x z(deviation), moving average, z-scored again over the protein."""
    from scipy.stats import zscore

    d = np.where(np.isnan(signal["divergence"]), 0.0, signal["divergence"])
    cd = signal["consensus_dev"]
    d_z = zscore(d, nan_policy="omit") if d.std() > 1e-9 else np.zeros_like(d)
    cd_z = zscore(cd, nan_policy="omit") if cd.std() > 1e-9 else np.zeros_like(cd)
    rel = moving_average(np.asarray(d_z) * np.asarray(cd_z))
    return ((rel - rel.mean()) / (rel.std() + 1e-9)).astype(np.float32)


def runs(mask: np.ndarray) -> list:
    """(start, length) of every run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def card(gene: str, inp: dict | None = None) -> list:
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    inp = inp or card_inputs()
    info, signal = gate_values(gene, inp), inp["signals"][gene]
    L, human = signal["length"], signal["human_seq"]
    ident = identity_rows(human, [inp["sequences"].get(info["row"].get(f"{sp}_gene_id"), "") for sp in SPECIES])
    attr = np.abs(np.asarray(inp["attr"][gene]["attribution"]))
    share = {g: float(attr[[i for i, k in enumerate(INPUT_GROUP_OF) if k == g]].sum()) for g in INPUT_GROUP}
    total = sum(share.values()) or 1.0
    share = {g: v / total for g, v in share.items()}
    rel, real = relevance(signal), inp["real"][gene]
    x = np.arange(L)

    W, left, right = 486.0, 40.0, 4.0
    heights, lead, gap = [38.0, 40.0, 34.0, 48.0, 56.0], 11.0, 7.0   # track heights; room for a heading; white between tracks
    tops, t = [], 13.0
    for h in heights:
        tops.append(t + lead)
        t += lead + h + gap
    strip_top = tops[-1] + heights[-1] + 23.0 + lead
    fig = sheet(W, strip_top + 12.0 + 16.0)
    aw, name, tick = W - left - right, C.SIZE["label"], C.SIZE["tick"]
    axes = []
    for top, h in zip(tops, heights):
        axes.append(axes_at(fig, left, top, aw, h, **({"sharex": axes[0]} if axes else {})))
    ax_seq, ax_div, ax_dev, ax_rel, ax_real = axes
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in axes[1:]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

    gw = width_of(fig, gene, C.SIZE["title"] + 0.5, fontweight="bold")
    put(fig, left, 8.5, gene, fontsize=C.SIZE["title"] + 0.5, fontweight="bold")
    delta = f"{info['delta']:+.2f}".replace("-", "−")
    put(fig, left + gw, 8.5, f"  ·  {CURATED_CLASS[info['curated']]}  ·  α = {info['alpha']:.2f}, π = {info['prior']:.2f}, Δα = {delta}, {L} aa",
        fontsize=C.SIZE["title"] + 0.5)

    # 1) identity of each ortholog to the human protein, residue by residue
    ax_seq.add_patch(Rectangle((0, 0), L, len(SPECIES), facecolor=GRAY_BG, edgecolor="none", zorder=0))
    for r, row in enumerate(ident):
        y0 = len(SPECIES) - 1 - r
        ax_seq.broken_barh(runs(row == 0), (y0, 1), facecolors=GRAY_DARK, edgecolor="none", linewidth=0)
        ax_seq.broken_barh(runs(row == -1), (y0, 1), facecolors=GRAY_LITE, edgecolor="none", linewidth=0)
    ax_seq.set_xlim(0, L)
    ax_seq.set_ylim(0, len(SPECIES))
    ax_seq.set_yticks(np.arange(len(SPECIES)) + 0.5)
    ax_seq.set_yticklabels(list(SPECIES.values())[::-1])
    ax_seq.tick_params(axis="y", length=0)
    put(fig, left, tops[0] - 4, "identity to human per species", fontsize=name)
    key_row(fig, W - right, tops[0] - 4, [("patch", GRAY_BG, "#888888", "match"), ("patch", GRAY_DARK, "none", "mismatch"),
                                         ("patch", GRAY_LITE, "none", "gap")], right=True)

    # 2) divergence between species in the embedding of the protein language model
    d = np.where(np.isnan(signal["divergence"]), 0.0, signal["divergence"])
    d_smooth = moving_average(d)
    ax_div.fill_between(x, 0, d_smooth, color=GRAY_DARK, alpha=0.30, linewidth=0)
    ax_div.plot(x, d_smooth, color=GRAY_DARK, linewidth=1.0)
    ax_div.set_ylim(0, max((float(np.nanmax(d)) if np.any(np.isfinite(d)) else 1.0) * 1.15, 0.05))
    put(fig, left, tops[1] - 4, "divergence between species in the ESM-2 35M embedding", fontsize=name)

    # 3) deviation of each residue from the mean embedding of the protein
    cd_smooth = moving_average(signal["consensus_dev"])
    ax_dev.fill_between(x, 0, cd_smooth, color=GRAY_MID, alpha=0.30, linewidth=0)
    ax_dev.plot(x, cd_smooth, color=GRAY_MID, linewidth=1.0)
    ax_dev.set_ylim(0, max(float(cd_smooth.max()) * 1.15, 0.5))
    put(fig, left, tops[2] - 4, "deviation from the protein's mean embedding", fontsize=name)

    # 4) structural-proxy relevance; thin lines: the residues with the highest relevance
    marked_rel = dict(color=GRAY_DARK, linewidth=0.3, alpha=0.35)
    ax_rel.plot(x, rel, color=GRAY_DARK, linewidth=1.0)
    ax_rel.fill_between(x, 0, rel, where=(rel > 0), color=GRAY_DARK, alpha=0.45, linewidth=0)
    ax_rel.fill_between(x, 0, rel, where=(rel < 0), color=GRAY_LITE, alpha=0.55, linewidth=0)
    for o in np.argsort(rel)[::-1][:N_MARKED]:
        ax_rel.axvline(o, zorder=0, **marked_rel)
    ax_rel.axhline(0, color="#444444", linewidth=0.5)
    put(fig, left, tops[3] - 4, "structural-proxy relevance (z)", fontsize=name)
    key_row(fig, W - right, tops[3] - 4, [("line", dict(marked_rel, linewidth=0.6), f"the {N_MARKED} residues with the highest relevance")], right=True)

    # 5) attribution of the gate network per residue; dashed lines: the residues with the largest |attribution| before the average
    marked_real = dict(color="#222222", linewidth=0.3, alpha=0.6, linestyle=(0, (3, 2)))
    ra = moving_average(np.where(np.isnan(real), 0.0, real))
    ax_real.plot(x, ra, color=GRAY_DARK, linewidth=1.0)
    ax_real.fill_between(x, 0, ra, where=(ra > 0), color=GRAY_DARK, alpha=0.55, linewidth=0)
    ax_real.fill_between(x, 0, ra, where=(ra < 0), color=GRAY_LITE, alpha=0.65, linewidth=0)
    finite = ~np.isnan(real)
    for tp in np.where(finite)[0][np.argsort(np.abs(real[finite]))[::-1][:N_MARKED]]:
        ax_real.axvline(tp, zorder=0, **marked_real)
    ax_real.axhline(0, color="#444444", linewidth=0.5)
    ax_real.set_xlabel("residue position (human reference)")
    put(fig, left, tops[4] - 4, "per-residue attribution, signed", fontsize=name)
    key_row(fig, W - right, tops[4] - 4, [("line", dict(marked_real, linewidth=0.6), f"the {N_MARKED} residues with the largest |attribution| before smoothing")],
            right=True)

    # strip: share of |attribution| of the gene by group of gate inputs
    ax = axes_at(fig, left, strip_top, aw, 12.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cur = 0.0
    for g, (_label, color, ink) in INPUT_GROUP.items():
        f = share[g]
        if f <= 0:
            continue
        ax.barh(0.5, f, left=cur, height=1.0, color=color, edgecolor="#222222", linewidth=0.4)
        if f >= 0.08:
            ax.text(cur + f / 2, 0.5, f"{f * 100:.0f}%", ha="center", va="center_baseline", fontsize=tick, color=ink)
        cur += f
    put(fig, left, strip_top - 4, "share of |attribution| by input group", fontsize=name)
    key_row(fig, left, strip_top + 12.0 + 11.5, [("patch", color, "none", label) for label, color, _ in INPUT_GROUP.values()], gap=12.0)
    print(f"{gene}: {CURATED_CLASS[info['curated']]}, alpha {info['alpha']:.4f}, prior {info['prior']:.4f}, L {L}; shares "
          + ", ".join(f"{INPUT_GROUP[g][0]} {v * 100:.1f}%" for g, v in share.items()))
    return [C.save_panel(fig, f"supp_residue_card_{gene.lower()}", "card")]


def cards() -> list:
    inp = card_inputs()
    return [p for g in ("ETV1", "HPRT1", "RUNX1") for p in card(g, inp)]


PANELS = {
    "mixed_effects": mixed_effects, "cliffs_delta": cliffs_delta, "attention_by_feature": attention_by_feature,
    "card_etv1": lambda: card("ETV1"), "card_hprt1": lambda: card("HPRT1"), "card_runx1": lambda: card("RUNX1"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
