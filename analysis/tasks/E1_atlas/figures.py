"""E1 panels (exploratory): gene programs that recur across eight tissues, programs chosen for each tissue, named genes.
Four panels of three figures, saved through figures/style.py to figures/panels/<figure>/; reads the result tables of
results/analysis/E1 only.

Every panel is drawn at its size on the page and saved at that exact size, so the figures compose at scale 1:
supp_programs_recurrent (atlas, distance), supp_programs_by_tissue (tissue) and supp_programs_named_genes (genes).

    PYTHONPATH=$PWD python -m analysis.tasks.E1_atlas.figures [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                          # dry run: panels under <folder>/panels
    PYTHONPATH=$PWD python -m analysis.tasks.E1_atlas.figures --rows-table     # the table of the rows of the atlas panel (results/analysis/E1/panels/e1_atlas_rows.csv)
"""

from __future__ import annotations

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from analysis.tasks.E1_atlas import rules as RU  # noqa: E402
from analysis.tasks.H1_harmonise.figures import SET_LABEL  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
ROWS_TABLE = RU.OUT / "panels" / "e1_atlas_rows.csv"  # the rows of the atlas panel, as a table
WIDTH = style.TEXTWIDTH  # bp; every panel is as wide as the text
S_TITLE, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_TICK, style.SIZE_TEXT  # axis labels take SIZE_LABEL from style.apply()
INK, RULE = style.INK, style.RULE
UP, DOWN = style.ABOVE, style.BELOW  # called above and below the protein prior
NO_CALL = "#CCCCCC"  # a gene tested in a tissue and not called
NA_CELL = "#D9D9D9"  # heat-map cell without a value; darker than the house gray so it cannot be taken for the white center of the map
GATE, PER_GENE, PROFILE = style.READOUT["delta_alpha"], style.READOUT["delta_alpha_per_gene"], style.READOUT["profile_correlation"]
IDENTITY, DNDS = style.READOUT["protein_identity"], style.READOUT["dnds"]
TISSUE_NAME = {t: style.TISSUE[t].name for t in RU.TISSUES}
LADDERS = ("liver", "testis", "eye", "cortex")  # tissues with two-species models at several divergence times
# the four groups of the atlas panel: protein side, deployment side, name on the figure
QUAD = [("fast", "above", "fast-evolving proteins,\nabove the prior"), ("conserved", "above", "conserved proteins,\nabove the prior"),
        ("conserved", "below", "conserved proteins,\nbelow the prior"), ("fast", "below", "fast-evolving proteins,\nbelow the prior")]
# the source of a gene set, by the library key of the tables
SOURCE = {"keyword": "UniProt", "goslim": "GO", "go_tissue": "GO", "reactome": "Reactome", "reactome_top": "Reactome", "hallmark": "Hallmark"}
SHORT = {"Cleavage on pair of basic residues [PTM]": "Propeptide cleavage [PTM]"}
SOURCE_TAG = re.compile(r" (\[[^\]]+\]|\((CC|MF|BP|category)\))$")  # the tag a library puts behind a name
PROPER = {"Golgi", "Notch", "Hedgehog", "Hippo"}  # names that keep their capital inside a set name
# outline of the cell of a program's own tissue: solid where the own tissue is above the other tissues (q < 0.05), dashed otherwise
OUTLINE = {True: dict(fill=False, lw=1.1, ec=INK, ls="-"), False: dict(fill=False, lw=0.7, ec=INK, ls=(0, (2, 1.3)))}


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<figure>/<name>.pdf."""
    return style.save_panel(fig, name, exact=True)


def sentence_case(name: str) -> str:
    """A set name in sentence case: words in Title Case after the first become lower case; acronyms, words with a digit, a
    hyphen or an inner capital, and proper names stay."""
    words = name.split(" ")
    words = [w.lower() if i and w.isalpha() and w[0].isupper() and w[1:].islower() and w not in PROPER else w for i, w in enumerate(words)]
    if words[0][:1].islower() and words[0][1:].islower():
        words[0] = words[0].capitalize()
    return " ".join(words)


def program_label(name: str, library: str) -> str:
    """The name of a gene set on a figure: sentence case, its source in parentheses ("Secreted (UniProt)")."""
    n = str(name)
    if library == "class":
        n = SET_LABEL.get(n, n.replace("broad:", ""))
        source = "category" if "(category)" in n else "gene class"
    else:
        source = SOURCE[library]
    n = SOURCE_TAG.sub("", SHORT.get(n, n))
    return f"{sentence_case(n)} ({source})"


def heat_axes(fig, box: tuple, columns: list, rows: list | None = None):
    """Axes for a heat map with the tissue names along the top and, if given, the row names at the left."""
    ax = fig.add_axes(box)
    ax.set_xticks(range(len(columns)), columns, fontsize=S_TICK, rotation=45, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)) if rows else [], rows or [], fontsize=S_TEXT)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def color_key(fig, box: tuple, vmax: float, title: str) -> None:
    """Horizontal color bar of a diverging map from -vmax to vmax with its title above."""
    cax = fig.add_axes(box)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=style.DIVERGING), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=S_TEXT, length=1.5, pad=1.5)
    labels = cb.ax.get_xticklabels()  # the end labels stay inside the bar, so a bar at the panel edge loses nothing
    labels[0].set_ha("left")
    labels[-1].set_ha("right")
    cb.outline.set_linewidth(0.4)
    cax.set_title(title, fontsize=S_TEXT, pad=2.5, loc="left", linespacing=1.1)


def protein_side(A):
    """conserved: identity recurrent above matched-expression genes with positive Stouffer z; fast: the mirror."""
    return np.where(A.pi_recurrent & (A.pi_direction == "above") & (A.pi_stouffer > 0), "conserved",
                    np.where(A.pi_recurrent & (A.pi_direction == "below") & (A.pi_stouffer < 0), "fast", "neither"))


def atlas_rows(k=6) -> tuple[pd.DataFrame, list]:
    """Rule: STRICT sets, pruned for redundancy, split by protein side (pi recurrent above = conserved, below = fast) and deployment
    side; top k per quadrant by the mean Stouffer z of the two gates."""
    A = pd.read_csv(RU.OUT / "atlas.csv", index_col=0)
    A = A[A.strict & A.redundant_with.isna()].copy()
    A["prot"] = protein_side(A)
    rows, groups = [], []
    for p, d, lab in QUAD:
        x = A[(A.prot == p) & (A.can_direction == d)].sort_values("joint_stouffer", ascending=(d == "below"))
        x = x[x.groupby("library").cumcount() < 2].head(k)  # at most two sets per library, so one vocabulary does not fill a group
        rows.append(x)
        groups.append((lab, len(x)))
    return pd.concat(rows), groups


def panel_atlas():
    """Programs that recur with both gates, in four groups (protein side by deployment side): the Δα effect in each tissue
    under each gate, * where q < 0.05; at the right the z of protein identity, dN/dS and profile correlation, combined
    over the tissues."""
    R, groups = atlas_rows()
    T = RU.TISSUES
    nr = len(R)
    # bp from the left edge: group names, row names, two blocks of eight tissues, three columns of numbers
    W, wg, wn, ws, gap, rh, top, bot = WIDTH, 64, 124, 28, 5, 8.4, 50, 3
    x0 = wg + wn + 3
    wc = (W - 4 - 3 * ws - 2 * gap - x0) / 16
    H = top + rh * nr + bot
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    vmax = 1.0
    cmap = plt.get_cmap(style.DIVERGING).copy()
    cmap.set_bad(NA_CELL)
    names = [program_label(r["name"], r["library"]) for _, r in R.iterrows()]
    missing = False
    for b, (gate, pre) in enumerate(((GATE, "can"), (PER_GENE, "free"))):
        xb = x0 + b * (8 * wc + gap)
        ax = heat_axes(fig, [xb / W, bot / H, 8 * wc / W, rh * nr / H], [TISSUE_NAME[t] for t in T], names if b == 0 else None)
        M = R[[f"{pre}_e_{t}" for t in T]].to_numpy(float)
        Q = R[[f"{pre}_q_{t}" for t in T]].to_numpy(float)
        missing |= bool(np.isnan(M).any())
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        for i in range(nr):
            for j in range(8):
                if np.isfinite(M[i, j]) and Q[i, j] < 0.05:
                    ax.text(j, i + 0.2, style.SIGNIFICANT, ha="center", va="center", fontsize=style.SIZE_FLOOR, color="#ffffff" if abs(M[i, j]) > 0.6 * vmax else INK)
        fig.text((xb + 4 * wc) / W, 1 - 2 / H, gate.name, ha="center", va="top", fontsize=S_TITLE)
    xz = x0 + 16 * wc + 2 * gap
    ax2 = fig.add_axes([xz / W, bot / H, 3 * ws / W, rh * nr / H])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(nr - 0.5, -0.5)
    for j, hd in enumerate([IDENTITY.name.replace(" ", "\n", 1), DNDS.name, PROFILE.name.replace(" ", "\n")]):
        ax2.text(j, -0.8, hd, fontsize=S_TEXT, ha="center", va="bottom", linespacing=1.05)
    for i, (_, r) in enumerate(R.iterrows()):
        for j, v in enumerate((r.pi_stouffer, r.dNdS_stouffer, r.B1_stouffer)):
            ax2.text(j, i, f"{v:+.0f}".replace("-", "−"), fontsize=S_TEXT, ha="center", va="center")
    ax2.axis("off")
    fig.text((xz + 1.5 * ws) / W, 1 - 2 / H, "combined z\nover tissues", ha="center", va="top", fontsize=S_TITLE, linespacing=1.05)
    y = 0
    for k, (name, n) in enumerate(groups):
        fig.text(2 / W, (bot + rh * (nr - y - n / 2)) / H, name, fontsize=S_TEXT, ha="left", va="center", linespacing=1.1)
        y += n
        if k < len(groups) - 1:  # a rule under each group, across the whole panel
            fig.add_artist(Line2D([2 / W, 1 - 2 / W], [(bot + rh * (nr - y)) / H] * 2, color=INK, lw=0.6))
    color_key(fig, [2 / W, 1 - 30 / H, 84 / W, 5 / H], vmax, f"Δα effect against matched genes (SD)\ncolors end at ±{vmax:.1f}")
    fig.text(96 / W, 1 - 27.5 / H, f"{style.SIGNIFICANT} q < 0.05", fontsize=S_TEXT, ha="left", va="center")
    if missing:  # key of the cells without a value (a program with fewer than MIN_SET genes in a tissue)
        fig.add_artist(Rectangle((132 / W, 1 - 30.25 / H), 7 / W, 5.5 / H, fc=NA_CELL, ec="none"))
        fig.text(142 / W, 1 - 27.5 / H, f"fewer than {RU.MIN_SET} genes", fontsize=S_TEXT, ha="left", va="center")
    return save(fig, "supp_programs_recurrent/atlas")


def write_atlas_rows() -> None:
    """The rows of the atlas panel with their combined z values, as a table beside the results."""
    R, _ = atlas_rows()
    ROWS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    R[["name", "library", "can_stouffer", "free_stouffer", "pi_stouffer", "dNdS_stouffer", "B1_stouffer"]].to_csv(ROWS_TABLE)


def own_outline(ax, j: int, i: int, hit: bool) -> None:
    """Outline of the cell of a program's own tissue (``OUTLINE``), on a white line that keeps it visible on dark cells."""
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=OUTLINE[hit]["lw"] + 0.9, ec="white", clip_on=False))
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, clip_on=False, **OUTLINE[hit]))


def panel_tissue():
    """Programs chosen for each tissue before the analysis: set z in every tissue, for the two gates and for profile
    correlation; the cell of the program's own tissue is outlined."""
    t = pd.read_csv(RU.OUT / "tissue_programmes.csv")
    s = pd.read_csv(RU.OUT / "sets" / "sets.csv").set_index("key")
    T = RU.TISSUES
    rows = []
    for own, ids in RU.TISSUE_PROGRAMMES.items():
        for k in t[(t.tissue == own) & (t.readout == "D_can")].key:
            if k not in [r[1] for r in rows if r[0] == own]:
                rows.append((own, k))
    rows = [r for r in rows if np.isfinite(t[(t.tissue == r[0]) & (t.key == r[1]) & (t.readout == "D_can")].z_own).all()]
    nr = len(rows)
    blocks = (("D_can", GATE.name), ("D_free", PER_GENE.name), ("B1", PROFILE.name))
    # bp from the left edge: tissue of the group, row names, three blocks of eight tissues
    W, wt, wn, gap, rh, top, bot = WIDTH, 30, 190, 5, 8.25, 50, 3
    x0 = wt + wn + 3
    wc = (W - 10 - 2 * gap - x0) / 24  # 10 bp at the right for the name of the last tissue
    H = top + rh * nr + bot
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    vmax = 12.0
    cmap = plt.get_cmap(style.DIVERGING).copy()
    cmap.set_bad(NA_CELL)
    names = [program_label(s.loc[k, "name"], s.loc[k, "library"]) for _, k in rows]
    for b, (r_, title) in enumerate(blocks):
        xb = x0 + b * (8 * wc + gap)
        ax = heat_axes(fig, [xb / W, bot / H, 8 * wc / W, rh * nr / H], [TISSUE_NAME[x] for x in T], names if b == 0 else None)
        M = np.array([[t[(t.readout == r_) & (t.tissue == own) & (t.key == k)][f"z_{x}"].iloc[0] for x in T] for own, k in rows], float)
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        prev = rows[0][0]
        for i, (own, _) in enumerate(rows):  # a white rule between the groups of two tissues
            if own != prev:
                ax.axhline(i - 0.5, color="#ffffff", lw=1.2)
                prev = own
        for i, (own, k) in enumerate(rows):
            rr = t[(t.readout == r_) & (t.tissue == own) & (t.key == k)].iloc[0]
            own_outline(ax, T.index(own), i, bool((rr.q_contrast < 0.05) and (rr.contrast > 0)))
        fig.text((xb + 4 * wc) / W, 1 - 2 / H, title, ha="center", va="top", fontsize=S_TITLE)
    for own in dict.fromkeys(o for o, _ in rows):  # the tissue of each group, once, at the left
        idx = [i for i, (o, _) in enumerate(rows) if o == own]
        fig.text(2 / W, (bot + rh * (nr - (idx[0] + idx[-1] + 1) / 2)) / H, TISSUE_NAME[own].replace(" ", "\n"), fontsize=S_TICK, weight="bold", ha="left",
                 va="center", linespacing=1.05)
    color_key(fig, [2 / W, 1 - 30 / H, 84 / W, 5 / H], vmax, f"set z against matched genes\ncolors end at ±{vmax:.0f}")
    # key of the outlines and of the cells without a value, in bp from the top left corner
    kx, ky, kw, kh = 98, (7.5, 16.5, 25.5), 7, 5.5
    key = fig.add_axes([0, 0, 1, 1], facecolor="none")
    key.set_xlim(0, W)
    key.set_ylim(H, 0)
    key.axis("off")
    for y, text in zip(ky, ("own tissue above the others, q < 0.05", "own tissue, not above", f"fewer than {RU.MIN_SET} genes: no z")):
        key.text(kx + kw + 3, y, text, fontsize=S_TEXT, ha="left", va="center")
    key.add_patch(Rectangle((kx, ky[0] - kh / 2), kw, kh, **OUTLINE[True]))
    key.add_patch(Rectangle((kx, ky[1] - kh / 2), kw, kh, **OUTLINE[False]))
    key.add_patch(Rectangle((kx, ky[2] - kh / 2), kw, kh, fc=NA_CELL, ec="none"))
    return save(fig, "supp_programs_by_tissue/tissue")


def panel_distance():
    """Two-species models of four tissues against the divergence time of the partner species from human: mean effect of the
    programs that recur above and below the prior, and agreement of the model's program order with the other tissues.
    Symbol and color by tissue; filled with a solid line for the protein-feature gate, open with a dashed line for the
    per-gene gate. Models of one tissue at the same divergence time are averaged."""
    d = pd.read_csv(RU.OUT / "distance_cohorts.csv")
    d = d[d.pair & d.tissue.isin(LADDERS) & ~d.cohort.isin(["brain_m1_hmouse"])]
    W, H, left, aw, gap, top, bottom = WIDTH, 108, 40, 140, 56, 5, 26
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    axs = [fig.add_axes([(left + j * (aw + gap)) / W, bottom / H, aw / W, (H - top - bottom) / H]) for j in range(2)]
    ylabels = ("mean effect of the\nrecurrent programs (SD)", "agreement with the\nprogram order of the\nother tissues (ρ)")
    for ax, col, ylab in zip(axs, ("pole", "rho_consensus"), ylabels):
        for lad in LADDERS:
            m = style.TISSUE[lad]
            for gate, filled in (("canonical", True), ("per-gene", False)):
                x = d[(d.tissue == lad) & (d.gate == gate)].sort_values("divergence_my")
                x = x.groupby("divergence_my", as_index=False).mean(numeric_only=True)
                kw = dict(color=m.color, marker=m.marker, ms=3.4, lw=0.9, mfc=m.color if filled else "white", mew=0.7, ls="-" if filled else "--")
                if col == "pole":
                    ax.plot(x.divergence_my, x.above_pole, **kw)
                    ax.plot(x.divergence_my, x.below_pole, **kw)
                else:
                    ax.plot(x.divergence_my, x.rho_consensus, **kw)
        ax.set_xscale("log")
        ax.set_xticks([6.4, 29, 90, 429], ["6", "29", "90", "429"])
        ax.set_xlabel("divergence from human (million years)", labelpad=2)
        ax.set_ylabel(ylab, linespacing=1.1)
    axs[0].axhline(0, color=RULE, lw=style.LW_REF, zorder=0)
    axs[0].text(7, 0.57, "programs above the prior", fontsize=S_TEXT, va="bottom")
    axs[0].text(7, -0.43, "programs below the prior", fontsize=S_TEXT, va="top")
    axs[0].set_ylim(-0.56, 0.68)  # holds every model (lowest -0.35, highest 0.48) and the two notes
    axs[0].set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    axs[1].set_ylim(0.4, 1.0)  # holds every model (lowest 0.44, highest 0.91)
    h = [Line2D([], [], color=style.TISSUE[lad].color, marker=style.TISSUE[lad].marker, ms=3.4, lw=0.9, label=TISSUE_NAME[lad]) for lad in LADDERS]
    h += [Line2D([], [], color=INK, marker="o", ms=3.4, lw=0.9, label=GATE.name),
          Line2D([], [], color=INK, marker="o", mfc="white", mew=0.7, ls="--", ms=3.4, lw=0.9, label=PER_GENE.name)]
    axs[1].legend(handles=h, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=S_TICK, handlelength=2.6, handletextpad=0.5, borderaxespad=0,
                  labelspacing=0.5)
    return save(fig, "supp_programs_recurrent/distance")


def panel_genes(k_up=24, k_dn=14):
    """Named genes called on the same side of the prior, with both gates, in three or more of the eight tissues: the 24 most
    recurrent above (left) and the 14 most recurrent below (right). A dot for every tissue in which the gene was tested;
    beside each block the protein identity and the number of cohorts in which the gene is called, of those that tested it."""
    R = pd.read_csv(RU.OUT / "genes" / "recurrent.csv", index_col=0)
    R = R[~R.is_mt]
    K = pd.read_parquet(RU.OUT / "genes" / "calls.parquet")
    prim = [RU.PRIMARY[t] for t in RU.TISSUES]
    up = R[R.direction == "above"].sort_values(["n_tissues", "n_cohorts_up", "int_can"], ascending=False).head(k_up)
    dn = R[R.direction == "below"].sort_values(["n_tissues", "n_cohorts_down", "int_can"], ascending=[False, False, True]).head(k_dn)
    th = K[K.cohort.isin(prim) & K.hsapiens_gene_id.isin(up.index.union(dn.index))]
    T = RU.TISSUES
    # bp: gene names, eight tissues, two columns of numbers; the two blocks side by side
    W, wn, wc, wi, wk, rh, top, bot = WIDTH, 34, 16, 30, 32, 7.8, 50, 3
    block = wn + 8 * wc + wi + wk
    H = top + rh * max(len(up), len(dn)) + bot
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    for b, (G, title) in enumerate(((up, "recurrently above the prior"), (dn, "recurrently below the prior"))):
        xb, nr = 2 + b * (W - 4 - block), len(G)
        box = [(xb + wn) / W, (H - top - rh * nr) / H, 8 * wc / W, rh * nr / H]
        ax = heat_axes(fig, box, [TISSUE_NAME[x] for x in T], list(G.symbol))
        ax.set_yticklabels(G.symbol, fontsize=S_TICK, style="italic")
        for i, g in enumerate(G.index):
            for j, tt in enumerate(T):
                x = th[(th.hsapiens_gene_id == g) & (th.tissue == tt)]
                if not len(x):
                    continue
                x = x.iloc[0]
                if x.call != 0:
                    ax.scatter(j, i, s=16, c=UP if x.call > 0 else DOWN, edgecolors="none")
                else:
                    ax.scatter(j, i, s=5, c=NO_CALL, edgecolors="none")
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(nr - 0.5, -0.5)
        ax2 = fig.add_axes([box[0] + box[2], box[1], (wi + wk) / W, box[3]])
        ax2.set_xlim(0, wi + wk)
        ax2.set_ylim(nr - 0.5, -0.5)
        ax2.axis("off")
        xi, xk = wi / 2 + 3, wi + wk / 2
        ax2.text(xi, -0.8, IDENTITY.name.replace(" ", "\n", 1), fontsize=S_TEXT, ha="center", va="bottom", linespacing=1.05)
        ax2.text(xk, -0.8, "cohorts\ncalled /\ntested", fontsize=S_TEXT, ha="center", va="bottom", linespacing=1.05)
        for i, (_, r) in enumerate(G.iterrows()):
            ax2.text(xi, i, f"{r.pi_raw:.2f}", fontsize=S_TEXT, ha="center", va="center")
            ax2.text(xk, i, f"{int(r.n_cohorts_up if r.direction == 'above' else r.n_cohorts_down)}/{int(r.n_cohorts_tested)}", fontsize=S_TEXT,
                     ha="center", va="center")
        fig.text((xb + wn + 4 * wc) / W, 1 - 2 / H, title, ha="center", va="top", fontsize=S_TITLE)
    h = [Line2D([], [], ls="", marker="o", ms=4, mew=0, color=UP, label="called above the prior with both gates"),
         Line2D([], [], ls="", marker="o", ms=4, mew=0, color=DOWN, label="called below the prior with both gates"),
         Line2D([], [], ls="", marker="o", ms=2.2, mew=0, color=NO_CALL, label="tested, not called"),
         Line2D([], [], ls="", marker="", label="no dot: not tested in the tissue")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=((xb + wn) / W, (H - top - rh * len(dn) - 16) / H), fontsize=S_TICK, handlelength=1.0,
               handletextpad=0.4, borderaxespad=0, labelspacing=0.5)
    return save(fig, "supp_programs_named_genes/genes")


# panel name in figures/panels -> the function that draws it
DRAW = {"supp_programs_recurrent/atlas": panel_atlas, "supp_programs_recurrent/distance": panel_distance,
        "supp_programs_by_tissue/tissue": panel_tissue, "supp_programs_named_genes/genes": panel_genes}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E1 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all four). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


if __name__ == "__main__":
    if sys.argv[1:] == ["--rows-table"]:
        write_atlas_rows()
    else:
        for path in draw_panels(sys.argv[1:] or None):
            print(path)
