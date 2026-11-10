"""The house style of the figures as code: type, lines, colors, names, and the one function panels are saved with.

Every panel script imports this module (``from figures import style``), calls ``style.apply()`` once and saves through
``style.save_panel``. A panel is drawn at the size it has on the page of the paper (scale 1.0 in its figure spec), so
the font sizes below are the printed sizes. The checks of ``figures/check.py`` enforce the result: no text under
5.0 pt, embedded fonts, one font family, no internal names or British spelling in the artwork.

    from figures import style
    style.apply()
    fig, ax = plt.subplots(figsize=(style.inch(232), style.inch(150)))
    ax.plot(x, y, color=style.READOUT["delta_alpha"].color, marker=style.READOUT["delta_alpha"].marker)
    style.save_panel(fig, "main_validation/prespecified_tests")        # figures/panels/main_validation/prespecified_tests.pdf

Panels go to ``figures/panels/<family>/<name>.pdf``; ``<family>`` is the key of the figure that owns the panel, without
its part suffix. Set ``CHIMERA_FIGURES_OUT=<folder>`` for a dry run: panels are then written under that folder and
nothing in the repository changes.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent  # .../figures
OUT = Path(os.environ.get("CHIMERA_FIGURES_OUT", ROOT))
PANELS = OUT / "panels"
PANEL_PREVIEW = OUT / "_work" / "panels"  # PNG previews of panels, not tracked

# ----------------------------------------------------------------------------------------------- page and type
TEXTWIDTH = 486.0  # bp, 6.75 in: a figure across both columns, and every supplementary figure
COLUMNWIDTH = 234.0  # bp, 3.25 in
LANDSCAPE = 648.0  # bp, 9.0 in
GAP = 7.0  # bp between panels of a row, as the layout tool places them

FONT = "DejaVu Sans"
SIZE_TITLE = 7.0  # column, cohort or tissue heading (bold only for a group heading)
SIZE_LABEL = 6.5  # axis labels
SIZE_TICK = 6.0  # tick labels, legends
SIZE_TEXT = 5.5  # text inside the plot: annotations, cell values, gene names, rows of long lists
SIZE_FLOOR = 5.0  # nothing smaller at print size (single characters such as subscripts: 3.5)

INK, MUTED, RULE = "#222222", "#777777", "#BBBBBB"
NULL_BAND = "#E5E5E5"  # seed floors, label nulls
LW_AXES, LW_TICK, LW_DATA, LW_REF, LW_CI = 0.6, 0.5, 1.0, 0.6, 1.1


def inch(bp: float) -> float:
    """Big points to inches, for ``figsize``."""
    return bp / 72.0


def apply() -> None:
    """Set matplotlib for panels drawn at print size. Call once, before the first figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": FONT, "mathtext.fontset": "dejavusans", "mathtext.default": "it",
        "font.size": SIZE_LABEL, "axes.labelsize": SIZE_LABEL, "axes.titlesize": SIZE_TITLE, "figure.titlesize": SIZE_TITLE,
        "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK, "legend.fontsize": SIZE_TICK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": LW_AXES,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": LW_TICK, "ytick.major.width": LW_TICK,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": False, "legend.frameon": False, "legend.borderpad": 0.2, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "lines.linewidth": LW_DATA, "lines.markersize": 3.5, "patch.linewidth": 0.4, "axes.unicode_minus": True,
    })


# ----------------------------------------------------------------------------------------------- colors and markers
class Mark(NamedTuple):
    color: str
    marker: str
    name: str  # the name on the figure
    open: bool = False  # drawn with a white face


# One entity, one color and marker, in every figure. Identity never rests on color alone.
READOUT = {
    "delta_alpha": Mark("#D55E00", "o", "Δα, protein-feature gate"),  # "Δα" when it is the only gate on the figure
    "delta_alpha_per_gene": Mark("#882255", "h", "Δα, per-gene gate"),
    "count_evidence": Mark("#0072B2", "s", "count evidence $S_g$"),
    "direct_margin": Mark("#009E73", "D", "labeled direct margin, no model"),
    "profile_correlation": Mark("#222222", "^", "profile correlation"),
    "rank_matched_level": Mark("#666666", "v", "rank-matched level per cell type"),
    "specificity_similarity": Mark("#999999", "<", "specificity similarity"),
    "latent_cluster_correlation": Mark("#CC79A7", "P", "profile correlation, clusters of Chimera's latent"),
    "gene_features": Mark("#E69F00", "X", "gene features alone"),
    "fold_change": Mark("#E69F00", "X", "fold change", open=True),
    "expression": Mark("#999999", "<", "expression level", open=True),
    "samap": Mark("#44AA99", "d", "SAMap"),
    "other_tool": Mark("#222222", "D", "gene score of another tool", open=True),  # SATURN, CAMEX, ...: named on the figure
    "protein_identity": Mark("#A6761D", "o", "protein identity π"),
    "dnds": Mark("#A6761D", "s", "dN/dS"),
    "loeuf": Mark("#A6761D", "^", "LOEUF"),
}
# Chimera readouts first, then direct baselines, other tools, sequence measures. Rows keep this order unless a panel is
# a ranking (and then its caption says "sorted by").
ORDER = ["count_evidence", "delta_alpha", "delta_alpha_per_gene", "direct_margin", "profile_correlation", "rank_matched_level",
         "specificity_similarity", "latent_cluster_correlation", "fold_change", "gene_features", "expression", "samap", "other_tool",
         "protein_identity", "dnds", "loeuf"]

ABOVE, BELOW = "#D55E00", "#0072B2"  # above / below the protein prior; the red and the blue end of a diverging map
DIVERGING, SEQUENTIAL = "RdBu_r", "Blues"  # Δα, z and effect sizes centered on 0; scores from 0 to 1
COUNT_EVIDENCE, IMPUTED, TREND = "#2b6ca3", "#9a9a9a", "#c8501e"  # genes with count evidence, without it, median trend
BACKGROUND_GENES = "#BBBBBB"

SPECIES = {"human": "#222222", "macaque": "#0072B2", "monkey": "#0072B2", "mouse": "#D55E00", "zebrafish": "#009E73",
           "chimpanzee": "#56B4E9", "marmoset": "#E69F00"}
TISSUE = {  # one point per cohort: color and marker by tissue
    "liver": Mark("#0072B2", "o", "Liver"), "testis": Mark("#D55E00", "^", "Testis"), "bone marrow": Mark("#CC79A7", "s", "Bone marrow"),
    "kidney": Mark("#009E73", "D", "Kidney"), "islet": Mark("#E69F00", "v", "Islet"), "eye": Mark("#5B3E80", "P", "Eye"),
    "cortex": Mark("#56B4E9", "X", "Cortex"), "lung": Mark("#882255", "h", "Lung"), "muscle": Mark("#7F7F7F", "<", "Skeletal muscle"),
    "tumor": Mark("#222222", "o", "Lung-tumor myeloid", open=True),
}
LIVER_CELL_TYPE = {"hepatocytes": "#E69F00", "endothelial": "#56B4E9", "stellate and fibroblasts": "#009E73",
                   "Kupffer cells and macrophages": "#CC79A7", "other": "#BBBBBB"}
GENE_THEME = {"urea cycle and gluconeogenesis": "#0072B2", "mitochondrially encoded": "#5B3E80", "supporting tissue": "#D55E00",
              "housekeeping": "#7F7F7F"}
TRANSFER = {"transfers": "#0072B2", "does not transfer": "#C9C9C9", "opposite direction": "#E69F00"}

# ----------------------------------------------------------------------------------------------- names
# The seven liver and testis cohorts. Other cohorts take the label of their registry entry through cohort_name().
COHORT = {
    "liv_hmonkey": "Liver Human+Monkey", "liv_hmouse": "Liver Human+Mouse", "liv_hzeb": "Liver Human+Zebrafish", "liv_4sp": "Liver 4 species",
    "test_hmac": "Testis Human+Macaque", "test_hmouse": "Testis Human+Mouse", "test_3sp": "Testis 3 species",
}
_SPELL = [(r"\bH\+", "Human+"), (r"\b4-sp\b", "4 species"), (r"\b3-sp\b", "3 species"), (r"\+Mac\b", "+Macaque"), (r"\+Zeb\b", "+Zebrafish"),
          (r"\bTumour\b", "Tumor"), (r"\btumour\b", "tumor")]


def cohort_name(key_or_label: str, lower: bool = False) -> str:
    """The name of a cohort on a figure: tissue first, species set with capitals and a plus, nothing abbreviated
    ("Liver Human+Mouse", "Liver 4 species"). Takes a cohort key of ``COHORT`` or a label such as "Liver H+Mouse".
    ``lower=True`` for use inside a longer phrase ("Δα (liver Human+Mouse)")."""
    s = COHORT.get(key_or_label, key_or_label)
    for pat, rep in _SPELL:
        s = re.sub(pat, rep, s)
    return s[0].lower() + s[1:] if lower else s


SIGNIFICANT = "*"  # the one mark for q < 0.05

# ----------------------------------------------------------------------------------------------- no machine paths
_PTEX_NAME = re.compile(rb"/PTEX\.FileName\s*\(((?:[^()\\]|\\.)*)\)")
MACHINE_PATH = re.compile(rb"ictstr\d*|/lustre/|/home/icb|kemal|/Users/")  # nothing of a working machine in a tracked file


def blank_file_names(pdf: Path) -> int:
    """A PDF made with pdfTeX records the path of every PDF it includes (``/PTEX.FileName``), and that entry travels
    into any file that includes it in turn. Overwrite such paths with blanks of the same length (nothing else in the
    file moves). Returns the number of entries found."""
    data = Path(pdf).read_bytes()
    out, n = _PTEX_NAME.subn(lambda m: m.group(0)[: m.start(1) - m.start(0)] + b" " * len(m.group(1)) + b")", data)
    if n and out != data:
        Path(pdf).write_bytes(out)
    return n


def machine_paths(path: Path, deep: bool = True) -> list:
    """Traces of a working machine in a file (``MACHINE_PATH``): in its bytes and, for a PDF with ``deep``, also in its
    uncompressed form (object streams and stream data opened with qpdf, when qpdf is on PATH). There must be none."""
    import shutil
    import subprocess
    import tempfile

    path = Path(path)
    hits = {m.group(0).decode("latin-1") for m in MACHINE_PATH.finditer(path.read_bytes())}
    qpdf = shutil.which("qpdf")
    if deep and qpdf and path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "u.pdf"
            subprocess.run([qpdf, "--stream-data=uncompress", "--object-streams=disable", "--decode-level=generalized", str(path), str(tmp)],
                           capture_output=True)
            if tmp.exists():
                hits |= {m.group(0).decode("latin-1") for m in MACHINE_PATH.finditer(tmp.read_bytes())}
    return sorted(hits)


# ----------------------------------------------------------------------------------------------- saving
_METADATA = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}  # the same file on every run


def panel_path(name: str) -> Path:
    """figures/panels/<family>/<name>.pdf for "family/name"."""
    if not re.fullmatch(r"[a-z0-9_]+/[a-z0-9_]+", name):
        raise ValueError(f"panel name {name!r}: use 'family/name', lower case, words joined by '_'")
    return PANELS / (name + ".pdf")


def save_panel(fig, name: str, exact: bool = False, preview: bool = True, close: bool = True) -> Path:
    """Save a panel as ``figures/panels/<family>/<name>.pdf`` (vector, fonts embedded as text).

    ``exact=False`` crops the page to the drawn content. ``exact=True`` keeps the figure size: use it when the axes of the
    panel must line up with a neighbor, and place it with ``X(...)`` in the figure spec. A panel carries no panel letter."""
    import matplotlib.pyplot as plt

    path = panel_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if exact else dict(bbox_inches="tight", pad_inches=0.02)
    with plt.rc_context({"savefig.bbox": "standard"} if exact else {}):  # exact: the figure size, whatever the caller's defaults
        fig.savefig(path, format="pdf", metadata=_METADATA, **kw)
        blank_file_names(path)
        if preview:
            png = PANEL_PREVIEW / (name + ".png")
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, format="png", dpi=200, **kw)
    if close:
        plt.close(fig)
    return path


def save_axes(fig, name: str, axes: list, pad: float = 0.02) -> Path:
    """One panel of a figure that draws several: only these axes, cropped to them. The figure stays open."""
    from matplotlib.transforms import Bbox

    others = [a for a in fig.axes if a not in axes and a.get_visible()]
    for a in others:
        a.set_visible(False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = Bbox.union([a.get_tightbbox(r) for a in axes]).transformed(fig.dpi_scale_trans.inverted())
    path = panel_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata=_METADATA, bbox_inches=box.padded(pad))
    blank_file_names(path)
    for a in others:
        a.set_visible(True)
    return path


def copy_panel(src: Path, name: str) -> Path:
    """Take a finished panel file made elsewhere as ``figures/panels/<family>/<name>.pdf``. Paths that pdfTeX recorded
    in it are blanked; a file that still carries a trace of the machine it was made on is refused."""
    import shutil

    path = panel_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, path)
    blank_file_names(path)
    found = machine_paths(path)
    if found:
        path.unlink()
        raise ValueError(f"{src}: carries {', '.join(found)}; not taken as a panel")
    return path


# ----------------------------------------------------------------------------------------------- old names
def panel_for(old_name: str) -> str:
    """The panel ("family/name") that a script still knows under its former file name; see ``panel_names.tsv``. A script
    that calls ``save_panel`` with the new name does not need this and its line leaves the table."""
    with (ROOT / "panel_names.tsv").open() as fh:
        table = {r["old_name"].lower(): r["panel"] for r in csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t")}
    key = Path(old_name).stem.lower()
    if key not in table:
        raise KeyError(f"panel {old_name!r} has no entry in figures/panel_names.tsv: save it with style.save_panel('family/name')")
    return table[key]
