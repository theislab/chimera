"""Shared paths, styles and checks for the figure panels drawn by the analysis tasks.

Every panel is one PDF in ``figures/panels/<family>/`` (fonts embedded as TrueType, no Type 3), saved through
``figures.style``; the figures of the paper are composed from the panels by ``figures/build.sh``. A PNG preview of each
panel goes to ``results/analysis/T8_4/preview/``. Scripts read result tables only; no model is loaded.

``save``, ``save_axes`` and ``copy_pdf`` take the former file name of a panel and look up its place in
``figures/panel_names.tsv``; a name that is not in the table is an error. New code calls
``figures.style.save_panel(fig, "family/name")`` directly. Set ``CHIMERA_FIGURES_OUT=<folder>`` for a dry run.

Run all from ``chimera/``::

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T8_4_figures.make_all
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis"
PREVIEW = RES / "T8_4" / "preview"

sys.path.insert(0, str(REPO / "notebooks"))
import _figure_design as fd  # noqa: E402

sys.path.insert(0, str(REPO))
from figures import style  # noqa: E402


def _poppler() -> Path | None:
    """Folder of the poppler tools for the font check: POPPLER_BIN, else PATH, else the figures environment beside this one."""
    if os.environ.get("POPPLER_BIN"):
        return Path(os.environ["POPPLER_BIN"])
    found = shutil.which("pdffonts")
    if found:
        return Path(found).parent
    beside = Path(sys.prefix).parent / "chimera_figures_env" / "bin"
    return beside if (beside / "pdffonts").exists() else None


POPPLER = _poppler()

# One colour and marker per readout, shared with X6 so every validation panel reads the same. Labels are the
# reader names of the manuscript; the internal codes (keys) never reach a figure.
# Okabe-Ito hues; markers differ as well, so identity never rests on colour alone.
STY = {
    "D": ("#D55E00", "o", "Δα (Chimera)"),
    "S": ("#0072B2", "s", "count-evidence margin $S_g$ (Chimera)"),
    "Sdir": ("#009E73", "D", "direct margin, cell-type labels, no model"),
    "B1b": ("#222222", "^", "profile correlation"),
    "B1c": ("#666666", "v", "rank-matched level per cell type"),
    "B2b": ("#999999", "<", "specificity similarity"),
    "B1z": ("#CC79A7", "P", "profile correlation, clusters of Chimera's latent"),
    "B3a": ("#E69F00", "X", "gene features alone"),
}
INK = "#222222"
MUTED = "#777777"
RULE = "#BBBBBB"

COHORT = dict(style.COHORT)  # "Liver Human+Mouse", "Liver 4 species", ...: the names every figure uses
GOLD = {
    "H1": "liver expression level\n(EVEE, 4 mammals)",
    "H2": "testis expression level\n(EVEE, 4 mammals)",
    "H3": "liver cell-type pattern\n(human vs mouse atlases)",
    "H4": "testis cell-type pattern\n(8 held-out species)",
}
GOLD_SHORT = {"H1": "liver level", "H2": "testis level", "H3": "liver pattern", "H4": "testis pattern"}
RHO_P = "partial rank correlation with independent data, $\\rho_p$"


def setup() -> None:
    fd.apply_design()
    plt.rcParams.update({"axes.titlesize": 7, "legend.fontsize": 6.5})
    PREVIEW.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str, exact: bool = False) -> Path:
    """Panel PDF to ``figures/panels`` (``name``: its former file name, see the module docstring), PNG preview to
    results; returns the PDF path.

    The page is cropped to the drawn content with a 0.02 in pad. ``exact`` is kept for the existing calls and changes
    nothing: under the design of ``_figure_design.apply_design`` (``savefig.bbox`` tight) those panels were always saved
    cropped. They still line up in a row because their axes are drawn at fixed places; the figure spec places them with
    ``X(...)``.
    """
    PREVIEW.mkdir(parents=True, exist_ok=True)
    pdf = style.save_panel(fig, style.panel_for(name), exact=False, preview=False, close=False)
    fig.savefig(PREVIEW / f"{name.lower()}.png", format="png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf


def save_axes(fig: plt.Figure, name: str, axes: list, out_dir: Path | None = None, pad: float = 0.02) -> Path:
    """One panel of a many-panel figure as its own PDF: only these axes, cropped to them. With ``out_dir`` the file is
    written there under ``name``; without it, to ``figures/panels`` at the place of its former name."""
    if out_dir is None:
        return style.save_axes(fig, style.panel_for(name), axes, pad=pad)
    others = [a for a in fig.axes if a not in axes and a.get_visible()]
    for a in others:
        a.set_visible(False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = Bbox.union([a.get_tightbbox(r) for a in axes]).transformed(fig.dpi_scale_trans.inverted())
    pdf = out_dir / f"{name.lower()}.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches=box.padded(pad))
    for a in others:
        a.set_visible(True)
    return pdf


def copy_pdf(src: Path, name: str) -> Path:
    """Take a panel made by its own task as the panel that had the file name ``name``."""
    dst = style.copy_panel(src, style.panel_for(name))
    png = src.with_suffix(".png")
    if png.exists():
        PREVIEW.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, PREVIEW / f"{name}.png")
    return dst


def check_pdf(path: Path) -> dict:
    """Fonts embedded and none of Type 3; no internal run name on the panel. Without the poppler tools the check is
    skipped (the binding checks are those of ``figures/build.sh``)."""
    if POPPLER is None:
        return dict(file=path.name, n_fonts="", type3="", unembedded="", panc="", ok=True, note="poppler not found: not checked")
    fonts = subprocess.run([str(POPPLER / "pdffonts"), str(path)], capture_output=True, text=True).stdout
    rows = [ln for ln in fonts.splitlines()[2:] if ln.strip()]
    type3 = [ln for ln in rows if " Type 3 " in ln]
    unembedded = [ln for ln in rows if len(ln.split()) >= 5 and ln.split()[-5] != "yes"]
    text = subprocess.run([str(POPPLER / "pdftotext"), str(path), "-"], capture_output=True, text=True).stdout
    raw = path.read_bytes()
    bad = sorted(set(re.findall(r"panc_\w*", text))) + (["panc_ (bytes)"] if b"panc_" in raw else [])
    return dict(file=path.name, n_fonts=len(rows), type3=len(type3), unembedded=len(unembedded),
                panc=";".join(bad), ok=not type3 and not unembedded and not bad)
