"""Shared paths, styles and checks for the revision figure panels (T8.4).

Every panel is one PDF in ``chimera-manuscript/figures/revision/`` rendered through
``notebooks/_figure_design.py`` (fonts embedded as TrueType, no Type 3). A PNG preview of each panel
goes to ``results/revision/T8_4/preview/``. Scripts read result CSVs only; no model is loaded.

Run all from ``chimera/``::

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.T8_4_figures.make_all
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

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "revision"
MS = Path(os.environ.get("CHIMERA_MANUSCRIPT", REPO.parent / "chimera-manuscript"))
FIG_OUT = MS / "figures" / "revision"
TAB_OUT = MS / "tables"
PREVIEW = RES / "T8_4" / "preview"

sys.path.insert(0, str(REPO / "notebooks"))
import _figure_design as fd  # noqa: E402

# Poppler tools for the font and text checks (any env that ships them; override with POPPLER_BIN).
POPPLER = Path(os.environ.get("POPPLER_BIN", Path.home() / "tools/apps/mamba/envs/tardis_env/bin"))

# One colour and marker per readout, shared with X6 so every validation panel reads the same.
# Okabe-Ito hues; markers differ as well, so identity never rests on colour alone.
STY = {
    "D": ("#D55E00", "o", "Δα (Chimera)"),
    "S": ("#0072B2", "s", "count-evidence margin $S_g$ (Chimera)"),
    "Sdir": ("#009E73", "D", "direct margin, cell-type labels, no model"),
    "B1b": ("#222222", "^", "profile correlation (B1b)"),
    "B1c": ("#666666", "v", "level match per cell type (B1c)"),
    "B2b": ("#999999", "<", "specificity similarity (B2b)"),
    "B1z": ("#CC79A7", "P", "profile correlation, clusters of Chimera latent (B1z)"),
    "B3a": ("#E69F00", "X", "gene features alone (B3a)"),
}
INK = "#222222"
MUTED = "#777777"
RULE = "#BBBBBB"

COHORT = {
    "liv_hmonkey": "liver, human + monkey", "liv_hmouse": "liver, human + mouse",
    "liv_hzeb": "liver, human + zebrafish", "liv_4sp": "liver, 4 species",
    "test_hmac": "testis, human + macaque", "test_hmouse": "testis, human + mouse",
    "test_3sp": "testis, 3 species",
}
GOLD = {
    "H1": "liver expression level\n(EVEE, 4 mammals)",
    "H2": "testis expression level\n(EVEE, 4 mammals)",
    "H3": "liver cell-type pattern\n(human vs mouse atlases)",
    "H4": "testis cell-type pattern\n(8 held-out species)",
}
GOLD_SHORT = {"H1": "liver level", "H2": "testis level", "H3": "liver pattern", "H4": "testis pattern"}
RHO_P = "partial rank correlation with independent measure, $\\rho_p$"


def setup() -> None:
    fd.apply_design()
    plt.rcParams.update({"axes.titlesize": 7, "legend.fontsize": 6.5})
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str, exact: bool = False) -> Path:
    """PDF to the manuscript, PNG preview to results; returns the PDF path.

    ``exact``: keep the figure size instead of cropping to content, so a panel drawn with fixed axes
    lines up with its neighbours when F1_figures/compose.py places it with ``ink=False``.
    """
    if exact:
        pdf = FIG_OUT / f"{name.lower()}.pdf"
        fig.savefig(pdf, format="pdf")
        fig.savefig(PREVIEW / f"{name.lower()}.png", format="png", dpi=200)
        plt.close(fig)
        return pdf
    pdf = fd.save_figure(fig, name, formats=("pdf",), out_dir=FIG_OUT)[0]
    fd.save_figure(fig, name, formats=("png",), out_dir=PREVIEW, close=True)
    return pdf


def save_axes(fig: plt.Figure, name: str, axes: list, out_dir: Path | None = None, pad: float = 0.02) -> Path:
    """One panel of a many-panel figure as its own PDF, for compose.py: only these axes, cropped to them."""
    others = [a for a in fig.axes if a not in axes and a.get_visible()]
    for a in others:
        a.set_visible(False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = Bbox.union([a.get_tightbbox(r) for a in axes]).transformed(fig.dpi_scale_trans.inverted())
    pdf = (out_dir or FIG_OUT) / f"{name.lower()}.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches=box.padded(pad))
    for a in others:
        a.set_visible(True)
    return pdf


def copy_pdf(src: Path, name: str) -> Path:
    """Copy an existing panel made by its own task into the manuscript folder."""
    dst = FIG_OUT / f"{name}.pdf"
    shutil.copyfile(src, dst)
    png = src.with_suffix(".png")
    if png.exists():
        shutil.copyfile(png, PREVIEW / f"{name}.png")
    return dst


def check_pdf(path: Path) -> dict:
    """Fonts embedded and none of Type 3; no internal run name on the panel."""
    fonts = subprocess.run([str(POPPLER / "pdffonts"), str(path)], capture_output=True, text=True).stdout
    rows = [ln for ln in fonts.splitlines()[2:] if ln.strip()]
    type3 = [ln for ln in rows if " Type 3 " in ln]
    unembedded = [ln for ln in rows if len(ln.split()) >= 5 and ln.split()[-5] != "yes"]
    text = subprocess.run([str(POPPLER / "pdftotext"), str(path), "-"], capture_output=True, text=True).stdout
    raw = path.read_bytes()
    bad = sorted(set(re.findall(r"panc_\w*", text))) + (["panc_ (bytes)"] if b"panc_" in raw else [])
    return dict(file=path.name, n_fonts=len(rows), type3=len(type3), unembedded=len(unembedded),
                panc=";".join(bad), ok=not type3 and not unembedded and not bad)
