"""Shared paths, names and style of the scripts in this folder that redraw panels at print size.

The scripts ``alpha``, ``categories``, ``integration_scores``, ``integration_umap``, ``gate_network``, ``atlas`` and
``residue`` draw the panels of the supplementary figures on the gate, the cell integration, the gene categories, the
gate network, the protein-space atlas and the residue-level analysis. Each reads files that already exist (stored
model outputs under ``training/runs/``, the processed data objects, tables under ``notebooks/cache/``) and writes its
panels to ``figures/panels/<figure key>/``. Nothing is trained and no notebook is executed. A script stops when an
input is missing (``need``).

Run from the repository root, in the analysis environment::

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.<script> [panel ...]

then ``bash figures/build.sh`` composes the figures. ``CHIMERA_ROOT`` and ``CHIMERA_FIGURES_OUT`` have the meaning
they have in ``figures/compose.py`` and ``figures/style.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from figures import style

FIGURES = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("CHIMERA_ROOT", FIGURES.parent))
PANELS = style.PANELS
NB = REPO / "notebooks"
NB_FIGURES = NB / "figures"
NB_CACHE = NB / "cache"
RUNS = REPO / "training" / "runs"
DATA = REPO / "data" / "processed"

# Stored runs behind these figures (run folder names are fixed identifiers).
RUN = {
    "liver_hmonkey": "iter90-150ep-s42",
    "liver_hmouse": "xds-hmouse-iter90-150ep-s42",
    "liver_hzeb": "xds-hzeb-iter85-s42",
    "liver_4sp": "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42",
    "testis_hmac": "testis-hmac-iter109-pclamp085-200ep-s42",
    "testis_hmouse": "testis-hmouse-iter111-pclamp-coc-200ep-s42",
    "testis_3sp": "testis-unified-iter115-pclamp-coc-100ep-s42",
}
# Data objects of the same cohorts ("pancreas" in one file name is a legacy key of the liver data).
ADATA = {
    "liver_hmonkey": "camex_human_monkey.h5ad", "liver_hmouse": "camex_human_mouse.h5ad",
    "liver_hzeb": "camex_human_zebrafish.h5ad", "liver_4sp": "camex_pancreas_unified.h5ad",
    "testis_hmac": "camex_testis_human_macaque.h5ad", "testis_hmouse": "camex_testis_human_mouse.h5ad",
    "testis_3sp": "camex_testis_unified.h5ad",
}
# Keys used in notebook file names and cached tables -> the keys above ("panc" is the same legacy key of liver).
NOTEBOOK_KEY = {"panc_hmonkey": "liver_hmonkey", "panc_hmouse": "liver_hmouse", "panc_hzeb": "liver_hzeb", "panc_4sp": "liver_4sp",
                "test_hmac": "testis_hmac", "test_hmouse": "testis_hmouse", "test_3sp": "testis_3sp"}

# Names shown on figures. Cohort names come from the style module, so that every figure writes them the same way.
_STYLE_KEY = {"liver_hmonkey": "liv_hmonkey", "liver_hmouse": "liv_hmouse", "liver_hzeb": "liv_hzeb", "liver_4sp": "liv_4sp",
              "testis_hmac": "test_hmac", "testis_hmouse": "test_hmouse", "testis_3sp": "test_3sp"}
COHORT = {k: style.cohort_name(v) for k, v in _STYLE_KEY.items()}                 # "Liver Human+Monkey", "Liver 4 species", ...
COHORT_SHORT = {k: v.split(" ", 1)[1] for k, v in COHORT.items()}                 # "Human+Monkey", "4 species", ...
METHOD = {"PCA": "PCA", "Harmony": "Harmony", "scVI-sp": "scVI (species batch)", "scVI-donor": "scVI (donor batch)",
          "sysVI": "sysVI", "scANVI": "scANVI", "Chimera": "Chimera"}
# The 72 inputs of the gate network: eight single numbers and 64 protein-language components.
GATE_INPUT = {
    "log_mean_expr": "log mean expression", "log_expr_var": "log expression variance", "frac_nonzero": "detection rate",
    "mean_identity": "mean ortholog identity", "std_identity": "ortholog identity, SD", "min_identity": "ortholog identity, minimum",
    "max_identity": "ortholog identity, maximum", "esm2_intersp_var": "ESM-2 distance between species",
}
GATE_GROUP = {"ESM-2 PCs": "ESM-2 components", "esm2_intersp_var": "ESM-2 distance between species", "mean_identity": "mean ortholog identity",
              "expression": "expression", "identity_spread": "identity spread"}
# One color per group of gate inputs, wherever shares of the groups are drawn (attribution shares, strips of the residue cards).
GATE_GROUP_COLOR = {"ESM-2 PCs": "#cccccc", "esm2_intersp_var": "#56B4E9", "mean_identity": "#0072B2", "expression": "#D55E00",
                    "identity_spread": "#888888"}
# Gene categories of ``notebooks/_gene_categories.py``: key -> name on a figure.
CATEGORY = {
    "apoptosis": "Apoptosis", "autophagy": "Autophagy", "cell_cycle": "Cell cycle", "chaperones_hsp": "Chaperones (HSP)",
    "coagulation_serum": "Coagulation and serum", "complement": "Complement", "cytochrome_p450": "Cytochrome P450",
    "cytokines_chemokines": "Cytokines and chemokines", "cytoskeleton_actin": "Actin cytoskeleton",
    "cytoskeleton_intermediate": "Intermediate filaments", "cytoskeleton_microtubule": "Microtubules",
    "dna_replication_repair": "DNA replication and repair", "ecm_collagen": "ECM: collagens", "ecm_other": "ECM: other",
    "glutathione_redox": "Glutathione and redox", "hedgehog_signaling": "Hedgehog signaling", "histone": "Histones",
    "housekeeping": "Housekeeping", "immediate_early": "Immediate early genes", "interferon_response": "Interferon response",
    "kinases_signaling": "Signaling kinases", "metabolism_glycolysis": "Glycolysis", "metabolism_lipid": "Lipid metabolism",
    "mitochondrial_dna_encoded": "Mitochondrially encoded", "notch_signaling": "Notch signaling",
    "nuclear_receptors": "Nuclear receptors", "oxphos_nuclear_encoded": "OXPHOS, nuclear-encoded", "proteasome": "Proteasome",
    "ribosomal_mt": "Mitochondrial ribosome", "ribosomal_structural": "Cytosolic ribosome", "ribosome_biogenesis": "Ribosome biogenesis",
    "splicing_factors": "Splicing factors", "tf_forkhead_etc": "Forkhead and other TFs", "tf_homeobox": "Homeobox TFs",
    "tf_zinc_finger": "Zinc-finger TFs", "tgfb_bmp_signaling": "TGF-β/BMP signaling",
    "translation_elongation": "Translation elongation", "translation_initiation": "Translation initiation",
    "wnt_signaling": "Wnt signaling", "mhc_class_I": "MHC class I", "mhc_class_II": "MHC class II",
}
INK, MUTED, RULE = "#222222", "#777777", "#BBBBBB"
GREY_BAR = "#CCCCCC"
UP, DOWN = "#D55E00", "#0072B2"   # above and below the prior

SIZE = dict(label=7.0, tick=6.5, legend=6.5, text=5.8, title=7.0)   # pt at print size; nothing under 5.5


def gate_input(name: str) -> str:
    """Name of one gate input on a figure: 'PC13' -> 'ESM-2 component 13'."""
    if name in GATE_INPUT:
        return GATE_INPUT[name]
    if name.upper().startswith("PC") and name[2:].isdigit():
        return f"ESM-2 component {int(name[2:])}"
    return name


def category_names() -> dict:
    """Gene-category key -> name on a figure."""
    return dict(CATEGORY)


def category(key: str, names: dict | None = None) -> str:
    names = names if names is not None else CATEGORY
    return names.get(key, key.replace("_", " ").capitalize())


def need(*paths) -> None:
    """Stop when an input file is missing: nothing is recomputed in its place."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("missing input(s), nothing drawn:\n  " + "\n  ".join(missing))


def run_dir(cohort: str) -> Path:
    d = RUNS / RUN[cohort]
    need(d / "final_state.npz")
    return d


def design():
    """The notebooks' design module (axis ranges, density scatter), set to the type sizes of panels drawn at print size."""
    if str(NB) not in sys.path:
        sys.path.insert(0, str(NB))
    import _figure_design as fd

    fd.apply_design()
    mpl.rcParams.update({
        "font.size": SIZE["tick"], "axes.labelsize": SIZE["label"], "axes.titlesize": SIZE["title"],
        "xtick.labelsize": SIZE["tick"], "ytick.labelsize": SIZE["tick"], "legend.fontsize": SIZE["legend"],
        "savefig.dpi": 300, "pdf.fonttype": 42, "axes.unicode_minus": True,
    })
    return fd


def save_panel(fig: plt.Figure, family: str, name: str, exact: bool = False) -> Path:
    """One panel as ``figures/panels/<family>/<name>.pdf``: cropped to content, or at its exact size (``exact=True``)."""
    return style.save_panel(fig, f"{family}/{name}", exact=exact)
