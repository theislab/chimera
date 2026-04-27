"""Broad-category vocabulary for paper-friendly Δα biology figures.

Companion to [`_gene_categories.py`](_gene_categories.py) (which has 43
narrow categories — too many for paper figures). This module rolls
those 43 into **10 paper-friendly umbrella categories** by reusing the
narrow `Category` definitions and grouping them. The classification
API is identical (``classify_broad`` / ``annotate_broad``) so notebook
04's simpler companion can be a near-identical drop-in.

**Design rule.** First, classify with the narrow vocabulary in
``_gene_categories.classify_extended``. Then map the narrow label to
its broad-category umbrella via ``NARROW_TO_BROAD``. This guarantees
that priority-ordering of narrow categories is preserved (so
``RPL13A`` resolves to ``ribosomal_structural`` → ``Translation & ribosome``,
not ``housekeeping``), and the broad system inherits all the curated
gene-set members of the narrow system without re-curating anything.

**Categories** (priority order = narrow-priority order):

0. ``Translation & ribosome``   — RPL/RPS, MRPL/MRPS, EIF, EEF, ribosome-biogenesis curated.
1. ``Mitochondria & OXPHOS``   — MT-*, NDUF, UQCR, ATP5, COX*, SDH*, mitophagy genes.
2. ``Chromatin & DNA processing``  — histones, splicing factors (SRSF/HNRNP/SNRP), DNA replication & repair (MCM/MSH/MLH/RAD51/BRCA/PARP/TOP*).
3. ``Cell cycle``              — CDK*, CCN*, AURK*, BUB*, PLK*, MKI67, FOXM1, E2F, BIRC.
4. ``Protein homeostasis``     — HSP*/DNAJ chaperones, proteasome (PSMA-G), autophagy (ATG, GABARAP, ULK).
5. ``Cytoskeleton & ECM``      — actin/tubulin/IF cytoskeleton, collagens, laminins, MMPs, ADAMTS, FN1, ELN.
6. ``Immunity & inflammation`` — MHC-I/II, complement (C1Q-C9, FCN), cytokines/chemokines (IL/TNF/CCL/CXCL), IFN response, immediate-early (FOS/JUN/EGR/NR4A).
7. ``Signaling pathways``      — kinase cascades (MAPK/AKT/PIK3/JAK/MTOR), Wnt, Notch, TGF-β/BMP, Hedgehog.
8. ``Transcription factors``   — homeobox, zinc-finger, forkhead/SOX/GATA/NFkB/STAT/ETS, nuclear receptors (NR/ESR/RAR/PPAR/AR).
9. ``Metabolism``              — cytochrome P450, glutathione/redox, glycolysis (HK/PKM/GAPDH/ENO/LDH), lipid (FASN/SCD/SREBF/LDLR/APO).

Genes that match no narrow category fall through to the ``other``
sentinel; that bin contains apoptosis, hemoglobin, coagulation_serum,
housekeeping, and unannotated genes — biology that doesn't fit
cleanly under any of the 10 paper-headline umbrellas. The
``classify_broad`` function returns the broad name directly.

If you need to re-tune the umbrellas, edit ``NARROW_TO_BROAD`` only —
the underlying gene-membership stays in ``_gene_categories.py``.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# 10 broad-category umbrella names (priority order matches narrow priority)
# ---------------------------------------------------------------------------

BROAD_NAMES: tuple[str, ...] = (
    "Translation & ribosome",
    "Mitochondria & OXPHOS",
    "Chromatin & DNA processing",
    "Cell cycle",
    "Protein homeostasis",
    "Cytoskeleton & ECM",
    "Immunity & inflammation",
    "Signaling pathways",
    "Transcription factors",
    "Metabolism",
)

OTHER = "other"


# ---------------------------------------------------------------------------
# Map narrow category → broad umbrella
# (narrow names from `_gene_categories.CATEGORIES`, in the same order)
# ---------------------------------------------------------------------------

NARROW_TO_BROAD: dict[str, str] = {
    # Translation & ribosome
    "ribosomal_structural":    "Translation & ribosome",
    "ribosomal_mt":            "Translation & ribosome",
    "ribosome_biogenesis":     "Translation & ribosome",
    "translation_initiation":  "Translation & ribosome",
    "translation_elongation":  "Translation & ribosome",
    # Mitochondria & OXPHOS
    "mitochondrial_dna_encoded": "Mitochondria & OXPHOS",
    "oxphos_nuclear_encoded":   "Mitochondria & OXPHOS",
    # Chromatin & DNA processing
    "histone":                  "Chromatin & DNA processing",
    "splicing_factors":         "Chromatin & DNA processing",
    "dna_replication_repair":   "Chromatin & DNA processing",
    # Cell cycle
    "cell_cycle":               "Cell cycle",
    # Protein homeostasis
    "chaperones_hsp":           "Protein homeostasis",
    "proteasome":               "Protein homeostasis",
    "autophagy":                "Protein homeostasis",
    # Cytoskeleton & ECM
    "cytoskeleton_actin":         "Cytoskeleton & ECM",
    "cytoskeleton_microtubule":   "Cytoskeleton & ECM",
    "cytoskeleton_intermediate":  "Cytoskeleton & ECM",
    "ecm_collagen":               "Cytoskeleton & ECM",
    "ecm_other":                  "Cytoskeleton & ECM",
    # Immunity & inflammation
    "mhc_class_I":              "Immunity & inflammation",
    "mhc_class_II":             "Immunity & inflammation",
    "complement":               "Immunity & inflammation",
    "cytokines_chemokines":     "Immunity & inflammation",
    "interferon_response":      "Immunity & inflammation",
    "immediate_early":          "Immunity & inflammation",
    # Signaling pathways
    "kinases_signaling":        "Signaling pathways",
    "wnt_signaling":            "Signaling pathways",
    "notch_signaling":          "Signaling pathways",
    "tgfb_bmp_signaling":       "Signaling pathways",
    "hedgehog_signaling":       "Signaling pathways",
    # Transcription factors
    "tf_homeobox":              "Transcription factors",
    "tf_zinc_finger":           "Transcription factors",
    "tf_forkhead_etc":          "Transcription factors",
    "nuclear_receptors":        "Transcription factors",
    # Metabolism
    "cytochrome_p450":          "Metabolism",
    "glutathione_redox":        "Metabolism",
    "metabolism_glycolysis":    "Metabolism",
    "metabolism_lipid":         "Metabolism",
    # Narrow categories that fall through to `other` in the broad system —
    # apoptosis, hemoglobin, coagulation_serum, housekeeping. Listing them
    # explicitly here so any future addition / rename to the narrow list
    # surfaces as a missing-key error if not deliberately mapped.
    "apoptosis":                OTHER,
    "hemoglobin":               OTHER,
    "coagulation_serum":        OTHER,
    "housekeeping":             OTHER,
}


# Public ordering list for figures (10 broad + "other" trailer)
CATEGORY_ORDER_BROAD: tuple[str, ...] = BROAD_NAMES + (OTHER,)


# ---------------------------------------------------------------------------
# Classification API
# ---------------------------------------------------------------------------

def classify_broad(symbol: str) -> str:
    """Return the broad-category umbrella for a gene symbol.

    Resolution chain: ``symbol -> narrow category (priority) -> broad
    umbrella``. Genes that match no narrow category, or whose narrow
    category is in the ``OTHER`` bucket (apoptosis / hemoglobin /
    coagulation_serum / housekeeping), return ``"other"``.
    """
    # Use the narrow classifier from the sibling module; we re-import
    # here at call time so the lazy-import-friendly form works whether
    # this module is loaded directly (no `chimera` package context) or
    # as a package member.
    try:
        from ._gene_categories import classify_extended  # type: ignore
    except ImportError:
        from _gene_categories import classify_extended   # type: ignore
    narrow = classify_extended(symbol)
    return NARROW_TO_BROAD.get(narrow, OTHER)


def annotate_broad(var_names: Iterable[str], symbol_lookup) -> pd.DataFrame:
    """Build a [var_name, symbol, category] DataFrame using broad categories.

    Same calling convention as `_gene_categories.annotate_extended` —
    `symbol_lookup` may be a callable, a Series, or a dict-like; the
    Chimera workspace's ortholog table maps `ortho_X` → human gene
    symbol (use ``_paper_utils.annotate_genes_with_names``).
    """
    var_names = pd.Index(list(var_names))
    if hasattr(symbol_lookup, "loc"):
        sym = symbol_lookup.reindex(var_names)
    elif callable(symbol_lookup):
        sym = pd.Series([symbol_lookup(v) for v in var_names], index=var_names)
    else:
        sym = pd.Series([symbol_lookup.get(v, v) for v in var_names], index=var_names)
    cats = sym.map(lambda s: classify_broad(s) if isinstance(s, str) else OTHER)
    return pd.DataFrame({
        "var_name": var_names,
        "symbol":   sym.values,
        "category": cats.values,
    }, index=var_names)


# ---------------------------------------------------------------------------
# Self-test (run as `python _gene_categories_broad.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    spot_checks = {
        # Translation & ribosome
        "RPL13A":   "Translation & ribosome",
        "MRPS5":    "Translation & ribosome",
        "EIF4A1":   "Translation & ribosome",
        # Mitochondria & OXPHOS
        "MT-CO1":   "Mitochondria & OXPHOS",
        "NDUFB1":   "Mitochondria & OXPHOS",
        "PRKN":     "Mitochondria & OXPHOS",
        # Chromatin & DNA processing
        "HIST1H1A": "Chromatin & DNA processing",
        "SRSF1":    "Chromatin & DNA processing",
        "MSH6":     "Chromatin & DNA processing",
        "BRCA1":    "Chromatin & DNA processing",
        # Cell cycle
        "CDK1":     "Cell cycle",
        "MKI67":    "Cell cycle",
        # Protein homeostasis
        "HSPA5":    "Protein homeostasis",
        "PSMB5":    "Protein homeostasis",
        "ATG5":     "Protein homeostasis",
        # Cytoskeleton & ECM
        "ACTB":     "Cytoskeleton & ECM",
        "TUBA1B":   "Cytoskeleton & ECM",
        "COL1A1":   "Cytoskeleton & ECM",
        "FN1":      "Cytoskeleton & ECM",
        # Immunity & inflammation
        "HLA-A":    "Immunity & inflammation",
        "C1QA":     "Immunity & inflammation",
        "IL6":      "Immunity & inflammation",
        "ISG15":    "Immunity & inflammation",
        "FOS":      "Immunity & inflammation",
        # Signaling pathways
        "MAPK1":    "Signaling pathways",
        "WNT3A":    "Signaling pathways",
        "NOTCH1":   "Signaling pathways",
        "TGFB1":    "Signaling pathways",
        "SHH":      "Signaling pathways",
        # Transcription factors
        "HOXA1":    "Transcription factors",
        "ZNF1":     "Transcription factors",
        "FOXA1":    "Transcription factors",
        "PPARG":    "Transcription factors",
        # Metabolism
        "CYP3A4":   "Metabolism",
        "GSTM1":    "Metabolism",
        "GAPDH":    "Metabolism",
        "FASN":     "Metabolism",
        # other (narrow categories that map to OTHER)
        "BAX":      "other",       # apoptosis → other
        "HBA1":     "other",       # hemoglobin → other
        "FGA":      "other",       # coagulation_serum → other
        "ACTG":     "other",       # un-curated, falls through
    }

    pass_count, fail_count = 0, 0
    for sym, expected in spot_checks.items():
        actual = classify_broad(sym)
        ok = actual == expected
        flag = "  " if ok else "✗ "
        print(f"{flag}{sym:12s} → {actual:30s}  (expected: {expected})")
        if ok: pass_count += 1
        else:  fail_count += 1
    print(f"\n{pass_count}/{pass_count+fail_count} pass")
