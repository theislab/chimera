"""Depth scaling of the gate anchor, shared by D1_sparse_genes and B2_benchmark Track B.

The fit reward is a per-cell mean log-likelihood difference, so it scales with count depth, and the anchor that sets the scale of Delta-alpha (Law 2, tables/k_scaling_laws.tex) was set at the
recipe's depth. On bone marrow the anchor was therefore scaled by the depth ratio (analysis/tasks/T12_2b_recipe/README.md: 0.001 x 3,672 / 1,097 = 0.0033). The same rule here, in both directions:

    lambda_anchor(object) = lambda_anchor(recipe) x L(object) / L(recipe object)

L = mean over species of the species median library size, computed on each object's counts. Nothing is read from a fit or a gold; this is a formula, not a tuning.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ANCHOR = 0.003  # chimera/_constants.py LAMBDA_ANCHOR, used when the recipe sets none


def depth_stat(X, species) -> float:
    lib = np.asarray(X.sum(1)).ravel()
    species = np.asarray(species)
    return float(np.mean([np.median(lib[species == s]) for s in sorted(set(species))]))


def scaled_anchor(recipe_anchor: float | None, X_object, species_object, X_recipe, species_recipe) -> tuple[float, float]:
    """Returns (anchor, depth ratio)."""
    base = DEFAULT_ANCHOR if recipe_anchor is None else float(recipe_anchor)
    ratio = depth_stat(X_object, species_object) / depth_stat(X_recipe, species_recipe)
    return base * ratio, ratio


def recipe_anchor(args: str) -> float | None:
    w = args.split()
    return float(w[w.index("--lambda-anchor") + 1]) if "--lambda-anchor" in w else None
