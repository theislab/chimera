"""Build the small assets that ``overview.py`` draws from. Run once; the figure script reads only ``assets/``.

    python build_assets.py silhouettes                 # assets/silhouettes.json
    python build_assets.py data --repo <chimera repo>  # assets/liver_human_monkey_*.csv

Silhouettes
    The species outlines (human, macaque, mouse, zebrafish) are traced from the background picture of the
    hand-made overview page that this figure replaces (its background picture, extracted with
    ``pdfimages -all``, is kept as ``assets/source/overview_page_background.jpg``, 3000 x 1674 px).
    Each shape is separated from the paper-white background and its soft edge is traced on an eight times
    finer grid and thinned to a few hundred points, so the figure draws it as a filled vector outline in the
    species color of the other figures. A shape is a list of rings (the outline first, then enclosed gaps);
    coordinates are in units of the outline's height, origin at its lower left, y upward.

Data (panel d)
    Real values of the liver Human+Monkey reference model, the run the paper introduces alpha with
    (``training/runs/iter90-150ep-s42``, seed 42; 12,608 cells, 8,430 one-to-one orthologs):
      liver_human_monkey_delta_alpha.csv   counts of delta alpha = alpha - prior in the 80 bins of the
                                           paper's delta-alpha axis (-0.4 to 0.4), and the totals
      liver_human_monkey_umap.csv          1,500 cells drawn without replacement (seed 0): the cached UMAP
                                           of the latent (``notebooks/cache/chimera_umap/
                                           panc_hmonkey__chimera__umap.npy``), species and cell type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
BACKGROUND = ASSETS / "source" / "overview_page_background.jpg"

# Boxes in the background picture (left, top, right, bottom), pixels.
SPECIES_BOX = {
    "human": (130, 160, 220, 362),
    "macaque": (296, 180, 460, 330),
    "mouse": (496, 204, 664, 326),
    "zebrafish": (686, 220, 864, 306),
}
LEVEL = {"human": 55.0, "macaque": 40.0, "mouse": 40.0, "zebrafish": 40.0}  # distance from paper white at the outline
PAPER = np.array([254.0, 254.0, 250.0])  # background of the picture
RUN = "iter90-150ep-s42"
ADATA = "data/processed/camex_human_monkey.h5ad"
UMAP = "notebooks/cache/chimera_umap/panc_hmonkey__chimera__umap.npy"
N_CELLS_SHOWN = 1500
DELTA_ALPHA_BINS = np.linspace(-0.4, 0.4, 81)  # notebooks/_figure_design.py


def _simplify(p: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline."""
    keep = np.zeros(len(p), bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = p[b] - p[a]
        n = np.hypot(*seg)
        q = p[a + 1:b] - p[a]
        d = np.abs(seg[0] * q[:, 1] - seg[1] * q[:, 0]) / n if n > 0 else np.hypot(q[:, 0], q[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack += [(a, k), (k, b)]
    return p[keep]


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def silhouettes() -> None:
    from contourpy import contour_generator
    from PIL import Image
    from scipy import ndimage

    img = np.asarray(Image.open(BACKGROUND).convert("RGB")).astype(float)
    up = 8  # trace on a finer grid, from the soft edge of the picture, so outlines come out smooth
    out = {}
    for name, (x0, y0, x1, y1) in SPECIES_BOX.items():
        crop = img[y0:y1, x0:x1]
        dist = np.abs(crop - PAPER).max(axis=2)  # how far a pixel is from the paper white
        lab, n = ndimage.label(dist > LEVEL[name])
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        body = ndimage.binary_dilation(lab == 1 + int(np.argmax(sizes)), iterations=3)
        fine = ndimage.zoom(np.where(body, dist, 0.0), up, order=3)
        fine = ndimage.gaussian_filter(fine, 0.45 * up)
        rings = [r for r in contour_generator(z=fine).lines(LEVEL[name]) if len(r) > 8 and np.allclose(r[0], r[-1])]
        outer = max(rings, key=lambda r: abs(_area(r)))
        keep = [r for r in rings if abs(_area(r)) >= 0.004 * abs(_area(outer))]  # the outline and real gaps
        xs, ys = outer[:, 0], outer[:, 1]
        h = ys.max() - ys.min()
        paths = []
        for r in sorted(keep, key=lambda r: -abs(_area(r))):
            r = _simplify(r, 0.035 * up)
            paths.append(np.round(np.c_[(r[:, 0] - xs.min()) / h, (ys.max() - r[:, 1]) / h], 4).tolist())
        out[name] = {"aspect": round(float((xs.max() - xs.min()) / h), 4), "rings": paths}
        print(f"{name}: {[len(q) for q in paths]} points, width/height {out[name]['aspect']}")
    (ASSETS / "silhouettes.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")


def data(repo: Path) -> None:
    import anndata as ad
    import pandas as pd

    fs = np.load(repo / "training" / "runs" / RUN / "final_state.npz", allow_pickle=False)
    alpha, prior = fs["alpha"].astype(float), fs["prior"].astype(float)
    da = alpha - prior
    counts, _ = np.histogram(da, bins=DELTA_ALPHA_BINS)
    h = pd.DataFrame({"bin_left": DELTA_ALPHA_BINS[:-1].round(4), "bin_right": DELTA_ALPHA_BINS[1:].round(4),
                      "n_genes": counts})
    with (ASSETS / "liver_human_monkey_delta_alpha.csv").open("w") as f:
        f.write(f"# liver Human+Monkey reference model ({RUN}); delta alpha = alpha - prior\n")
        f.write(f"# n_genes_total={len(da)} n_above_0={(da > 0).sum()} n_below_0={(da < 0).sum()} "
                f"n_outside_axis={(np.abs(da) > 0.4).sum()}\n")
        h.to_csv(f, index=False)
    print(f"delta alpha: {len(da)} genes, above 0 {(da > 0).mean():.3f}, below 0 {(da < 0).mean():.3f}, "
          f"outside the axis {(np.abs(da) > 0.4).sum()}")

    xy = np.load(repo / UMAP)
    a = ad.read_h5ad(repo / ADATA, backed="r")
    ct_col = next(c for c in ("cell_ontology_class", "cell_type") if c in a.obs.columns)
    obs = a.obs[["species", ct_col]].copy()
    a.file.close()
    assert len(obs) == len(xy) == fs["z_mean"].shape[0], (len(obs), len(xy), fs["z_mean"].shape)
    idx = np.sort(np.random.default_rng(0).choice(len(xy), N_CELLS_SHOWN, replace=False))
    u = pd.DataFrame({"umap1": xy[idx, 0].round(3), "umap2": xy[idx, 1].round(3),
                      "species": obs["species"].astype(str).to_numpy()[idx],
                      "cell_type": obs[ct_col].astype(str).to_numpy()[idx]})
    with (ASSETS / "liver_human_monkey_umap.csv").open("w") as f:
        f.write(f"# liver Human+Monkey reference model ({RUN}); {N_CELLS_SHOWN} of {len(xy)} cells, seed 0; "
                f"UMAP of the latent from {UMAP}\n")
        u.to_csv(f, index=False)
    print(u.species.value_counts().to_dict(), u.cell_type.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["silhouettes", "data"])
    ap.add_argument("--repo", type=Path, default=None, help="chimera repository (for 'data')")
    args = ap.parse_args()
    if args.what == "silhouettes":
        silhouettes()
    else:
        data(args.repo)
