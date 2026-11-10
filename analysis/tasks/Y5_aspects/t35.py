"""Y5 T3.5 (descriptive): per author label, share of cells in cross-species clusters and share of neighbours from the other species, each latent."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.Y5_aspects import common as K, f3 as F3

OUT = K.OUT / "F3"
rows = []
for c in ("liv_hmouse", "test_3sp", "test_hmouse"):
    X, obs, var = F3.load_cells(c)
    spc = obs.species.to_numpy()
    lab = obs.cell_type.map(lambda s: T42.TESTIS_COARSE.get(s, s)).to_numpy() if c.startswith("test") else obs.cell_type.to_numpy()
    for nm, z in F3.latents(c).items():
        if z.shape[0] != X.shape[0]:
            continue
        f = OUT / f"clusters__{c}__{nm}.parquet"
        shared = None
        if f.exists():
            cl = pd.read_parquet(f).cl.to_numpy()
            tab = pd.crosstab(cl, spc)
            ok = (tab >= T42.MIN_CELLS_CLUSTER).sum(1).ge(2)
            shared = pd.Series(cl).map(ok).to_numpy(bool)
        ind = NearestNeighbors(n_neighbors=31).fit(z).kneighbors(z, return_distance=False)[:, 1:]
        other = (spc[ind] != spc[:, None]).mean(1)
        for k in np.unique(lab):
            m = lab == k
            rows.append(dict(cohort=c, latent=nm, label=k, n=int(m.sum()), n_species=len(set(spc[m])), share_in_shared=float(shared[m].mean()) if shared is not None else np.nan, other_species_neighbours=float(other[m].mean())))
pd.DataFrame(rows).to_csv(OUT / "t35_labels.csv", index=False)
t = pd.DataFrame(rows)
t["tool"] = t.latent.str.replace(r"_s\d+$", "", regex=True)
pd.set_option("display.width", 220)
print(t.pivot_table(index=["cohort", "label"], columns="tool", values="other_species_neighbours").round(2).to_string())
print(t.pivot_table(index=["cohort", "label"], columns="tool", values="share_in_shared").round(2).to_string())
