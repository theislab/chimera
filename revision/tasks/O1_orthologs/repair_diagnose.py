"""O2, post hoc and gold-free, after the gate failed on liver H+Monkey and 4-sp: what moved? Reads existing runs only. No training, no gold.

    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.repair_diagnose
"""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import controls as K
from revision.tasks.O1_orthologs import rules as RU
from revision.tasks.O1_orthologs.repair_compare import OUT, SEEDS, obj, repaired_set
from revision.tasks.T4_3_scoreboard import inputs as I

sys.path.insert(0, str(RU.REPO / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402


def row(run, a, label):
    x = K.read_run(run)
    z = np.load(C.RUNS_DIR / run / "final_state.npz")
    g = compute_integration_metrics(z["z_mean"], a.obs["species"].to_numpy(), a.obs["cell_type"].to_numpy(), k=15, seed=42)["global"]
    d = x["D"] - np.median(x["D"])
    return dict(arm=label, run=run, adv=float(z["adversarial_accuracy_final"]), entropy=g["knn_mixing_entropy"], purity=g["knn_species_purity"], label_transfer=g["label_transfer_acc"], sil_type=g["sil_celltype"], sd_D=float(d.std()), tail_neg=float((d < -0.05).mean()),
                tail_pos=float((d > 0.05).mean()), frac_S_pos=float((x["S_g_ema"] > 0).mean()), rho_D_S=float(spearmanr(x["D"], x["S_g_ema"])[0]), failed=";".join(x["summary"].get("failed_expectations", [])))


def main():
    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_rows", 80)
    frames, mats = [], {}
    for c, can in (("liv_hmonkey", {s: r for s, r in I.seed_runs_used("liv_hmonkey").items()}), ("liv_4sp", {s: r for s, r in I.seed_runs_used("liv_4sp").items()}), ("liv_hmouse", {s: r for s, r in I.seed_runs_used("liv_hmouse").items() if s in SEEDS})):
        a0, a1 = C.read_adata(c), ad.read_h5ad(obj(c))
        rows = [row(r, a0, f"canonical s{s}") for s, r in sorted(can.items())]
        if c == "liv_hmonkey":
            rows += [row(f"o2-ctrl-{c}-s{s}", a0, f"control s{s}") for s in SEEDS]
        rows += [row(f"o2-{c}-s{s}", a1, f"repaired s{s}") for s in SEEDS]
        f = pd.DataFrame(rows)
        f.insert(0, "cohort", c)
        frames.append(f)
        shared = pd.Index(a0.var_names.astype(str)).intersection(pd.Index(a1.var_names.astype(str)))
        p0, p1 = pd.Index(a0.var_names.astype(str)).get_indexer(shared), pd.Index(a1.var_names.astype(str)).get_indexer(shared)
        D = {}
        for r in f.itertuples():
            x = K.read_run(r.run)["D"]
            D[r.arm] = x[p1] if r.arm.startswith("repaired") else x[p0]
        if c == "liv_hmonkey":
            m = pd.DataFrame(D).corr("spearman").round(2)
            m.to_csv(OUT / "diagnose_hmonkey_rho.csv")
            print(m.to_string())
        # object-level comparison, gold-free
        rep = set(repaired_set(c).var_name)
        for s in sorted(set(a0.obs["species"].astype(str))):
            k0, k1 = (a0.obs["species"].astype(str) == s).to_numpy(), (a1.obs["species"].astype(str) == s).to_numpy()
            l0, l1 = np.asarray(a0.X[k0].sum(1)).ravel(), np.asarray(a1.X[k1].sum(1)).ravel()
            t1 = np.asarray(a1.X[k1].sum(0)).ravel()
            top = pd.Series(t1, index=a1.var_names.astype(str))
            tr = top[top.index.isin(rep)].sort_values(ascending=False)
            print(c, s, "median library", np.median(l0), "->", np.median(l1), "share of UMI in repaired genes", round(float(tr.sum() / t1.sum()), 4), "largest", tr.head(3).round(0).to_dict())
        F0, F1 = np.asarray(a0[:, shared].varm["gene_features"]), np.asarray(a1[:, shared].varm["gene_features"])
        print(c, "features: max abs change by block", {k: round(float(np.abs(F0[:, v] - F1[:, v]).max()), 3) for k, v in C.GIN_COLUMNS.items()}, "mean abs expr block", round(float(np.abs(F0[:, 66:69] - F1[:, 66:69]).mean()), 4))
    F = pd.concat(frames, ignore_index=True)
    F.to_csv(OUT / "diagnose_runs.csv", index=False)
    print(F.drop(columns=["run"]).round(3).to_string())


if __name__ == "__main__":
    main()
