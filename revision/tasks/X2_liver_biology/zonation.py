"""X2 (EXPLORATORY): porto-central zonation of liver hepatocytes against Chimera's D and margin.

Inputs: zone-resolved forward sums (``forward.py``), per-gene tables (``build.py``), the frozen
sets (``sets.py``). Species enter only if their landmark axis passes the check fixed in
forward.py (Spearman(PC part, PP part) <= -0.3): Human and Mouse; Monkey (+0.35) and Zebrafish
(no periportal landmark detected, one pericentral landmark at 8 %) do not.

Per gene and species (hepatocytes):
* ``obs_slope``: observed log2(1 + CPM) of the PC tertile minus the PP tertile (pooled counts);
* ``pred_slope``: the same for the shared decoder's predicted composition on the same cells
  (what the species-blind decoder reproduces of that species' zonation);
* ``Sz_<zone>`` / ``Wz_<zone>``: per-cell margin (model null / within-type null) per zone.

Outputs (results/revision/X2/zonation/): ``gene_zonation_reference.csv``, ``axis_check.csv``,
``latent_zonation.csv`` (does z encode and align the zonation score across species?),
``literature_check.csv`` (do in-vocabulary literature genes have the expected slope sign?),
``conservation.csv`` (human vs mouse zonation agreement: the simple baseline),
``decoder_fidelity.csv`` (pred vs obs slope), ``set_tests.csv`` (D / S / B1b matched excess of
zonation sets), ``discordance.csv`` (D of genes whose zonation flips between human and mouse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import gene_ids

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sets as ST  # noqa: E402
import stats as SX  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "X2"
FW = OUT / "forward"
ZD = OUT / "zonation"
AXIS_SPECIES = ["Human", "Mouse"]
LABELS = {"liv_hmouse": ["liv_hmouse", "liv_hmouse_s43", "liv_hmouse_s44"],
          "liv_4sp": ["liv_4sp", "liv_4sp_s43", "liv_4sp_s44", "liv_4sp_s45", "liv_4sp_s46"]}


def gene_zonation(label: str) -> pd.DataFrame:
    z = np.load(FW / f"{label}.npz")
    grp = pd.DataFrame([g.split("|") for g in z["groups"]], columns=["species", "cell_type", "zone"])
    out = {}
    for s in AXIS_SPECIES:
        for zone in ("PP", "mid", "PC"):
            m = ((grp.species == s) & (grp.cell_type == "Hepatocytes") & (grp.zone == zone)).to_numpy()
            if m.sum() != 1:
                raise AssertionError(f"{label}: {s} {zone} groups {m.sum()}")
            i = np.where(m)[0][0]
            n = z["n"][i]
            out[f"x_{s}_{zone}"] = np.log2(1 + 1e6 * z["sum_obs"][i] / z["sum_lib"][i])
            out[f"p_{s}_{zone}"] = np.log2(1 + 1e6 * z["sum_pred"][i] / n)
            out[f"Sz_{s}_{zone}"] = z["sum_l"][i] / n
            out[f"Wz_{s}_{zone}"] = z["sum_wt"][i] / n
        out[f"obs_slope_{s}"] = out[f"x_{s}_PC"] - out[f"x_{s}_PP"]
        out[f"pred_slope_{s}"] = out[f"p_{s}_PC"] - out[f"p_{s}_PP"]
    df = pd.DataFrame(out, index=pd.Index(z["var_names"], name="var_name"))
    return df


def zonated_flags(gz: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=gz.index)
    for s in AXIS_SPECIES:
        det = g.set_index("var_name")[f"f__{s}__Hepatocytes"].reindex(gz.index)
        sl = gz[f"obs_slope_{s}"]
        f[f"zPC_{s}"] = (sl >= ST.ZONE_SLOPE) & (det >= ST.ZONE_DET)
        f[f"zPP_{s}"] = (sl <= -ST.ZONE_SLOPE) & (det >= ST.ZONE_DET)
    return f


def latent_check(label: str, cohort: str) -> list[dict]:
    """Does z encode each species' zonation score, and does the direction transfer across species?"""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    from revision.common import cohorts as C

    a = C.read_adata(cohort, backed=True)
    z = np.load(FW / f"{label}__zmean.npy")
    zt = pd.read_csv(FW / f"{label}__zones.csv").set_index("cell")
    pos = pd.Series(np.arange(a.n_obs), index=a.obs_names)
    alphas = np.logspace(-3, 3, 13)
    rows, models, data = [], {}, {}
    for s in AXIS_SPECIES:
        d = zt[zt.species == s]
        Z, y = z[pos[d.index].to_numpy()], d["score"].to_numpy()
        pred = cross_val_predict(RidgeCV(alphas=alphas), Z, y, cv=5)
        models[s] = RidgeCV(alphas=alphas).fit(Z, y)
        data[s] = (Z, y)
        rows.append(dict(cohort=cohort, label=label, train=s, test=s, spearman=spearmanr(pred, y)[0], kind="cv5"))
    for tr, te in (("Mouse", "Human"), ("Human", "Mouse")):
        Z, y = data[te]
        rows.append(dict(cohort=cohort, label=label, train=tr, test=te,
                         spearman=spearmanr(models[tr].predict(Z), y)[0], kind="transfer"))
    return rows


def main():
    ZD.mkdir(parents=True, exist_ok=True)
    rows_axis = []
    for lab in sorted({x for v in LABELS.values() for x in v}):
        zl = pd.read_csv(FW / f"{lab}__zone_landmarks.csv")
        rows_axis.append(zl[zl.status == "axis_check"].assign(label=lab))
    pd.concat(rows_axis).to_csv(ZD / "axis_check.csv", index=False)

    # Human and Mouse hepatocytes (and their zones) are the same cells in every cohort, so gene
    # zonation and flags come from the 4-sp table and are joined to other cohorts by Ensembl ID.
    g4 = SX.load_genes("liv_4sp")
    gz4 = gene_zonation("liv_4sp")
    flags = zonated_flags(gz4, g4)
    ref = gz4.join(flags).join(g4.set_index("var_name")[["symbol", "hsapiens_gene_id"]])
    for s in AXIS_SPECIES:
        ref[f"det_{s}"] = g4.set_index("var_name")[f"f__{s}__Hepatocytes"].reindex(ref.index)
    ref.to_csv(ZD / "gene_zonation_reference.csv")
    ref = ref.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")

    lit_rows = []
    for set_name, genes in ST.ZONATION_SETS.items():
        exp = +1 if set_name.startswith("PC") else (-1 if set_name.startswith("PP") else 0)
        for gene in genes:
            m = ref["symbol"] == gene
            if not m.any():
                lit_rows.append(dict(set=set_name, gene=gene, in_vocab=False))
                continue
            r = dict(set=set_name, gene=gene, in_vocab=True, expected_sign=exp)
            for s in AXIS_SPECIES:
                r[f"obs_slope_{s}"] = float(ref.loc[m, f"obs_slope_{s}"].iloc[0])
                r[f"det_{s}"] = float(ref.loc[m, f"det_{s}"].iloc[0])
            lit_rows.append(r)
    pd.DataFrame(lit_rows).to_csv(ZD / "literature_check.csv", index=False)

    H, M = ref["obs_slope_Human"], ref["obs_slope_Mouse"]
    both = (ref.det_Human >= ST.ZONE_DET) & (ref.det_Mouse >= ST.ZONE_DET)
    zon_H = (ref.zPC_Human | ref.zPP_Human) & both
    zon_M = (ref.zPC_Mouse | ref.zPP_Mouse) & both
    zon_any = zon_H | zon_M
    cons_rows = []
    for name, mk in (("expressed_both", both), ("zonated_either", zon_any), ("zonated_human", zon_H),
                     ("zonated_mouse", zon_M)):
        cons_rows.append(dict(genes=name, n=int(mk.sum()), spearman_H_M=spearmanr(H[mk], M[mk])[0],
                              sign_concordance=float((np.sign(H[mk]) == np.sign(M[mk])).mean())))
    pd.DataFrame(cons_rows).to_csv(ZD / "conservation.csv", index=False)

    fid_rows, lat_rows = [], []
    for cohort, labs in LABELS.items():
        g = SX.load_genes(cohort).set_index("var_name")
        ens = g["hsapiens_gene_id"]
        for lab in labs:
            gz = gene_zonation(lab).join(ens)
            gz = gz.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index(
                "hsapiens_gene_id")
            for s in AXIS_SPECIES:
                for name, mk in (("zonated_either", zon_any), ("zonated_human", zon_H), ("zonated_mouse", zon_M)):
                    ids = mk.index[mk.to_numpy()].intersection(gz.index)
                    o, p = gz.loc[ids, f"obs_slope_{s}"], gz.loc[ids, f"pred_slope_{s}"]
                    fid_rows.append(dict(cohort=cohort, label=lab, species=s, genes=name, n=len(ids),
                                         spearman_pred_obs=spearmanr(o, p)[0],
                                         ols_slope_pred_on_obs=float(np.polyfit(o, p, 1)[0]),
                                         sign_agreement=float((np.sign(o) == np.sign(p)).mean())))
            ids = zon_any.index[zon_any.to_numpy()].intersection(gz.index)
            fid_rows.append(dict(cohort=cohort, label=lab, species="pred_Human_vs_pred_Mouse", genes="zonated_either",
                                 n=len(ids), spearman_pred_obs=spearmanr(gz.loc[ids, "pred_slope_Human"],
                                                                         gz.loc[ids, "pred_slope_Mouse"])[0]))
            lat_rows += latent_check(lab, cohort)
    pd.DataFrame(fid_rows).to_csv(ZD / "decoder_fidelity.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(ZD / "latent_zonation.csv", index=False)

    set_rows, disc_rows = [], []
    conc = ((ref.zPC_Human & ref.zPC_Mouse) | (ref.zPP_Human & ref.zPP_Mouse))
    disc = zon_any & (H * M < 0)
    for cohort in ("liv_4sp", "liv_hmonkey", "liv_hmouse", "liv_hzeb"):
        gi = SX.load_genes(cohort)
        e = gi["hsapiens_gene_id"]
        sets = {f"lit:{k}": gi["symbol"].isin(v).to_numpy() for k, v in ST.ZONATION_SETS.items()}
        for s in AXIS_SPECIES:
            for side in ("PC", "PP"):
                sets[f"data:z{side}_{s}"] = e.map(ref[f"z{side}_{s}"]).fillna(False).to_numpy(bool)
        sets["data:zonation_concordant_HM"] = e.map(conc).fillna(False).to_numpy(bool)
        sets["data:zonation_discordant_HM"] = e.map(disc).fillna(False).to_numpy(bool)
        scores = ["D", "S_ema_cons", "B1b_raw", "B1b"] + [c for c in gi.columns if c.startswith("Cs__")]
        scores += [c for c in gi.columns if c.startswith("W__") and c.endswith("__Hepatocytes")]
        scores += [c for c in gi.columns if c.startswith("lfc__") and c.endswith("__Hepatocytes")]
        set_rows += SX.set_excess(gi, cohort, sets, scores, B=2000, seed=11)
        mk = e.map(zon_any).fillna(False).to_numpy(bool)
        dd = e.map((H - M).abs()).to_numpy(dtype=float)
        resid = SX.residualise(gi, "D").to_numpy()
        ok = mk & np.isfinite(resid)
        disc_rows.append(dict(cohort=cohort, n=int(ok.sum()),
                              spearman_D_vs_abs_slope_diff=spearmanr(gi.loc[ok, "D"], dd[ok])[0],
                              spearman_Dresid_vs_abs_slope_diff=spearmanr(resid[ok], dd[ok])[0],
                              spearman_B1braw_vs_abs_slope_diff=spearmanr(gi.loc[ok, "B1b_raw"], dd[ok],
                                                                          nan_policy="omit")[0]))
    pd.DataFrame(set_rows).to_csv(ZD / "set_tests.csv", index=False)
    pd.DataFrame(disc_rows).to_csv(ZD / "discordance.csv", index=False)
    print("zonation done", flush=True)


if __name__ == "__main__":
    main()
