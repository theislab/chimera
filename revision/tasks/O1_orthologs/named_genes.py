"""O1 part B6: every human symbol named in main.tex, captions, tables and the Supplement, crossed with the flags of the cohorts; raw counts per species.
Symbols are tokens of the .tex sources that equal a human symbol of any table, or the case-insensitive match of a mouse symbol. Text in comments is skipped.

    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.named_genes
"""

from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.tasks.O1_orthologs import rules as RU

warnings.filterwarnings("ignore")
MS = RU.REPO.parent / "chimera-manuscript"
SKIP = {"SET", "CAT", "MAX", "MET", "REST", "CAPS", "ACE", "SHE", "CD", "IMPACT", "GC", "AR", "ALL", "BID", "NHS", "ICE", "PC", "CA", "CP", "ROCK", "ATM", "AMH", "TANK", "RAN", "WAS", "SI", "HR", "COIL", "MALL", "MARS", "MAGI", "CAST", "MAP", "PAM", "CLOCK", "NEAT", "DES", "DBP", "SARS", "GAN", "VAMP", "ESR", "NOTCH", "PLAN", "CAD", "CARD", "FAT", "MICE", "RELA", "BOD", "BAD", "SON", "TEC", "EGG", "SPRED", "IDS", "IDE", "GIN", "ESM", "MMD", "UMI", "TF", "PI", "NB", "LUC", "VAE", "PCA", "UMAP", "AUROC", "ROC", "CI", "SD", "BH", "QC", "OR", "ANOVA", "GO", "KO", "PDF", "TOP", "BMP", "NOT", "AND", "FOR", "IMP", "NET", "LAMP", "COPE", "HOOK", "PAGE", "WARS", "OAT", "SMS", "PLOD", "PEAK", "AIM", "STAR", "TRAP", "SAFE", "POLE", "SHOT", "TANGO", "TACTIC", "GAP", "CHIC", "ERF", "EVEE"}


def tokens() -> pd.DataFrame:
    files = [MS / "main.tex"] + sorted(Path(p) for p in glob.glob(str(MS / "supplementary/*.tex"))) + sorted(Path(p) for p in glob.glob(str(MS / "tables/*.tex")))
    rows = []
    for f in files:
        for i, ln in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            ln2 = re.sub(r"(?<!\\)%.*$", "", ln)
            for m in re.finditer(r"(?<![A-Za-z0-9_\\-])([A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*)(?![A-Za-z0-9_])", ln2):
                rows.append((f.name, i, m.group(1), ln2[max(0, m.start() - 60): m.end() + 60]))
    return pd.DataFrame(rows, columns=["file", "line", "token", "context"])


def main():
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    P = pd.read_parquet(RU.RES / "table_pairs_flags.parquet")
    M = pd.read_parquet(RU.RES / "mapping_rows.parquet")
    syms = set(G.symbol.dropna())
    mouse = P[P.species == "mmusculus"].dropna(subset=["species_gene_name"])
    m2h = dict(zip(mouse.species_gene_name, mouse.hsapiens_gene_name))
    T = tokens()
    T["symbol"] = np.where(T.token.isin(syms), T.token, T.token.map(m2h))
    T = T[T.symbol.notna() & ~T.symbol.isin(SKIP)]
    T = T[T.token.str.len() >= 3]
    T.to_csv(RU.RES / "named_tokens.csv", index=False)
    seen = T.groupby("symbol").agg(n=("line", "size"), files=("file", lambda x: ";".join(sorted(set(x)))), context=("context", "first")).reset_index()
    fl = G[G.symbol.isin(seen.symbol)]
    tiers = ["T1", "T2", "T2L", "M_wrong_row", "M_false_zero", "T3", "zero_any"]
    long = fl.melt(id_vars=["cohort", "symbol", "hsapiens_gene_id", "var_name"], value_vars=tiers, var_name="flag", value_name="on")
    long = long[long.on]
    zero_cols = [c for c in G.columns if c.startswith("zero_") and c != "zero_any"]
    wide = long.groupby(["symbol", "flag"]).cohort.agg(lambda x: ",".join(x)).unstack().reset_index()
    out = seen.merge(wide, on="symbol", how="left")
    any_flag = out[[c for c in tiers if c in out and c != "zero_any"]].notna().any(axis=1)
    out["flagged"] = any_flag
    pairs = P[P.hsapiens_gene_name.isin(seen.symbol) & P.T2]
    d = pairs.groupby("hsapiens_gene_name").apply(lambda x: "; ".join(sorted({f"{a}:{b or c}(id{e:.0f},goc{f:.0f},wga{g:.0f},hc{h:.0f})" for a, b, c, e, f, g, h in zip(x.species, x.species_gene_name.fillna(""), x.gene_id, x.identity.fillna(-1), x.goc_score.fillna(-1), x.wga_coverage.fillna(-1), x.is_high_confidence.fillna(-1))})))
    out["pair_detail"] = out.symbol.map(d)
    mm = M[(M.M_wrong_row | M.M_false_zero)]
    t = pd.concat([pd.read_csv(RU.TABLES["liver"])[["ortholog_group_id", "hsapiens_gene_name"]].assign(table="liver"), pd.read_csv(RU.TABLES["testis"])[["ortholog_group_id", "hsapiens_gene_name"]].assign(table="testis")])
    mm = mm.merge(t, on=["table", "ortholog_group_id"])
    mm = mm[mm.hsapiens_gene_name.isin(seen.symbol)]
    e = mm.groupby("hsapiens_gene_name").apply(lambda x: "; ".join(f"{a}/{b}:{'wrong' if w else 'false0'} obj={o:.0f} id_row={i:.0f} alt={k:.0f}" for a, b, w, o, i, k in zip(x.table, x.obj_species, x.M_wrong_row, x.obj_total, x.id_total_sub.fillna(-1), x.alt_counts.fillna(-1))))
    out["map_detail"] = out.symbol.map(e)
    out.to_csv(RU.RES / "named_genes_all.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400, "display.max_colwidth", 70)
    print(len(seen), "symbols;", int(out.flagged.sum()), "flagged")
    print(out[out.flagged].reindex(columns=["symbol", "n", "files", "T2", "T2L", "M_false_zero", "M_wrong_row", "zero_any", "pair_detail", "map_detail"]).to_string())
    print(sorted(seen.symbol))


if __name__ == "__main__":
    main()
