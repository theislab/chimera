"""Y6 use 1, PH5 (post hoc, see ledger): count-free wrong-ortholog label from gene names. T_named_else = the species gene's own name differs from the human symbol and
another gene of that species in Ensembl 110 carries the human symbol. Also the whole vocabulary count, per table.

    PYTHONPATH=$PWD python -m revision.tasks.Y6_uses.use1_named
"""

from __future__ import annotations

import pandas as pd

from revision.tasks.Y6_uses.annot_genome import GTF
from revision.tasks.Y6_uses.candidates import gtf_names
from revision.tasks.Y6_uses.use1 import OUT


def main():
    D = pd.read_parquet(OUT / "posthoc_rows.parquet")
    lab = {}
    for sp in D.species.unique():
        names = gtf_names(GTF[sp][110])
        inv = {}
        for g, (n, _, _) in names.items():
            if n:
                inv.setdefault(n.upper(), set()).add(g)
        lab[sp] = (names, inv)
    def f(r):
        names, inv = lab[r.species]
        sym = str(r.symbol).upper() if pd.notna(r.symbol) else ""
        own = names.get(r.gene_id, ("", "", ""))[0].upper()
        if not sym:
            return float("nan")
        return float(own != sym and len(inv.get(sym, set()) - {r.gene_id}) > 0)
    D["T_named_else"] = [f(r) for r in D[["species", "symbol", "gene_id"]].itertuples()]
    D["named_true_gene"] = [";".join(sorted(lab[r.species][1].get(str(r.symbol).upper(), set()) - {r.gene_id})) for r in D[["species", "symbol", "gene_id"]].itertuples()]
    D["own_name"] = [lab[r.species][0].get(r.gene_id, ("", "", ""))[0] for r in D[["species", "gene_id"]].itertuples()]
    print(D.groupby(["fit", "species_label"]).T_named_else.agg(["sum", "mean", "count"]).round(4), flush=True)
    # Too few per chromosome half for the odd/even split, so per row on all genes: median percentile and how many reach the top 1% of U_expr.
    rows = []
    for (fit, l), x in D.groupby(["fit", "species_label"]):
        u = x[x.det_other >= 0.05]
        hit = u.T_named_else.eq(1)
        if hit.sum() < 3:
            continue
        r = dict(fit=fit, species_label=l, n=len(u), n_pos=int(hit.sum()))
        for s in ["S_row", "gap_abs", "gap_log", "zero_gap", "zero_s", "id_neg"]:
            p = u[s].fillna(u[s].min() - 1).rank(pct=True)
            r[f"{s}_medpct"], r[f"{s}_top1pct"] = p[hit].median(), int((p[hit] >= 0.99).sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "posthoc_named.csv", index=False)
    D[D.T_named_else == 1][["fit", "species_label", "species", "symbol", "gene_id", "own_name", "named_true_gene", "identity_s", "S_gs", "det_s", "det_other", "cp_s", "cp_other", "goc", "wga", "high_conf", "homology_type_110", "homology_type_116", "T_lowconf", "T_later", "split"]].sort_values(["fit", "species_label", "S_gs"]).to_csv(OUT / "named_else_pairs.csv", index=False)


if __name__ == "__main__":
    main()
