"""Y6 use 1, PH5 (post hoc, see ledger): count-free wrong-ortholog label from gene names. T_named_else = the species gene's own name differs from the human symbol and
another gene of that species in Ensembl 110 carries the human symbol. Also the whole vocabulary count, per table.

    PYTHONPATH=$PWD python -m revision.tasks.Y6_uses.use1_named
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from revision.tasks.Y6_uses.annot_genome import GTF
from revision.tasks.Y6_uses.candidates import gtf_names
from revision.tasks.Y6_uses.use1 import OUT
from revision.tasks.Y6_uses.use1_posthoc import boot_row


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
    jobs = []
    for (fit, l), m in D.groupby(["fit", "species_label"]):
        for u in ("expr", "hi"):
            jobs.append((m, "T_named_else", u, ["S_row", "gap_abs", "gap_log", "zero_gap", "zero_s", "id_neg"]))
    res = Parallel(n_jobs=12)(delayed(boot_row)(*j) for j in jobs)
    pd.DataFrame([r for x in res for r in x]).to_csv(OUT / "posthoc_named.csv", index=False)
    D[D.T_named_else == 1][["fit", "species_label", "species", "symbol", "gene_id", "own_name", "named_true_gene", "identity_s", "S_gs", "det_s", "det_other", "cp_s", "cp_other", "goc", "wga", "high_conf", "homology_type_110", "homology_type_116", "T_lowconf", "T_later", "split"]].sort_values(["fit", "species_label", "S_gs"]).to_csv(OUT / "named_else_pairs.csv", index=False)


if __name__ == "__main__":
    main()
