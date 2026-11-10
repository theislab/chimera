"""E1 step 1: gene-set libraries by human Ensembl ID over the union vocabulary of the 26 cohorts (definitions in rules.py).

Writes results/analysis/E1/sets/{membership.parquet, sets.csv}. Reads no read-out.

    PYTHONPATH=$PWD python -m analysis.tasks.E1_atlas.sets
"""

from __future__ import annotations

import gzip
import zipfile
from collections import defaultdict

import pandas as pd

from analysis.tasks.E1_atlas import rules as RU
from analysis.tasks.H1_harmonise import rules as H1R
from analysis.tasks.H1_harmonise.recurrence import gene_sets as h1_gene_sets
from analysis.tasks.X3_cross_cohort import annotations as AN


def vocabulary() -> pd.DataFrame:
    rows = []
    for c in H1R.cohorts():
        for d in (RU.CAN, RU.FREE):
            rows.append(pd.read_parquet(d / f"{c}.parquet", columns=["hsapiens_gene_id", "symbol"]))
    u = pd.concat(rows).dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")
    u["sym"] = u.symbol.astype(str).str.upper()
    return u


def sym_map(u: pd.DataFrame) -> dict[str, list[str]]:
    m = defaultdict(list)
    for e, s in zip(u.hsapiens_gene_id, u.sym):
        m[s].append(e)
    return m


# ------------------------------------------------------------------------ Reactome
def reactome(u: pd.DataFrame) -> tuple[list, dict]:
    sm = sym_map(u)
    names = pd.read_csv(RU.DATA / "ReactomePathways.txt", sep="\t", header=None, names=["id", "name", "species"])
    names = names[names.species == "Homo sapiens"].set_index("id")["name"]
    rel = pd.read_csv(RU.DATA / "ReactomePathwaysRelation.txt", sep="\t", header=None, names=["parent", "child"])
    rel = rel[rel.parent.str.startswith("R-HSA") & rel.child.str.startswith("R-HSA")]
    parents = rel.groupby("child").parent.apply(set).to_dict()
    roots = set(names.index) - set(rel.child)
    gmt = zipfile.ZipFile(RU.FUNC / "ReactomePathways.gmt.zip").read("ReactomePathways.gmt").decode().splitlines()
    mem = {}
    for ln in gmt:
        x = ln.split("\t")
        mem[x[1]] = {e for s in x[2:] if s for e in sm.get(s.upper(), [])}

    def top(pid, seen=None):
        seen = seen or set()
        if pid in roots:
            return {pid}
        out = set()
        for p in parents.get(pid, ()):
            if p not in seen:
                seen.add(p)
                out |= top(p, seen)
        return out

    rows, top_of = [], {}
    for pid, genes in mem.items():
        if pid not in names.index:
            continue
        tops = top(pid)
        top_of[pid] = ";".join(sorted(names.get(t, t) for t in tops))
        is_root = pid in roots
        if is_root or RU.REACTOME_MIN <= len(genes) <= RU.REACTOME_MAX:
            lib = "reactome_top" if is_root else "reactome"
            rows += [(lib, pid, names[pid], e) for e in genes]
    return rows, top_of


# ------------------------------------------------------------------------ GO
def read_obo(path) -> tuple[dict, dict, dict]:
    """id -> name, namespace, parents (is_a and part_of); obsolete terms skipped."""
    name, ns, par = {}, {}, defaultdict(set)
    for stanza in open(path).read().split("\n\n"):
        lines = stanza.strip().splitlines()
        if not lines or lines[0] != "[Term]":
            continue
        cur = {"par": set()}
        for ln in lines[1:]:
            k, _, v = ln.partition(": ")
            if k in ("id", "name", "namespace"):
                cur[k] = v
            elif k == "is_obsolete" and v.strip() == "true":
                cur["obsolete"] = True
            elif k == "is_a":
                cur["par"].add(v.split(" ! ")[0].split(" {")[0].strip())
            elif k == "relationship" and v.startswith("part_of "):
                cur["par"].add(v.split()[1])
        if "id" in cur and not cur.get("obsolete"):
            name[cur["id"]] = cur.get("name")
            ns[cur["id"]] = cur.get("namespace", "x_x")
            par[cur["id"]] |= cur["par"]
    return name, ns, par


def go_sets(u: pd.DataFrame, extra_ids: set) -> list:
    sm = sym_map(u)
    name, ns, par = read_obo(RU.DATA / "go-basic.obo")
    slim, _, _ = read_obo(RU.DATA / "goslim_generic.obo")
    anc_cache = {}

    def anc(t):
        if t in anc_cache:
            return anc_cache[t]
        out = {t}
        for p in par.get(t, ()):
            out |= anc(p)
        anc_cache[t] = out
        return out

    gene_terms = defaultdict(set)
    with gzip.open(RU.DATA / "goa_human.gaf.gz", "rt") as f:
        for ln in f:
            if ln.startswith("!"):
                continue
            x = ln.split("\t")
            if "NOT" in x[3]:
                continue
            for e in sm.get(x[2].upper(), []):
                gene_terms[e].add(x[4])
    want = (set(slim) | extra_ids) & set(name)
    rows = []
    for e, terms in gene_terms.items():
        a = set().union(*(anc(t) for t in terms if t in name)) if terms else set()
        for t in a & want:
            lib = "goslim" if t in slim else "go_tissue"
            rows.append((lib, t, f"{name[t]} ({''.join(w[0].upper() for w in ns[t].split('_'))})", e))
    return rows


# ------------------------------------------------------------------------ UniProt keywords
def keyword_sets(u: pd.DataFrame) -> list:
    up = AN.uniprot_by_gene().reindex(pd.Index(u.hsapiens_gene_id))
    kt = AN.keyword_table()
    keep = {k for k, r in kt.iterrows() if r.kw_category in RU.KEYWORD_CATEGORIES}
    sets = defaultdict(set)
    for g, kws in zip(up.index, up["kw"]):
        if isinstance(kws, set):
            for k in kws & keep:
                sets[k].add(g)
    rows = []
    for k, v in sets.items():
        if len(v) >= RU.KEYWORD_MIN:
            rows += [("keyword", k, f"{kt.loc[k, 'kw_name']} [{kt.loc[k, 'kw_category']}]", e) for e in v]
    return rows


def main() -> int:
    out = RU.OUT / "sets"
    out.mkdir(parents=True, exist_ok=True)
    u = vocabulary()
    print("vocabulary", len(u))
    h = h1_gene_sets(u.rename(columns={"sym": "_s"}).drop(columns="_s"))
    rows = [("hallmark" if f == "hallmark" else "class", s, s, e) for f, s, e in h.itertuples(index=False) if f in ("hallmark", "class")]
    rows += [("class", f"broad:{s}", f"{s} (category)", e) for f, s, e in h.itertuples(index=False) if f == "broad"]
    r, top_of = reactome(u)
    rows += r
    tissue_go = {x for v in RU.TISSUE_PROGRAMMES.values() for x in v if x.startswith("GO:")}
    rows += go_sets(u, tissue_go)
    rows += keyword_sets(u)
    m = pd.DataFrame(rows, columns=["library", "set_id", "name", "hsapiens_gene_id"]).drop_duplicates(["library", "set_id", "hsapiens_gene_id"])
    m["key"] = m.library + "|" + m.set_id
    m.to_parquet(out / "membership.parquet")
    s = m.groupby(["key", "library", "set_id", "name"]).size().rename("n_vocab").reset_index()
    s["reactome_top"] = s.set_id.map(top_of)
    s.to_csv(out / "sets.csv", index=False)
    print(s.groupby("library").size())
    missing = [x for v in RU.TISSUE_PROGRAMMES.values() for x in v
               if not ((x.startswith("hallmark:") and (s.library.eq("hallmark") & s.set_id.eq(x.split(":", 1)[1])).any())
                       or (s.set_id == x).any())]
    print("tissue programmes missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
