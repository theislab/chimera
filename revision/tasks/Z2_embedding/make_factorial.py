"""Writes factorial.tsv: run, cohort, level, seed, input mode, run_experiment args. Row order is priority order and only appends, since array tasks read their row when they start:
liver seeds 42, 43 in level order; testis seeds 42, 43; then liver seed 44. python -m revision.tasks.Z2_embedding.make_factorial"""
from pathlib import Path
from revision.tasks.Z2_embedding import rules as RU


def row(c, lv, s):
    mode, extra = RU.LEVELS[lv]
    base = RU.BASE[c]
    if lv == "gmmd":
        base = base.replace(" --celltype-conditional-mmd --celltype-key cell_ontology_class", "")
    return "\t".join([RU.run_name(c, lv, s), c, lv, str(s), mode, f"{base} {extra} --seed {s}".replace("  ", " ")])


rows = ["run\tcohort\tlevel\tseed\tinput\targs"]
L, T = RU.COHORT_LEVELS["liv_hmouse"], RU.COHORT_LEVELS["test_hmouse"]
rows += [row("liv_hmouse", lv, s) for lv in L for s in (42, 43)]
rows += [row("test_hmouse", lv, s) for lv in T for s in (42, 43)]
rows += [row("liv_hmouse", lv, 44) for lv in L]
rows += [row("liv_hmouse", lv, s) for lv in ("adv50mmd10", "adv20mmd10", "adv50mmd5", "adv20mmd5") for s in (42, 43)]   # amendment 12
# amendment 13
rows += [row("liv_hmouse", "novamp", s) for s in (45, 46, 47)] + [row("test_hmouse", "novamp", s) for s in (42, 43, 44)] + [row("islet", "novamp", s) for s in (42, 43)] + [row("kidney", "novamp", 42)]
rows += [row("liv_hmouse", lv, s) for lv in ("novamp_adv400", "novamp_mmd100") for s in (42, 43)]
p = Path(__file__).parent / "factorial.tsv"
p.write_text("\n".join(rows) + "\n")
print(len(rows) - 1)
