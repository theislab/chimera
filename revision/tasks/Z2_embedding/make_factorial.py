"""Writes factorial.tsv: run, cohort, level, seed, input mode, run_experiment args. python -m revision.tasks.Z2_embedding.make_factorial"""
from pathlib import Path
from revision.tasks.Z2_embedding import rules as RU

rows = ["run\tcohort\tlevel\tseed\tinput\targs"]
for c, levels in RU.COHORT_LEVELS.items():
    for lv in levels:
        mode, extra = RU.LEVELS[lv]
        base = RU.BASE[c]
        if lv == "gmmd":
            base = base.replace(" --celltype-conditional-mmd --celltype-key cell_ontology_class", "")
        for s in RU.FACTORIAL_SEEDS[c]:
            rows.append("\t".join([RU.run_name(c, lv, s), c, lv, str(s), mode, f"{base} {extra} --seed {s}".replace("  ", " ")]))
p = Path(__file__).parent / "factorial.tsv"
p.write_text("\n".join(rows) + "\n")
print(len(rows) - 1)
