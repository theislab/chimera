"""Print per-run and per-cohort diagnostics. python analysis/tasks/Z1_retune/table.py <diag.tsv> [--runs]"""
import sys

import pandas as pd

t = pd.read_csv(sys.argv[1], sep="\t")
cols = ["rho_DS", "rho_DS_partial", "rho_DSbar", "rho_cS", "frac_support", "tail_up_raw", "tail_dn_raw", "sat_hi", "n_active_z",
        "adv_acc", "mix_entropy", "lr_freeze_epoch", "last_epoch", "max_learned_var", "rho_S_expr", "rho_D_expr"]
pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
pd.set_option("display.float_format", lambda x: f"{x:.3g}")
if "--runs" in sys.argv:
    print(t[["cohort", "run", "no_nan", "healthy"] + cols].to_string(index=False))
ok = t[t.get("no_nan", True) == True]
g = ok.groupby("cohort", sort=False)
s = g[cols].mean()
s.insert(0, "n", g.size())
s.insert(1, "rho_min", g["rho_DS"].min())
s.insert(2, "rho_max", g["rho_DS"].max())
print(s.to_string())
