import pandas as pd
pd.set_option('display.width',200,'display.max_rows',100)
r=pd.read_csv('results/analysis/Z2/reading.csv'); l=pd.read_csv('results/analysis/Z2/reading_loo.csv')
print(l.groupby('cohort')[['rho_loo','rho_loo_all']].agg(['median','min','max']).round(3).to_string())
print(r.groupby(['cohort','level'])[['healthy','rho_DS','rho_DS_partial','rho_D_consensus','rho_S_consensus','frac_S_pos','sat_hi','tail_up_raw','tail_dn_raw','adv_acc','mix_entropy']].mean(numeric_only=False).round(3).to_string())
