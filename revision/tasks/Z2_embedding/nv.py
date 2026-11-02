import pandas as pd
pd.set_option('display.width',200)
q=pd.read_csv('results/revision/Z2/quick.csv')
print(q[q.level.str.startswith('novamp')|q.level.eq('adv0mmd0')][['cohort','run','mix','lt','rself','rcross','auc','held_auc','held_auc_depth','wv','pc1_share','within_z_var_by_lib']].round(3).to_string())
