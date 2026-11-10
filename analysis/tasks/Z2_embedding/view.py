import sys, glob, pandas as pd, numpy as np
pd.set_option('display.width',220,'display.max_rows',300,'display.max_columns',40)
c=sys.argv[1]
d=pd.concat(pd.read_parquet(f) for f in glob.glob(f'results/analysis/Z2/measure/{c}__*.parquet'))
d=d[~d.latent.str.startswith('z2-')]
t=d[~d.task.isin(['labelfree','global'])]
print(t.groupby(['task','metric','method']).value.mean().unstack('method').round(3).to_string())
l=d[d.task=='labelfree']
print(l.groupby(['kind','metric','method']).apply(lambda x: np.average(x.value,weights=x.n),include_groups=False).unstack('method').round(3).to_string())
g=d[(d.task=='global')&d.metric.isin(['knn_mixing_entropy','knn_species_purity','label_transfer_acc','sil_celltype','sil_species','knn_celltype_purity'])]
print(g.groupby(['metric','method']).value.mean().unstack('method').round(3).to_string())
if c=='liv_hmouse':
    x=d[(d.method=='Chimera')&(d.task=='hep_zone')&(d.metric=='rho_self')].groupby('latent').value.mean()
    y=d[(d.method=='Chimera')&(d.metric=='knn_mixing_entropy')].set_index('latent').value
    print(pd.concat([x,y],axis=1).round(3))
