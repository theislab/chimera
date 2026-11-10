#!/usr/bin/env bash
#SBATCH --job-name=c1_mtg
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
# Jorstad 2023 MTG (collection 4dca242c): human, macaque, marmoset, reference only, never in training.
D=data/analysis/new_usecases/brain_mtg
for f in 224db6b4-2895-4c4e-8503-bb647fae9b29 261dd26f-1507-41a5-94e5-b96903de74a9 c556b8e8-e348-49b9-b4da-08b05cd43350; do
  u=https://datasets.cellxgene.cziscience.com/$f.h5ad
  curl -sS -f -L --retry 5 -C - -o $D/$f.h5ad $u && printf 'brain_mtg\t%s.h5ad\t%s\t%s\t%s\n' $f $u "$(date -u +%FT%TZ)" "$(stat -c %s $D/$f.h5ad)" >> data/analysis/new_usecases/MANIFEST_C1.tsv
done
