#!/usr/bin/env bash
#SBATCH --job-name=c1_ens
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
# Ensembl 110 reference files that the cohorts of this folder need and no earlier task fetched.
E=https://ftp.ensembl.org/pub/release-110
D=data/analysis/new_usecases/ensembl110
mkdir -p $D
get() { curl -sS -f -L --retry 5 -o "$D/$2" "$1" && printf '%s\t%s\t%s\t%s\n' "$2" "$1" "$(date -u +%FT%TZ)" "$(sha256sum "$D/$2" | cut -d' ' -f1)" >> $D/MANIFEST.tsv; }
for sp in pan_troglodytes callithrix_jacchus macaca_mulatta danio_rerio gallus_gallus; do
  n=$(curl -s $E/gtf/$sp/ | grep -o 'href="[^"]*\.110\.gtf\.gz"' | cut -d'"' -f2)
  get $E/gtf/$sp/$n "$n"
  c=$(curl -s $E/fasta/$sp/pep/ | grep -o 'href="[^"]*\.pep\.all\.fa\.gz"' | cut -d'"' -f2)
  get $E/fasta/$sp/pep/$c "$c"
done
for sp in macaca_mulatta danio_rerio gallus_gallus; do
  get $E/tsv/ensembl-compara/homologies/$sp/Compara.110.protein_default.homologies.tsv.gz "Compara.110.protein_default.homologies.$sp.tsv.gz"
done
ls -la $D
