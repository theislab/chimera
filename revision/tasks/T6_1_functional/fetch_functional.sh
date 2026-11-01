#!/usr/bin/env bash
# T6.1 downloads. Usage: fetch_functional.sh <chimera_repo>
C="$1"; F="$C/revision/tasks/T4_1_gold/fetch_item.sh"; D="$C/data/revision/functional"; M="$D/MANIFEST.tsv"
S=https://stringdb-downloads.org/download
"$F" HuRI https://interactome-atlas.org/data/HuRI.tsv "$D/HuRI.tsv" "$M"
for t in 9606 10090; do
    "$F" STRING "$S/protein.physical.links.full.v12.0/$t.protein.physical.links.full.v12.0.txt.gz" "$D/$t.protein.physical.links.full.v12.0.txt.gz" "$M"
    "$F" STRING "$S/protein.info.v12.0/$t.protein.info.v12.0.txt.gz" "$D/$t.protein.info.v12.0.txt.gz" "$M"
    "$F" STRING "$S/protein.aliases.v12.0/$t.protein.aliases.v12.0.txt.gz" "$D/$t.protein.aliases.v12.0.txt.gz" "$M"
done
"$F" Reactome https://reactome.org/download/current/ReactomePathways.gmt.zip "$D/ReactomePathways.gmt.zip" "$M"
"$F" CollecTRI "https://omnipathdb.org/interactions?datasets=collectri&organisms=9606&fields=sources,references&genesymbols=1" "$D/collectri_human.tsv" "$M"
for g in hg38 mm10; do
    for tf in HNF4A CEBPA FOXA1 FOXA2 HNF1A ONECUT1 RXRA PPARA NR1H3; do
        name="$tf"; [ "$g" = mm10 ] && name="$(echo "${tf:0:1}$(echo "${tf:1}" | tr 'A-Z' 'a-z')")"
        "$F" "ChIP-Atlas" "https://chip-atlas.dbcls.jp/data/$g/target/$name.5.tsv" "$D/chipatlas/$g/$name.5.tsv" "$M"
    done
done
# Skinnider et al. 2021 Cell 184:4073 (PII S0092867421007042), Elsevier supplementary files.
# mmc2 = "Table S3: PCP-SILAM interactomes of seven mouse tissues" (its own legend sheet);
# mmc1 = per-tissue protein identifications, needed to restrict to proteins detected in liver.
for n in 1 2; do
    "$F" Skinnider2021 "https://ars.els-cdn.com/content/image/1-s2.0-S0092867421007042-mmc$n.xlsx" "$D/skinnider2021/Skinnider2021_mmc$n.xlsx" "$M"
done
