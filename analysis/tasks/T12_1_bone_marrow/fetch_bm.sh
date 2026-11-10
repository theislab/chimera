#!/usr/bin/env bash
# T12.1 downloads (login node). Usage: fetch_bm.sh <chimera_repo>
# CELLxGENE h5ads (dataset ids in scouting S2): Tabula Sapiens Bone_Marrow, Tabula Microcebus
# Bone marrow, Tabula Muris Senis Bone marrow 10x. Ensembl 110 GTFs for mouse and mouse lemur
# (gene symbols for the ortholog table; the Compara 110 homologies TSV and the human GTF are
# already in data/analysis/ensembl110/). GEO platform annotations for the COE recomputation.
set -u
C="$1"; F="$C/analysis/tasks/T4_1_gold/fetch_item.sh"
D="$C/data/analysis/bone_marrow/raw"; M="$D/MANIFEST.tsv"
E="$C/data/analysis/bone_marrow/ensembl110"; ME="$E/MANIFEST.tsv"
mkdir -p "$D" "$E"
CX=https://datasets.cellxgene.cziscience.com
"$F" TS_BM "$CX/4aedb645-7487-4ab0-ad95-ee4db1b8eccd.h5ad" "$D/human_TS_bone_marrow.h5ad" "$M" &
"$F" TMC_BM "$CX/1af01471-603b-4055-a372-d85aab92eb5f.h5ad" "$D/lemur_TMicrocebus_bone_marrow.h5ad" "$M" &
"$F" TMS_BM "$CX/88dcbe63-d6eb-4fae-9362-e21b98365a42.h5ad" "$D/mouse_TMS_bone_marrow_10x.h5ad" "$M" &
wait
EF=https://ftp.ensembl.org/pub/release-110/gtf
"$F" E110_GTF_MM "$EF/mus_musculus/Mus_musculus.GRCm39.110.gtf.gz" "$E/Mus_musculus.GRCm39.110.gtf.gz" "$ME"
"$F" E110_GTF_MM "$EF/mus_musculus/CHECKSUMS" "$E/CHECKSUMS.mus_musculus" "$ME"
"$F" E110_GTF_MIC "$EF/microcebus_murinus/Microcebus_murinus.Mmur_3.0.110.gtf.gz" "$E/Microcebus_murinus.Mmur_3.0.110.gtf.gz" "$ME"
"$F" E110_GTF_MIC "$EF/microcebus_murinus/CHECKSUMS" "$E/CHECKSUMS.microcebus_murinus" "$ME"
# Section 12 spleen rule (lemur NK under 150 cells, decided on counts): spleen from all three atlases.
# Tabula Sapiens Spleen cee11228-9f0b-4e57-afe2-cfe15ee56312; Tabula Microcebus Spleen
# bf694f39-4342-4e75-aec0-d46db17f2036; Tabula Muris Senis Spleen 10x e2b469d4-b5c3-4a35-9d19-ee71ce61cae0.
if [ "${WITH_SPLEEN:-0}" = 1 ]; then
"$F" TS_SPL "$CX/5498457f-6a18-4766-b9a3-a2c8c4662703.h5ad" "$D/human_TS_spleen.h5ad" "$M" &
"$F" TMC_SPL "$CX/038459b5-a2d5-49be-afe7-a52db60f01a3.h5ad" "$D/lemur_TMicrocebus_spleen.h5ad" "$M" &
"$F" TMS_SPL "$CX/45c07569-0bba-4fc3-ad5d-adf3ebae4fe4.h5ad" "$D/mouse_TMS_spleen_10x.h5ad" "$M" &
wait
fi
