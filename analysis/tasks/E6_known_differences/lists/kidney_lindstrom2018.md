# kidney_lindstrom2018: Lindström et al. 2018, mouse nephron anchor genes in human fetal kidney

**Citation.** Lindström NO, McMahon JA, Guo J, et al. Conserved and divergent features of human and mouse kidney organogenesis. *J Am Soc Nephrol* 29:785–805 (2018). DOI 10.1681/ASN.2017080887. PMC5827606.
**DOI check.** `curl -sIL https://doi.org/10.1681/ASN.2017080887` redirects through ovid.com to the JASN article, then 200. Crossref metadata matches.

**File.** `chimera/data/analysis/E6/kidney_lindstrom2018/ASN.2017080887t2.jpg`: Table 2, "Validation of mouse anchor genes in human fetal samples". PMC renders this table only as an image. It was **manually transcribed** into the derive script; the sha256 of the image is recorded in each row's note.

**Rule (Table 2 columns).** "Reproduces primary expression" (Yes/No) × "Additional sites of expression" (Yes/No). The table footnote says green rows = conserved.
| Reproduces | Additional | Set | Direction |
|---|---|---|---|
| Yes | No | conserved_control | conserved |
| Yes | Yes | conserved_with_additional_human_sites | conserved |
| No | Yes | species_divergent | up_in_mouse |
| No | No | not_detected_in_human | up_in_mouse |

Direction is relative to the mouse domain: a mouse marker not reproduced in that human domain counts as up_in_mouse. Cell type = the mouse domain (EPT, LOH, RC, RV). The human domain is in `note`.

**Counts (unique genes, n = 26).**
| Set | Genes |
|---|---|
| conserved_control (3) | SLC22A6, ENTPD5, UMOD |
| conserved_with_additional_human_sites (9) | BDH2, SLC27A2, SLC3A1, TCN2, METTL7B, FBP1, GLYCTK, GPSM3, TDRD5 |
| species_divergent (8) | ACE, ACSM1, CRYL1, PRODH2, SORD, C2, AQP11, VIP |
| not_detected_in_human (6) | FMO2, MOGAT2, SLC18A1, SLC6A20, SPP2, NPY |

**Caveats.**
- The paper text gives 3 / 6 / 11 / 6 for these categories. The table as transcribed gives 3 / 9 / 8 / 6, so 3 genes differ between the "additional" and "different population" categories. The table is used here as the primary source, but transcription error is possible; recheck the image for TDRD5, GPSM3 and GLYCTK.
- Human data are week 14–16 fetal SISH, i.e. developmental and qualitative.
- **Circularity: no.**

**Script.** `derive/derive_kidney_lindstrom2018.py <table2_jpg> <out_tsv>`
