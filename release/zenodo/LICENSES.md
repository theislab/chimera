# Third-party data in the Chimera deposit: source, license, and whether it is redistributed

Rule applied: redistribute only processed derivatives of sources with an open license or no use
restriction, always with attribution; link (do not redistribute) where the license is unstated or
unclear; never redistribute raw downloads that the source itself serves. Every linked download
has a manifest (`MANIFEST*.tsv`: URL, date, sha256) in `data/analysis/**` (deposited in the bundle
`data_other_cohorts`); the files that the data preparation of liver and testis downloads (CAMEX files,
SATURN protein embeddings, eye count matrices) are listed in `data/raw/MANIFEST.tsv` (bundle
`data_liver_testis`). "To verify" marks a license that was not confirmed from the source page.
Bundle names in the last column are those of `release/zenodo/bundles.yaml` in the code repository.

| Source | Used for | License / terms | In the deposit |
|---|---|---|---|
| CAMEX processed bundle (Guo, Huang, Zhang 2026), archived at figshare doi:10.6084/m9.figshare.31131808.v1; the data preparation downloads it from the CAMEX authors' Google Drive folder (folder id in `chimera/_data_utils.py`) | Raw liver and testis matrices of the 7 canonical cohorts | MIT (figshare record) | Processed h5ad only (`data_liver_testis`); raw bundle linked, each downloaded file with its figshare address, size, MD5 and sha256 in `data/raw/MANIFEST.tsv` (the files are byte-identical in both places) |
| Guilliams et al. 2022, GEO GSE192742 | Liver human, macaque, mouse (via CAMEX); mouse SD/WD diet series | GEO: NCBI places no restriction on use; submitters may hold rights (attribution) | Processed objects, pseudobulks, obs with re-annotation; raw linked |
| Wang et al. 2023, GEO GSE198832 | Liver zebrafish (via CAMEX) | GEO terms as above | Processed only |
| Murat et al. 2023, ArrayExpress E-MTAB-11063/-11068/-11071 | Testis cohorts; held-out species (H4 gold) | EMBL-EBI terms of use, freely available (attribution) | Processed only; derived H4 per-gene values in results |
| Van Zyl et al. 2020, GEO GSE146186, GSE146187, GSE148371, GSE148373, GSE148374 | Eye cohort (aqueous-humor outflow tissues) | GEO terms as above | Processed objects (`data_other_cohorts`); raw linked |
| Tabula Sapiens, Tabula Microcebus, Tabula Muris Senis (CELLxGENE) | Bone marrow (+ spleen), H3 liver gold (TS, TMS liver) | CC BY 4.0 (CELLxGENE Discover) | Bone-marrow object; H3 profiles in results; raw linked |
| KPMP human and mouse kidney (CELLxGENE collection 9c9d04c4) | Kidney AKI case | CC BY 4.0 (CELLxGENE Discover) | Healthy training object, pseudobulks; raw linked |
| Human islet 48-donor atlas, Hrovatin mouse islet (CELLxGENE) | Islet T2D case | CC BY 4.0 (CELLxGENE Discover) | Healthy training object, pseudobulks, obs; raw linked |
| GSE202379 (Gribben 2024), GSE135251, GSE136103, GSE129516 (GEO) | Liver MASH, bulk direction, second contrast | GEO terms as above | Pseudobulks, author metadata table (`disease/meta`); raw linked |
| Shay et al. 2013, GEO GSE24759, GSE15907 | Bone-marrow COE gold | GEO terms as above | Recomputed COE table; portal cross-check not deposited |
| SATURN ESM-2 protein embeddings (`esm2_t48_15B_UR50D`, snap.stanford.edu) | Gene features | Code MIT; data license not stated | **Not redistributed**; derived GIN features inside the h5ad files; the downloaded archive is listed in `data/raw/MANIFEST.tsv` |
| EVEE evolutionary statistics (Chen et al. 2019, Broad portal, Wayback copy) | H1, H2 gold | **License unstated** | **Not redistributed**: per-gene EVEE columns (`G_H1`, `G_H2`, `evee_mean_*`) dropped from every deposited table (`scrubbed.tsv`); summary statistics and correlations kept; link in `data/analysis/gold/G1/MANIFEST.tsv` |
| Ensembl 99-110 (BioMart, Compara, GTF, CDS, REST) | Ortholog tables, prior, dN/dS, CDS covariates, sequences | No restrictions | Ortholog tables and derived values; dumps linked |
| gnomAD v4.1 constraint | LOEUF gold | ODbL 1.0 per gnomAD (to verify) | Per-gene LOEUF in derived gold tables (attribution); raw linked |
| AlphaFold DB, UniProt | Structures and annotations in `notebooks/cache/protein_structure` | CC BY 4.0 | Cached files redistributed with attribution |
| 3Dmol.js (`3Dmol-min.js` in the same cache) | Structure rendering | BSD 3-Clause | Redistributed |
| STRING 12.0 | F2/F3 network, prior swap | CC BY 4.0 | Derived statistics; raw linked |
| ChIP-Atlas | F4 | CC BY 4.0 | Derived statistics; raw linked |
| Reactome | Gene sets | CC0 | Derived statistics; raw linked |
| Enrichr libraries (MSigDB Hallmark 2020, Reactome 2024) | Gene sets | Enrichr free use; MSigDB terms (to verify) | GSEA results only; libraries linked |
| HGNC, GO (goa_human, go-basic) | Gene groups, V3 set | HGNC free use; GO CC BY 4.0 | Set memberships in derived tables; raw linked |
| HuRI, CollecTRI (OmniPath), Skinnider 2021, Lambert 2018, Housekeeping (Eisenberg), Cippà 2018, Gharahkhani 2021, Hagai 2018 | Functional layer, sets, anchors | Unverified / per-resource / article supplements | Derived statistics only; raw linked |
| Bakken et al. 2021 (M1), Ma et al. 2022 (DLPFC), SEA-AD MTG; Travaglini 2020, Tabula Microcebus, Tabula Muris Senis lung; Kedlian 2024 muscle (all CELLxGENE Discover) | Cortex, lung, skeletal-muscle cohorts; held-out MTG; Alzheimer transfer | CC BY 4.0 (CELLxGENE Discover) | Training objects, ortholog tables, scalers (`data_other_cohorts`, `data/analysis/new_usecases/c1_build/objects`); raw linked (`data/analysis/new_usecases/MANIFEST_SHA256.tsv`) |
| Zilionis et al. 2019 GEO GSE127465; lung fibrosis GSE135893, GSE141259; 5xFAD GSE140511 (GEO) | Tumor myeloid cohort; disease transfers | GEO terms as above | Tumor training object; disease DE summaries in results; raw linked |
| ESM-2 embeddings computed in SATURN format for chimpanzee and marmoset (`data/analysis/new_usecases/esm`), with the model cache | Gene features of the DLPFC cohorts | Derived from ESM-2 (model MIT); treated as the SATURN embeddings | **Not redistributed**; derived GIN features inside the h5ad files |
| Ensembl 98, 110, 116 GTF and Compara homologies; Ensembl Compara extracts of the ortholog audit | Identifier-joined objects (O1), cohort ortholog checks | No restrictions | Identifier-joined objects and their column logs (`data_identifier_joined`); dumps linked |
| GO (go-basic, GO slim, goa_human), Reactome, UniProt, Pfam, Complex Portal | Program atlas (E1), protein architecture (E2) | GO, UniProt, Pfam CC BY 4.0; Reactome, Complex Portal CC0 | Set memberships and annotations in result tables; raw linked |
| Human Protein Atlas (proteinatlas.tsv), GTEx v8 medians, Collins 2022 dosage scores, 2R ohnolog lists (OHNOLOGS) | Protein localization, tissue breadth, dosage, duplication class (E2) | HPA CC BY-SA 3.0; GTEx open access; Collins 2022 Zenodo CC BY 4.0 (to verify); OHNOLOGS academic, unstated (to verify) | Per-gene annotation columns in E2 tables (attribution); raw linked |
| MGI, IMPC, Disease Ontology, GenCC, Genomics England PanelApp, Open Targets, HPO, ClinGen | Translational reading (E4), functional uses (Y2, Y6) | MGI, IMPC, Open Targets CC BY 4.0; DO CC0; GenCC CC0 (to verify); PanelApp, HPO, ClinGen own terms (to verify) | Derived statistics and gene classes in result tables; raw linked |
| ENCODE SCREEN cCRE V4 and contact domains, TargetScanHuman 8.0, OmniPath DoRothEA A-C and CollecTRI, Eisenberg-Levanon housekeeping, MSigDB 2025.1 C2 CGP | Regulatory architecture (E5) | ENCODE unrestricted; TargetScan free use (to verify); DoRothEA GPL-3, CollecTRI CC BY 4.0 (academic license filter); MSigDB CC BY 4.0 for C2 CGP (to verify) | Per-gene counts and set memberships in E5 tables; raw linked |
| Zoonomia phyloP, UCSC hg38, DepMap 24Q4, ABC model, NCBI gene2ensembl/gene2pubmed, EPD; allele-specific expression studies (Agoglia 2021, Wang 2024), Minikel 2024 | Functional and regional layers (Y2, Y4, Y6) | Zoonomia, UCSC, NCBI unrestricted; DepMap CC BY 4.0; ABC, EPD, article supplements unverified | Derived statistics only; raw linked |
| Literature species-difference lists (E6: Jorstad 2023, Hodge 2019, Bakken 2021, Krienen 2020, Ma 2022, Khrameeva 2020, Caglayan 2023, Benner 2014, Baron 2016, Aizarani 2019, Guilliams 2022, Shami 2020, kidney and lung studies) | Recovery of known differences (E6) | Article supplements (journal terms; unverified) | Gene lists derived in the code release with source and rule; per-list results in E6 tables; supplements linked |
