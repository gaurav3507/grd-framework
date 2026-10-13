E12c: deviation from the E12b pre-registration (Norman depth source)

- What went wrong: E12b took Norman depth from obs["UMI_count"], which the prereg's column-name list matched. That column is the guide UMI count from Norman et al.'s cell_identities table, not the cell library size: the mean log value in BASIS cells is 3.74 (about 42 UMIs).
- Change: depth = log of the row sum of layers["counts"] (integer UMI counts over the 5,000 retained genes). This tracks library size, but it is not full-transcriptome.
- Everything else is unchanged: seed 0, d=10, B=9999, the same environments, REF draws, fakes, R3/R4 definitions, BH families and the S1/S2/S3 thresholds.
- Reporting: E12b's Norman depth result is reported as invalid (wrong column). E12c is reported as the Norman depth result, labelled as a post hoc correction.
- K562 and RPE1 are unchanged: depth cannot be tested from the available files (normalised selected genes only, no barcodes).
- The global verdict stays S3 (rule (i), K562 on the depth fallback). E12c reports Norman's own per-dataset rule outcome (meets_ii, meets_iii, R3 retention).
- Sanity stop: if the median BASIS row sum of the counts layer is below 500, the run stops (the guide-UMI column gave about 42; real library size over 5,000 genes is in the thousands).
- Prediction (Claude's): Norman R3 retains >= 70% of the E12 T1 BH count.
