E12b amendment 2 (19 Oct 2026): K562 depth from the raw per-cell UMI count (E12d)

Written before any E12d run. The E12b pre-registration's hypotheses, readouts, BH families, decision rules S1 to S3 and batch rule are unchanged. Only K562's depth source changes.

Why
- E12b took K562 depth from the X row sum (fallback 3 of the prereg) because causalbench_k562.h5ad holds 1,158 selected genes and no library size. By rule (i) the global verdict was S3, so D1 (the mean-shift headline) stayed open for K562 even though R3 retained 98% of the E12 T1 detections and R4 97%.
- The data build on the home PC (commit fbb81c8, experiments/verify_causalbench_data.py) read the raw figshare file 35773219 and extracted obs["UMI_count"], the per-cell total UMI count of Replogle et al. 2022, for the 240,148 cells kept by CausalBench, aligned row for row to causalbench_k562.h5ad (gene_id labels equal guide_ids on every row; raw rows 310,385 = npz rows, 0 zero-sum rows dropped). Kept rows: min 1,817, median 13,354, max 62,866; correlation with the raw X row sums 0.99994. Saved as $GRD_DATA_ROOT/build/k562_umi_count_aligned.npy. This is library size, source 1 of the prereg (an obs total-count column), supplied from outside the h5ad.
- Norman's depth was corrected the same way in E12c (deviation note 5268eb3). RPE1 keeps the E12b fallback result (reported only, no decision role).

Change
- E12d = E12b for K562 with depth = log(raw UMI_count), depth_fallback = False. Implemented without editing e12b_confound_check.py: experiments/e12d_k562_depth.py imports it, replaces only the depth returned by inspect_h5ad for K562 (length and positivity checked, STOP otherwise), routes outputs to results/e12d_k562_depth/, and reuses e12b's sanity stop (median BASIS library size must be at least 500).
- Aggregate: K562 from E12d, Norman from E12c, verdict by the unchanged rules (i) to (iv). RPE1 is listed as missing in that aggregate by design; its E12b block stands.
- Seed 0, d = 10, B = 9999, the same environments, REF draws, fakes, streams k = 10 and 11, as E12b.
- Reproduction tolerance: e12b recomputes the E12 T1 statistic for every environment and stops if it differs from the stored value by more than 1e-8. causalbench_k562.h5ad on the home PC is the rebuilt file (notes/e13_amendment_1.md): statistics differ from the A100 file by about 3e-8 and every checked p-value is identical. As in E13 amendment 1, the K562 tolerance is 1e-6 (set by the wrapper, recorded in the output). Norman and RPE1 are not rerun here.
- The alignment of the external vector to the h5ad rows is established by label equality on every row plus the 0.99994 correlation with the raw row sums, not by a row identifier (the h5ad has none).

Predictions (recorded so they can fail)
- K562 R3 retains at least 70% of the E12 T1 BH count (E12b with the fallback: 98%).
- K562 R4 retains at least 70% (E12b: 97%).
- |corr(PC1, depth)| in BASIS cells stays above 0.5 (E12b with the X row sum: -0.88), so R4 stays decision-relevant.
- If all three hold and R3 fakes are at most 3/50, rule (iii) gives S1: the mean-shift headline survives depth and PC1 removal in both decision datasets; D1 is confirmed.

Not allowed
- No change to thresholds, B, seed, d, the 500 sanity floor, or the BH families. No rerun with another depth transform. A failed prediction is reported as a result.

Reporting
- E12b's K562 block is reported as "fallback depth, superseded by E12d". E12d carries the K562 depth result, labelled "raw UMI_count, amendment 2". E12c carries Norman. The global verdict comes from the E12d aggregate.

Amendment 2a (19 Oct 2026, after the E12d smoke run stopped; written before any full run)
- The 1e-6 absolute tolerance stopped the smoke run: ENSG00000006712 has a stored T1 of 354.2 and reproduced to 4.3e-6 absolute, 1.2e-8 relative. Relative differences over the 5 gate environments were 2.6e-9 to 1.5e-7, the same float32-level noise as E13 amendment 1; an absolute bound is the wrong shape for Hotelling's T1, which reaches 2388.8 in K562.
- Change: the absolute gate becomes 2.5e-3 (1e-6 times the largest stored K562 T1), and the wrapper adds a relative check after the run, |diff| / |stored T1| <= 1e-6 on every one of the 435 environments, STOP otherwise. Both tolerances and the observed maxima are recorded (reproduction block and k562_seed0_d10_repro_relative.json).
- Why this cannot change a result: R3 and R4 retention are computed from the stored E12 p-values (T1_e12_p), not from the recomputed statistics; the gate only confirms the code path reproduces E12 on this data.
- Nothing else changes.
