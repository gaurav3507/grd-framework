E13 amendment 1 (19 Oct 2026): reproduction tolerance for the rebuilt K562 and RPE1 files

Written before any E13 Part A or Part B run. Nothing in notes/e13_prereg.md changes except the reproduction tolerance below.

What happened
- E13 runs on the home PC, not the A100. Norman2019_raw.h5ad is the original file (SHA-256 equal to the A100 manifest). causalbench_k562.h5ad and causalbench_rpe1.h5ad were rebuilt from the figshare raw files with CausalBench 1a2143cf and the A100's 20_to_anndata logic (commit fbb81c8). They match the A100 files on shape, control count, guide count, control mean and RMS, and file size in bytes; their SHA-256 differs.
- e13_power_sign.py --verify (Step 0d, tolerance 1e-8 on |stat diff| + |p diff|) passed on Norman (3.3e-15) and failed on K562 (3.4e-8) and RPE1 (2.5e-8).
- Diagnostic (commit 6fd090b, outputs in ~/grd_data/build/diag on the home PC):
  - the 50 fakes, which skip target correction, show the same size of difference (max relative stat diff 1.2e-7 K562, 8.7e-8 RPE1);
  - every p-value checked is identical to the stored E12 value (|p diff| = 0 for 55 environments per dataset);
  - numpy 1.24.4 / Python 3.10 (the A100 versions) and numpy 2.5.4 / Python 3.14 give the same difference to 8 significant digits, and both reproduce Norman exactly;
  - a second rebuild with the A100 build pins on Python 3.10 is bit-identical to the first.
- Conclusion: the rebuilt K562/RPE1 matrices differ from the A100 files at float32 rounding level (relative differences of 1 to 2 float32 ulps, eps = 6e-8). The cause is not established. The file sizes are byte-identical (uncompressed csr, so the sparsity pattern is the same) and the split files written by the build (per-perturbation cell counts, usable set, train/held-out lists) must equal the A100's results/splits/*_zeroshot_split.json exactly, which pins the row composition; what remains is the values. The leading candidate is CPU-dependent summation order in the float32 per-cell normalisation (numpy reductions block by SIMD width, so float32 sums can round differently on different CPUs); not verified, since the A100 files cannot be compared directly.

Change
- For K562 and RPE1 only, the reproduction tolerance in Step 0d (--verify) and in the Part B check of max(T2down, T2up) against the stored E12 T2 statistic becomes 1e-6 absolute (about 10x the largest observed difference, far below the 1e-4 step of a B=9999 p-value, so any change in a p-value still fails). Norman keeps 1e-8.
- Implemented without editing e13_power_sign.py: experiments/e13_rebuilt_data.py imports it, sets its TOL to 1e-6 and calls its main(), and refuses --dataset norman. The tolerance used is written into each output (tol field of step0d and e12_T2_check).
- Everything else is unchanged: seeds, streams, B, grids, sizes, tests, decision rules, predictions.

Reporting
- Every K562 and RPE1 E13 number is labelled "computed on the rebuilt files, amendment 1".
- E12 conclusions are unaffected: the rebuilt files reproduce every checked E12 p-value exactly.
