# E15 pre-registration (23 Oct 2026): a direction-independent one-sided gate

Committed before any E15 code.

## Question
E13 Part A showed the GRD gate's power depends on the direction of a variance decrease relative to the control spectrum (T0h power at v=0.5, median n_e, alpha 0.01: PC1 0.00 to 0.01, PC10 1.00). The statistic lambda_max(Omega_e - Omega_0) is not affine-invariant. The two-sided affine-invariant test (G8 in E14, T2 in E12/E13) sees variance increases too and certifies nearly everything on the synthetic benchmark (E14: coverage 0.97 clean, 0.96 contaminated). E15 tests the statistic between the two: one-sided like the gate, affine-invariant like G8.

## The new gate
G10, readout "geneig_down": T = -log(lambda_min) of the generalised eigenvalues of cov(Y_env) v = lambda cov(Y_ref) v (scipy.linalg.eigh(S_env, S_ref)). Population target: lambda_min(Sigma_0^{-1/2} Sigma_e Sigma_0^{-1/2}) < 1 if and only if Sigma_e is not Loewner-above Sigma_0, the same condition as lambda_max(Omega_e - Omega_0) > 0 (Lemma 1 of the manuscript). So G10 tests the same precondition as the gate, with a statistic invariant to any common invertible linear map. NaN or non-positive eigenvalues are handled exactly as geneig_signal handles them. Everything else is the gate's: same null design, same null draws (same rng stream as the precision readout in each setting), BH at q.

## Three arms, each reusing committed code paths
A. Synthetic matched coverage (E14 design, unchanged): the 900 E10 units, readout called through E14's _call with the same rng (910000 + seed), B=500, BH within each E10 family over E10.GRID["G2"], decisions and curves exactly as G8. G0, G2, G5, G8, G9 are read from results/e14 (not recomputed), except that the data pass recomputes the precision p-values and asserts equality with E10's stored values (E14's regression); any mismatch voids the run. Bootstrap: E14's 1000 draws over the 200 (arm, level, seed) systems, default_rng(120000), G10 added to every comparison. Scopes ABD and C as in E14.

B. Real panel (E12 held-out construction, unchanged): for K562, RPE1, Norman and seeds 0, 1, 2, tasks and shared from e12_standard_tests.build_tasks(panel, seed, 9999, 10, False); per retained real environment and per fake, G10 against the full corrected REF exactly as T0h is computed in e12_common.run_environment (same REF correction), through precision_readout.detect_with_pvalues with readout="geneig_down", B=9999, rng_for(seed, i, K_NULL) (the T0h stream, so null index draws are paired). BH at q=0.05 over retained reals plus 50 fakes in file order, as e12_aggregate does. T0h p-values are read from the stored E12 files for the paired comparison.

C. Planted power (E13 Part A design, unchanged): the E13 replicate structure (FAKE-pool cells, CRN index i = 1000000 + 10000 s + 1000 q + r, k=1 cells, k=2 REF rows, k=3 labels), directions PC1, PC5, PC10, random, sizes 200 and median n_e, arms A0 (300 reps per size), A1 (v in 0.9, 0.8, 0.7, 0.6, 0.5, 0.3), A2 (v in 1.1, 1.25, 1.5, 2, 3), A3 (delta in 0.05, 0.1, 0.2, 0.3, 0.5), 100 reps per cell; A4 is not run. G10 only, against the full REF (as T0h), B=999, rng stream k=10 (T0h's stream in E13, paired). Rejection at alpha 0.01, 0.05, 0.001. T0h, T2 power read from results/e13_power_sign. K562 and RPE1 run on the rebuilt files (E13 amendment 1 applies: any reproduction check against stored values uses tolerance 1e-6 for those two).

## Predictions (recorded so they can fail)
- P1 direction independence (C): at median n_e, alpha 0.01, A1 v=0.3: G10 power >= 0.8 in every direction and every dataset. (T0h at v=0.3 along PC1: 0.01, 0.00, 0.01.)
- P2 power cost (C, recorded as a trade-off, held or failed): at median n_e, alpha 0.01, A1 v=0.5: mean over directions of G10 power < mean over directions of T0h power, in every dataset.
- P3 blindness kept (C): mean G10 power over all A2 and A3 cells at alpha 0.05 <= 0.08 in every dataset (as T0h).
- P4 calibration (B and C): G10 BH fakes <= 3/50 in every dataset and seed; A0 rejection at alpha 0.05 in [0.02, 0.09] for every dataset and size (a value below the band is reported as conservative).
- P5 synthetic (A): on ABD, AURC(G10) <= AURC(G2) + 0.02 and matched risk(G10) <= matched risk(G2) + 0.02; on C, |matched risk(G10) - matched risk(G2)| <= 0.02 (or undetermined if G10's curve does not reach G2's coverage).
- P6 real (B, descriptive with one threshold): K562 G10 BH count <= 20 of 385 in every seed (E13 found 4 significant decreases against a size-matched REF draw).

## Consequences fixed in advance
- P4 fails: G10 is uncalibrated; reported, no recommendation, and nothing else from E15 is used in the manuscript's claims.
- P1 and P5 hold (and P4): G10 is presented as a direction-independent alternative gate for the same precondition, with P2's power trade-off stated; the manuscript recommends reporting both when the control spectrum is anisotropic.
- P1 fails: direction dependence at these sizes is shared by the affine-invariant one-sided statistic; reported as a sample-size limitation of covariance-based gates, not a defect of the precision statistic alone.
- P5 fails: G10 is not recommended as a replacement on the synthetic benchmark; reported.
- P3 fails: G10 responds to mean shifts or increases (a bug or a misunderstanding); STOP and report before interpreting anything else.

## Not allowed
No change to B, q, sizes, directions, levels, streams, the readout definition or the predictions after the first run; no added readouts; no dropped datasets, seeds or units; scripts and results committed together.
