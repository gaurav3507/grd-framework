E12 pre-registration (predictions fixed in advance, before any code)

Q1: is the K562/Norman abstention a data property or a statistic blind spot?
Q2: are GRD real-data detections a basis-overlap artefact?

Perturb-seq (K562, Norman single-gene, RPE1)
- Control split per seed into disjoint parts: BASIS 30% / REF 40% / FAKE 30%.
- Exclusion: environments with n_e > |REF|//2 are excluded from the primary family for ALL tests and listed by name. Without this the GRD null silently switches to the bootstrap path. Expected: exactly one per dataset, Norman KLF1 and RPE1 ENSG00000108064; none for K562.
- Target gene: if the perturbation's target is feature column j, column j is set to the projection's centring mean in that environment and in that environment's reference copy. Applied as Y_corr = Y - (X[:, j] - mu_j) * P[j, :]. Fakes use the uncorrected REF. The count of corrected environments is recorded.
- 50 fakes drawn from FAKE at the dataset-median n_e.
- Tests, permutation p-values, B=9999, d=10 primary and d=30 secondary:
  T0h  GRD: PCA on BASIS; per environment, precision_readout.detect_with_pvalues([Y_i], Yref_i, rng=default_rng(SeedSequence([seed, i]))); reference = full corrected REF (the E3 design).
  T0i  the same, with PCA fit on REF itself (the E3 construction); reference = full corrected REF, not size-matched.
  T0m  GRD at matched size: env vs n_e REF rows drawn without replacement, null="pooled", held-out basis.
  T1   Hotelling T2 (also analytic F p, secondary).
  T2   Roy two-sided: max |log lambda_i| of the generalized eigenvalues of (S_e, S_ref).
  T3   Energy distance.
  T4   Box's M, analytic chi-square p; descriptive only.
- T0m and T1 to T4 use the held-out basis and the same size-matched reference draw per environment. T1 to T3 permute over the pooled 2*n_e rows.
- BH q=0.05 via precision_readout.bh_fdr, one family per dataset per test: retained real perturbations plus the 50 fakes.
- Environment index i counts ALL candidate perturbations in E3 order (load_panel order), including excluded ones, so the k=0 stream equals E3's [seed, i] stream.
- RNG: per environment i, SeedSequence([seed, i, k]) with k=0 for T0h/T0i (the E3 null stream), 1 for the size-matched REF draw, 2 for T0m, 3/4/5 for T1/T2/T3. Fakes use i + 100000. The control split uses SeedSequence([seed, 7777]).
- Raw rejection means p <= 0.05, for every test.
- Seeds 0, 1, 2.

HCP (GRD only)
- A new loader returns {task: {subject_id: connectivity vector}}.
- C (18 subjects) is drawn per seed only from subjects who have all 7 tasks, so every environment has n=18. A (basis) and B (reference), |A| = |B| = 37, are drawn from the remaining WM subjects; the rest are unused.
- Environments: for each non-WM task, the rows of C subjects that have that task (n recorded).
- Fakes: 50 random subsets of size 14 from the C subjects' WM rows.
- Tests:
  T0h: PCA on A, reference proj(B)
  T0i: PCA on B, reference proj(B)
- Cross-fit: repeat with the A and B roles swapped; report both.
- Assert n_env <= |B|//2 everywhere.
- Separate BH families for environments and fakes, as in E3.
- Seeds 0, 1, 2.
- T0i is the in-sample reproduction check. E3 itself ran on the bootstrap fallback, so an exact match is not expected.

Measured before code (repo gate, pure null, B=199, p=4000, rank-20 signal with N(0,1) loadings plus unit noise, ref 99, env 40, separate basis 99): in-sample basis 55-62% false detections, held-out 7.5-9.5%.

Predictions, evaluation and decisions (all BH counts over retained real perturbations, mean over seeds unless stated)

Predictions are Claude's forecasts, recorded so they can fail. Only P2 has its own consequence; every decision comes from D1 to D7.

| ID | Prediction | Output key |
|----|------------|------------|
| P1 | K562: T1 and T3 >= 30%, and T0h <= 2% | bh_frac_real |
| P2 | Fakes BH-selected <= 3 of 50 for T0h, T0m, T1, T2, T3, every dataset and seed; if false, the failing test is reported as uncalibrated | bh_count_fake |
| P3 | HCP: T0i >= 4/6 and T0h <= 1/6, in both cross-fit directions | hcp_bh_count |
| P4 | RPE1: T0h / T0i <= 0.60 (ratio of seed-mean BH counts; undefined and reported if T0i = 0) | bh_count_real |
| P5 | K562: T2 > T0h | bh_count_real |
| P6 | Fakes pooled over 3 seeds x 50: T0i raw rejection > T0h raw rejection, one-sided McNemar p < 0.05, every Perturb-seq dataset | raw_reject_fake |

Decision rules (all evaluated at d=10; d=30 is reported only)
- D1: K562 T1 or T3 >= 20% while T0h <= 2% and T0m <= 2% -> withdraw the thin-signal claim; pivot to a mean-aware screen.
- D2: K562 and Norman T1, T2 and T3 all <= 5% -> thin signal stands; GRD's abstention is corroborated by T1 to T3; keep GRD.
- D3: neither D1 nor D2 -> mixed; report all tests and decide after discussion. No silent choice.
- D4: RPE1 T0h / T0i < 0.5 -> the basis artefact affects Perturb-seq; every E3 real-data number is reworked on the held-out basis. D4 is not triggered if the ratio is undefined (T0i = 0); that is reported.
- D5: HCP T0i >= 4/6 and T0h <= 1/6 in both cross-fit directions -> artefact confirmed; the fMRI section is removed.
- D6: HCP T0i < 4/6 in either direction -> inconclusive at this sample size; the fMRI section is removed, since it cannot be defended.
- D7: HCP T0h >= 2/6 and T0i >= 4/6 in both cross-fit directions -> detections survive the held-out basis; keep the fMRI section, reported on held-out numbers only.
- Any HCP outcome not covered by D5 to D7 -> report as mixed and remove the fMRI section.
Failed predictions are reported.
