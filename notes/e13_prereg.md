E13 pre-registration (fixed before code)

Q1: the direction-dependent power of the calibrated GRD gate (T0h) for planted changes at real cell counts.
Q2: the sign of the real covariance changes that Roy (T2) detects.
Setting: K562, Norman, RPE1; seed 0; d=10; the E12 held-out basis, REF and FAKE pool. median n_e = design["fake_n"] from build_tasks (K562 270, Norman 516, RPE1 262).

Part A: planted power grid (semi-synthetic, real control noise)
- Whitening: S = cov(projected uncorrected REF), W = S^(-1/2) (symmetric). Store the REF eigenvalues lambda_1..lambda_10 per dataset.
- Directions, indexed by q:
  - q=0: PC1 of S
  - q=1: PC5 of S
  - q=2: PC10 of S
  - q=3: a random unit vector in whitened space, a = z/|z| with z = rng_for(0, 900000, 0).standard_normal(10), the same vector for every dataset
  For q=0 to 2, a is the eigenvector itself (identical in whitened and original coordinates).
- Planting: cells are drawn without replacement from the FAKE pool; m = FAKE-pool mean; row-vector convention.
  - A1/A2 (covariance only): Y' = m + (Y - m) W (I + (sqrt(v) - 1) a a') W^-1. The generalized eigenvalues against S are then {v, 1, ..., 1} in population.
  - A3 (mean only): Y' = Y + delta * a' W^-1, so delta is the Mahalanobis shift.
  - A4 (responder mixture): exactly round(pi * n_e) cells, chosen at random, get + delta * a' W^-1. This produces a mean shift pi*delta and a variance increase pi(1-pi)delta^2 along a.
- Arms:
  - A0 null: 300 replicates per size, direction-free (q=4).
  - A1 decrease: v in {0.9, 0.8, 0.7, 0.6, 0.5, 0.3}.
  - A2 increase: v in {1.1, 1.25, 1.5, 2, 3}.
  - A3 mean: delta in {0.05, 0.1, 0.2, 0.3, 0.5}.
  - A4 mixture: pi in {0.3, 0.6} x delta in {0.5, 1, 2}.
  A1 to A4 use 100 replicates per (arm, level, direction, size).
- Sizes, indexed by s: s=0 is n_e=200, s=1 is median n_e.
- Tests, all E12 functions, B=999:
  - T0h: grd vs the full REF, disjoint null.
  - T0m: grd vs a size-matched REF draw, pooled null.
  - T1 Hotelling, T2 Roy, T3 energy: vs the same size-matched REF draw.
- Common random numbers: replicate r at (s, q) uses index i = 1000000 + 10000*s + 1000*q + r.
  - k=1 draws its cells, k=2 its REF rows, k=3 the mixture labels.
  - Tests: k=10 T0h, k=11 T0m, k=12 T1, k=13 T2, k=14 T3.
  - These are shared across arms and levels, so power curves are smoother across levels. A0 uses q=4 with r = 0 to 299.
- Power = rejection rate at alpha = 0.01, also reported at 0.05 and 0.001.
  - alpha = 0.01 is the decision level because it is close to the BH cutoffs E12 actually reached: 0.024 for Roy on K562 and 0.020 on Norman.
  - At B=999, alpha=0.001 means zero exceedances.
  - T0h's first BH discovery on real data would need p near 0.05/435, about 1.1e-4, which B=999 cannot resolve, so Part A power is an upper bound on real-data detectability.

Part B: sign readout on the real environments (all retained real environments plus 50 fakes, exactly as E12 seed 0 d=10)
- Uses the tasks and shared dict returned by build_tasks(panel, 0, 9999, 10, False).
- The REF for each environment is correct(shared["ref_heldout"], shared["ref_target_columns"][j], task["mu_h_j"], task["P_h_j"]) when j is not None, and shared["ref_heldout"] otherwise.
- Using the E12 size-matched REF draw (stream k=1) and one shared permutation pass over the pooled 2*n_e rows (B=9999, stream k=20):
  - T2down = -log(min generalized eigenvalue of (S_env, S_ref)).
  - T2up = log(max generalized eigenvalue).
  - Each p = (1 + #>=obs)/(B+1).
- BH q=0.05 per dataset per test, over real plus fakes.
- Also stored per environment, against that environment's full corrected REF (covariance S_e_ref):
  - the min and max generalized eigenvalues
  - delta_hat, the pooled-covariance Hotelling Mahalanobis distance
  - the generalized eigenvector w for lambda_min, and lambda_eff = w'S_e_ref w / w'w (scale-free in w), with its rank within lambda_1..lambda_10 (the eigenvalues of the uncorrected REF covariance)
- Summary: the median lambda_eff among T2down BH discoveries vs the median of lambda_1..lambda_10, per dataset.

Decision rules (K562 and Norman decide; RPE1 is reported only; median n_e; alpha=0.01)
Classify each dataset, in this order:
- R1, powered in every direction: min over directions of T0h power >= 0.8 at v=0.7.
- R2, underpowered in every direction: max over directions < 0.5 at v=0.5.
- R-dir, direction-dependent: at v=0.5, max over directions >= 0.8 and min < 0.5.
- R3, intermediate: otherwise.
Global R branch: the shared branch if K562 and Norman agree; "split" otherwise.
- S-up: T2up BH count >= 3 x T2down BH count and >= 10, in both datasets, with fakes <= 3/50 for both tests.
- S-down: T2down BH count >= 3 x T2up BH count and >= 10, in both datasets, with fakes <= 3/50 for both tests.
- S-mixed: otherwise.
- Calibration: A0 rejection at alpha=0.05 must lie in [0.02, 0.09] for every test and size (300 replicates; a calibrated test falls outside this band about 0.4% of the time per check, about 10% across all checks). A test failing this is flagged as uncalibrated.

Interpretation fixed in advance:
- R1 and S-up: "the gate is blind by construction to mean shifts and variance increases; the real covariance changes are increases; a powered gate still passes ~0". A T2up excess is also what partial-responder mixtures (A4) produce, so it does not by itself distinguish heterogeneous knockdown from a regulatory variance change.
- R-dir: "the gate's power depends on the direction of the change relative to the control spectrum (it is not affine-invariant); an affine-invariant test (Roy) does not have this defect". It is offered as an explanation for 0/385 ONLY IF, in both K562 and Norman, the median lambda_eff among T2down BH discoveries is above the median lambda_k (if a dataset has no T2down discoveries, the condition is not met); otherwise the R and S results are reported as is.
- If R-dir and S-up both hold, report both clauses.
- R3 or "split": report as is, decide after discussion.
- R2: "the gate is underpowered at these cell counts".
- Anything else: report as is, decide after discussion.

Predictions (Claude's, recorded so they can fail):
- K562 is classified R-dir, with T0h power along PC1 < 0.5 at v=0.5.
- S-up holds in K562.
- Mean T0h power over all A2 and A3 cells is <= 0.08 at alpha=0.05.
- T2 power >= 0.8 for A4 with pi=0.3, delta=2, in every direction, at median n_e and alpha=0.01.
