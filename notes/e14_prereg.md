# E14 pre-registration: synthetic matched-coverage comparison (16 Oct 2026)

Committed before any E14 code, following the E10 precedent (commit 308c090).

## Question
Does the GRD precision gate make better recovery decisions than standard shift tests when every gate is compared at the same coverage? The external review (9 Oct) rated the paper 5/10 partly because simpler gates appear competitive; E10 showed raw-threshold, uncalibrated and bootstrap-stability gates are indistinguishable from GRD on clean arms. E14 adds the two tests a reviewer names first: an affine-invariant two-sided covariance test and an energy-distance (scPerturb-style) test.

## Relation to E12
E12 (e12_standard_tests, e12b, e12c) is the Perturb-seq standard-tests experiment. E14 is its synthetic counterpart on the E10 units, where ground-truth recovery is known. The two share the question (does GRD's gate beat standard shift tests at matched coverage) and nothing else: no code, no results.

## Scope: synthetic, Mac
Unit: one (arm, level, seed, intervened direction) of the four E2 arms, exactly the 900 units of results/e10_risk_coverage/units.json. Ground truth, coverage, risk, AURC, matched risk, scopes and the q-grid are E10's, unchanged. Clean = arms A, B, D pooled (ABD); contaminated = arm C.

Gates:
- G2 GRD: lambda_max of the precision difference, size-matched null, BH at q (E10).
- G8 geneig: max_i |log lambda_i| of the generalised eigenvalues cov(Y_env) v = lambda cov(Y_obs) v, same null design, same null index draws (same rng seed), BH at q.
- G9 energy: multivariate energy distance (Szekely and Rizzo), each side reduced to at most 1000 rows by an evenly strided index, same null design and draws, BH at q.
- G0 (none) and G5 (bootstrap stability) carried from E10 for reference.

Primary metric: risk at matched coverage (each gate at G2's q = 0.05 coverage) on ABD and on C; secondary: AURC. Intervals: 95% percentile bootstrap over the 200 (arm, level, seed) systems, 1000 draws, decisions fixed, metrics and the matched-coverage target recomputed per draw. A hypothesis whose inputs are undefined (gate curve does not span the target coverage) is recorded as undetermined, neither held nor failed.

Regression: the data pass recomputes the precision p-values with the same rng and asserts equality with E10's stored values; any mismatch voids the run.

## Hypotheses
- H1 (clean): |AURC(G8) - AURC(G2)| <= 0.02 on ABD. Expected near tie: Lemma 2's one-sidedness only matters under variance inflation, which ABD does not generate.
- H2 (contaminated): matched risk(G8) < matched risk(G2) - 0.02 on C. Expected yes: diagonal gains with g_i > 1 inflate variance, invisible to G2 by Lemma 2, visible to G8.
- H3 (both): G9 coverage at q = 0.05 >= G2's and G9 matched risk >= G2's, on ABD and on C. Expected: the E-test fires on any distributional change, including ones the backbone cannot use.
- H4 (clean): matched risk(G5) <= matched risk(G2) + 0.02 on ABD (E10 P4, one-sided).

## Kill criteria and consequences
- K1: if on ABD no gate in (G2, G5, G8, G9) beats G0 on AURC by more than the half-width of the 95% interval of that difference, the gating line is dead on this simulator; report, do not rescue.
- K2: if matched risk(G9) <= matched risk(G2) on both ABD and C, GRD's gate has no advantage over the standard perturbation test; the paper's contribution narrows to the attribution results (C3), the RPE1 confound finding and the gating-what-Chen-assumes framing, and the gate becomes a component, not the headline.
- K3: if on C matched risk(G8) < matched risk(G2) - 0.02 with the interval of the difference excluding 0, and on ABD AURC(G8) <= AURC(G2) + 0.02 (ABD risk is near zero for every gate, E10: G2 0.0019, so AURC is the clean-arm criterion), G8 replaces G2 as the gate in the paper and Lemma 2 moves from limitation to motivation.
- Caveat written in advance: every synthetic unit uses the historical two-bootstrap null (n_env equals n_obs in all arms), which resamples with replacement while the observed statistic uses distinct rows. At small n this inflates every readout's null, precision most. Coverage at q = 0.05 for G2, G8, G9 is reported separately at arm B levels 20 and 40, and H3 is not over-read there.

## Not allowed
No tuning of B, q, d_proj, the 0.90 recovery threshold or the energy row cap after the first run; no added gates after the first run; no dropped units; no /tmp scripts; script and results committed together.

## Deferred
- Real-data counterpart of G8/G9 at matched coverage is E12's business, not E14's. Before any further real-data gate run: cell-dependence audit (guide, batch, replicate labels) decides the permutation unit.
- Two new synthetic arms for detectability versus recoverability: redundant interventions (several environments all targeting one latent) and mean-only shifts. Need new E2-style builders; pre-registered separately when built.

## Amendments
(none)
