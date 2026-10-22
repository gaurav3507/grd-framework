# E16 pre-registration (24 Oct 2026): the gate in front of a neural backbone

Committed before any E16 code.

## Question
Every result so far puts the gate in front of a linear backbone: a PCA projection fitted on the basis environment, then the precision-difference unmixing of src/recover/backbone.py. E16 asks whether the gate's verdict still tracks recovery when the projection is a trained autoencoder and the mixing is nonlinear. That is the standard two-stage neural pipeline of Ahuja et al. (2023, "Interventional causal representation learning", ICML): an autoencoder with a polynomial decoder, which identifies the latents up to an affine map, followed by a linear interventional step. If the autoencoder gives affine codes, the gate's linear theory applies to the codes, so the prediction is that the gate keeps working. That prediction can fail.

## Design: the E10 benchmark with nonlinear mixing and an autoencoder
Unit, arms, levels, seeds, ground truth and metrics are E10's: the four E2 arms (A environment starvation 5 to 1, B rows per environment 4000 to 20, C measurement contamination 0 to 5, D weak signal 0.1 to 1.0), seeds 0 to 9, one unit per (arm, level, seed, intervened direction), 900 units per mixing. A unit is recoverable when its recovered column's best-match |corr| with a true latent exceeds 0.90 (E2.DIR_CORR), on the obs environment, as in E10.

### Data
For each (arm, level, seed), the latent draws are E10's exactly: the builder repeats SIM.simulate's random-number order (make_scm, the draw of A, then per environment sample_latent, mixing, observation noise; arm C's contamination stream default_rng(seed + 7)), and only the deterministic mixing step changes. Mixing parameters come from a separate stream, default_rng(160_000 + seed), so the latent stream is untouched. Regression: Z in every environment must equal the linear E2 builder's Z exactly (max abs diff 0); any mismatch voids the run. Observation noise sd is fixed from the first environment's signal scale, as in simulate. Arm C contamination applies the per-dimension gain g.uniform(0.5, 1.5, D) to the nonlinear signal, in the same position and draw order as arm_C.

linear (regression path only, not an E16 arm): X_sig = SIM.mix_linear(Z, A) on the raw Z, A from the main stream; this must reproduce the E2 builders bit for bit.

For poly and mlp, let u = Z / s, with s_j the square root of the j-th diagonal entry of the observational population latent covariance (SIM.population_latent_cov(B, noise_var)), a fixed constant per seed shared by every environment.

- **poly (primary):** X_sig = phi(u) G^T, phi(u) = [u_1..u_5, kappa * (u_i u_j for (i, j) in np.triu_indices(5), row-major)] (5 + 15 = 20 features), kappa = 0.5, G = default_rng(160_000 + seed).standard_normal((200, 20)). G has full column rank with probability 1, so the map is injective. About half the signal variance is in the quadratic terms, which a linear projection cannot undo. This is the generator class of Ahuja et al.'s polynomial-decoder result; their theorem is stated for noiseless data and perfect reconstruction, and with 10% observation noise it applies approximately.
- **mlp (secondary, exploratory):** X_sig = SIM.mix_mlp(u, A1, A2, s=1.0, slope=0.1), A1 = rng.standard_normal((64, 5)) / sqrt(5) then A2 = rng.standard_normal((200, 64)) / sqrt(64), in that order, rng = default_rng(160_000 + seed). Injective (full-column-rank A1 and A2, leaky ReLU invertible), but outside the polynomial-decoder theory.

### Autoencoder (replaces BK.fit_pca and BK.project)
Trained on the basis environment only, as the PCA is now, so the obs and interventional environments are never seen in training.
- Precision: torch float32 throughout training; codes cast to float64 before every numpy step.
- Input: X standardised per dimension with the basis mean and standard deviation (numpy, ddof=0), computed in float64 then cast.
- Encoder: 200 -> 128 -> 128 -> 5, LeakyReLU(0.1), linear output.
- Decoder: poly arm: Linear over [1, z_hat, z_hat_i z_hat_j for i <= j] (21 features) -> 200. mlp arm: 5 -> 128 -> 128 -> 200, LeakyReLU(0.1).
- Loss: mean squared reconstruction error. Adam (torch defaults), learning rate 1e-3, batch min(256, n_basis), exactly 6000 gradient steps regardless of n_basis, no weight decay, no early stopping. Batches: epoch-wise torch.randperm(n_basis, generator=gen), consecutive slices of the permutation, last short batch kept, new permutation each epoch, stop at step 6000.
- Seed: immediately before building each model, inside the worker, torch.manual_seed(S) and gen = torch.Generator().manual_seed(S), S = 160_000 + 1000 * seed + 100 * arm_index + level_index, arm_index A0 B1 C2 D3, level_index the position in E2.ARMS[arm]["levels"]. Modules are constructed encoder first, then decoder, with default PyTorch initialisation. torch.set_num_threads(1) and torch.use_deterministic_algorithms(True), and OMP/MKL/OPENBLAS threads 1, set in the pool initializer before torch is imported.
- Codes: Y = encoder(standardised X) for the obs and every interventional environment. Everything downstream is E10's and E14's code applied to Y in place of the PCA projection.

### Gates and analysis
- One call per (arm, level, seed) and readout: precision_readout.detect_with_pvalues(Y_int, Y_obs, alpha=0.05, B=500, rng=default_rng(910_000 + seed), q=0.05, readout=r) for r in precision (G2), geneig (G8), energy (G9): E14's _call on the codes.
- E10's unit record on the codes: backbone rows BK.unmixing_rows(Y_obs, Y_int, nodes, 5, readout="precision"), best-match |corr|, the 50 bootstrap angles (default_rng(77_000 + 1000 * seed + node)).
- Gates G0, G2, G5, G8, G9 with E14.decisions, curves_for, summary_of and bootstrap on the E16 units (knob grids, scopes, AURC, matched risk at G2's q = 0.05 coverage, 1000 draws over the 200 systems, default_rng(120_000)), exactly as E14. G1, G3, G4 from E10's curves, point estimates only. E10's P1 to P6 and E14's H1 to H4 and K1 to K3 are not evaluated or reported for E16. Each mixing is analysed separately.
- H1's interval: in the same bootstrap draw loop with the same system picks, risk(G0) - risk(G2 at q=0.05) per draw, risk from summary_of(cv)["ABD"]["G2"]["risk_q05"]; a draw where G2 certifies nothing has no G2 risk and is excluded; the number of defined draws is reported and the interval is undetermined if fewer than 900 of 1000 are defined. The same 900-draw minimum applies to every interval below.
- Units are stored and analysed in E14.systems() order, whatever order workers finish in.
- Failed units: a system that raises (non-finite loss, singular covariance) is recorded in failed_units.json. Its units count as not recoverable and certified by G0 only. If more than 5% of systems in a mixing fail, every prediction for that mixing is undetermined.
- Diagnostics recorded per system: affine R^2, the minimum over j of the R^2 from an OLS regression of the true z_j on [1, Y] on the obs environment; the same affine R^2 on each interventional environment that is not contaminated (diagnostic only: the encoder never saw these latent regions in training, so this measures extrapolation, and no kill rule uses it); full-batch reconstruction loss on basis after the last step; the same loss on obs (held out).

### Pre-flight (runs before the full data pass)
Poly arm, arm A at level 5 (2000 rows per environment, every latent intervened), seeds 0 to 9: train, record affine R^2.

## Kill criterion
- K0: if the median over the 10 pre-flight seeds of affine R^2 is below 0.90, the autoencoder has not reached the affine recovery the theory needs. STOP before the full run and report. No gate result from E16 is produced or claimed. The architecture and training settings above are not tuned to rescue it.

## Predictions (poly arm unless stated; recorded so they can fail)
- H1 (ABD): risk(G2 at q=0.05) < risk(G0), with the lower end of the 95% bootstrap interval of risk(G0) - risk(G2 at q=0.05) above 0.
- H2 (ABD): risk(G2 at q=0.05) <= 0.05.
- H3 (ABD): AURC(G2) < AURC(G0), with the lower end of the 95% bootstrap interval of AURC(G0) - AURC(G2) above 0.
- H4 (ABD): matched risk(G9) > matched risk(G2), with the lower end of the 95% interval of matched risk(G9) - matched risk(G2) above 0; undetermined if G9's curve does not reach G2's coverage.
- H5 (ABD): coverage(G2 at q=0.05) >= 0.60 (linear backbone, E10: 0.805).
- H6 (C): risk(G2 at q=0.05) >= 0.10. This predicts the known failure mode: contamination fools a detectability test (linear backbone, E14: 0.160).
- H7 (mlp arm, exploratory, point estimates only): risk(G2 at q=0.05) < risk(G0) on ABD.

## Consequences fixed in advance
- K0 triggers: reported in the manuscript's appendix as a negative result; no neural-backbone claim; the venue decision is revisited.
- H1, H2 and H3 hold: the manuscript adds a section and a contribution: the gate tracks recovery for a neural CRL pipeline at its calibrated operating point, under the polynomial-decoder assumptions. H4 to H6 are reported with it whatever their outcome.
- H1 or H3 fails: the gate's guarantee does not transfer through a trained encoder at these sizes. Reported as a failure in the same section; no claim that the gate covers neural pipelines.
- H2 or H5 fails alone: reported as a limitation (risk or coverage cost of the neural pipeline); H1 and H3 decide the claim.
- H7: reported as exploratory, labelled as outside the theory's assumptions, whatever it shows. No claim rests on it.
- Every E16 number is reported next to the corresponding linear-backbone number from E10 and E14.

## Caveats written in advance
- Arm B's small levels train the autoencoder on 20 to 100 basis rows. Recovery there is expected to fail for reasons unrelated to the gate. Those units count as they fall; arm B coverage and risk at 20 and 40 rows are reported separately and not over-read (as in E14).
- The synthetic null resamples with replacement (E14 caveat); unchanged.

## Not allowed
No change to the architecture, optimiser, step count, seeds, kappa, B, q, the 0.90 threshold, the grids or the predictions after the pre-flight starts; no added gates or mixings; no dropped units, seeds or levels; no retraining with a different seed after a bad fit; scripts and results committed together.

## Deferred
- Discrepancy-VAE (Zhang et al., 2023) on Norman: no ground truth, so it would test stability, not recovery. Pre-registered separately if pursued.
