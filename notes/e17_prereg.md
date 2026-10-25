# E17 pre-registration (25 Oct 2026): restart selection in front of the gate, on fresh seeds

Committed before any E17 code.

## Why
E16 (prereg 1325510, results 8482158) put the gate in front of an autoencoder on polynomially mixed data. H1 and H3 held (the gate lowers risk relative to certifying everything), but H2 failed: risk at q=0.05 was 0.381 on ABD. A post hoc split of the E16 units by the encoder's affine R^2 on obs, made after seeing the results, located the failure upstream of the gate:

| affine R^2 (obs) | ABD units | recoverable | G2 coverage q=0.05 | G2 risk q=0.05 |
|---|---|---|---|---|
| < 0.90 | 251 | 0.13 | 0.63 | 0.885 |
| 0.90 to 0.99 | 171 | 0.67 | 0.77 | 0.321 |
| >= 0.99 | 228 | 0.90 | 0.82 | 0.000 |

29% of encoders trained on 2000 or more basis rows (160 systems, all four arms) did not reach affine recovery (R^2 < 0.90), and held-out reconstruction loss on obs tracked affine R^2 closely (Spearman -0.93 over the same 160 systems). The obvious ground-truth-free remedy is standard practice: train several restarts and keep the one with the lowest held-out reconstruction loss. E17 tests, on seeds E16 never used, whether the gate controls risk at its calibrated operating point once the encoder is selected this way.

## Design
Everything is E16's poly arm (notes/e16_prereg.md and experiments/e16_neural_backbone.py at b29139e) except what is listed here. The mlp arm is not run.

- Seeds 10 to 19 (E16's SCM and main streams used seeds 0 to 9; arm C's contamination stream default_rng(seed + 7) is inherited from E2 and overlaps some main streams across seeds, unchanged because the replay regressions require it). All four arms, every level: 200 systems, 900 units. Data, mixing and latent streams are E16's builders with these seeds; the Z and arm C replay regressions apply as in E16.
- Validation split of the basis environment: perm = default_rng(175_000 + 1000 * seed + 100 * arm_index + level_index).permutation(n_basis); the first max(1, floor(0.2 * n_basis)) indices are validation, the rest training; rows are taken in their original order, Xb[np.sort(perm[n_val:])] for training and Xb[np.sort(perm[:n_val])] for validation. n_basis is the number of basis rows before the split. Standardisation mean and sd (ddof=0) are computed on the training rows only. obs and the interventional environments are never used in training or selection.
- Restarts: r = 0 to 4, model seed S_r = 170_000 + 100_000 * r + 1000 * seed + 100 * arm_index + level_index, used exactly as E16 uses its model seed (torch.manual_seed and the batch generator, set immediately before building each model; encoder built before decoder). Each restart trains with E16's architecture, optimiser and 6000 steps on the training rows, batch min(256, n_train).
- Selection: the restart with the lowest full-batch reconstruction MSE on the validation rows; ties go to the lower r. Only the selected encoder produces codes. Ground truth plays no part in selection.
- Downstream, unchanged from E16: codes for obs and every interventional environment, E14's precision, geneig and energy calls (B=500, default_rng(910_000 + seed)), E16's unit record, gates G0, G2, G5, G8, G9, curves, matched risk, and E16's bootstrap16 (1000 draws over the 200 systems, default_rng(120_000)), units in E14.systems(range(10, 20)) order.
- Diagnostics per system: for every restart, validation loss, affine R^2 on obs and the index r; for the selected encoder, affine R^2 on obs and on each uncontaminated interventional environment, and reconstruction loss on obs.
- Secondary rule G2a (abstention): G2 at q=0.05, and in addition no environment of a system is certified if that system's selected validation loss exceeds 0.02. The 0.02 was read off the E16 results after seeing them (E16 obs loss; in-sample E16 ABD risk at 0.02 was 0.000 at coverage 0.29). It is therefore a test of a threshold chosen on E16, evaluated on fresh seeds. The quantity also changes: E16's 0.02 was a loss on 2000 obs rows from a model trained on the full basis; E17 applies it to the validation loss on 20% of the basis rows (4 to 80 rows on arm B), from a model trained on the other 80%. G2a with the selected encoder's obs loss in place of the validation loss is reported as a descriptive line, not a prediction.
- Failed restarts and systems: a restart fails on a non-finite training or validation loss and is excluded from selection; if all five fail, the system fails. Any exception after selection (non-finite codes, gate calls, unit record) fails the system as in E16, with no fallback to another restart. Failed systems as E16: units not recoverable, G0 only; more than 5% failed makes every prediction undetermined.

## Predictions (ABD unless stated; recorded so they can fail)
- P1: risk(G2 at q=0.05) <= 0.05.
- P2: risk(G0) - risk(G2 at q=0.05) has the lower end of its 95% bootstrap interval above 0 (minimum 900 defined draws, as E16).
- P3 (all four arms): among the 160 systems with n_basis >= 2000 before the split (A, C, D every level; B level 4000), the share whose selected encoder has affine R^2 (obs) >= 0.90 is at least 0.90; a failed system counts as below 0.90 (E16, single run, same set: 0.71).
- P4: coverage(G2 at q=0.05) >= 0.60.
- P5 (secondary): risk(G2a) <= 0.05 and coverage(G2a) >= 0.30.
- P6 (C): risk(G2 at q=0.05) >= 0.10. Contamination still fools the gate.

## Consequences fixed in advance
- P1, P2 and P3 hold: the manuscript's neural section claims that, with restart selection on held-out reconstruction loss, the gate controls risk at its calibrated operating point for this neural pipeline. E16 is reported in full beside it, including H2's failure, and the E16 stratified table is labelled post hoc.
- P3 holds and P1 fails: the gate does not control risk at q=0.05 even with well-recovered encoders on fresh seeds; the E16 stratified result is not confirmed. Reported as a failure; the neural section claims only E16's H1 and H3.
- P3 fails: restart selection does not fix the encoder at this setting. Reported; the neural section claims only E16's H1 and H3, and the stratified table stays post hoc and unconfirmed. P1 and P2 are reported but carry no claim.
- P5: reported as the test of a threshold chosen on E16. If it holds, the manuscript may describe the abstention rule as a candidate check, not a validated one, because 0.02 depends on this simulator's noise level.
- P4 or P6 failing alone: reported as a limitation or a change from E16; no claim rests on them.

## Caveats written in advance
- Arm B's small levels split 20 to 100 basis rows into training and validation (4 to 20 validation rows). Selection there is noisy. Arm B at 20 and 40 rows is reported separately and not over-read, as in E14 and E16.
- E17 trains on 80% of the basis rows, E16 on 100%. E17 is compared with E16 descriptively, not as a paired experiment.

## Not allowed
No change to seeds, restart count, split, selection rule, architecture, steps, B, q, the 0.02 threshold or the predictions after the first training run; no added restarts after seeing results; no dropped units, seeds or levels; no rerun of a system with another seed; scripts and results committed together.
