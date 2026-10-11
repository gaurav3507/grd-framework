E12b pre-registration (fixed before code)

Question: are the E12 T1 (Hotelling) mean-shift detections driven by sequencing depth, batch, or a single leading axis, rather than by perturbation-specific expression change?

Setting: seed 0, d=10, held-out basis, the same environments, REF draws, fakes and BH family as E12. T1 with B=9999 permutations.

Depth covariate
- Depth = log of each cell's library size, from the first available source:
  1. an obs total-count column
  2. the row sum of a full-gene counts layer
  3. only as a last resort, the row sum of X; X holds only 1,158 / 651 / 5,000 selected genes, so this is not library size, and the dataset is then reported as S3 whatever the readouts say
- The source used is recorded per dataset.

Readouts per environment and fake
- R1 depth_diff: mean depth(env) minus mean depth(size-matched REF draw).
- R3 T1_depthreg: least squares Y = a + beta*depth, fit on BASIS cells projected on the held-out basis (uncorrected coordinates). Apply Y - beta*(depth - mean BASIS depth) to the target-corrected env and REF-draw coordinates, then run T1, B=9999, stream k=10.
- R4 T1_noPC1: T1 recomputed on PCs 2-10 only, stream k=11.
- R2 pc1_share = (T1 - T1_noPC1) / T1, which lies in [0, 1]. The marginal diff_1^2 / S_11 is stored separately.
- corr(PC1, depth) in BASIS cells, recorded per dataset.
- R5 batch: if a batch column exists, the batch composition of env cells vs REF cells (total variation distance).

Outputs:
- BH counts for R3 and R4 (same family: real retained envs plus 50 fakes).
- Partial Spearman(T1/n_e, |depth_diff|) controlling for n_e.
- Median pc1_share for BH-T1-detected envs.
- Median |depth_diff| for real envs vs fakes.

Decision rule
Decisions use K562 and Norman; RPE1 is reported only. Evaluate the rules in order; the first that matches is the verdict:
- (i) Any decision dataset (K562 or Norman) took depth from the X row-sum fallback -> S3.
- (ii) In K562 or Norman, R3 keeps < 30% of the E12 T1 BH count, or R4 is decision-relevant (|corr(PC1, depth)| > 0.5 in BASIS) and keeps < 30% -> S2.
- (iii) In both K562 and Norman, R3 keeps >= 70%, R4 is not decision-relevant or keeps >= 70%, and fakes in the R3 BH family are <= 3/50 -> S1.
- (iv) Otherwise -> S3.

What each verdict means:
- S1, survives: the mean-shift headline stands; D1 is confirmed.
- S2, depth-aligned: report "detections are depth-aligned; confound vs genuine depth phenotype unresolved". D1 is suspended pending discussion. In CRISPRi of essential genes, lower library size can itself be a real phenotype, so S2 is not called a confound by itself.
- S3, mixed: report per dataset and decide after discussion. R4 outside the decision-relevant case is descriptive only (PC1 may be real biology).
- Batch: if a batch column exists and the median env-vs-REF total variation distance is > 0.3 in any dataset, that dataset's T1 result is flagged as batch-exposed whatever S1 to S3 says. With no batch column, batch_tvd is null (unchecked), never 0.

Prediction (Claude's, recorded so it can fail): K562 R3 retains >= 70%; Norman R3 retains >= 30%; median pc1_share < 0.5 in K562.
