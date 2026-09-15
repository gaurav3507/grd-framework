# grd-framework

Gate-Recover-Discover: a precondition-gated CRL+CD framework; the Gate pre-tests identifiability preconditions, Recover runs an identifiable estimator on certified environments, and Discover reports the latent graph only at the ambiguity-permitted resolution.

**STATUS: Experimental arc complete. Framework built and validated (E0 oracle, E1 recovery, Path A gate-backbone coupling, M3 discover, E2 violation-injection calibration). Real-data gate screens on K562, RPE1, Norman (Perturb-seq) and HCP/ABIDE (fMRI) with 5-seed stability and a semi-synthetic positive control. See notes/TRACK_SHEET.md and results/. Next: writeup.**

Experiments note: `experiments/ilcs_baseline.py` is an ungated baseline whose iLCS detector is reimplemented from Chen et al. 2024 (arXiv 2410.24059) Algorithm 1, because the original repository github.com/TianyuCodings/iLCS is unavailable (404). It runs on the same D=10 control-fit projected inputs as the E3 gate, with a size-matched null.
