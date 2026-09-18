# grd-framework

Gate-Recover-Discover: a precondition-gated CRL+CD framework; the Gate pre-tests identifiability preconditions, Recover runs an identifiable estimator on certified environments, and Discover reports the latent graph only at the ambiguity-permitted resolution.

**STATUS: Experimental arc complete; manuscript preparation in progress.** The
repository contains the E0 oracle, full and genuinely capped E1 pipelines,
non-tautological E2 starvation calibration, corrected-null/BH-FDR E3 screens,
real-data attribution limits, and the validated three-seed iLCS baseline. The projected-attribution
rescue failed its predeclared synthetic gate and was not applied to real data.
See `notes/TRACK_SHEET.md` and `results/` for the evidence trail.

Experiments note: `experiments/ilcs_baseline.py` is an ungated baseline whose iLCS detector is reimplemented from Chen et al. 2024 (arXiv 2410.24059) Algorithm 1, because the original repository github.com/TianyuCodings/iLCS is unavailable (404). It runs on the same D=10 control-fit projected inputs as the E3 gate, with a size-matched null.

## Reproduction

Synthetic experiments require only the Python dependencies in `pyproject.toml`.
Real-data paths are configured through `GRD_DATA_ROOT`; see `DATA.md`.

Print the complete E3 regeneration plan without running it:

    python experiments/reproduce_e3.py

Execute E3, including an external-data SHA-256 manifest:

    python experiments/reproduce_e3.py --execute

Include the three-seed iLCS baseline:

    python experiments/reproduce_e3.py --execute --include-ilcs

The compact gate decisions used for baseline overlap calculations are under
`results/e3_decisions/`, each tied to its full E3 source JSON by SHA-256.
