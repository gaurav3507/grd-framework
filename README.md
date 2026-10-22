# Gate, Recover, Discover (GRD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: preprint](https://img.shields.io/badge/status-preprint-orange.svg)](#citation)

Reference implementation and committed result artifacts for

> **GRD: A calibrated test of the intervention assumption in causal representation learning.**
> Gaurav Goyal, Shailendra Tiwari, and Manju Khurana. Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, India. Manuscript in preparation, 2026.

GRD reframes latent causal recovery as a **selective decision problem**. Modern causal
representation learning (CRL) results identify latent variables and their graph only under
explicit conditions on environments, power, mechanism shifts, and measurement stability, yet
an estimator returns *something* for every input. GRD adds a data-side decision layer:

1. **Gate**: test estimator-coupled preconditions with a size-matched precision-difference
   null and BH-FDR within each declared family; return a certified environment set `C` and a
   recoverable resolution `d_rec = min(|C|, d*)`.
2. **Recover**: run the identifiable backbone **only** on `C`; withhold uncertified latents
   rather than numerically completing them.
3. **Discover**: report a partial graph with explicit *decided* and *undecided* edge states.

The verdict is the first output: `PROCEED`, `PROCEED-CAPPED`, or `ABSTAIN`. Abstention and
unresolved attribution are treated as scientific results, not failures.

---

## Headline results

Every number below is read from committed JSON in `results/`. Every real-data number uses the
held-out construction (projection fitted on held-out control cells, reference and negative
controls drawn from disjoint control subsets; `experiments/e12_common.py`). Real-data numbers
from earlier versions of this README used an in-sample construction that selects up to 48 of
50 pure-control pseudo-environments; they are void and have been removed (decision D4,
`notes/d4_prereg.md`). Experiments from E10 onwards are pre-registered in `notes/*_prereg.md`;
amendments and the one post hoc correction are listed in `notes/TRACK_SHEET.md`.

**Synthetic calibration** (ten seeds, `d = 5` latents). Under environment starvation the gate
first restricts at `m = 4` while recovery is still `0.9908`, which drops below `0.90` only at
`m = 2`. Measurement contamination fools the gate (declared failure class).

**Matched coverage against standard gates** (900 units, `results/e14`, `results/e15/synthetic`).
On clean arms at q = 0.05 the gate certifies 80.5% of units with risk 0.0019; the energy-distance
test has matched risk 0.032, the two-sided affine-invariant test and the one-sided
affine-invariant variant certify 96 to 97% and never reach the gate's coverage. Bootstrap
stability and the raw statistic rank as well or better (lower AURC); the gate's advantage is a
calibrated operating point.

**Real Perturb-seq screens** (BH q = 0.05, B = 9,999, three seeds, 50 pure-control
pseudo-environments per seed; `results/e12_standard_tests`).

| Dataset | Envs. | Gate (held-out) | Fakes | In-sample construction, fakes | Hotelling mean test |
|---|---:|---:|---:|---:|---:|
| K562 (CRISPRi)   | 385 | 0, 0, 0    | 0/50 | 33, 34, 24 /50 | 318, 316, 310 |
| RPE1 (CRISPRi)   | 145 | 10, 14, 8  | 0/50 | 46, 48, 48 /50 | 101, 93, 92 |
| Norman (CRISPRa) | 100 | 1, 3, 1    | 0/50 | 0, 2, 0 /50    | 100, 100, 100 |

**Why the gate abstains.** The mean shifts survive per-cell depth correction (verdict S1;
`results/e12c_norman_depth`, `results/e12d_k562_depth`). Real covariance changes are mostly
increases (K562 222 vs 4; `results/e13_power_sign`), invisible to the one-sided gate, and its
power for decreases depends on direction: a planted halving of variance is detected 0 to 1% of
the time along the leading control direction and always along the trailing one. RPE1's
certified changes align with the leading control axes (p = 0.001 in all seeds); shared-subspace
attribution rejects nothing; an uncalibrated latent shift detector (iLCS) fires on every
pseudo-environment, and size-matched calibration does not bring it to 0.15 everywhere
(`results/d4_heldout`).

**Theory** (`notes/theory`). Validity of the pooled null under exchangeability, consistency of
the verdict under BH or Benjamini-Yekutieli with the resolution condition (B+1) k q >= m c(m),
and lower bounds that make the gate rate-optimal up to logarithmic factors on a hard subfamily.

---

## Repository layout

```
src/           Gate / Recover / Discover modules and the precision-difference readout
sim/           multi-environment linear-SCM simulator
experiments/   one runnable script per experiment (E0-E15, D4, baselines, certificate)
results/       committed JSON artifacts (source of every number in the paper)
paper/         figure-generation script and the six manuscript figures
notes/         track sheet, Davis-Kahan certificate note, theory notes (notes/theory)
docs/          framework design document
```

## Reproduction

Synthetic experiments need only the dependencies in `pyproject.toml` (NumPy, SciPy,
scikit-learn; Python >= 3.10). Real-data paths are configured through the `GRD_DATA_ROOT`
environment variable; `DATA.md` documents the expected filenames. The source datasets
(Replogle Perturb-seq via CausalBench, Norman, HCP, ABIDE) are **not** redistributed here.

```bash
pip install -e .

# print the full E3 regeneration plan without running it
python experiments/reproduce_e3.py

# execute E3 with an external-data SHA-256 manifest
python experiments/reproduce_e3.py --execute

# add the three-seed iLCS baseline
python experiments/reproduce_e3.py --execute --include-ilcs

# regenerate the manuscript figures from the committed JSONs
python paper/make_all_figures.py --figure all
```

Every stochastic script takes an explicit seed. Compact gate decisions in `results/e3_decisions/`
are linked to their full E3 source JSON by SHA-256, and `results/repro/data_manifest.json`
records external-data hashes.

## Artifact map

| Result | Artifacts | Pre-registration |
|---|---|---|
| Clean and capped pipelines | `results/e1_full`, `results/e1_capped` | expectations in script |
| Violation sweeps, starvation, weak signal | `results/e2`, `e2c`, `e11_weak_signal` | expectations in script |
| Risk-coverage comparison (E10) | `results/e10_risk_coverage`, `results/e10b_metrics` | `notes/e10_prereg.md` |
| Standard tests on Perturb-seq (E12) | `results/e12_standard_tests` | `notes/e12_prereg.md` |
| Depth and leading-PC checks (E12b, E12c, E12d) | `results/e12b_confound`, `e12c_norman_depth`, `e12d_k562_depth` | `notes/e12b_prereg.md`, `e12c_deviation.md`, `e12b_amendment_2.md` |
| Power by direction and sign of changes (E13) | `results/e13_power_sign` | `notes/e13_prereg.md`, `e13_amendment_1.md` |
| Matched coverage with energy and two-sided tests (E14) | `results/e14` | `notes/e14_prereg.md` |
| Held-out rework: alignment, attribution, iLCS (D4) | `results/d4_heldout` | `notes/d4_prereg.md` |
| Affine-invariant one-sided variant (E15) | `results/e15` | `notes/e15_prereg.md` |
| Superseded (in-sample construction, kept for history) | `results/e3*`, `e5_split_control`, `e6_resolution`, `e8_pooled_null`, `e9_alignment_baseline`, `ilcs_baseline` | void under D4 |

K562 and RPE1 analyses after E12 use files rebuilt from figshare with
`experiments/build_causalbench_data.py`; they match the original files on every content
fingerprint and differ at float32 rounding level (`notes/e13_amendment_1.md`).

The iLCS detector in `experiments/ilcs_baseline.py` is reimplemented from Algorithm 1 of
Chen et al. 2024 (arXiv 2410.24059) because the original repository is unavailable; it runs on
the same `D = 10` control-fit projected inputs as the gate, with a size-matched null.

## Scope and limitations

The implemented backbone assumes linear mixing, Gaussian latent noise, and known atomic
reduced-variance hard-intervention targets. The gate is re-derived per backbone; this is one
exact regime, not a universal certificate. The real pipeline runs in a control-fit
10-dimensional projection, so linear exactness holds only there. The direction certificate is
valid but vacuous below `n ~ 16,000`.

## Citation

If you use this code or the committed artifacts, please cite the paper. A machine-readable entry
is in [`CITATION.cff`](CITATION.cff) (rendered as a "Cite this repository" button on GitHub).

```bibtex
@unpublished{goyal2026grd,
  author = {Goyal, Gaurav and Tiwari, Shailendra and Khurana, Manju},
  title  = {{GRD}: A calibrated test of the intervention assumption
            in causal representation learning},
  year   = {2026},
  note   = {Manuscript in preparation. \url{https://github.com/gaurav3507/grd-framework}}
}
```

## License

MIT, see [`LICENSE`](LICENSE). The source datasets retain their own licenses and are not
redistributed here.
