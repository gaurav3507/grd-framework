# GRD Framework Track Sheet

One row per produced artefact. Fill commit_hash after the commit that produced the row.

| date | machine (Mac/A100) | script | produced_artefact | commit_hash | notes |
|------|--------------------|--------|-------------------|-------------|-------|
| 2026-09-01 | Mac | (none) | M0 scaffolding + design doc | (fill) | repo created |
| 2026-09-08 | Mac | sim/simulator.py, src/gate/rank_readout.py, experiments/e0_oracle.py | results/e0/e0_report.json (e0_spectra.npz untracked) | 278dd98 | M1 simulator port + E0 oracle green on Mac CPU; E0-A/E0-B/E0-C all PASS across seeds 0-9; venv .venv with numpy/scipy/scikit-learn |
| 2026-09-09 | Mac | src/recover/backbone.py, experiments/e1_recover_check.py | results/e1/e1_recover_report.json | 2e93c78 | M2 recover backbone. Bing "ours" (external/, .venv-recover) ran but did not recover on our variance-changing interventions (MCC 0.63, DO-control 0.999 proves estimator ok); adopted design-doc R2 fallback: covariance/precision-difference linear estimator, pure numpy/scipy, no torch, runs in .venv. E1 recovery MCC 0.999 all seeds 0-9 vs E0-C blind 0.42-0.75 |
| 2026-09-09 | Mac | experiments/e2p5_regime_coupling.py | results/e2p5/regime_coupling_report.json | e26403c | M2.5 gate-backbone regime coupling (gates M3). STATUS=FAIL by design (exits 1). Q1 FAILED: P4 gate does NOT read per-env atomic rank cleanly in the reducing regime (LFC k_hat correct 5/10 seeds) nor increasing (7/10); misreads are +/-1 at low sv-gap. Backbone recovers reducing only (MCC 0.999) not increasing (mean 0.79). NOT cleanly coupled: backbone's only working regime (reducing) is the gate's WEAKER regime. M3 must not assume the gate can vet reducing-regime data at per-env resolution |
