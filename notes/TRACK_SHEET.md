# GRD Framework Track Sheet

One row per produced artefact. Fill commit_hash after the commit that produced the row.

| date | machine (Mac/A100) | script | produced_artefact | commit_hash | notes |
|------|--------------------|--------|-------------------|-------------|-------|
| 2026-09-01 | Mac | (none) | M0 scaffolding + design doc | (fill) | repo created |
| 2026-09-08 | Mac | sim/simulator.py, src/gate/rank_readout.py, experiments/e0_oracle.py | results/e0/e0_report.json (e0_spectra.npz untracked) | 278dd98 | M1 simulator port + E0 oracle green on Mac CPU; E0-A/E0-B/E0-C all PASS across seeds 0-9; venv .venv with numpy/scipy/scikit-learn |
