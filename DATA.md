# Data

The synthetic experiments (E0, E1, E2, and the E3 null-size regression check) need no
external data: they generate everything from `sim/simulator.py` and run on CPU.

The E3 real-data gate screens and the iLCS baseline read external single-cell and
neuroimaging datasets that are **not** included in this repository (they are large and
have their own licences).

## Expected location

Scripts expect the external data under a root given by the environment variable

    GRD_DATA_ROOT      (default: /workspace/external)

## Expected files

Perturb-seq (CausalBench / Norman), under `discrepancy_vae/datasets/`:

    causalbench_k562.h5ad        (K562 CRISPRi)
    causalbench_rpe1.h5ad         (RPE1 CRISPRi)
    Norman2019_raw.h5ad          (Norman CRISPRa, single-gene)

Each carries per-cell perturbation labels in `obs['guide_ids']`, with the control
cells labelled by the empty string `''`. fMRI (HCP task, ABIDE site) inputs are read
by the corresponding `experiments/e3_fmri_*` and `experiments/e3_stability_hcp.py`
scripts.

So, for the default root, the K562 file is expected at:

    /workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad

## Path portability

`GRD_DATA_ROOT` is honoured by **new** scripts. The original E3 scripts
(`experiments/e3_perturbseq_panel.py`, `experiments/e3_rpe1_confound_check.py`,
`experiments/e3_stability_perturbseq.py`, and the other `e3_*` scripts) still contain
hardcoded `/workspace/external/...` paths; migrating them to `GRD_DATA_ROOT` is
pending and intentionally out of scope here (those scripts and their committed outputs
are left untouched). Until then, place the data under `/workspace/external` or adjust
those paths locally when running the original E3 scripts.
