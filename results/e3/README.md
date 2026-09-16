# E3 real-data gate screens

Gate detection outputs for the E3 real datasets (Perturb-seq K562 / RPE1 / Norman,
fMRI HCP / ABIDE), plus stability and positive-control artefacts under the sibling
`results/e3_stability/` and `results/e3_poscontrol/` directories.

## Retained pre-fix artefact (do not read as a current number)

`e3_k562_gate_VOID_sample_size_artifact.json` is a **deliberately retained** artefact
from **before** the sample-size-matched null correction (Lesson 2). It is the K562
screen run with the old, non-size-matched null, which inflated detection to about
**99 percent** of powered perturbations. After the size-matched-null fix the same
screen detects about **13 percent**.

It is kept as **evidence for the measurement-integrity result** (the whole point of
the precondition gate: a mismatched null manufactures detections out of sampling
noise), NOT as a live result. Do not cite the numbers in the VOID file as a current
detection rate. The `VOID` in the filename marks it as void.

## Live K562 result

The current, size-matched K562 gate screen is:

    results/e3/e3_k562_gate_fixed.json

(RPE1's live screen is `e3_rpe1_gate_fixed.json`.) These `*_gate_fixed.json` files use
the size-matched per-environment null and are the numbers to cite.
