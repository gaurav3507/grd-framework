"""D4-A: top-2 control-PC alignment of the held-out detected mean shifts, with a
1000-replicate disjoint baseline (pre-registration: notes/d4_prereg.md).

Observed, per detected environment: e3_gate_compare.shift_alignment([Y_heldout], R_j,
[True])["top2_energy_median"], with R_j the environment's corrected REF copy built as in
e12_common.run_environment; the observed value is the median over the detected set,
as e9_alignment_baseline.align_all does.

Baseline: replicate r uses one generator E.rng_for(seed, 400000 + r, E.K_REFDRAW); for
each detected environment, in detected order, it draws n_e rows from the uncorrected
projected REF R without replacement and scores shift_alignment([R[rows]], R[complement],
[True]); the replicate value is the median over the set. p = (1 + #(median_r >=
observed)) / (N_REP + 1). The T0h-detected set carries P2; the T1-detected set is
reported descriptively with the same streams.

Output: results/d4_heldout/alignment/<ds>_seed<s>.json. --selftest: synthetic panel,
first 3 retained reals as the detected set, 20 replicates, nothing written.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d4_common as C  # noqa: E402
import e3_gate_compare as GC  # noqa: E402

E, S, T2 = C.E, C.S, C.T2
N_REP = 1000
BASE_I = 400_000
OUT = C.OUT_ROOT / "alignment"


def _stat(Y, R):
    return GC.shift_alignment([Y], R, [True])["top2_energy_median"]


def align_block(detected, shared, seed, n_rep):
    if not detected:
        return dict(n_detected=0, note="no detected environments")
    R = shared["ref_heldout"]
    nR = len(R)
    obs = [_stat(t["Y_heldout"], C.ref_for(t, shared)) for t in detected]
    observed = float(np.median(obs))
    sizes = [len(t["Y_heldout"]) for t in detected]
    meds = np.empty(n_rep)
    for r in range(n_rep):
        rng = E.rng_for(seed, BASE_I + r, E.K_REFDRAW)
        vals = []
        for n_e in sizes:
            rows = rng.choice(nR, n_e, replace=False)
            keep = np.ones(nR, dtype=bool)
            keep[rows] = False
            vals.append(_stat(R[rows], R[keep]))
        meds[r] = np.median(vals)
    q05, q95 = (float(np.quantile(meds, q)) for q in (0.05, 0.95))
    return dict(
        n_detected=len(detected), labels=[t["label"] for t in detected], n_e=sizes,
        target_corrected=[t["target_column"] is not None for t in detected],
        observed_per_env=obs, observed_median=observed,
        baseline=dict(n_rep=n_rep, median=float(np.median(meds)), mean=float(np.mean(meds)),
                      q05=q05, q95=q95),
        p_one_sided=float((1 + np.count_nonzero(meds >= observed)) / (n_rep + 1)),
        observed_exceeds_q95=bool(observed > q95))


def run(ds, seed):
    t0 = time.time()
    _panel, tasks, shared, design = C.load_heldout(ds, seed)
    by_label = {t["label"]: t for t in tasks if not t["is_fake"]}
    out = dict(experiment="D4-A", dataset=ds, seed=seed, d=C.D, n_rep=N_REP,
               baseline_stream=f"rng_for(seed, {BASE_I} + r, K_REFDRAW), one generator per replicate",
               n_ref=int(len(shared["ref_heldout"])), fake_n=design["fake_n"], sets={})
    for test in ("T0h", "T1"):
        b = C.bh_sets(ds, seed, test)
        block = align_block([by_label[l] for l in b["labels"]], shared, seed, N_REP)
        block["role"] = "P2" if test == "T0h" else "descriptive (mean-shift headline set)"
        out["sets"][test] = block
        print(f"[{ds} seed {seed}] {test}: n {block['n_detected']} obs "
              f"{block.get('observed_median')} base q95 {block.get('baseline', {}).get('q95')} "
              f"p {block.get('p_one_sided')}", flush=True)
    doc = C.document(out, __file__, t0, extra_files=["experiments/e3_gate_compare.py",
                                                     "experiments/e9_alignment_baseline.py"])
    path = OUT / f"{ds}_seed{seed}.json"
    T2.write_json(path, doc)
    print(f"-> {path}", flush=True)


def selftest():
    panel = E.make_selftest_panel_full("k562", seed=0)
    tasks, _records, shared, _design = S.build_tasks(panel, 0, 199, C.D, False)
    det = [t for t in tasks if not t["is_fake"]][:3]
    block = align_block(det, shared, 0, 20)
    worst = 0.0
    for t, v in zip(det, block["observed_per_env"]):
        R = shared["ref_heldout"]
        j = t["target_column"]
        if j is not None:
            R = E.correct(R, shared["ref_target_columns"][j], t["mu_h_j"], t["P_h_j"])
        direct = GC.shift_alignment([t["Y_heldout"]], R, [True])["top2_energy_median"]
        worst = max(worst, abs(direct - v))
    print(f"[d4_alignment] selftest: n {block['n_detected']} corrected {block['target_corrected']} "
          f"obs {block['observed_median']} base {block['baseline']} p {block['p_one_sided']} "
          f"direct_max_abs_diff {worst}", flush=True)
    if worst != 0.0:
        raise SystemExit("STOP: observed statistic differs from shift_alignment applied directly")
    print("[d4_alignment] selftest passed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=C.DATASETS)
    ap.add_argument("--seed", type=int, choices=C.SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.dataset is None or a.seed is None:
        ap.error("--dataset and --seed are required")
    run(a.dataset, a.seed)


if __name__ == "__main__":
    main()
