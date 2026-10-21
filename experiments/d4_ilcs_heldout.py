"""D4-I: the iLCS ungated baseline on the E12 held-out construction (pre-registration:
notes/d4_prereg.md).

Reuses ilcs_baseline unchanged: IL._eval_env, IL._ica_rows, IL.ilcs_L_given_ctrl,
IL.ilcs_fires, IL._bucket_index and the constants NAIVE_ALPHA, B_NULL, N_BUCKETS and the
ICA settings. The D4 replacements are marked "D4:" below:
  (1) REF and projection from d4_common.load_heldout: R = shared["ref_heldout"] plays
      Xc with proj = identity (already projected); environments are the retained real
      tasks' Y_heldout (target-corrected, environment-only); fakes are the 50 E12
      FAKE-pool draws; the control ICA is fitted once on R (uncorrected);
  (2) random and structured arms removed;
  (3) bucket edges from the retained sizes (inline logic of ilcs_baseline.run_dataset);
  (4) d4_calib_alpha_bucket replaces IL.calib_alpha_bucket: null draw b of bucket k uses
      E.rng_for(seed, 500000 + k * IL.B_NULL + b, E.K_REFDRAW) on R; the per-draw ICA
      seed stays IL.SEED + b as in the original body;
  (5) IL.SEED = seed;
  (6) Jaccard computed here against d4_common.bh_sets(ds, seed, "T0h"); when the union
      of the calibrated set and the T0h set is empty, jaccard = None and
      jaccard_union_empty = True (D4 pre-run clarification).
Run from the repo root (ilcs_baseline loads src/gate/precision_readout.py by relative
path). --smoke copies ilcs_baseline's smoke counts: 5 fakes, 20 perts, B = 10.

Output: results/d4_heldout/ilcs/<ds>_seed<s>.json (smoke: results/d4_heldout/ilcs/smoke/).
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
import ilcs_baseline as IL  # noqa: E402

E, T2 = C.E, C.T2
OUT = C.OUT_ROOT / "ilcs"
CALIB_BASE_I = 500_000


def make_calib(seed, rep_sizes):
    """D4: IL.calib_alpha_bucket with the held-out REF draws on fixed streams."""
    if len(set(rep_sizes)) != len(rep_sizes):
        raise SystemExit(f"STOP: bucket representative sizes are not unique: {rep_sizes}")

    def d4_calib_alpha_bucket(proj, Xc, control_rows, rep_size, B, rng):
        # D4: bucket index recovered from its representative size (unique, checked above);
        # rng is unused, the draws come from fixed per-(bucket, draw) streams.
        k = rep_sizes.index(rep_size)
        N = len(Xc)
        m = rep_size if rep_size < N else N
        maxL = np.empty(B)
        for b in range(B):
            # D4: null draw b of bucket k uses rng_for(seed, 500000 + k * B_NULL + b, K_REFDRAW).
            idx = E.rng_for(seed, CALIB_BASE_I + k * IL.B_NULL + b, E.K_REFDRAW).choice(
                N, m, replace=False)
            Lb, _, _, _ = IL.ilcs_L_given_ctrl(control_rows, proj(Xc[idx]), IL.SEED + b)
            maxL[b] = float(Lb.max())
        return float(np.percentile(maxL, 95))

    return d4_calib_alpha_bucket


def run(ds, seed, smoke):
    t0 = time.time()
    IL.SEED = seed  # D4: (5)
    _panel, tasks, shared, design = C.load_heldout(ds, seed)
    R = shared["ref_heldout"]  # D4: (1) plays Xc
    proj = lambda M: M  # noqa: E731  D4: (1) inputs are already projected
    reals = [t for t in tasks if not t["is_fake"]]
    fakes = [t for t in tasks if t["is_fake"]]
    sizes = [len(t["Y_heldout"]) for t in reals]

    # D4: (3) bucket edges, inline logic of ilcs_baseline.run_dataset on the retained sizes.
    smin, smax = int(min(sizes)), int(max(sizes))
    edges = np.unique(np.geomspace(smin, max(smax, smin + 1), IL.N_BUCKETS + 1).astype(int))
    while len(edges) < IL.N_BUCKETS + 1:
        edges = np.unique(np.append(edges, edges[-1] + 1))
    edges = edges[:IL.N_BUCKETS + 1]
    rep_sizes = [int(round(np.sqrt(edges[i] * edges[i + 1]))) for i in range(IL.N_BUCKETS)]

    IL.calib_alpha_bucket = make_calib(seed, rep_sizes)  # D4: (4), before any evaluation

    B = IL.B_NULL
    if smoke:
        fakes, reals, B = fakes[:5], reals[:20], 10

    control_rows = IL._ica_rows(R, IL.SEED)  # D4: (1) control ICA once on uncorrected REF
    alpha_cache = {}
    rng = np.random.default_rng(seed)  # passed through IL._eval_env; unused by the D4 null
    records = []
    for t in fakes:
        Y = t["Y_heldout"]
        records.append(IL._eval_env(Y, control_rows, proj, R, len(Y), rng, B, alpha_cache,
                                    edges, rep_sizes, t["label"], "fake"))
    print(f"[{ds} seed {seed}] fakes done {time.time() - t0:.1f}s", flush=True)
    for t in reals:
        Y = t["Y_heldout"]
        records.append(IL._eval_env(Y, control_rows, proj, R, len(Y), rng, B, alpha_cache,
                                    edges, rep_sizes, t["label"], "pert"))
    print(f"[{ds} seed {seed}] reals done {time.time() - t0:.1f}s", flush=True)

    def rate(kind, key):
        rows = [r for r in records if r["kind"] == kind]
        return float(np.mean([r[key] for r in rows])) if rows else None

    # D4: (6) Jaccard against the held-out T0h BH set, on the evaluated reals.
    evaluated = {str(r["id"]) for r in records if r["kind"] == "pert"}
    calib = {str(r["id"]) for r in records if r["kind"] == "pert" and r["calibrated_fires"]}
    gate = set(C.bh_sets(ds, seed, "T0h")["labels"]) & evaluated
    union = calib | gate
    # D4 pre-run clarification: an empty union gives jaccard = None (P4 Jaccard part undetermined).
    jaccard = len(calib & gate) / len(union) if union else None

    nonconv = int(sum(1 for r in records if not r["ica_converged"]))
    ctrl_conv = bool(control_rows[2])
    total_fits = len(records) + 1
    nonconv_total = nonconv + (0 if ctrl_conv else 1)
    results = dict(
        experiment="D4-I", dataset=ds, seed=seed, smoke=bool(smoke), d=C.D,
        naive_alpha=IL.NAIVE_ALPHA, B_null=B, n_buckets=IL.N_BUCKETS,
        bucket_edges=[int(x) for x in edges], bucket_rep_sizes=rep_sizes,
        n_fakes=len(fakes), n_reals=len(reals), fake_n=design["fake_n"], n_ref=int(len(R)),
        naive_fake_rate=rate("fake", "naive_fires"), calib_fake_rate=rate("fake", "calibrated_fires"),
        naive_real_rate=rate("pert", "naive_fires"), calib_real_rate=rate("pert", "calibrated_fires"),
        n_calib_real=len(calib), n_gate_T0h=len(gate), jaccard_calib_vs_heldout_T0h=jaccard,
        jaccard_union_empty=not union,
        control_ica_converged=ctrl_conv, ica_nonconverged_count=nonconv_total,
        ica_nonconverged_frac=nonconv_total / total_fits, total_ica_fits=total_fits,
        ica_nonconverged_note="denominator = env fits + 1 control fit (null-draw fits not counted), as ilcs_baseline",
        ica_deflation_count=int(sum(1 for r in records if r["ica_used_deflation"])),
        alpha_by_bucket={str(k): v for k, v in sorted(alpha_cache.items())},
        records=records)
    doc = C.document(results, __file__, t0, extra_files=["experiments/ilcs_baseline.py"])
    path = (OUT / "smoke" if smoke else OUT) / f"{ds}_seed{seed}.json"
    T2.write_json(path, doc)
    print(f"[{ds} seed {seed}] naive fake {results['naive_fake_rate']} calib fake "
          f"{results['calib_fake_rate']} naive real {results['naive_real_rate']} calib real "
          f"{results['calib_real_rate']} jaccard {jaccard} nonconv {results['ica_nonconverged_frac']:.4f} "
          f"seconds {doc['meta']['seconds']} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--seed", required=True, type=int, choices=C.SEEDS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.dataset, a.seed, a.smoke)


if __name__ == "__main__":
    main()
