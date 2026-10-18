"""Diagnostic for the E13 --verify failures (K562, RPE1): fakes vs reals.

Builds tasks exactly as e13_power_sign.py --verify does (recompute(panel), then
E.init_worker(shared)) and runs E.run_environment on the 50 FAKE tasks (uncorrected,
so the target-correction path is not involved) and on the 5 reals that verify_0d
uses (pick_reals). T0h and T2 stat and p are compared with the stored E12 values in
results/e12_standard_tests/<ds>/seed0_d10.json, keyed by label, as verify_0d does.
B = 9999 as stored. Diagnostic only: nothing is written under results/.

Output: $GRD_DATA_ROOT/build/diag/step1_<ds>.json

Usage:
  python -u experiments/diag_e13_verify.py --dataset k562 --workers 24
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402
import e13_power_sign as M  # noqa: E402
from data_paths import data_root  # noqa: E402

TESTS = ("T0h", "T2")


def compare(recs, stored):
    rows = []
    for rec, t in recs:
        st = stored[t["label"]]
        row = dict(label=t["label"], target_corrected=t["target_column"] is not None)
        for k in TESTS:
            ds = abs(rec[k]["stat"] - st[k]["stat"])
            row[k] = dict(stat=rec[k]["stat"], stat_stored=st[k]["stat"], abs_stat_diff=ds,
                          rel_stat_diff=ds / abs(st[k]["stat"]) if st[k]["stat"] != 0 else None,
                          p=rec[k]["p"], p_stored=st[k]["p"], abs_p_diff=abs(rec[k]["p"] - st[k]["p"]))
        rows.append(row)
    rel = [r[k]["rel_stat_diff"] for r in rows for k in TESTS if r[k]["rel_stat_diff"] is not None]
    summary = dict(
        n=len(rows),
        max_abs_stat_diff=max(r[k]["abs_stat_diff"] for r in rows for k in TESTS),
        max_abs_p_diff=max(r[k]["abs_p_diff"] for r in rows for k in TESTS),
        max_rel_stat_diff=max(rel) if rel else None,
        per_test={k: dict(max_abs_stat_diff=max(r[k]["abs_stat_diff"] for r in rows),
                          max_abs_p_diff=max(r[k]["abs_p_diff"] for r in rows))
                  for k in TESTS})
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=("k562", "rpe1", "norman"))
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if "GRD_DATA_ROOT" not in os.environ:
        sys.exit("STOP: GRD_DATA_ROOT is not set")
    t0 = time.time()
    panel = E.load_panel_full(a.dataset)
    tasks, shared, _design, _pool, check0c = M.recompute(panel)
    print(f"[{a.dataset}] step0c {json.dumps(check0c)} load+recompute {time.time() - t0:.1f}s", flush=True)
    stored = M.load_stored(a.dataset)

    fakes = [t for t in tasks if t["is_fake"]]
    reals = M.pick_reals(tasks)
    todo = fakes + reals
    t1 = time.time()
    recs = M.run_pool(E.run_environment, E.init_worker, shared, todo, a.workers)
    print(f"[{a.dataset}] ran {len(todo)} envs in {time.time() - t1:.1f}s", flush=True)
    pairs = list(zip(recs, todo))
    fs, frows = compare(pairs[:len(fakes)], stored)
    rs, rrows = compare(pairs[len(fakes):], stored)

    import platform
    out = dict(dataset=a.dataset, B=9999, data_root=str(data_root()), step0c=check0c,
               fakes=dict(summary=fs, envs=frows), reals=dict(summary=rs, envs=rrows),
               provenance=dict(python=platform.python_version(), numpy=np.__version__,
                               scipy=scipy.__version__),
               seconds=round(time.time() - t0, 1))
    d = data_root() / "build" / "diag"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step1_{a.dataset}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(fakes=fs, reals=rs), indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
