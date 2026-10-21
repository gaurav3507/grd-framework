"""D4 shared helpers (pre-registration: notes/d4_prereg.md).

- load_heldout(ds, seed): the E12 held-out construction, by calling
  e12_standard_tests.build_tasks unchanged (split, basis, REF, target corrections,
  fakes); nothing is reimplemented.
- stored_e12(ds, seed): the stored E12 seed file (d = 10).
- bh_sets(ds, seed, test): BH at q = 0.05 recomputed from the stored p-values over the
  retained reals plus the fakes in file order, exactly as e12_aggregate.family does,
  and asserted equal to summary.json's bh_selected_real (order-insensitive) and
  bh_count_real; STOP otherwise.
- ref_for(task, shared): the per-environment corrected REF copy, as
  e12_common.run_environment builds it.

--selftest: bh_sets for every dataset, seed and test in (T0h, T1), assertions printed.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402
import e12_standard_tests as S  # noqa: E402
import e12_aggregate as AG  # noqa: E402

T2 = E.T2
DATASETS = ("k562", "norman", "rpe1")
SEEDS = (0, 1, 2)
D = 10
B_E12 = 9999
E12_ROOT = E.REPO / "results" / "e12_standard_tests"
OUT_ROOT = E.REPO / "results" / "d4_heldout"
SOURCES = ["experiments/d4_common.py", "experiments/e12_common.py",
           "experiments/e12_standard_tests.py", "experiments/e12_aggregate.py"]


def load_heldout(ds, seed):
    """E12's held-out construction via build_tasks, checked against the stored design."""
    panel = E.load_panel_full(ds)
    tasks, _records, shared, design = S.build_tasks(panel, seed, B_E12, D, False)
    r = stored_e12(ds, seed)["results"]
    if design["excluded"] != r["excluded"] or design["fake_n"] != r["fake_n"]:
        raise SystemExit(f"STOP: {ds} seed {seed}: build_tasks design differs from stored E12 "
                         f"(excluded {design['excluded']} vs {r['excluded']}, "
                         f"fake_n {design['fake_n']} vs {r['fake_n']})")
    stored_reals = [e["label"] for e in r["environments"] if not e["is_fake"] and not e["excluded"]]
    task_reals = [t["label"] for t in tasks if not t["is_fake"]]
    if stored_reals != task_reals:
        raise SystemExit(f"STOP: {ds} seed {seed}: retained real labels differ from stored E12")
    return panel, tasks, shared, design


def stored_e12(ds, seed):
    return json.loads((E12_ROOT / ds / f"seed{seed}_d{D}.json").read_text())


def _summary():
    return json.loads((E12_ROOT / "summary.json").read_text())["results"]


def bh_sets(ds, seed, test, summary=None):
    """Recomputed BH decisions (e12_aggregate.family's rule), asserted against summary.json."""
    r = stored_e12(ds, seed)["results"]
    reals = [e for e in r["environments"] if not e["is_fake"] and not e["excluded"]]
    fakes = [e for e in r["environments"] if e["is_fake"]]
    p_real = [AG.p_of(e, test) for e in reals]
    p_fake = [AG.p_of(e, test) for e in fakes]
    bh = E.PR.bh_fdr(p_real + p_fake, q=E.Q)
    n = len(reals)
    labels = [e["label"] for e, s in zip(reals, bh[:n]) if s]
    st = (summary or _summary())["perturbseq"][ds][f"d{D}"]["tests"][test]["per_seed"][str(seed)]
    checks = dict(labels_equal=sorted(labels) == sorted(st["bh_selected_real"]),
                  count_equal=len(labels) == st["bh_count_real"],
                  fake_count_equal=int(bh[n:].sum()) == st["bh_count_fake"],
                  n_real_equal=n == st["n_real"])
    if not all(checks.values()):
        disagree = sorted(set(labels) ^ set(st["bh_selected_real"]))
        raise SystemExit(f"STOP: {ds} seed {seed} {test}: recomputed BH differs from summary.json "
                         f"{checks}; disagreeing labels {disagree}")
    return dict(
        dataset=ds, seed=seed, test=test, labels=labels, n_real=n, n_fake=len(fakes),
        bh_count_real=len(labels), bh_count_fake=int(bh[n:].sum()),
        decision={e["label"]: bool(s) for e, s in zip(reals, bh[:n])},
        pvalue={e["label"]: float(p) for e, p in zip(reals, p_real)},
        n_e={e["label"]: int(e["n_e"]) for e in reals},
        order=[e["label"] for e in reals], checks=checks)


def ref_for(task, shared):
    """The environment's REF copy, exactly as e12_common.run_environment builds it."""
    j = task["target_column"]
    if j is None:
        return shared["ref_heldout"]
    return E.correct(shared["ref_heldout"], shared["ref_target_columns"][j],
                     task["mu_h_j"], task["P_h_j"])


def document(results, script, t0, extra_meta=None, extra_files=()):
    results = json.loads(json.dumps(results))
    meta = dict(seconds=round(time.time() - t0, 3), argv=sys.argv[1:],
                provenance=T2.provenance(script, list(dict.fromkeys(SOURCES + list(extra_files)))))
    meta.update(extra_meta or {})
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


def selftest():
    summary = _summary()
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            for test in ("T0h", "T1"):
                b = bh_sets(ds, seed, test, summary)
                rows.append((ds, seed, test, b["bh_count_real"], b["bh_count_fake"], b["n_real"]))
                print(f"[d4_common] {ds} seed {seed} {test}: real {b['bh_count_real']}/{b['n_real']} "
                      f"fake {b['bh_count_fake']}/{b['n_fake']} checks {b['checks']}", flush=True)
    print(f"[d4_common] selftest: {len(rows)} BH families recomputed, all assertions passed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        ap.error("only --selftest is available")


if __name__ == "__main__":
    main()
