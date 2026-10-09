"""E12 HCP: GRD on a held-out basis versus the in-sample (E3) basis.
Pre-registration: notes/e12_prereg.md (committed before this code).

Per seed (SeedSequence([seed, 7777])): C = 18 subjects drawn from those with all 7
tasks; A and B = 37 subjects each drawn from the remaining WM subjects. Environments
are the C subjects' rows for each non-WM task (n = 18); fakes are 50 subsets of 14 of
the C subjects' WM rows. Direction AB: T0h = PCA on A, reference proj(B); T0i = PCA on
B, reference proj(B). Direction BA swaps A and B. GRD only, always through
precision_readout.detect_with_pvalues; BH is applied separately to environments and to
fakes, per direction and test, here and again in e12_aggregate.py.

RNG: environment t (0..5, non-WM task order) uses SeedSequence([seed, t, 0]); fake f
uses i = 100000 + f, with k = 0 for its null and k = 6 for its row selection. Both
directions and both tests reuse the same streams (paired comparisons).

Output: results/e12_standard_tests/hcp/seed<S>.json (--smoke: .../smoke/hcp/, B=199).
--selftest runs synthetic subjects (code path only, written only with --out).
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402

T2 = E.T2
D = 10
N_C, N_A, N_FAKE_ROWS = 18, 37, 14
OUT_ROOT = E.REPO / "results" / "e12_standard_tests"
SMOKE_B = 199


def split_subjects(data, seed):
    tasks = E.HCP_TASKS
    complete = sorted(set.intersection(*(set(data[t]) for t in tasks)))
    assert len(complete) >= N_C, f"only {len(complete)} subjects have all 7 tasks (need {N_C})"
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), E.SPLIT_SALT]))
    C = sorted(rng.choice(complete, N_C, replace=False).tolist())
    rest = sorted(set(data["WM"]) - set(C))
    assert len(rest) >= 2 * N_A, f"only {len(rest)} remaining WM subjects (need {2 * N_A})"
    AB = rng.choice(rest, 2 * N_A, replace=False).tolist()
    return dict(n_complete=len(complete), n_wm=len(data["WM"]), C=C,
                A=sorted(AB[:N_A]), B=sorted(AB[N_A:]))


def gate(Y, Yref, seed, i, B):
    n_e = len(Y)
    assert n_e <= len(Yref) // 2, f"n_env {n_e} > |ref|//2 = {len(Yref) // 2}"
    assert PR_disjoint(len(Yref), n_e), "GRD null would fall back to the bootstrap path"
    s, p = E.grd(Y, Yref, B, E.rng_for(seed, i, E.K_NULL))
    return dict(stat=s, p=p)


def PR_disjoint(n_ref, n_e):
    return bool(E.PR._use_disjoint_null(n_ref, n_e, D, True))


def run_direction(data, split, basis_key, ref_key, seed, B, n_fake):
    rows = lambda task, ids: np.asarray([data[task][s] for s in ids])  # noqa: E731
    X_basis, X_ref = rows("WM", split[basis_key]), rows("WM", split[ref_key])
    proj = {"T0h": E.fit_projection(X_basis, D), "T0i": E.fit_projection(X_ref, D)}
    refs = {t: E.project(X_ref, *proj[t]) for t in proj}
    envs = []
    for t, task in enumerate(E.HCP_TASKS[1:]):
        ids = [s for s in split["C"] if s in data[task]]
        X = rows(task, ids)
        rec = dict(label=task, index=t, n_e=len(ids), is_fake=False)
        for test in proj:
            rec[test] = gate(E.project(X, *proj[test]), refs[test], seed, t, B)
        envs.append(rec)
    X_wm = rows("WM", split["C"])
    fakes = []
    for f in range(n_fake):
        i = E.FAKE_OFFSET + f
        pick = E.rng_for(seed, i, E.K_FAKE_ROWS).choice(len(X_wm), N_FAKE_ROWS, replace=False)
        rec = dict(label=f"fake_{f:02d}", index=i, n_e=N_FAKE_ROWS, is_fake=True)
        for test in proj:
            rec[test] = gate(E.project(X_wm[pick], *proj[test]), refs[test], seed, i, B)
        fakes.append(rec)
    bh = {}
    for test in proj:
        bh[test] = dict(
            hcp_bh_count=int(E.PR.bh_fdr([r[test]["p"] for r in envs], q=E.Q).sum()),
            bh_count_fake=int(E.PR.bh_fdr([r[test]["p"] for r in fakes], q=E.Q).sum()),
            raw_count=int(sum(r[test]["p"] <= E.ALPHA for r in envs)),
            raw_count_fake=int(sum(r[test]["p"] <= E.ALPHA for r in fakes)))
    return dict(basis=basis_key, reference=ref_key, n_ref=len(X_ref), environments=envs,
                fakes=fakes, bh=bh)


def run(data, seed, B, smoke, source):
    t0 = time.time()
    split = split_subjects(data, seed)
    n_fake = 10 if smoke else E.N_FAKE
    directions = {"AB": run_direction(data, split, "A", "B", seed, B, n_fake),
                  "BA": run_direction(data, split, "B", "A", seed, B, n_fake)}
    results = dict(
        experiment="E12", dataset="HCP", seed=int(seed), d=D, B=int(B), smoke=bool(smoke),
        alpha=E.ALPHA, q=E.Q, tasks=E.HCP_TASKS, n_subjects_per_task={
            t: len(data[t]) for t in E.HCP_TASKS},
        n_complete=split["n_complete"], C=split["C"], A=split["A"], B_subjects=split["B"],
        n_fake=n_fake, fake_rows=N_FAKE_ROWS,
        bh_families="per direction and test: environments and fakes separately",
        directions=directions)
    results = json.loads(json.dumps(results))
    meta = dict(source=source, argv=sys.argv[1:], seconds=round(time.time() - t0, 3),
                provenance=T2.provenance(__file__, ["experiments/e12_common.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="synthetic subjects, code path only")
    ap.add_argument("--out", default=None, help="selftest output path (never under results/)")
    args = ap.parse_args()
    B = SMOKE_B if args.smoke else args.B
    if args.selftest:
        doc = run(E.make_selftest_hcp(args.seed), args.seed, B, args.smoke, "synthetic selftest")
        if args.out:
            T2.write_json(args.out, doc)
        print(json.dumps(dict(results_sha256=doc["results_sha256"], bh={
            k: v["bh"] for k, v in doc["results"]["directions"].items()})))
        return
    data = E.load_hcp_subjects()
    doc = run(data, args.seed, B, args.smoke, str(E.hcp_ts_root()))
    path = (OUT_ROOT / "smoke" / "hcp" if args.smoke else OUT_ROOT / "hcp") / f"seed{args.seed}.json"
    T2.write_json(path, doc)
    r = doc["results"]
    print(f"[HCP] seed={args.seed} B={B} smoke={args.smoke} complete={r['n_complete']} "
          f"per_task={r['n_subjects_per_task']} -> {path}")
    for k, v in r["directions"].items():
        print(f"  {k}: {json.dumps(v['bh'])}")
    print(f"results_sha256={doc['results_sha256']} seconds={doc['meta']['seconds']}")


if __name__ == "__main__":
    main()
