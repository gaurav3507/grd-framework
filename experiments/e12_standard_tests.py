"""E12 Perturb-seq: GRD (held-out, in-sample, size-matched) against standard two-sample
tests. Pre-registration: notes/e12_prereg.md (committed before this code).

Per seed the controls are split BASIS 30% / REF 40% / FAKE 30%. Environments with
n_e > |REF|//2 are excluded from every test and listed by name. Each retained
environment gets T0h, T0i, T0m, T1, T2, T3 (permutation p, B draws) and T1 analytic
F and T4 Box's M (analytic). 50 fakes are drawn from FAKE at the dataset median n_e
(median over all powered perturbations). BH families are built by e12_aggregate.py.

Output: results/e12_standard_tests/<dataset>/seed<S>_d<D>.json; --smoke writes under
results/e12_standard_tests/smoke/ instead (first 10 retained plus 10 fakes, B=199).
--selftest runs a synthetic panel (code path only, written only with --out).
--oracles runs the Step 3 self-checks and writes results/e12_standard_tests/oracles.json.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402

T2 = E.T2
OUT_ROOT = E.REPO / "results" / "e12_standard_tests"
SMOKE_RETAINED, SMOKE_FAKES, SMOKE_B = 10, 10, 199
STRICT_MAPPING = ("k562", "norman")


def build_tasks(panel, seed, B, d, smoke):
    """Split, project and target-correct in the parent; workers get small arrays only."""
    Xc = panel["Xc"]
    basis, ref, fake_pool = E.control_split(len(Xc), seed)
    mu_h, P_h = E.fit_projection(Xc[basis], d)
    mu_i, P_i = E.fit_projection(Xc[ref], d)
    X_ref = Xc[ref]
    limit = len(ref) // 2
    sizes = [len(X) for X in panel["Xperts"]]
    fake_n = int(np.median(sizes))
    column = panel["mapping"]["column"]

    records, tasks, excluded, ref_columns = [], [], [], {}
    for i, (label, X) in enumerate(zip(panel["perts"], panel["Xperts"])):
        n_e = len(X)
        j = column.get(label)
        if n_e > limit:
            excluded.append(label)
            records.append(dict(label=label, index=i, n_e=n_e, is_fake=False, excluded=True,
                                target_corrected=j is not None, target_column=j))
            continue
        Yh, Yi = E.project(X, mu_h, P_h), E.project(X, mu_i, P_i)
        task = dict(seed=seed, index=i, B=B, label=label, is_fake=False, target_column=j)
        if j is not None:
            Yh = E.correct(Yh, X[:, j], mu_h[j], P_h[j])
            Yi = E.correct(Yi, X[:, j], mu_i[j], P_i[j])
            task.update(mu_h_j=float(mu_h[j]), P_h_j=P_h[j].copy(),
                        mu_i_j=float(mu_i[j]), P_i_j=P_i[j].copy())
            ref_columns[j] = X_ref[:, j].copy()
        task.update(Y_heldout=Yh, Y_insample=Yi)
        tasks.append(task)
    if smoke:
        tasks = tasks[:SMOKE_RETAINED]
        ref_columns = {t["target_column"]: ref_columns[t["target_column"]]
                       for t in tasks if t["target_column"] is not None}
    n_fake = SMOKE_FAKES if smoke else E.N_FAKE
    X_pool = Xc[fake_pool]
    for f in range(n_fake):
        i = E.FAKE_OFFSET + f
        rows = E.rng_for(seed, i, E.K_FAKE_ROWS).choice(len(X_pool), fake_n, replace=False)
        Xf = X_pool[rows]
        tasks.append(dict(seed=seed, index=i, B=B, label=f"fake_{f:02d}", is_fake=True,
                          target_column=None, Y_heldout=E.project(Xf, mu_h, P_h),
                          Y_insample=E.project(Xf, mu_i, P_i)))
    shared = dict(ref_heldout=E.project(X_ref, mu_h, P_h),
                  ref_insample=E.project(X_ref, mu_i, P_i),
                  ref_target_columns=ref_columns)
    design = dict(n_controls=int(len(Xc)), n_basis=int(len(basis)), n_ref=int(len(ref)),
                  n_fake_pool=int(len(fake_pool)), exclusion_limit=int(limit),
                  excluded=excluded, fake_n=fake_n, fake_n_rule="median n_e over all powered perturbations",
                  n_fake=n_fake)
    return tasks, records, shared, design


def run_tasks(tasks, shared, workers):
    if workers <= 1:
        E.init_worker(shared)
        return [E.run_environment(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=E.init_worker,
                             initargs=(shared,)) as pool:
        return list(pool.map(E.run_environment, tasks, chunksize=1))


def run(panel, dataset, seed, B, d, smoke, workers):
    t0 = time.time()
    tasks, records, shared, design = build_tasks(panel, seed, B, d, smoke)
    t1 = time.time()
    done = run_tasks(tasks, shared, workers)
    t2 = time.time()
    for rec in done:
        assert rec["disjoint_applied"]["T0h"] and rec["disjoint_applied"]["T0i"], (
            f"{rec['label']}: GRD null fell back to the bootstrap path")
    environments = sorted(records + done, key=lambda r: r["index"])
    mapping = {k: v for k, v in panel["mapping"].items() if k != "column"}
    results = dict(
        experiment="E12", dataset=dataset, name=panel["name"], seed=int(seed), d=int(d),
        B=int(B), smoke=bool(smoke), alpha=E.ALPHA, q=E.Q, n_cells=panel["n_cells"],
        n_genes=panel["n_genes"], control_label=panel["control_label"],
        n_powered=len(panel["perts"]), **design,
        target_mapping=mapping,
        n_target_corrected_retained=int(sum(
            r["target_corrected"] for r in environments if not r["excluded"] and not r["is_fake"])),
        tests=list(E.TESTS) + ["T1_analytic_F", "T4_analytic"],
        rng="SeedSequence([seed, i, k]); k 0 GRD null, 1 REF draw, 2 T0m, 3/4/5 T1/T2/T3, "
            "6 fake rows; fakes i = 100000 + f; split SeedSequence([seed, 7777])",
        environments=environments)
    results = json.loads(json.dumps(results))
    meta = dict(source=panel["source"], workers=int(workers), argv=sys.argv[1:],
                seconds_prepare=round(t1 - t0, 3), seconds_tests=round(t2 - t1, 3),
                var_names_head=panel["var_names_head"], var_columns=panel["var_columns"],
                provenance=T2.provenance(__file__, ["experiments/e12_common.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ Step 3 self-checks
ORACLE_PATH = OUT_ROOT / "oracles.json"
ORACLE_SALT = 12_000
ORACLE_REPS, ORACLE_B = 200, 199
NEW_FILES = ["experiments/e12_common.py", "experiments/e12_standard_tests.py",
             "experiments/e12_hcp_heldout.py", "experiments/e12_aggregate.py"]


def _oracle_tests(Y, R, o, rep, B, tests):
    """Production test calls on one synthetic (env, REF) pair; same k streams."""
    rng = lambda k: E.rng_for(ORACLE_SALT + o, rep, k)  # noqa: E731
    Rm = R[rng(E.K_REFDRAW).choice(len(R), len(Y), replace=False)]
    p = {}
    if "T0h" in tests:
        assert E.PR._use_disjoint_null(len(R), len(Y), Y.shape[1], True)
        p["T0h"] = E.grd(Y, R, B, rng(E.K_NULL))[1]
    if "T0m" in tests:
        p["T0m"] = E.grd(Y, Rm, B, rng(E.K_T0M), null="pooled")[1]
    for name, k in (("T1", E.K_T1), ("T2", E.K_T2), ("T3", E.K_T3)):
        if name in tests:
            p[name] = E.permutation_test(Y, Rm, B, rng(k), name)[1]
    return p


def _rates(o, draw, tests, reps=ORACLE_REPS, B=ORACLE_B):
    hits = {t: 0 for t in tests}
    for rep in range(reps):
        Y, R = draw(np.random.default_rng(np.random.SeedSequence([ORACLE_SALT, o, rep])))
        assert np.isfinite(Y).all() and np.isfinite(R).all()
        for t, p in _oracle_tests(Y, R, o, rep, B, tests).items():
            hits[t] += int(p <= E.ALPHA)
    return {t: hits[t] / reps for t in tests}


def _o4(reps=ORACLE_REPS, B=ORACLE_B, p=4000, rank=20, d=10, n_ref=99, n_env=40, n_basis=99):
    """Pure null; T0i = PCA fit on REF (in-sample), T0h = PCA fit on a separate basis."""
    hits = {"T0i": 0, "T0h": 0}
    for rep in range(reps):
        g = np.random.default_rng(np.random.SeedSequence([ORACLE_SALT, 4, rep]))
        W = g.standard_normal((rank, p))
        X = g.standard_normal((n_basis + n_ref + n_env, rank)) @ W + g.standard_normal(
            (n_basis + n_ref + n_env, p))
        assert np.isfinite(X).all()
        Xb, Xr, Xe = X[:n_basis], X[n_basis:n_basis + n_ref], X[n_basis + n_ref:]
        for test, Xfit in (("T0i", Xr), ("T0h", Xb)):
            mu, P = E.fit_projection(Xfit, d)
            Y, R = E.project(Xe, mu, P), E.project(Xr, mu, P)
            assert E.PR._use_disjoint_null(len(R), len(Y), d, True)
            hits[test] += int(E.grd(Y, R, B, E.rng_for(ORACLE_SALT + 4, rep, E.K_NULL))[1] <= E.ALPHA)
    return {t: v / reps for t, v in hits.items()}


def _target_correction_check():
    worst = 0.0
    n_checked = 0
    for d in (10, 30):
        panel = E.make_selftest_panel_full("k562", seed=0)
        Xc = panel["Xc"]
        basis, ref, _ = E.control_split(len(Xc), 0)
        for fit in (Xc[basis], Xc[ref]):
            mu, P = E.fit_projection(fit, d)
            for label, X in zip(panel["perts"], panel["Xperts"]):
                j = panel["mapping"]["column"][label]
                if j is None:
                    continue
                for M in (X, Xc[ref]):
                    fast = E.correct(E.project(M, mu, P), M[:, j], mu[j], P[j])
                    M2 = M.copy()
                    M2[:, j] = mu[j]
                    worst = max(worst, float(np.max(np.abs(fast - E.project(M2, mu, P)))))
                    n_checked += 1
    return dict(max_abs_diff=worst, n_checked=n_checked, tol=1e-10, passed=worst <= 1e-10)


def _identity_checks():
    g = np.random.default_rng(np.random.SeedSequence([ORACLE_SALT, 9]))
    Ya, Yb = g.standard_normal((60, 10)) + 0.2, g.standard_normal((60, 10))
    t_perm = E.permutation_test(Ya, Yb, 9, E.rng_for(0, 0, 3), "T1")[0]
    t_direct = E.hotelling_direct(Ya, Yb)
    P = np.vstack([Ya, Yb])
    from scipy.spatial.distance import cdist
    e_direct = (2 * cdist(Ya, Yb).mean() - cdist(Ya, Ya).mean() - cdist(Yb, Yb).mean())
    e_perm = E.permutation_test(Ya, Yb, 9, E.rng_for(0, 0, 5), "T3")[0]
    stream = bool(np.array_equal(E.rng_for(3, 7, E.K_NULL).random(5), np.random.default_rng(
        np.random.SeedSequence([3, 7])).random(5)))
    out = dict(hotelling_batch_vs_direct_rel=abs(t_perm - t_direct) / t_direct,
               energy_batch_vs_direct_abs=abs(e_perm - e_direct),
               k0_stream_equals_e3_stream=stream, pooled_rows=int(len(P)))
    out["passed"] = bool(out["hotelling_batch_vs_direct_rel"] < 1e-10
                         and out["energy_batch_vs_direct_abs"] < 1e-10 and stream)
    return out


def _determinism_check():
    import tempfile

    import e12_aggregate
    import e12_hcp_heldout
    panel = E.make_selftest_panel_full("k562", seed=0)
    a = run(panel, "selftest", 0, SMOKE_B, 10, True, 1)["results_sha256"]
    b = run(panel, "selftest", 0, SMOKE_B, 10, True, 4)["results_sha256"]
    data = E.make_selftest_hcp(0)
    h1 = e12_hcp_heldout.run(data, 0, SMOKE_B, True, "selftest")["results_sha256"]
    h2 = e12_hcp_heldout.run(data, 0, SMOKE_B, True, "selftest")["results_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        T2.write_json(root / "k562" / "seed0_d10.json", run(panel, "selftest", 0, SMOKE_B, 10, True, 2))
        T2.write_json(root / "hcp" / "seed0.json", e12_hcp_heldout.run(data, 0, SMOKE_B, True, "selftest"))
        s1 = E.sha256_json(e12_aggregate.aggregate(root))
        s2 = E.sha256_json(e12_aggregate.aggregate(root))
    return dict(standard_tests_workers1=a, standard_tests_workers4=b, hcp_run1=h1, hcp_run2=h2,
                aggregate_run1=s1, aggregate_run2=s2, passed=bool(a == b and h1 == h2 and s1 == s2))


def _file_checks():
    import py_compile
    import re
    dash = re.compile("[\u2014\u2013\u2012\u2015\u2e3a\u2e3b]")
    defs = {"experiments/e12_common.py": "def run_environment",
            "experiments/e12_standard_tests.py": "def oracles",
            "experiments/e12_hcp_heldout.py": "def run_direction",
            "experiments/e12_aggregate.py": "def decisions"}
    out = {}
    for f in NEW_FILES + ["notes/e12_prereg.md"]:
        path = E.REPO / f
        text = path.read_text(encoding="utf-8")
        row = dict(sha256=T2.sha256_file(path), lines=text.count("\n"),
                   dash_hits=[n for n, line in enumerate(text.splitlines(), 1) if dash.search(line)])
        if f.endswith(".py"):
            py_compile.compile(str(path), doraise=True)
            row.update(py_compile="ok", grep_back=defs[f] in text)
        out[f] = row
    passed = all(not r["dash_hits"] and r.get("grep_back", True) for r in out.values())
    return dict(files=out, passed=passed)


def oracles():
    t0 = time.time()
    I10 = lambda g, n: g.standard_normal((n, 10))  # noqa: E731

    def o2(g):
        Y = I10(g, 200)
        Y[:, 0] += 0.5
        return Y, I10(g, 600)

    def o3(g):
        Y = I10(g, 200)
        Y[:, 0] *= 0.5
        return Y, I10(g, 600)

    o1_rates = _rates(1, lambda g: (I10(g, 200), I10(g, 600)), ("T0h", "T0m", "T1", "T2", "T3"))
    o2_rates = _rates(2, o2, ("T0h", "T0m", "T1", "T2", "T3"))
    o3_rates = _rates(3, o3, ("T0h", "T0m", "T1", "T2", "T3"))
    o4_rates = _o4()
    checks = dict(
        O1=dict(design="Gaussian d=10, env 200, REF 600; T0h vs full REF, others vs a 200-row REF draw",
                rates=o1_rates, criterion="every rate in [0.02, 0.09]",
                passed=all(0.02 <= r <= 0.09 for r in o1_rates.values())),
        O2=dict(design="mean shift +0.5 SD on one axis, env 200; T1 to T3 and T0m vs a 200-row REF draw",
                rates=o2_rates, criterion="T1 >= 0.85 and T3 >= 0.80 (T0h and T0m reported)",
                passed=o2_rates["T1"] >= 0.85 and o2_rates["T3"] >= 0.80),
        O3=dict(design="one axis scaled to 0.5 SD, env 200", rates=o3_rates,
                criterion="T0h >= 0.90 and T2 >= 0.90",
                passed=o3_rates["T0h"] >= 0.90 and o3_rates["T2"] >= 0.90),
        O4=dict(design="pure null, p=4000, rank-20 W ~ N(0,1) plus unit noise, REF 99, env 40, "
                       "separate basis 99, d=10; W redrawn per rep",
                rates=o4_rates, criterion="T0i in [0.45, 0.80] and T0h <= 0.15",
                t0i_inflated=o4_rates["T0i"] > 0.09,
                passed=0.45 <= o4_rates["T0i"] <= 0.80 and o4_rates["T0h"] <= 0.15),
        target_correction=_target_correction_check(),
        identities=_identity_checks(),
        determinism=_determinism_check(),
        files=_file_checks())
    results = dict(experiment="E12 self-checks", reps=ORACLE_REPS, B=ORACLE_B, alpha=E.ALPHA,
                   rng=f"data SeedSequence([{ORACLE_SALT}, o, rep]); tests SeedSequence([{ORACLE_SALT}+o, rep, k])",
                   checks=checks,
                   summary={k: dict(passed=v["passed"], **({"rates": v["rates"]} if "rates" in v else {}))
                            for k, v in checks.items()},
                   all_pass=all(v["passed"] for v in checks.values()))
    results = json.loads(json.dumps(results))
    meta = dict(seconds=round(time.time() - t0, 1),
                provenance=T2.provenance(__file__, NEW_FILES[:1] + NEW_FILES[2:]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


def out_path(dataset, seed, d, smoke):
    base = OUT_ROOT / "smoke" if smoke else OUT_ROOT
    return base / dataset / f"seed{seed}_d{d}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(T2.DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selftest", action="store_true", help="synthetic panel, code path only")
    ap.add_argument("--out", default=None, help="selftest output path (never under results/)")
    ap.add_argument("--oracles", action="store_true", help="Step 3 self-checks")
    args = ap.parse_args()
    if args.oracles:
        doc = oracles()
        T2.write_json(ORACLE_PATH, doc)
        print(json.dumps(doc["results"]["summary"], indent=1))
        print(f"all_pass={doc['results']['all_pass']} -> {ORACLE_PATH}")
        if not doc["results"]["all_pass"]:
            raise SystemExit("STOP: a self-check failed")
        return
    B = SMOKE_B if args.smoke else args.B
    if args.selftest:
        panel = E.make_selftest_panel_full(args.dataset or "k562", seed=args.seed)
        doc = run(panel, "selftest", args.seed, B, args.d, args.smoke, args.workers)
        if args.out:
            T2.write_json(args.out, doc)
        print(json.dumps(dict(results_sha256=doc["results_sha256"], **{
            k: doc["results"][k] for k in ("excluded", "fake_n", "n_target_corrected_retained")})))
        return
    if args.dataset is None:
        ap.error("--dataset is required")
    panel = E.load_panel_full(args.dataset)
    m = panel["mapping"]
    print(f"[{args.dataset}] mapping namespace={m['namespace']} mapped={m['n_mapped']}/"
          f"{len(panel['perts'])} counts={m['match_counts']}", flush=True)
    if args.dataset in STRICT_MAPPING and m["n_mapped"] == 0:
        raise SystemExit(f"STOP: {args.dataset} maps 0 perturbation targets to feature "
                         f"columns (counts {m['match_counts']}); no mapping column found")
    doc = run(panel, args.dataset, args.seed, B, args.d, args.smoke, args.workers)
    path = out_path(args.dataset, args.seed, args.d, args.smoke)
    T2.write_json(path, doc)
    r = doc["results"]
    print(f"[{args.dataset}] seed={args.seed} d={args.d} B={B} smoke={args.smoke} "
          f"excluded={r['excluded']} fake_n={r['fake_n']} "
          f"corrected={r['n_target_corrected_retained']} -> {path}", flush=True)
    print(f"results_sha256={doc['results_sha256']} "
          f"seconds={doc['meta']['seconds_prepare'] + doc['meta']['seconds_tests']:.1f}")


if __name__ == "__main__":
    main()
