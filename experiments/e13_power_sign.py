"""E13: planted power grid (Part A) and covariance-change sign readout (Part B).
Pre-registration: notes/e13_prereg.md (committed before this code).

Every E12 statistic is imported, not reimplemented: grd (T0h, T0m), permutation_test
(T1, T2, T3), roy_stat, _memberships, correct, rng_for. The only new statistic is the
one-pass T2down / T2up pair, checked against roy_stat.

Data (seed 0, d=10): S.build_tasks(panel, 0, 9999, 10, False) plus a deterministic
recomputation of the split and held-out basis (Step 0c). Both 0c assertions run at
the start of every real run; --verify adds the Step 0d reproduction against the stored
E12 seed0_d10 values and exits nonzero on any mismatch.

Part A: cells drawn without replacement from the projected FAKE pool, planted
(A1/A2 covariance, A3 mean, A4 responder mixture) along PC1, PC5, PC10 or a random
whitened direction, tested at n_e = 200 and the median n_e with B=999.
Replicate r at (s, q) uses i = 1000000 + 10000*s + 1000*q + r; k=1 cells, 2 REF rows,
3 mixture labels, 10 to 14 T0h, T0m, T1, T2, T3. A0 (null) uses q=4, r = 0..299.

Part B: every retained real environment plus the 50 fakes, against the E12
size-matched REF draw (k=1); T2down = -log(min generalized eigenvalue), T2up =
log(max), one shared permutation pass over the pooled rows (k=20), B=9999. Also,
against the full corrected REF: min / max generalized eigenvalues, delta_hat, and
lambda_eff = w'S w / w'w for the lambda_min eigenvector w, with its rank among the
uncorrected REF eigenvalues (rank 1 = above every eigenvalue).

Modes: --dataset with --part A or B, --verify, --smoke (Part A: 2 cells x 5 reps, B=99;
Part B: 10 envs + 10 fakes, B=199; written under results/e13_power_sign/smoke/),
--selftest (synthetic panel, never written under results/), --oracles (Step 3),
--aggregate (summary.json, branches computed).
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
from scipy import linalg  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402
import e12_standard_tests as S  # noqa: E402

T2 = E.T2
SEED, D = 0, 10
OUT_ROOT = E.REPO / "results" / "e13_power_sign"
E12_ROOT = E.REPO / "results" / "e12_standard_tests"
DATASETS = ("k562", "norman", "rpe1")
DECISION = ("k562", "norman")
SCRIPT = "experiments/e13_power_sign.py"
TOL = 1e-8

# Part A
B_A, REPS, REPS_NULL, BASE_I = 999, 100, 300, 1_000_000
K_CELLS, K_REFROWS, K_LABELS = 1, 2, 3
K_TEST = {"T0h": 10, "T0m": 11, "T1": 12, "T2": 13, "T3": 14}
TESTS_A = tuple(K_TEST)
ALPHAS = (0.01, 0.05, 0.001)
ALPHA_DECIDE = 0.01
DIRECTIONS = {0: "PC1", 1: "PC5", 2: "PC10", 3: "random (whitened)"}
Q_NULL = 4
RANDOM_DIR_I = 900_000
ARMS = {
    "A1": [("v", v) for v in (0.9, 0.8, 0.7, 0.6, 0.5, 0.3)],
    "A2": [("v", v) for v in (1.1, 1.25, 1.5, 2.0, 3.0)],
    "A3": [("delta", x) for x in (0.05, 0.1, 0.2, 0.3, 0.5)],
    "A4": [("pi_delta", (p, x)) for p in (0.3, 0.6) for x in (0.5, 1.0, 2.0)],
}
SMOKE_A_CELLS = [("A1", 0.5, 0, 1), ("A4", (0.3, 2.0), 3, 0)]
SMOKE_A_REPS, SMOKE_A_B = 5, 99

# Part B
B_B, K_UPDOWN, SMOKE_B_B = 9999, 20, 199


def level_label(arm, level):
    if arm == "A0":
        return "null"
    if arm == "A4":
        return f"pi={level[0]},delta={level[1]}"
    return f"{'v' if arm in ('A1', 'A2') else 'delta'}={level}"


# ------------------------------------------------------------------ Step 0c / 0d
def recompute(panel, B=9999, smoke=False):
    """Step 0c: E12 tasks plus the projected REF and FAKE pool, with both assertions."""
    tasks, records, shared, design = S.build_tasks(panel, SEED, B, D, smoke)
    Xc = panel["Xc"]
    basis, ref, fake_pool = E.control_split(len(Xc), SEED)
    mu_h, P_h = E.fit_projection(Xc[basis], D)
    ref_ok = bool(np.array_equal(E.project(Xc[ref], mu_h, P_h), shared["ref_heldout"]))
    fakes_ok = []
    for t in tasks:
        if t["is_fake"]:
            f = t["index"] - E.FAKE_OFFSET
            rows_f = E.rng_for(SEED, E.FAKE_OFFSET + f, E.K_FAKE_ROWS).choice(
                len(fake_pool), design["fake_n"], replace=False)
            fakes_ok.append(bool(np.array_equal(E.project(Xc[fake_pool][rows_f], mu_h, P_h),
                                                t["Y_heldout"])))
    check = dict(ref_heldout_equal=ref_ok, n_fakes_checked=len(fakes_ok), fakes_equal=all(fakes_ok))
    if not (ref_ok and fakes_ok and all(fakes_ok)):
        raise SystemExit(f"STOP: Step 0c recomputation differs from build_tasks: {check}")
    pool = E.project(Xc[fake_pool], mu_h, P_h)
    return tasks, shared, design, pool, check


def ref_for(task, shared):
    """Part B REF: E12's per-environment target correction (run_environment 282-286)."""
    j = task["target_column"]
    if j is None:
        return shared["ref_heldout"]
    return E.correct(shared["ref_heldout"], shared["ref_target_columns"][j], task["mu_h_j"], task["P_h_j"])


def pick_reals(tasks, n=5):
    """The first target-corrected and the first uncorrected real task (when they exist),
    then the earliest remaining real tasks, n in total, in E12 order."""
    reals = [t for t in tasks if not t["is_fake"]]
    chosen = []
    for want in (True, False):
        first = next((t for t in reals if (t["target_column"] is not None) == want), None)
        if first is not None:
            chosen.append(first)
    for t in reals:
        if len(chosen) >= n:
            break
        if all(t is not c for c in chosen):
            chosen.append(t)
    return sorted(chosen, key=lambda t: t["index"])


def verify_0d(tasks, shared, stored=None, B=9999):
    """run_environment against the stored E12 values (real data) and against this
    script's own REF construction followed by E.grd / E.permutation_test (always)."""
    E.init_worker(shared)
    rows = []
    for t in pick_reals(tasks):
        rec = E.run_environment(t)
        i, Yh = t["index"], t["Y_heldout"]
        Rh = ref_for(t, shared)
        s0, p0 = E.grd(Yh, Rh, B, E.rng_for(SEED, i, E.K_NULL))
        draw = E.rng_for(SEED, i, E.K_REFDRAW).choice(len(Rh), len(Yh), replace=False)
        s2, p2 = E.permutation_test(Yh, Rh[draw], B, E.rng_for(SEED, i, E.K_T2), "T2")
        own = dict(T0h=(s0, p0), T2=(s2, p2))
        row = dict(label=t["label"], target_corrected=t["target_column"] is not None,
                   own_max_abs_diff=max(abs(rec[k]["stat"] - own[k][0]) + abs(rec[k]["p"] - own[k][1])
                                        for k in own))
        if stored is not None:
            st = stored[t["label"]]
            row["stored_max_abs_diff"] = max(abs(rec[k]["stat"] - st[k]["stat"]) + abs(rec[k]["p"] - st[k]["p"])
                                             for k in ("T0h", "T2"))
        rows.append(row)
    worst = max(max(r["own_max_abs_diff"], r.get("stored_max_abs_diff", 0.0)) for r in rows)
    out = dict(envs=rows, max_abs_diff=worst, tol=TOL, compared_to_stored=stored is not None,
               passed=bool(worst <= TOL))
    if not out["passed"]:
        raise SystemExit(f"STOP: Step 0d reproduction mismatch: {out}")
    return out


def load_stored(dataset):
    path = E12_ROOT / dataset / f"seed{SEED}_d{D}.json"
    return {e["label"]: e for e in json.loads(path.read_text())["results"]["environments"]}


# ------------------------------------------------------------------ Part A
def whitening(ref):
    """S = cov(REF); eigenvalues descending, eigenvector signs fixed (largest entry > 0)."""
    Sm = np.cov(ref, rowvar=False)
    lam, V = np.linalg.eigh(Sm)
    lam, V = lam[::-1], V[:, ::-1]
    V = V * np.where(V[np.abs(V).argmax(0), np.arange(V.shape[1])] < 0, -1.0, 1.0)
    W = (V / np.sqrt(lam)) @ V.T
    Winv = (V * np.sqrt(lam)) @ V.T
    return lam, V, W, Winv


def directions(V, seed=SEED):
    z = E.rng_for(seed, RANDOM_DIR_I, 0).standard_normal(V.shape[0])
    return np.vstack([V[:, 0], V[:, 4], V[:, 9], z / np.linalg.norm(z)])


def plant(Y, arm, level, a, W, Winv, m, labels):
    """Row-vector convention. A1/A2: m + (Y - m) W (I + (sqrt(v) - 1) a a') W^-1;
    A3: Y + delta a' W^-1; A4: the first round(pi n) cells of `labels` get + delta a' W^-1."""
    if arm == "A0":
        return Y
    if arm in ("A1", "A2"):
        M = np.eye(len(a)) + (np.sqrt(level) - 1.0) * np.outer(a, a)
        return m + (Y - m) @ W @ M @ Winv
    shift = a @ Winv
    if arm == "A3":
        return Y + level * shift
    pi, delta = level
    k = int(round(pi * len(Y)))
    out = Y.copy()
    out[labels[:k]] += delta * shift
    return out


def run_tests(Y, ref, rows, B, seed, i, tests=TESTS_A):
    Rm = ref[rows]
    p = {}
    for t in tests:
        rng = E.rng_for(seed, i, K_TEST[t])
        if t == "T0h":
            p[t] = E.grd(Y, ref, B, rng)[1]
        elif t == "T0m":
            p[t] = E.grd(Y, Rm, B, rng, null="pooled")[1]
        else:
            p[t] = E.permutation_test(Y, Rm, B, rng, t)[1]
    return p


_A = {}


def init_a(shared):
    _A.clear()
    _A.update(shared)


def replicate_a(task):
    """One replicate (s, q, r): cells, REF rows and labels drawn once, then every
    requested (arm, level) is planted and tested on the same streams."""
    sh = _A
    seed, s, q, r, n_e = sh["seed"], task["s"], task["q"], task["r"], task["n_e"]
    i = BASE_I + 10_000 * s + 1_000 * q + r
    pool, ref = sh["pool"], sh["ref"]
    Y = pool[E.rng_for(seed, i, K_CELLS).choice(len(pool), n_e, replace=False)]
    rows = E.rng_for(seed, i, K_REFROWS).choice(len(ref), n_e, replace=False)
    labels = E.rng_for(seed, i, K_LABELS).permutation(n_e)
    a = sh["dirs"][q] if q < Q_NULL else None
    out = []
    for arm, level in task["levels"]:
        Yp = plant(Y, arm, level, a, sh["W"], sh["Winv"], sh["m"], labels)
        out.append(dict(arm=arm, level=level_label(arm, level), q=q, s=s, r=r,
                        p=run_tests(Yp, ref, rows, task["B"], seed, i, sh.get("tests", TESTS_A))))
    return out


def part_a_tasks(sizes, B, smoke):
    tasks = []
    if smoke:
        for arm, level, q, s in SMOKE_A_CELLS:
            tasks += [dict(s=s, q=q, r=r, n_e=sizes[s], B=B, levels=[(arm, level)])
                      for r in range(SMOKE_A_REPS)]
        return tasks
    all_levels = [(arm, level) for arm, levs in ARMS.items() for _, level in levs]
    for s in (0, 1):
        tasks += [dict(s=s, q=Q_NULL, r=r, n_e=sizes[s], B=B, levels=[("A0", None)]) for r in range(REPS_NULL)]
        for q in DIRECTIONS:
            tasks += [dict(s=s, q=q, r=r, n_e=sizes[s], B=B, levels=all_levels) for r in range(REPS)]
    return tasks


def run_pool(fn, init, shared, tasks, workers):
    if workers <= 1:
        init(shared)
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=init, initargs=(shared,)) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def summarize_cells(reps):
    cells = {}
    for rec in reps:
        key = (rec["arm"], rec["level"], rec["q"], rec["s"])
        cell = cells.setdefault(key, dict(arm=rec["arm"], level=rec["level"], q=rec["q"], s=rec["s"],
                                          r=[], p={t: [] for t in rec["p"]}))
        cell["r"].append(rec["r"])
        for t, v in rec["p"].items():
            cell["p"][t].append(v)
    out = []
    for key in sorted(cells, key=lambda k: (k[3], k[2], k[0], k[1])):
        c = cells[key]
        c["n_reps"] = len(c["r"])
        c["power"] = {t: {str(al): float(np.mean(np.asarray(v) <= al)) for al in ALPHAS} for t, v in c["p"].items()}
        out.append(c)
    return out


def part_a(panel, dataset, B, smoke, workers, seed=SEED, recomputed=None):
    t0 = time.time()
    tasks_e12, shared, design, pool, check0c = recomputed or recompute(panel)
    ref = shared["ref_heldout"]
    lam, V, W, Winv = whitening(ref)
    dirs = directions(V, seed)
    sizes = {0: 200, 1: int(design["fake_n"])}
    for n in sizes.values():
        assert n <= len(pool) and E.PR._use_disjoint_null(len(ref), n, D, True), n
    shared_a = dict(seed=seed, pool=pool, ref=ref, W=W, Winv=Winv, m=pool.mean(0), dirs=dirs)
    reps = [rec for out in run_pool(replicate_a, init_a, shared_a, part_a_tasks(sizes, B, smoke), workers)
            for rec in out]
    results = dict(
        experiment="E13", part="A", prereg="notes/e13_prereg.md", dataset=dataset, seed=seed, d=D, B=B,
        smoke=bool(smoke), sizes={str(k): v for k, v in sizes.items()}, n_ref=int(len(ref)),
        n_pool=int(len(pool)), ref_eigenvalues=lam.tolist(), directions=DIRECTIONS,
        direction_vectors_whitened=dirs.tolist(), alphas=list(ALPHAS), decision_alpha=ALPHA_DECIDE,
        step0c=check0c, rng="i = 1000000 + 10000*s + 1000*q + r; k 1 cells, 2 REF rows, 3 labels, "
                            "10..14 T0h, T0m, T1, T2, T3; A0 q=4",
        cells=summarize_cells(reps))
    results = json.loads(json.dumps(results))
    meta = dict(workers=workers, argv=sys.argv[1:], seconds=round(time.time() - t0, 1),
                provenance=T2.provenance(__file__, ["experiments/e12_common.py",
                                                    "experiments/e12_standard_tests.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ Part B
def _updown(Sa, Sb):
    lam = linalg.eigh(Sa, Sb, eigvals_only=True)
    return -np.log(lam.min()), np.log(lam.max())


def updown_test(Ya, Yb, B, rng):
    """T2down and T2up with one shared permutation pass over the pooled rows."""
    P = np.vstack([Ya, Yb])
    N, n_a = len(P), len(Ya)
    obs = _updown(np.cov(Ya, rowvar=False), np.cov(Yb, rowvar=False))
    exceed = np.zeros(2, dtype=np.int64)
    for Sm in E._memberships(N, n_a, B, rng):
        for s in Sm:
            v = _updown(np.cov(P[s], rowvar=False), np.cov(P[~s], rowvar=False))
            exceed += (np.asarray(v) >= np.asarray(obs))
    p = (1 + exceed) / (B + 1)
    return dict(stat=float(obs[0]), p=float(p[0])), dict(stat=float(obs[1]), p=float(p[1]))


def full_ref_readouts(Y, R, ref_eigs):
    Se, Sr = np.cov(Y, rowvar=False), np.cov(R, rowvar=False)
    lam, vecs = linalg.eigh(Se, Sr)
    w = vecs[:, 0]
    lam_eff = float(w @ Sr @ w / (w @ w))
    ne, nr = len(Y), len(R)
    Sp = ((ne - 1) * Se + (nr - 1) * Sr) / (ne + nr - 2)
    diff = Y.mean(0) - R.mean(0)
    return dict(gen_eig_min=float(lam[0]), gen_eig_max=float(lam[-1]),
                delta_hat=float(np.sqrt(diff @ np.linalg.solve(Sp, diff))),
                lambda_eff=lam_eff, lambda_eff_rank=int(1 + np.sum(np.asarray(ref_eigs) > lam_eff)))


def env_b(task):
    sh = E._SHARED
    i, Yh = task["index"], task["Y_heldout"]
    Rh = ref_for(task, sh)
    draw = E.rng_for(SEED, i, E.K_REFDRAW).choice(len(Rh), len(Yh), replace=False)
    down, up = updown_test(Yh, Rh[draw], sh["B_partB"], E.rng_for(SEED, i, K_UPDOWN))
    rec = dict(label=task["label"], index=i, is_fake=task["is_fake"], n_e=int(len(Yh)),
               target_corrected=task["target_column"] is not None, T2down=down, T2up=up,
               roy_from_updown=max(down["stat"], up["stat"]))
    rec.update(full_ref_readouts(Yh, Rh, sh["ref_eigs"]))
    return rec


def part_b(panel, dataset, B, smoke, workers, stored=None, recomputed=None):
    t0 = time.time()
    tasks, shared, design, _pool, check0c = recomputed or recompute(panel)
    if smoke:
        tasks = S.build_tasks(panel, SEED, 9999, D, True)[0]
    lam = whitening(shared["ref_heldout"])[0]
    shared_b = dict(shared, B_partB=B, ref_eigs=lam)
    done = run_pool(env_b, E.init_worker, shared_b, tasks, workers)
    roy_check = None
    if stored is not None:
        diffs = [abs(r["roy_from_updown"] - stored[r["label"]]["T2"]["stat"]) for r in done]
        roy_check = dict(n=len(diffs), max_abs_diff_vs_e12_T2=max(diffs), tol=TOL)
        if max(diffs) > TOL:
            raise SystemExit(f"STOP: max(T2down, T2up) differs from the stored E12 T2: {roy_check}")
    results = dict(
        experiment="E13", part="B", prereg="notes/e13_prereg.md", dataset=dataset, seed=SEED, d=D, B=B,
        smoke=bool(smoke), excluded=design["excluded"], fake_n=design["fake_n"],
        ref_eigenvalues=lam.tolist(), step0c=check0c, e12_T2_check=roy_check,
        rng="E12 REF draw k=1; T2down/T2up shared permutation pass k=20",
        environments=sorted(done, key=lambda r: r["index"]))
    results = json.loads(json.dumps(results))
    meta = dict(workers=workers, argv=sys.argv[1:], seconds=round(time.time() - t0, 1),
                provenance=T2.provenance(__file__, ["experiments/e12_common.py",
                                                    "experiments/e12_standard_tests.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ aggregate
def _power(cells, arm, level, q, s, test, alpha):
    for c in cells:
        if c["arm"] == arm and c["level"] == level and c["q"] == q and c["s"] == s:
            return c["power"][test][str(alpha)]
    return None


def classify_r(cells):
    p = lambda v: [_power(cells, "A1", f"v={v}", q, 1, "T0h", ALPHA_DECIDE) for q in DIRECTIONS]  # noqa: E731
    p07, p05 = p(0.7), p(0.5)
    if None in p07 or None in p05:
        return dict(branch=None, power_v07=p07, power_v05=p05)
    if min(p07) >= 0.8:
        br = "R1"
    elif max(p05) < 0.5:
        br = "R2"
    elif max(p05) >= 0.8 and min(p05) < 0.5:
        br = "R-dir"
    else:
        br = "R3"
    return dict(branch=br, power_v07=p07, power_v05=p05)


def bh_counts(envs):
    reals = [e for e in envs if not e["is_fake"]]
    fakes = [e for e in envs if e["is_fake"]]
    out = {}
    for t in ("T2down", "T2up"):
        sel = E.PR.bh_fdr([e[t]["p"] for e in reals + fakes], q=E.Q)
        out[t] = dict(real=int(sel[:len(reals)].sum()), fake=int(sel[len(reals):].sum()),
                      selected=[e["label"] for e, k in zip(reals, sel[:len(reals)]) if k])
    sel_down = set(out["T2down"]["selected"])
    lam_eff = [e["lambda_eff"] for e in reals if e["label"] in sel_down]
    out["median_lambda_eff_T2down"] = float(np.median(lam_eff)) if lam_eff else None
    out["n_real"], out["n_fake"] = len(reals), len(fakes)
    return out


def s_condition(b, up):
    a, c = (b["T2up"], b["T2down"]) if up else (b["T2down"], b["T2up"])
    return bool(a["real"] >= 3 * c["real"] and a["real"] >= 10
                and b["T2up"]["fake"] <= 3 and b["T2down"]["fake"] <= 3)


def calibration(cells):
    flags = []
    for c in cells:
        if c["arm"] == "A0":
            for t, pw in c["power"].items():
                rate = pw["0.05"]
                if not 0.02 <= rate <= 0.09:
                    flags.append(dict(size=c["s"], test=t, rate=rate))
    return flags


def aggregate(root=OUT_ROOT):
    root = Path(root)
    A, B, missing, inputs = {}, {}, [], {}
    for ds in DATASETS:
        for part, store in (("A", A), ("B", B)):
            path = root / f"{ds}_part{part}.json"
            if path.exists():
                doc = json.loads(path.read_text())
                inputs[path.name] = doc["results_sha256"]
                store[ds] = doc["results"]
            else:
                missing.append(path.name)
    per = {}
    for ds in DATASETS:
        row = {}
        if ds in A:
            cells = A[ds]["cells"]
            lam = A[ds]["ref_eigenvalues"]
            row.update(R=classify_r(cells), calibration_flags=calibration(cells),
                       median_ref_eigenvalue=float(np.median(lam)))
            a23 = [c["power"]["T0h"]["0.05"] for c in cells if c["arm"] in ("A2", "A3")]
            row["mean_T0h_power_A2_A3_alpha05"] = float(np.mean(a23)) if a23 else None
            row["T2_power_A4_pi03_delta2_median_ne"] = [
                _power(cells, "A4", "pi=0.3,delta=2.0", q, 1, "T2", ALPHA_DECIDE) for q in DIRECTIONS]
        if ds in B:
            b = bh_counts(B[ds]["environments"])
            row["B"] = b
            row["S_up"], row["S_down"] = s_condition(b, True), s_condition(b, False)
            med = float(np.median(B[ds]["ref_eigenvalues"]))
            row["lambda_eff_condition"] = (b["median_lambda_eff_T2down"] is not None
                                           and b["median_lambda_eff_T2down"] > med)
        per[ds] = row
    ready = all(ds in A and ds in B for ds in DECISION)
    R_k = per.get("k562", {}).get("R", {}).get("branch")
    R_n = per.get("norman", {}).get("R", {}).get("branch")
    R_global = None if not ready else (R_k if R_k == R_n else "split")
    if not ready:
        S_global = None
    elif all(per[ds]["S_up"] for ds in DECISION):
        S_global = "S-up"
    elif all(per[ds]["S_down"] for ds in DECISION):
        S_global = "S-down"
    else:
        S_global = "S-mixed"
    lam_cond = ready and all(per[ds]["lambda_eff_condition"] for ds in DECISION)
    clauses = []
    if ready:
        if R_global == "R1" and S_global == "S-up":
            clauses.append("the gate is blind by construction to mean shifts and variance increases; the real "
                           "covariance changes are increases; a powered gate still passes ~0. A T2up excess is also "
                           "what partial-responder mixtures (A4) produce, so it does not by itself distinguish "
                           "heterogeneous knockdown from a regulatory variance change.")
        if R_global == "R-dir":
            if lam_cond:
                clauses.append("the gate's power depends on the direction of the change relative to the control "
                               "spectrum (it is not affine-invariant); an affine-invariant test (Roy) does not have "
                               "this defect. Offered as an explanation for 0/385 (lambda_eff condition met).")
            else:
                clauses.append("R-dir, but the lambda_eff condition is not met: R and S results are reported as is.")
            if S_global == "S-up":
                clauses.append("S-up: the real covariance changes are increases. A T2up excess is also what "
                               "partial-responder mixtures (A4) produce, so it does not by itself distinguish "
                               "heterogeneous knockdown from a regulatory variance change.")
        if R_global in ("R3", "split"):
            clauses.append("report as is, decide after discussion")
        if R_global == "R2":
            clauses.append("the gate is underpowered at these cell counts")
        if not clauses:
            clauses.append("anything else: report as is, decide after discussion")
    k = per.get("k562", {})
    pred = dict(
        k562_R_dir_and_PC1_power_lt_05=None if k.get("R", {}).get("branch") is None else bool(
            k["R"]["branch"] == "R-dir" and k["R"]["power_v05"][0] < 0.5),
        k562_S_up=k.get("S_up"),
        mean_T0h_power_A2_A3_le_008={ds: (None if per[ds].get("mean_T0h_power_A2_A3_alpha05") is None else
                                          per[ds]["mean_T0h_power_A2_A3_alpha05"] <= 0.08) for ds in DATASETS},
        T2_A4_pi03_delta2_ge_08_every_direction={
            ds: (None if None in per[ds].get("T2_power_A4_pi03_delta2_median_ne", [None]) else
                 all(v >= 0.8 for v in per[ds]["T2_power_A4_pi03_delta2_median_ne"]))
            for ds in DATASETS})
    results = dict(experiment="E13", prereg="notes/e13_prereg.md", decision_alpha=ALPHA_DECIDE,
                   per_dataset=per, R_global=R_global, S_global=S_global, lambda_eff_condition=bool(lam_cond),
                   interpretation=clauses, predictions=pred,
                   uncalibrated={ds: sorted({f["test"] for f in per[ds].get("calibration_flags", [])})
                                 for ds in DATASETS},
                   missing_inputs=missing, input_results_sha256=inputs)
    return json.loads(json.dumps(results))


# ------------------------------------------------------------------ Step 3 self-checks
ORACLE_SEED = 13_000


def _gaussian(g, n, lam, Q):
    return (g.standard_normal((n, len(lam))) * np.sqrt(lam)) @ Q.T


def _planting_check():
    g = np.random.default_rng(np.random.SeedSequence([ORACLE_SEED, 1]))
    lam = np.geomspace(20, 1, D)
    Q = np.linalg.qr(g.standard_normal((D, D)))[0]
    ref, cells = _gaussian(g, 5000, lam, Q), _gaussian(g, 5000, lam, Q)
    _, V, W, Winv = whitening(ref)
    dirs = directions(V, ORACLE_SEED)
    m = cells.mean(0)
    C = np.cov(cells, rowvar=False)
    worst_cov, worst_mean = 0.0, 0.0
    for q in range(4):
        for arm in ("A1", "A2"):
            for _, v in ARMS[arm]:
                Yp = plant(cells, arm, v, dirs[q], W, Winv, m, None)
                ev = np.sort(linalg.eigh(np.cov(Yp, rowvar=False), C, eigvals_only=True))
                target = np.sort(np.r_[v, np.ones(D - 1)])
                worst_cov = max(worst_cov, float(np.max(np.abs(ev / target - 1))))
        for _, delta in ARMS["A3"]:
            s = plant(cells, "A3", delta, dirs[q], W, Winv, m, None).mean(0) - cells.mean(0)
            worst_mean = max(worst_mean, abs(float(np.sqrt(s @ np.linalg.solve(C, s))) / delta - 1))
    labels = g.permutation(len(cells))
    counts = {}
    for _, (pi, delta) in ARMS["A4"]:
        changed = int(np.sum(np.any(plant(cells, "A4", (pi, delta), dirs[3], W, Winv, m, labels) != cells, 1)))
        counts[f"pi={pi},delta={delta}"] = dict(changed=changed, expected=int(round(pi * len(cells))))
    return dict(cov_max_rel_err=worst_cov, mean_max_rel_err=worst_mean, a4_counts=counts,
                passed=bool(worst_cov <= 0.03 and worst_mean <= 0.03
                            and all(v["changed"] == v["expected"] for v in counts.values())))


def _identity_check():
    g = np.random.default_rng(np.random.SeedSequence([ORACLE_SEED, 2]))
    worst = 0.0
    for _ in range(20):
        Ya, Yb = g.standard_normal((80, D)), g.standard_normal((90, D)) * g.uniform(0.5, 2, D)
        down, up = updown_test(Ya, Yb, 1, E.rng_for(0, 0, K_UPDOWN))
        worst = max(worst, abs(max(down["stat"], up["stat"]) - E.roy_stat(Ya, Yb)))
    return dict(n=20, max_abs_diff=worst, passed=bool(worst <= 1e-10))


def _rules_check():
    """classify_r and s_condition on constructed inputs, one case per branch and boundary."""
    def cells(p07, p05):
        return [dict(arm="A1", level=lv, q=q, s=1, power={"T0h": {str(ALPHA_DECIDE): pw[q]}})
                for q in DIRECTIONS for lv, pw in (("v=0.7", p07), ("v=0.5", p05))]
    r_cases = {"R1": ([0.8, 0.9, 1, 1], [1, 1, 1, 1]), "R2": ([0, 0, 0, 0], [0.1, 0.2, 0.49, 0.3]),
               "R-dir": ([0, 0.3, 1, 0.5], [0.07, 0.25, 0.8, 0.6]), "R3": ([0.5, 0.6, 0.9, 0.7], [0.6, 0.7, 0.9, 0.8])}
    r_out = {k: classify_r(cells(*v))["branch"] for k, v in r_cases.items()}

    def bb(up, down, fu=0, fd=0):
        return {"T2up": {"real": up, "fake": fu}, "T2down": {"real": down, "fake": fd}}
    s_cases = {"up 30 vs 10": (bb(30, 10), True, True), "up 30 vs 11": (bb(30, 11), True, False),
               "up 9 vs 0": (bb(9, 0), True, False), "up, down fakes 4": (bb(30, 10, fd=4), True, False),
               "down 15 vs 5": (bb(5, 15), False, True), "down 15 vs 5, up fakes 4": (bb(5, 15, fu=4), False, False)}
    s_out = {k: s_condition(b, up) for k, (b, up, _) in s_cases.items()}
    return dict(R=r_out, S=s_out, passed=bool(all(r_out[k] == k for k in r_cases)
                                              and all(s_out[k] == v[2] for k, v in s_cases.items())))


def _oracle_power(lam, cells_spec, reps, B=199, n_e=300, n_ref=3000, n_pool=3000, workers=1, tag=0):
    g = np.random.default_rng(np.random.SeedSequence([ORACLE_SEED, 3, tag]))
    Q = np.linalg.qr(g.standard_normal((D, D)))[0]
    ref, pool = _gaussian(g, n_ref, lam, Q), _gaussian(g, n_pool, lam, Q)
    _, V, W, Winv = whitening(ref)
    shared = dict(seed=ORACLE_SEED + tag, pool=pool, ref=ref, W=W, Winv=Winv, m=pool.mean(0),
                  dirs=directions(V, ORACLE_SEED))
    out = {}
    for name, (arm, level, q, tests) in cells_spec.items():
        tasks = [dict(s=0, q=q, r=r, n_e=n_e, B=B, levels=[(arm, level)]) for r in range(reps)]
        shared["tests"] = tests
        recs = [x for o in run_pool(replicate_a, init_a, shared, tasks, workers) for x in o]
        out[name] = {t: float(np.mean([x["p"][t] <= 0.05 for x in recs])) for t in tests}
    return out


def oracles(workers=4):
    t0 = time.time()
    checks = {}
    # Step 0c / 0d on the selftest panel
    panel = E.make_selftest_panel_full("k562", seed=0)
    rec = recompute(panel)
    checks["step0c_selftest"] = dict(rec[4], passed=True)
    checks["step0d_selftest"] = verify_0d(rec[0], rec[1], stored=None)
    checks["identity_updown_vs_roy"] = _identity_check()
    checks["decision_rules"] = _rules_check()
    checks["planting"] = _planting_check()
    iso = np.ones(D)
    rates = _oracle_power(iso, {
        "A1_v05_PC1": ("A1", 0.5, 0, ("T0h",)),
        "A2_v3_random": ("A2", 3.0, 3, ("T2", "T0h")),
        "A3_delta04_random": ("A3", 0.4, 3, ("T1", "T0h"))}, reps=100, workers=workers, tag=1)
    null = _oracle_power(iso, {"A0": ("A0", None, Q_NULL, TESTS_A)}, reps=300, workers=workers, tag=2)["A0"]
    aniso = _oracle_power(np.geomspace(20, 1, D), {
        f"A1_v05_{DIRECTIONS[q]}": ("A1", 0.5, q, ("T0h",)) for q in DIRECTIONS}, reps=100, workers=workers, tag=3)
    checks["power_isotropic"] = dict(
        rates=rates, criterion="A1 v=0.5 PC1 T0h >= 0.9; A2 v=3 T2 >= 0.9, T0h <= 0.15; A3 delta=0.4 T1 >= 0.85, "
                               "T0h <= 0.15 (alpha=0.05, n_e=300, REF 3000, B=199, 100 reps)",
        passed=bool(rates["A1_v05_PC1"]["T0h"] >= 0.9 and rates["A2_v3_random"]["T2"] >= 0.9
                    and rates["A2_v3_random"]["T0h"] <= 0.15 and rates["A3_delta04_random"]["T1"] >= 0.85
                    and rates["A3_delta04_random"]["T0h"] <= 0.15))
    checks["calibration_A0"] = dict(rates=null, criterion="every test in [0.02, 0.09] (300 reps)",
                                    passed=all(0.02 <= v <= 0.09 for v in null.values()))
    checks["anisotropic_recorded"] = dict(
        rates={k: v["T0h"] for k, v in aniso.items()}, spectrum="geomspace(20, 1, 10)", gated=False, passed=True)
    # Selftest determinism across worker counts (smoke-sized Part A and Part B)
    h = {}
    for w in (1, workers):
        h[w] = (part_a(panel, "selftest", SMOKE_A_B, True, w, recomputed=rec)["results_sha256"],
                part_b(panel, "selftest", SMOKE_B_B, True, w, recomputed=rec)["results_sha256"])
    checks["determinism_selftest"] = dict(hashes={str(k): v for k, v in h.items()}, passed=h[1] == h[workers])
    import py_compile
    py_compile.compile(str(E.REPO / SCRIPT), doraise=True)
    files = {}
    for f in (SCRIPT, "notes/e13_prereg.md"):
        text = (E.REPO / f).read_text(encoding="utf8")
        bad = [c for c in text if ord(c) in (0x2012, 0x2013, 0x2014, 0x2015, 0x2E3A, 0x2E3B)]
        files[f] = dict(sha256=T2.sha256_file(E.REPO / f), lines=text.count("\n"), dash_count=len(bad))
    files[SCRIPT].update(py_compile="ok", grep_back="def updown_test" in (E.REPO / SCRIPT).read_text())
    checks["files"] = dict(files=files, passed=all(v["dash_count"] == 0 for v in files.values())
                           and files[SCRIPT]["grep_back"])
    results = dict(experiment="E13 self-checks", alpha=0.05, checks=checks,
                   all_pass=all(v["passed"] for v in checks.values()))
    results = json.loads(json.dumps(results))
    meta = dict(seconds=round(time.time() - t0, 1), provenance=T2.provenance(
        __file__, ["experiments/e12_common.py", "experiments/e12_standard_tests.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ main
def out_path(dataset, part, smoke):
    return (OUT_ROOT / "smoke" if smoke else OUT_ROOT) / f"{dataset}_part{part}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--part", choices=("A", "B"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--oracles", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.oracles:
        doc = oracles(max(2, args.workers))
        T2.write_json(OUT_ROOT / "oracles.json", doc)
        for k, v in doc["results"]["checks"].items():
            print(k, json.dumps({kk: vv for kk, vv in v.items() if kk in ("passed", "rates", "max_abs_diff",
                                                                         "cov_max_rel_err", "mean_max_rel_err")}))
        print(f"all_pass={doc['results']['all_pass']} seconds={doc['meta']['seconds']}")
        if not doc["results"]["all_pass"]:
            raise SystemExit("STOP: a self-check failed")
        return
    if args.aggregate:
        t0 = time.time()
        results = aggregate()
        doc = dict(results=results, results_sha256=E.sha256_json(results), meta=dict(
            seconds=round(time.time() - t0, 3), provenance=T2.provenance(__file__)))
        T2.write_json(OUT_ROOT / "summary.json", doc)
        print(json.dumps({k: results[k] for k in ("R_global", "S_global", "lambda_eff_condition",
                                                  "interpretation", "predictions", "uncalibrated",
                                                  "missing_inputs")}, indent=1))
        return
    if args.selftest:
        panel = E.make_selftest_panel_full("k562", seed=0)
        rec = recompute(panel)
        print("step0c", json.dumps(rec[4]))
        print("step0d", json.dumps(verify_0d(rec[0], rec[1], stored=None)))
        a = part_a(panel, "selftest", SMOKE_A_B, True, args.workers, recomputed=rec)
        b = part_b(panel, "selftest", SMOKE_B_B, True, args.workers, recomputed=rec)
        print("partA", a["results_sha256"], json.dumps([(c["arm"], c["level"], c["q"], c["s"], c["power"]["T2"])
                                                       for c in a["results"]["cells"]]))
        print("partB", b["results_sha256"], len(b["results"]["environments"]), "envs")
        return
    if args.dataset is None:
        ap.error("--dataset is required (or --aggregate / --selftest / --oracles)")
    panel = E.load_panel_full(args.dataset)
    rec = recompute(panel)
    print(f"[{args.dataset}] step0c {json.dumps(rec[4])}", flush=True)
    if args.verify:
        out = verify_0d(rec[0], rec[1], stored=load_stored(args.dataset))
        print(f"[{args.dataset}] step0d {json.dumps(out)}")
        print(f"[{args.dataset}] VERIFY PASS")
        return
    if args.part is None:
        ap.error("--part is required")
    if args.part == "A":
        doc = part_a(panel, args.dataset, SMOKE_A_B if args.smoke else B_A, args.smoke, args.workers, recomputed=rec)
    else:
        doc = part_b(panel, args.dataset, SMOKE_B_B if args.smoke else B_B, args.smoke, args.workers,
                     stored=load_stored(args.dataset), recomputed=rec)
    path = out_path(args.dataset, args.part, args.smoke)
    T2.write_json(path, doc)
    print(f"[{args.dataset}] part {args.part} smoke={args.smoke} -> {path}")
    print(f"results_sha256={doc['results_sha256']} seconds={doc['meta']['seconds']}")


if __name__ == "__main__":
    main()
