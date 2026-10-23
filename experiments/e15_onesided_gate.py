"""E15: a direction-independent one-sided gate, G10 = readout "geneig_down"
(-log lambda_min of the generalised eigenvalues of (S_env, S_ref)).

Pre-registration: notes/e15_prereg.md, committed before this script.

Modes (reuse, do not reimplement):
  --synthetic   E14 design: the 900 E10 units; G10 through E14._call (rng 910000 + seed,
                B = 500) once per (arm, level, seed), with the precision regression
                against E10's stored p-values (any mismatch voids the run). G0, G2, G5,
                G8, G9 are read from results/e14/units.json. G10 decisions through
                E14.decisions with E14.NEW_GATES extended ("exactly as G8"); curves and
                summary through E14.curves_for and E14.summary_of with E14.GATES extended;
                the bootstrap is E14.bootstrap's loop with G10 in every comparison
                (same rng default_rng(120000), same draws).
                -> results/e15/synthetic/{units,curves,predictions_partial}.json
  --real        E12 held-out construction (e12_standard_tests.build_tasks, B = 9999);
                G10 against the full corrected REF as e12_common.run_environment builds
                Rh, rng_for(seed, i, K_NULL) (T0h's stream). Stored T0h p from E12.
                -> results/e15/real/<ds>_seed<s>.json
  --planted     E13 Part A design via E13.recompute, whitening, directions, plant and
                replicate_a (streams k = 1, 2, 3, index 1000000 + 10000 s + 1000 q + r);
                only the test is replaced (E13.run_tests -> G10 against the full REF,
                B = 999, stream k = 10). Arms A0 to A3 (A4 not run). K562 and RPE1:
                E13.TOL = 1e-6 (E13 amendment 1). -> results/e15/planted/<ds>.json
  --aggregate   results/e15/summary.json: P1 to P6 and the tables.
  --selftest    synthetic panels for the real and planted paths, the readout identity
                and invariance checks, a stored-value regression on two E14 systems;
                writes nothing.
  --smoke       short real (k562 seed 0, 10 + 10 envs, B = 199), planted (k562, 2 cells
                x 5 reps, B = 99) and synthetic (2 systems, B = 500, with regression)
                runs, timed; writes under results/e15/smoke/.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402
import e12_standard_tests as S  # noqa: E402
import e13_power_sign as E13  # noqa: E402
import e14_matched_coverage as E14  # noqa: E402

T2 = E.T2
PR = E.PR
E10 = E14.E10
REPO = E.REPO
OUT = REPO / "results" / "e15"
READOUT = "geneig_down"
DATASETS = ("k562", "norman", "rpe1")
SEEDS = (0, 1, 2)
REBUILT = ("k562", "rpe1")
B_REAL, B_PLANT, B_SYN = 9999, E13.B_A, E10.B_NULL
K_G10 = E13.K_TEST["T0h"]  # 10: T0h's stream in E13, paired
GATES6 = ("G0", "G2", "G5", "G8", "G9", "G10")
WORKERS = 16
SOURCES = ["experiments/e15_onesided_gate.py", "experiments/e12_common.py",
           "experiments/e12_standard_tests.py", "experiments/e13_power_sign.py",
           "experiments/e14_matched_coverage.py", "experiments/e10_risk_coverage.py"]

warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)


def document(results, t0, extra=None):
    results = json.loads(json.dumps(results))
    meta = dict(seconds=round(time.time() - t0, 3), argv=sys.argv[1:],
                provenance=T2.provenance(__file__, SOURCES))
    meta.update(extra or {})
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


def pool_map(fn, items, workers, initializer=None, initargs=()):
    if workers <= 1:
        if initializer:
            initializer(*initargs)
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        return list(ex.map(fn, items, chunksize=1))


# ------------------------------------------------------------------ A. synthetic
def syn_system(job):
    arm, lvl, seed, B, regression = job
    Y_obs, Y_int, nodes = E14.projected(arm, lvl, seed)
    key = dict(arm=E10.ARM_KEY[arm], level=lvl, seed=seed)
    recs = [dict(key, node=int(n)) for n in nodes]
    failures = []
    try:
        g = E14._call(Y_int, Y_obs, seed, B, READOUT)
        for k, r in enumerate(recs):
            sig, thr = g["signals"][k], g["thresholds"][k]
            if not (np.isfinite(sig) and np.isfinite(thr)):
                failures.append(dict(r, readout=READOUT, reason=(
                    "non-finite signal" if not np.isfinite(sig) else "non-finite threshold (NaN null draw)")))
                r["geneig_down_signal"] = r["geneig_down_pvalue"] = None
            else:
                r["geneig_down_signal"] = float(sig)
                r["geneig_down_pvalue"] = float(g["pvalues"][k])
    except Exception as exc:  # recorded per unit, as E14
        failures += [dict(r, readout=READOUT, reason=f"{type(exc).__name__}: {exc}") for r in recs]
        for r in recs:
            r["geneig_down_signal"] = r["geneig_down_pvalue"] = None
    prec = None
    if regression:
        prec = [float(p) for p in E14._call(Y_int, Y_obs, seed, B, "precision")["pvalues"]]
    return recs, failures, prec


def _install_gates():
    """Thin wrapper: E14's module-level gate lists, extended by G10 (e14 file unchanged)."""
    E14.NEW_GATES = dict(E14.NEW_GATES, G10=READOUT)
    E14.GATES = GATES6


def summary6(cv):
    sm = E14.summary_of(cv)
    for scope in E14.SCOPES:
        p = E10._knob_point(cv["G10"][scope]["points"], E10.Q_DEFAULT)
        sm[scope]["G10"].update(coverage_q05=p["coverage"], risk_q05=p["risk"])
    return sm


def bootstrap6(dec, good, masks, fams, n_draws=E14.N_SYS_BOOT):
    """E14.bootstrap's loop (same rng, same draws), G10 added to every comparison."""
    rng = np.random.default_rng(E14.BOOT_SEED)
    fams = [np.asarray(f) for f in fams]
    keys = ([f"{g}-G2_matched" for g in ("G8", "G9", "G10")] + [f"{g}-G2_aurc" for g in ("G8", "G9", "G10")]
            + [f"G0-{g}_aurc" for g in ("G2", "G5", "G8", "G9", "G10")])
    draws = {s: {g: dict(aurc=[], matched=[]) for g in GATES6} for s in E14.SCOPES}
    diffs = {s: {k: [] for k in keys} for s in E14.SCOPES}
    for _ in range(n_draws):
        pick = rng.integers(0, len(fams), len(fams))
        idx = np.concatenate([fams[k] for k in pick])
        cv = E14.curves_for(dec, good, masks, idx)
        sm = E14.summary_of(cv)
        for s in E14.SCOPES:
            for g in GATES6:
                draws[s][g]["aurc"].append(sm[s][g]["aurc"])
                draws[s][g]["matched"].append(sm[s][g]["matched_risk"])
            for g in ("G8", "G9", "G10"):
                diffs[s][f"{g}-G2_matched"].append(E14._diff(sm[s][g]["matched_risk"], sm[s]["G2"]["matched_risk"]))
                diffs[s][f"{g}-G2_aurc"].append(E14._diff(sm[s][g]["aurc"], sm[s]["G2"]["aurc"]))
            for g in ("G2", "G5", "G8", "G9", "G10"):
                diffs[s][f"G0-{g}_aurc"].append(E14._diff(sm[s]["G0"]["aurc"], sm[s][g]["aurc"]))
    ci = {s: {g: dict(aurc=E14._pct(draws[s][g]["aurc"]), matched_risk=E14._pct(draws[s][g]["matched"]))
              for g in GATES6} for s in E14.SCOPES}
    ci_diff = {s: {k: E14._pct(v) for k, v in diffs[s].items()} for s in E14.SCOPES}
    return ci, ci_diff


def p5(sm):
    A = lambda g, s: sm[s][g]["aurc"]  # noqa: E731
    M = lambda g, s: sm[s][g]["matched_risk"]  # noqa: E731
    parts = {}
    parts["ABD_aurc"] = (None if A("G10", "ABD") is None or A("G2", "ABD") is None
                         else A("G10", "ABD") <= A("G2", "ABD") + 0.02)
    parts["ABD_matched"] = (None if M("G10", "ABD") is None or M("G2", "ABD") is None
                            else M("G10", "ABD") <= M("G2", "ABD") + 0.02)
    parts["C_matched"] = (None if M("G10", "C") is None or M("G2", "C") is None
                          else abs(M("G10", "C") - M("G2", "C")) <= 0.02)
    vals = list(parts.values())
    verdict = "failed" if False in vals else ("undetermined" if None in vals else "held")
    return dict(verdict=verdict, parts=parts,
                inputs=dict(aurc_G10_ABD=A("G10", "ABD"), aurc_G2_ABD=A("G2", "ABD"),
                            matched_G10_ABD=M("G10", "ABD"), matched_G2_ABD=M("G2", "ABD"),
                            matched_G10_C=M("G10", "C"), matched_G2_C=M("G2", "C"),
                            target_coverage_ABD=sm["ABD"]["target_coverage"],
                            target_coverage_C=sm["C"]["target_coverage"]))


def synthetic(workers, smoke=False):
    t0 = time.time()
    e14 = json.loads((REPO / "results" / "e14" / "units.json").read_text())
    if e14.get("status") != "complete":
        raise SystemExit("STOP: results/e14/units.json is not complete")
    e14_index = {}
    for u in e14["units"]:
        e14_index.setdefault(E14._key(u), []).append(u)
    _e10_doc, e10_index = E14.load_e10()
    if smoke:
        systems = list(E14.systems([0], 1))[:2]
    else:
        systems = list(E14.systems(E10.E2.SEEDS, None))
    jobs = [(arm, lvl, seed, B_SYN, True) for arm, lvl, seed in systems]
    out = pool_map(syn_system, jobs, workers)
    units, failures, reg = [], [], []
    for recs, fails, prec in out:
        failures += fails
        for r, p in zip(recs, prec):
            k = E14._key(r)
            h14, h10 = e14_index.get(k, []), e10_index.get(k, [])
            if len(h14) != 1 or len(h10) != 1:
                raise SystemExit(f"STOP: join {k}: {len(h14)} E14 and {len(h10)} E10 matches (need 1 each)")
            reg.append(dict(key=list(k), e10=h10[0]["pvalue"], e15=p, equal=bool(p == h10[0]["pvalue"])))
            units.append(dict(h14[0], geneig_down_signal=r["geneig_down_signal"],
                              geneig_down_pvalue=r["geneig_down_pvalue"]))
    bad = [r for r in reg if not r["equal"]]
    regression = dict(n_units=len(reg), n_equal=len(reg) - len(bad), identical=not bad,
                      max_abs_diff=max(abs(r["e10"] - r["e15"]) for r in reg))
    root = OUT / ("smoke" if smoke else "") / "synthetic"
    if bad:
        T2.write_json(root / "failed_units.json", dict(failed=failures, regression_mismatch=bad[:20]))
        raise SystemExit(f"STOP: precision p-values differ from E10 for {len(bad)} units; run void")
    keys = [E14._key(u) for u in units]
    if not smoke and (len(units) != 900 or len(set(keys)) != 900):
        raise SystemExit(f"STOP: join is not one-to-one over 900 units ({len(units)}, {len(set(keys))})")
    T2.write_json(root / "units.json", document(dict(
        experiment="E15", arm="A", readout=READOUT, B=B_SYN, null_rng="default_rng(910_000 + seed)",
        precision_regression_vs_e10=regression, failed_units=failures, n_units=len(units), units=units), t0))
    print(f"[synthetic] {len(units)} units, failed {len(failures)}, regression {regression}, "
          f"{time.time() - t0:.1f}s", flush=True)
    if smoke:
        return
    if failures:
        raise SystemExit(f"STOP: {len(failures)} units lack a geneig_down p-value; no dropped units allowed")
    _install_gates()
    good = np.array([u["good"] for u in units])
    dec = E14.decisions(units)
    masks = E14._masks(units)
    cv = E14.curves_for(dec, good, masks)
    sm = summary6(cv)
    e14c = json.loads((REPO / "results" / "e14" / "curves.json").read_text())["summary"]
    consistency = max(abs((sm[s][g][k] or 0) - (e14c[s][g][k] or 0)) for s in ("ABD", "C")
                      for g in ("G0", "G2", "G5", "G8", "G9") for k in ("aurc", "matched_risk"))
    ci, ci_diff = bootstrap6(dec, good, masks, E10._families(units))
    T2.write_json(root / "curves.json", document(dict(
        experiment="E15", arm="A", gates=GATES6, scopes=E14.SCOPES, summary=sm, ci=ci, ci_diff=ci_diff,
        e14_gates_max_abs_diff_vs_results_e14=consistency, curves=E14._jsonable(cv)), t0))
    T2.write_json(root / "predictions_partial.json", document(dict(P5=p5(sm)), t0))
    print(f"[synthetic] P5 {p5(sm)['verdict']} e14 consistency {consistency} {time.time() - t0:.1f}s", flush=True)


# ------------------------------------------------------------------ B. real
def real_env(task):
    """G10 for one environment against the full corrected REF, as run_environment builds Rh."""
    seed, i, B = task["seed"], task["index"], task["B"]
    Yh = task["Y_heldout"]
    Rh = E._SHARED["ref_heldout"]
    j = task["target_column"]
    if j is not None:
        Rh = E.correct(Rh, E._SHARED["ref_target_columns"][j], task["mu_h_j"], task["P_h_j"])
    n_e, d = Yh.shape
    r = PR.detect_with_pvalues([Yh], Rh, alpha=E.ALPHA, B=B, rng=E.rng_for(seed, i, E.K_NULL),
                               q=E.Q, readout=READOUT)
    return dict(label=task["label"], index=i, is_fake=task["is_fake"], n_e=int(n_e),
                target_corrected=j is not None, stat=float(r["signals"][0]), p=float(r["pvalues"][0]),
                finite=bool(np.isfinite(r["signals"][0]) and np.isfinite(r["thresholds"][0])),
                disjoint_applied=bool(PR._use_disjoint_null(len(Rh), n_e, d, True)))


def real(ds, seed, workers, smoke=False):
    t0 = time.time()
    panel = E.load_panel_full(ds)
    B = 199 if smoke else B_REAL
    tasks, _records, shared, design = S.build_tasks(panel, seed, B, 10, smoke)
    stored = json.loads((REPO / "results" / "e12_standard_tests" / ds / f"seed{seed}_d10.json").read_text())["results"]
    if design["excluded"] != stored["excluded"] or design["fake_n"] != stored["fake_n"]:
        raise SystemExit(f"STOP: {ds} seed {seed}: build_tasks design differs from stored E12")
    envs = pool_map(real_env, tasks, workers, E.init_worker, (shared,))
    by_label = {e["label"]: e for e in stored["environments"]}
    for e in envs:
        e["T0h_p_e12"] = by_label[e["label"]]["T0h"]["p"]
    nonfinite = [e["label"] for e in envs if not e["finite"]]
    if not all(e["disjoint_applied"] for e in envs):
        raise SystemExit(f"STOP: {ds} seed {seed}: G10 null fell back to the bootstrap path")
    reals = [e for e in envs if not e["is_fake"]]
    fakes = [e for e in envs if e["is_fake"]]
    n = len(reals)
    g10 = PR.bh_fdr([e["p"] for e in reals + fakes], q=E.Q)
    t0h = PR.bh_fdr([e["T0h_p_e12"] for e in reals + fakes], q=E.Q)
    g10_set = [e["label"] for e, k in zip(reals, g10[:n]) if k]
    t0h_set = [e["label"] for e, k in zip(reals, t0h[:n]) if k]
    results = dict(
        experiment="E15", arm="B", dataset=ds, seed=seed, B=B, smoke=bool(smoke), readout=READOUT,
        rebuilt_file=ds in REBUILT, n_real=n, n_fake=len(fakes), fake_n=design["fake_n"],
        excluded=design["excluded"], nonfinite=nonfinite,
        G10=dict(bh_count_real=int(g10[:n].sum()), bh_count_fake=int(g10[n:].sum()), selected=g10_set,
                 raw_reject_real=float(np.mean([e["p"] <= E.ALPHA for e in reals])),
                 raw_reject_fake=float(np.mean([e["p"] <= E.ALPHA for e in fakes]))),
        T0h=dict(bh_count_real=int(t0h[:n].sum()), bh_count_fake=int(t0h[n:].sum()), selected=t0h_set,
                 source="stored E12 T0h p"),
        overlap=dict(n_both=len(set(g10_set) & set(t0h_set)), n_G10_only=len(set(g10_set) - set(t0h_set)),
                     n_T0h_only=len(set(t0h_set) - set(g10_set))),
        environments=envs)
    doc = document(results, t0, dict(workers=workers))
    path = OUT / ("smoke" if smoke else "") / "real" / f"{ds}_seed{seed}.json"
    T2.write_json(path, doc)
    print(f"[real {ds} seed {seed}] G10 {results['G10']['bh_count_real']}/{n} fake "
          f"{results['G10']['bh_count_fake']}; T0h {results['T0h']['bh_count_real']} fake "
          f"{results['T0h']['bh_count_fake']}; overlap {results['overlap']} {time.time() - t0:.1f}s -> {path}",
          flush=True)


# ------------------------------------------------------------------ C. planted
def g10_run_tests(Y, ref, rows, B, seed, i, tests=None):
    """Replaces E13.run_tests: G10 only, against the full REF, stream k = 10 (paired with T0h)."""
    r = PR.detect_with_pvalues([Y], ref, alpha=E.ALPHA, B=B, rng=E.rng_for(seed, i, K_G10),
                               q=E.Q, readout=READOUT)
    return {"G10": float(r["pvalues"][0])}


def init_g10(shared):
    E13.run_tests = g10_run_tests
    E13.init_a(shared)


def planted_tasks(sizes, B, smoke):
    if smoke:
        cells = [("A1", 0.5, 0, 1), ("A0", None, E13.Q_NULL, 1)]
        return [dict(s=s, q=q, r=r, n_e=sizes[s], B=B, levels=[(arm, level)])
                for arm, level, q, s in cells for r in range(E13.SMOKE_A_REPS)]
    levels = [(arm, level) for arm, levs in E13.ARMS.items() if arm != "A4" for _, level in levs]
    tasks = []
    for s in (0, 1):
        tasks += [dict(s=s, q=E13.Q_NULL, r=r, n_e=sizes[s], B=B, levels=[("A0", None)])
                  for r in range(E13.REPS_NULL)]
        for q in E13.DIRECTIONS:
            tasks += [dict(s=s, q=q, r=r, n_e=sizes[s], B=B, levels=levels) for r in range(E13.REPS)]
    return tasks


def planted(ds, workers, smoke=False):
    t0 = time.time()
    if ds in REBUILT:
        E13.TOL = 1e-6  # E13 amendment 1
    panel = E.load_panel_full(ds)
    _tasks, shared, design, pool, check0c = E13.recompute(panel)
    ref = shared["ref_heldout"]
    lam, V, W, Winv = E13.whitening(ref)
    dirs = E13.directions(V, E13.SEED)
    sizes = {0: 200, 1: int(design["fake_n"])}
    for n in sizes.values():
        assert n <= len(pool) and PR._use_disjoint_null(len(ref), n, E13.D, True), n
    st = json.loads((REPO / "results" / "e13_power_sign" / f"{ds}_partA.json").read_text())["results"]
    repro = dict(sizes_equal=st["sizes"] == {str(k): v for k, v in sizes.items()},
                 ref_eigenvalues_max_abs_diff=float(np.max(np.abs(np.array(st["ref_eigenvalues"]) - lam))),
                 directions_max_abs_diff=float(np.max(np.abs(np.array(st["direction_vectors_whitened"]) - dirs))),
                 tol=E13.TOL)
    if not repro["sizes_equal"] or max(repro["ref_eigenvalues_max_abs_diff"], repro["directions_max_abs_diff"]) > E13.TOL:
        raise SystemExit(f"STOP: {ds}: E13 Part A setup does not reproduce the stored file: {repro}")
    B = 99 if smoke else B_PLANT
    shared_a = dict(seed=E13.SEED, pool=pool, ref=ref, W=W, Winv=Winv, m=pool.mean(0), dirs=dirs)
    reps = [rec for out in pool_map(E13.replicate_a, planted_tasks(sizes, B, smoke), workers,
                                    init_g10, (shared_a,)) for rec in out]
    cells = E13.summarize_cells(reps)
    results = dict(
        experiment="E15", arm="C", dataset=ds, seed=E13.SEED, B=B, smoke=bool(smoke), readout=READOUT,
        rebuilt_file=ds in REBUILT, sizes={str(k): v for k, v in sizes.items()}, n_ref=int(len(ref)),
        n_pool=int(len(pool)), directions=E13.DIRECTIONS, alphas=list(E13.ALPHAS), step0c=check0c,
        reproduction_vs_e13=repro,
        rng="E13 replicate_a: i = 1000000 + 10000*s + 1000*q + r; k 1 cells, 2 REF rows, 3 labels; "
            "G10 null stream k = 10 (T0h's); A0 q=4", arms="A0, A1, A2, A3 (A4 not run)", cells=cells)
    doc = document(results, t0, dict(workers=workers))
    path = OUT / ("smoke" if smoke else "") / "planted" / f"{ds}.json"
    T2.write_json(path, doc)
    print(f"[planted {ds}] {len(cells)} cells, repro {repro}, {time.time() - t0:.1f}s -> {path}", flush=True)


# ------------------------------------------------------------------ aggregate
def _pw(cells, arm, level, q, s, test, alpha):
    for c in cells:
        if c["arm"] == arm and c["level"] == level and c["q"] == q and c["s"] == s:
            return c["power"][test][str(alpha)]
    return None


def aggregate():
    t0 = time.time()
    missing = []

    def load(p):
        if not p.exists():
            missing.append(str(p.relative_to(REPO)))
            return None
        return json.loads(p.read_text())

    syn_c = load(OUT / "synthetic" / "curves.json")
    syn_p = load(OUT / "synthetic" / "predictions_partial.json")
    syn_table = None
    if syn_c:
        r = syn_c["results"]
        syn_table = {s: dict(target_coverage=r["summary"][s]["target_coverage"], gates={
            g: dict(aurc=r["summary"][s][g]["aurc"], aurc_ci=[r["ci"][s][g]["aurc"]["lo"], r["ci"][s][g]["aurc"]["hi"]],
                    matched_risk=r["summary"][s][g]["matched_risk"],
                    matched_ci=[r["ci"][s][g]["matched_risk"]["lo"], r["ci"][s][g]["matched_risk"]["hi"]],
                    coverage_q05=r["summary"][s][g].get("coverage_q05"))
            for g in GATES6}, G10_minus_G2=dict(
                matched=r["ci_diff"][s]["G10-G2_matched"], aurc=r["ci_diff"][s]["G10-G2_aurc"]))
            for s in ("ABD", "C")}

    real_table, p4_fakes, p6 = {}, [], []
    for ds in DATASETS:
        real_table[ds] = {}
        for s in SEEDS:
            d = load(OUT / "real" / f"{ds}_seed{s}.json")
            if d is None:
                p4_fakes.append(None)
                if ds == "k562":
                    p6.append(None)
                continue
            r = d["results"]
            real_table[ds][str(s)] = dict(G10_real=r["G10"]["bh_count_real"], G10_fake=r["G10"]["bh_count_fake"],
                                          T0h_real=r["T0h"]["bh_count_real"], T0h_fake=r["T0h"]["bh_count_fake"],
                                          n_real=r["n_real"], overlap=r["overlap"], nonfinite=len(r["nonfinite"]))
            p4_fakes.append(r["G10"]["bh_count_fake"] <= 3)
            if ds == "k562":
                p6.append(r["G10"]["bh_count_real"] <= 20)

    planted_table, calib, p1, p2, p3, p4_a0 = {}, {}, {}, {}, {}, []
    for ds in DATASETS:
        d = load(OUT / "planted" / f"{ds}.json")
        e13 = load(REPO / "results" / "e13_power_sign" / f"{ds}_partA.json")
        if d is None or e13 is None:
            p1[ds] = p2[ds] = p3[ds] = None
            p4_a0.append(None)
            continue
        cg, ce = d["results"]["cells"], e13["results"]["cells"]
        planted_table[ds] = {f"v={v}": {
            t: [_pw(cg if t == "G10" else ce, "A1", f"v={v}", q, 1, t, 0.01) for q in E13.DIRECTIONS]
            for t in ("G10", "T0h", "T2")} for v in (0.3, 0.5, 0.7)}
        a0 = {str(c["s"]): c["power"]["G10"]["0.05"] for c in cg if c["arm"] == "A0"}
        calib[ds] = dict(A0_rate_alpha05=a0, in_band={k: 0.02 <= v <= 0.09 for k, v in a0.items()},
                         conservative={k: v < 0.02 for k, v in a0.items()})
        p4_a0 += list(calib[ds]["in_band"].values())
        g03 = planted_table[ds]["v=0.3"]["G10"]
        p1[ds] = None if None in g03 else all(v >= 0.8 for v in g03)
        g05, t05 = planted_table[ds]["v=0.5"]["G10"], planted_table[ds]["v=0.5"]["T0h"]
        p2[ds] = None if None in g05 + t05 else float(np.mean(g05)) < float(np.mean(t05))
        a23 = [c["power"]["G10"]["0.05"] for c in cg if c["arm"] in ("A2", "A3")]
        p3[ds] = dict(mean_G10_A2_A3_alpha05=float(np.mean(a23)) if a23 else None,
                      held=None if not a23 else float(np.mean(a23)) <= 0.08)
        planted_table[ds]["mean_over_directions_v05"] = dict(G10=float(np.mean(g05)), T0h=float(np.mean(t05)))

    def v_all(vals):
        return "failed" if False in vals else ("undetermined" if None in vals or not vals else "held")

    p3_vals = [None if p3[ds] is None else p3[ds]["held"] for ds in DATASETS]
    p4_vals = p4_fakes + p4_a0
    preds = dict(
        P1=dict(verdict=v_all([p1[ds] for ds in DATASETS]), per_dataset=p1),
        P2=dict(verdict=v_all([p2[ds] for ds in DATASETS]), per_dataset=p2),
        P3=dict(verdict=v_all(p3_vals), per_dataset=p3),
        P4=dict(verdict=v_all(p4_vals), fakes_part=v_all(p4_fakes), A0_part=v_all(p4_a0),
                A0_outside_band_only_conservative=all(
                    calib[ds]["conservative"][k] for ds in calib for k, ok in calib[ds]["in_band"].items() if not ok)
                if calib else None),
        P5=(syn_p["results"]["P5"] if syn_p else dict(verdict="undetermined", note="synthetic missing")),
        P6=dict(verdict=v_all(p6), k562_G10_counts=[real_table["k562"].get(str(s), {}).get("G10_real")
                                                    for s in SEEDS]))
    results = dict(experiment="E15", prereg="notes/e15_prereg.md", synthetic=syn_table, real=real_table,
                   planted=planted_table, A0_calibration=calib, predictions=preds, missing_inputs=missing,
                   p3_stop_rule="P3 fails -> STOP and report before interpreting anything else")
    T2.write_json(OUT / "summary.json", document(results, t0))
    print(json.dumps(dict(predictions={k: v["verdict"] for k, v in preds.items()}, missing=missing), indent=1))
    if preds["P3"]["verdict"] == "failed":
        raise SystemExit("STOP: P3 failed (G10 responds to mean shifts or increases); report before interpreting")


# ------------------------------------------------------------------ selftest
def selftest():
    rng = np.random.default_rng(15_000)
    Ye, Yo = rng.standard_normal((300, 10)) * 0.8, rng.standard_normal((900, 10))
    import scipy.linalg
    lam = scipy.linalg.eigh(np.cov(Ye, rowvar=False), np.cov(Yo, rowvar=False), eigvals_only=True)
    direct = float(-np.log(lam.min()))
    d0 = abs(PR.geneig_down_signal(Ye, Yo) - direct)
    A = rng.standard_normal((10, 10)) + 3 * np.eye(10)
    d1 = abs(PR.geneig_down_signal(Ye @ A, Yo @ A) - PR.geneig_down_signal(Ye, Yo))
    print(f"[selftest] geneig_down vs direct -log(min eig): diff {d0}; invariance under a random "
          f"invertible map (cond {np.linalg.cond(A):.1f}): diff {d1:.3e}", flush=True)
    if d0 != 0.0 or d1 >= 1e-8:
        raise SystemExit("STOP: readout identity or invariance check failed")
    assert "geneig_down" in PR.READOUTS and PR.readout_signal(Ye, Yo, "geneig_down") == PR.geneig_down_signal(Ye, Yo)

    # regression on two E14 systems at B = 500: precision vs E10, geneig vs E14 (stored)
    e14 = {E14._key(u): u for u in json.loads((REPO / "results" / "e14" / "units.json").read_text())["units"]}
    worst_p = worst_g = 0.0
    for arm, lvl, seed in list(E14.systems([0], 1))[:2]:
        Y_obs, Y_int, nodes = E14.projected(arm, lvl, seed)
        pp = E14._call(Y_int, Y_obs, seed, B_SYN, "precision")["pvalues"]
        pg = E14._call(Y_int, Y_obs, seed, B_SYN, "geneig")["pvalues"]
        for n, a, b in zip(nodes, pp, pg):
            u = e14[(E10.ARM_KEY[arm], lvl, seed, int(n))]
            worst_p = max(worst_p, abs(a - u["pvalue"]))
            worst_g = max(worst_g, abs(b - u["geneig_pvalue"]))
    print(f"[selftest] stored-value regression (2 systems, B=500): precision vs E10 max diff {worst_p}, "
          f"geneig vs E14 max diff {worst_g}", flush=True)
    if worst_p != 0.0 or worst_g != 0.0:
        raise SystemExit("STOP: precision or geneig readout changed")

    panel = E.make_selftest_panel_full("k562", seed=0)
    tasks, _r, shared, _d = S.build_tasks(panel, 0, 49, 10, False)
    E.init_worker(shared)
    recs = [real_env(t) for t in tasks[:4] + [t for t in tasks if t["is_fake"]][:2]]
    print(f"[selftest] real path: {len(recs)} envs, corrected {[r['target_corrected'] for r in recs]}, "
          f"p {[round(r['p'], 3) for r in recs]}, disjoint {all(r['disjoint_applied'] for r in recs)}", flush=True)
    rec = E13.recompute(panel)
    ref = rec[1]["ref_heldout"]
    lam, V, W, Winv = E13.whitening(ref)
    sh = dict(seed=E13.SEED, pool=rec[3], ref=ref, W=W, Winv=Winv, m=rec[3].mean(0), dirs=E13.directions(V))
    init_g10(sh)
    out = [E13.replicate_a(dict(s=0, q=q, r=r, n_e=60, B=20, levels=[("A1", 0.3), ("A2", 2.0), ("A3", 0.5)]))
           for q in (0, 3) for r in range(2)] + [E13.replicate_a(dict(s=0, q=E13.Q_NULL, r=0, n_e=60, B=20,
                                                                       levels=[("A0", None)]))]
    cells = E13.summarize_cells([x for o in out for x in o])
    print(f"[selftest] planted path: {len(cells)} cells, tests {sorted(cells[0]['power'])}", flush=True)
    print("[selftest] passed (nothing written)", flush=True)


def smoke(workers):
    t = {}
    for name, fn in (("real_k562_seed0", lambda: real("k562", 0, workers, smoke=True)),
                     ("planted_k562", lambda: planted("k562", workers, smoke=True)),
                     ("synthetic_2_systems", lambda: synthetic(workers, smoke=True))):
        t0 = time.time()
        fn()
        t[name] = round(time.time() - t0, 1)
    print(f"[smoke] timings {json.dumps(t)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("synthetic", "real", "planted", "aggregate", "selftest", "smoke"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.workers > 16:
        ap.error("at most 16 workers (thermal limit)")
    if a.selftest:
        selftest()
    elif a.smoke:
        smoke(a.workers)
    elif a.synthetic:
        synthetic(a.workers)
    elif a.real:
        if a.dataset is None or a.seed is None:
            ap.error("--real needs --dataset and --seed")
        real(a.dataset, a.seed, a.workers)
    elif a.planted:
        if a.dataset is None:
            ap.error("--planted needs --dataset")
        planted(a.dataset, a.workers)
    else:
        aggregate()


if __name__ == "__main__":
    main()
