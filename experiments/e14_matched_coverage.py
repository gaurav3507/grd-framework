"""E14: affine-invariant (geneig) and energy-distance gates at matched coverage.

Pre-registration: notes/e14_prereg.md, committed before this script was written or run.

Synthetic only: the 900 units of results/e10_risk_coverage/units.json (four E2 arms,
every level, seeds 0-9). Ground truth, coverage, risk, AURC, matched risk, scopes and
the q-grid are E10's; this script imports them from e10_risk_coverage.py.

Data pass (default): for every (arm, level, seed), projected() repeats E10.unit_records
lines 86 to 93 (E2 builder, control-fit PCA to D_PROJ, projection). Each new readout
("geneig", "energy") is called ONCE per (arm, level, seed) with the full environment
list, precision_readout.detect_with_pvalues(Y_int, Y_obs, alpha=0.05, B=500,
rng=default_rng(910_000 + seed), q=0.05, readout=...): E10's precision call with only
the readout changed, so the null index draws are identical across readouts. Regression:
the same call with readout="precision" must reproduce E10's stored p-values exactly;
any mismatch stops the run. Non-finite signals or thresholds (a NaN null draw), or an
exception, are recorded in results/e14/failed_units.json and the run continues.
results/e14/units.json is rewritten atomically after every (arm, level, seed) (status
"partial"), then finally with the one-to-one join onto the E10 units (status
"complete").

Analysis (--analyse, reads results/e14/units.json only): decisions computed once on
the full unit set: E10.gate_decisions (G0..G7) plus G8 (geneig) and G9 (energy), BH
within each E10 family over E10.GRID["G2"], exactly as G2. Gates G0, G2, G5, G8, G9 on
scopes ABD, A, B, D, C; matched risk at G2's q=0.05 coverage per scope. Bootstrap: 1000
resamples of the 200 (arm, level, seed) systems with replacement (default_rng(120_000)),
decision masks fixed, metrics and the matched-coverage target recomputed per draw.
Writes results/e14/curves.json and results/e14/predictions.json.

--selftest: seed 0, first two levels per arm, B=20, data pass and analysis in memory,
writes nothing; the precision regression is skipped (B differs from E10's 500).
--time-one: times one (arm A, level 5, seed 0) energy call at B=500.

Usage:
    python experiments/e14_matched_coverage.py --selftest
    python experiments/e14_matched_coverage.py              # data pass -> units.json
    python experiments/e14_matched_coverage.py --analyse    # curves.json, predictions.json
"""

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import tier2_common as T2  # noqa: E402

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

E10 = T2.load_module("grd_e10_for_e14", "experiments/e10_risk_coverage.py")
E2, PR, BK = E10.E2, E10.PR, E10.BK
OUT = REPO / "results" / "e14"
E10_UNITS = REPO / "results" / "e10_risk_coverage" / "units.json"

NEW_READOUTS = ("geneig", "energy")
NEW_GATES = {"G8": "geneig", "G9": "energy"}
GATES = ("G0", "G2", "G5", "G8", "G9")
SCOPES = ("ABD", "A", "B", "D", "C")
N_SYS_BOOT = 1000
BOOT_SEED = 120_000
COV_GRID = np.linspace(0, 1, 101)
SELFTEST_B = 20
CAVEAT_LEVELS = (20, 40)
CAVEAT = ("Every synthetic unit uses the historical two-bootstrap null (n_env equals n_obs in "
          "all arms), which resamples rows with replacement while the observed statistic uses "
          "distinct rows. This inflates every readout's null at small n, and precision most "
          "(reviewer check at n=20: 95% null quantile ratio two-bootstrap/independent is about "
          "1.99 precision, 1.13 geneig, 1.18 energy). Coverage at q=0.05 for G2, G8 and G9 is "
          "reported separately at arm B levels 20 and 40, and H3 is not over-read there.")


# ------------------------------------------------------------------ data pass
def projected(arm, lvl, seed):
    ds, int_envs = E10.E2.ARMS[arm]["builder"](seed, lvl)
    mu, Wp = E10.BK.fit_pca(ds.environments["basis"].X, E10.E2.D_PROJ)
    Y_obs = E10.BK.project(ds.environments["obs"].X, mu, Wp)
    Y_int = [E10.BK.project(X, mu, Wp) for (_, X) in int_envs]
    nodes = [i for (i, _) in int_envs]
    return Y_obs, Y_int, nodes


def _call(Y_int, Y_obs, seed, B, readout):
    return E10.PR.detect_with_pvalues(Y_int, Y_obs, alpha=E10.ALPHA, B=B,
                                      rng=np.random.default_rng(910_000 + seed),
                                      q=E10.Q_DEFAULT, readout=readout)


def system_records(arm, lvl, seed, B, regression):
    """E14 fields for every unit of one (arm, level, seed), failures, regression rows."""
    Y_obs, Y_int, nodes = projected(arm, lvl, seed)
    key = dict(arm=E10.ARM_KEY[arm], level=lvl, seed=seed)
    recs = [dict(key, node=int(n)) for n in nodes]
    failures = []
    for readout in NEW_READOUTS:
        try:
            g = _call(Y_int, Y_obs, seed, B, readout)
        except Exception as exc:  # recorded per unit; the run continues
            failures += [dict(r, readout=readout, reason=f"{type(exc).__name__}: {exc}") for r in recs]
            for r in recs:
                r[f"{readout}_signal"] = r[f"{readout}_pvalue"] = None
            continue
        for k, r in enumerate(recs):
            sig, thr = g["signals"][k], g["thresholds"][k]
            if not (np.isfinite(sig) and np.isfinite(thr)):
                failures.append(dict(r, readout=readout, reason=(
                    "non-finite signal" if not np.isfinite(sig) else "non-finite threshold (NaN null draw)")))
                r[f"{readout}_signal"] = r[f"{readout}_pvalue"] = None
            else:
                r[f"{readout}_signal"] = float(sig)
                r[f"{readout}_pvalue"] = float(g["pvalues"][k])
    precision_p = None
    if regression:
        precision_p = [float(p) for p in _call(Y_int, Y_obs, seed, B, "precision")["pvalues"]]
    return recs, failures, precision_p, Y_obs


def _key(u):
    return (u["arm"], u["level"], u["seed"], u["node"])


def load_e10():
    doc = json.loads(E10_UNITS.read_text())
    index = {}
    for u in doc["units"]:
        index.setdefault(_key(u), []).append(u)
    return doc, index


def join(recs, e10_index):
    """E10 unit (every field) plus the four E14 fields; exactly one E10 match per unit."""
    out = []
    for r in recs:
        hits = e10_index.get(_key(r), [])
        if len(hits) != 1:
            raise SystemExit(f"[stop] join: {_key(r)} matches {len(hits)} E10 units (need exactly 1)")
        out.append(dict(hits[0], **{k: r[k] for k in r if k not in ("arm", "level", "seed", "node")}))
    return out


def _versions():
    import matplotlib
    import scipy
    import sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, matplotlib=matplotlib.__version__)


def provenance(wall):
    return dict(
        code_commit=T2.code_commit(), versions=_versions(), wall_seconds=wall,
        e10_units_sha256=T2.sha256_file(E10_UNITS),
        sha256={f: T2.sha256_file(REPO / f) for f in (
            "src/gate/precision_readout.py", "experiments/e14_matched_coverage.py",
            "experiments/e10_risk_coverage.py", "experiments/e2_calibration.py",
            "src/recover/backbone.py")},
        precision_readout_note=("src/gate/precision_readout.py differs from the hash pinned in the "
                                "E10/E11 provenance because of the additive geneig and energy readouts "
                                "(E14 step 1); no existing branch, constant or draw order changed"))


def systems(seeds, levels_per_arm):
    for arm, spec in E2.ARMS.items():
        levels = spec["levels"][:levels_per_arm] if levels_per_arm else spec["levels"]
        for lvl in levels:
            for seed in seeds:
                yield arm, lvl, seed


def data_pass(seeds, levels_per_arm, B, regression, write):
    e10_doc, e10_index = load_e10()
    units, failures, reg_rows, first_obs = [], [], [], None
    started = time.time()
    config = dict(alpha=E10.ALPHA, B=B, q=E10.Q_DEFAULT, seeds=list(seeds), readouts=list(NEW_READOUTS),
                  null_rng="default_rng(910_000 + seed)", energy_max_rows=PR.ENERGY_MAX_ROWS,
                  call="once per (arm, level, seed) with the full environment list")
    for arm, lvl, seed in systems(seeds, levels_per_arm):
        recs, fails, prec_p, Y_obs = system_records(arm, lvl, seed, B, regression)
        if first_obs is None:
            first_obs = Y_obs
        joined = join(recs, e10_index)
        if regression:
            for u, p in zip(joined, prec_p):
                reg_rows.append(dict(key=list(_key(u)), e10=u["pvalue"], e14=p, equal=bool(p == u["pvalue"])))
            bad = [r for r in reg_rows[-len(joined):] if not r["equal"]]
            if bad:
                if write:
                    T2.write_json(OUT / "failed_units.json", dict(failed=failures, count=len(failures)))
                raise SystemExit(f"[stop] precision p-values differ from E10 for {bad[:3]}; run void")
        units += joined
        failures += fails
        if write:
            T2.write_json(OUT / "units.json", dict(
                experiment="e14_matched_coverage", prereg="notes/e14_prereg.md", status="partial",
                config=config, units=units, provenance=provenance(round(time.time() - started, 1))))
            T2.write_json(OUT / "failed_units.json", dict(failed=failures, count=len(failures)))
        print(f"  {E10.ARM_KEY[arm]} level={lvl} seed={seed} units={len(units)} "
              f"failed={len(failures)} ({time.time() - started:.0f}s)", flush=True)
    wall = round(time.time() - started, 1)
    regression_summary = None
    if regression:
        regression_summary = dict(n_units=len(reg_rows), n_equal=sum(r["equal"] for r in reg_rows),
                                  identical=all(r["equal"] for r in reg_rows),
                                  max_abs_diff=max(abs(r["e10"] - r["e14"]) for r in reg_rows))
    return dict(units=units, failures=failures, regression=regression_summary, wall=wall,
                config=config, first_obs=first_obs, e10_n=len(e10_doc["units"]))


# ------------------------------------------------------------------ analysis
def decisions(units):
    """E10.gate_decisions plus G8 and G9, built exactly as gate_decisions builds G2."""
    out = E10.gate_decisions(units)
    fams = E10._families(units)
    for gate, readout in NEW_GATES.items():
        p = np.array([u[f"{readout}_pvalue"] for u in units], dtype=float)
        ops = []
        for q in E10.GRID["G2"]:
            dec = np.zeros(len(units), dtype=bool)
            for idx in fams:
                dec[idx] = E10.PR.bh_fdr(p[idx], q=q)
            ops.append((q, dec))
        out[gate] = ops
    return out


def _masks(units):
    m = {s: E10._scope_mask(units, s) for s in SCOPES}
    arms = np.array([u["arm"] for u in units])
    levels = np.array([u["level"] for u in units])
    for lv in CAVEAT_LEVELS:
        m[f"B_level{lv}"] = (arms == "B") & (levels == lv)
    return m


def curves_for(dec, good, masks, index=None, scopes=SCOPES):
    out = {}
    for gate in GATES:
        out[gate] = {}
        for scope in scopes:
            mask = masks[scope]
            if index is None:
                pts = [dict(knob=k, **E10._metrics(d, good, mask)) for k, d in dec[gate]]
            else:
                pts = [dict(knob=k, **E10._metrics(d[index], good[index], mask[index])) for k, d in dec[gate]]
            out[gate][scope] = dict(points=pts, curve=E10._curve(pts))
    return out


def summary_of(cv, scopes=SCOPES):
    """AURC, matched risk at G2's q=0.05 coverage, and coverage/risk at q=0.05."""
    s = {}
    for scope in scopes:
        target = E10._knob_point(cv["G2"][scope]["points"], E10.Q_DEFAULT)["coverage"]
        row = dict(target_coverage=target)
        for gate in GATES:
            c = cv[gate][scope]["curve"]
            row[gate] = dict(aurc=E10._aurc(c)["value"], matched_risk=E10._matched(c, target))
            if gate in ("G2", "G8", "G9"):
                p = E10._knob_point(cv[gate][scope]["points"], E10.Q_DEFAULT)
                row[gate].update(coverage_q05=p["coverage"], risk_q05=p["risk"])
        s[scope] = row
    return s


def _band(curve):
    if not curve:
        return np.full(len(COV_GRID), np.nan)
    c = np.array([x for x, _ in curve])
    r = np.array([y for _, y in curve])
    out = np.interp(COV_GRID, c, r)
    out[(COV_GRID < c[0] - E10.TOL) | (COV_GRID > c[-1] + E10.TOL)] = np.nan
    return out


def _pct(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return dict(lo=None, hi=None, median=None, n_defined=0)
    lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
    return dict(lo=float(lo), hi=float(hi), median=float(med), n_defined=int(v.size))


def _diff(a, b):
    return None if a is None or b is None else a - b


def bootstrap(dec, good, masks, fams, n_draws=N_SYS_BOOT):
    rng = np.random.default_rng(BOOT_SEED)
    fams = [np.asarray(f) for f in fams]
    draws = {s: {g: dict(aurc=[], matched=[]) for g in GATES} for s in SCOPES}
    diffs = {s: {k: [] for k in ("G8-G2_matched", "G9-G2_matched", "G8-G2_aurc", "G9-G2_aurc",
                                  "G0-G2_aurc", "G0-G5_aurc", "G0-G8_aurc", "G0-G9_aurc")} for s in SCOPES}
    bands = {s: {g: [] for g in GATES} for s in SCOPES}
    for _ in range(n_draws):
        pick = rng.integers(0, len(fams), len(fams))
        idx = np.concatenate([fams[k] for k in pick])
        cv = curves_for(dec, good, masks, idx)
        sm = summary_of(cv)
        for s in SCOPES:
            for g in GATES:
                draws[s][g]["aurc"].append(sm[s][g]["aurc"])
                draws[s][g]["matched"].append(sm[s][g]["matched_risk"])
                bands[s][g].append(_band(cv[g][s]["curve"]))
            for g in ("G8", "G9"):
                diffs[s][f"{g}-G2_matched"].append(_diff(sm[s][g]["matched_risk"], sm[s]["G2"]["matched_risk"]))
                diffs[s][f"{g}-G2_aurc"].append(_diff(sm[s][g]["aurc"], sm[s]["G2"]["aurc"]))
            for g in ("G2", "G5", "G8", "G9"):
                diffs[s][f"G0-{g}_aurc"].append(_diff(sm[s]["G0"]["aurc"], sm[s][g]["aurc"]))
    ci = {s: {g: dict(aurc=_pct(draws[s][g]["aurc"]), matched_risk=_pct(draws[s][g]["matched"]))
              for g in GATES} for s in SCOPES}
    ci_diff = {s: {k: _pct(v) for k, v in diffs[s].items()} for s in SCOPES}
    band_out = {}
    for s in SCOPES:
        band_out[s] = {}
        for g in GATES:
            arr = np.vstack(bands[s][g])
            n_def = np.sum(np.isfinite(arr), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q = np.nanpercentile(arr, [2.5, 50, 97.5], axis=0)
            band_out[s][g] = dict(
                coverage=COV_GRID.tolist(), n_defined=n_def.tolist(),
                **{name: [None if not np.isfinite(x) else float(x) for x in q[j]]
                   for j, name in enumerate(("p2_5", "p50", "p97_5"))})
    return ci, ci_diff, band_out


def _flag(name, inputs, test, threshold, kind="held"):
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return {**inputs, "threshold": threshold, kind: None, "note": f"undefined input: {', '.join(missing)}"}
    return {**inputs, "threshold": threshold, kind: bool(test(**inputs))}


def predictions(sm, ci_diff):
    M = lambda g, s: sm[s][g]["matched_risk"]  # noqa: E731
    A = lambda g, s: sm[s][g]["aurc"]  # noqa: E731
    C = lambda g, s: sm[s][g]["coverage_q05"]  # noqa: E731
    out = {}
    out["H1"] = _flag("H1", dict(aurc_G8_ABD=A("G8", "ABD"), aurc_G2_ABD=A("G2", "ABD")),
                      lambda aurc_G8_ABD, aurc_G2_ABD: abs(aurc_G8_ABD - aurc_G2_ABD) <= 0.02,
                      "|AURC(G8) - AURC(G2)| <= 0.02 on ABD")
    out["H2"] = _flag("H2", dict(matched_G8_C=M("G8", "C"), matched_G2_C=M("G2", "C")),
                      lambda matched_G8_C, matched_G2_C: matched_G8_C < matched_G2_C - 0.02,
                      "matched risk(G8) < matched risk(G2) - 0.02 on C")
    h3_in = {f"{k}_{s}": v for s in ("ABD", "C") for k, v in (
        ("coverage_G9", C("G9", s)), ("coverage_G2", C("G2", s)),
        ("matched_G9", M("G9", s)), ("matched_G2", M("G2", s)))}
    out["H3"] = _flag("H3", h3_in, lambda **v: all(
        v[f"coverage_G9_{s}"] >= v[f"coverage_G2_{s}"] and v[f"matched_G9_{s}"] >= v[f"matched_G2_{s}"]
        for s in ("ABD", "C")), "coverage(G9, q=0.05) >= coverage(G2, q=0.05) and matched risk(G9) "
                                ">= matched risk(G2), on ABD and on C")
    out["H4"] = _flag("H4", dict(matched_G5_ABD=M("G5", "ABD"), matched_G2_ABD=M("G2", "ABD")),
                      lambda matched_G5_ABD, matched_G2_ABD: matched_G5_ABD <= matched_G2_ABD + 0.02,
                      "matched risk(G5) <= matched risk(G2) + 0.02 on ABD")
    out["H4"]["info_risk_G2_q05_ABD"] = sm["ABD"]["G2"]["risk_q05"]
    k1_in = {}
    for g in ("G2", "G5", "G8", "G9"):
        d = _diff(A("G0", "ABD"), A(g, "ABD"))
        c = ci_diff["ABD"][f"G0-{g}_aurc"]
        k1_in[f"G0-{g}_aurc"] = d
        k1_in[f"G0-{g}_halfwidth"] = None if c["lo"] is None else (c["hi"] - c["lo"]) / 2.0
    out["K1"] = _flag("K1", k1_in, lambda **v: all(
        v[f"G0-{g}_aurc"] <= v[f"G0-{g}_halfwidth"] for g in ("G2", "G5", "G8", "G9")),
        "on ABD, AURC(G0) - AURC(g) <= half-width of its 95% interval for every g in "
        "(G2, G5, G8, G9); true means no gate beats certify-all", kind="triggered")
    out["K2"] = _flag("K2", dict(matched_G9_ABD=M("G9", "ABD"), matched_G2_ABD=M("G2", "ABD"),
                                 matched_G9_C=M("G9", "C"), matched_G2_C=M("G2", "C")),
                      lambda matched_G9_ABD, matched_G2_ABD, matched_G9_C, matched_G2_C:
                      matched_G9_ABD <= matched_G2_ABD and matched_G9_C <= matched_G2_C,
                      "matched risk(G9) <= matched risk(G2) on both ABD and C", kind="triggered")
    c3 = ci_diff["C"]["G8-G2_matched"]
    out["K3"] = _flag("K3", dict(matched_G8_C=M("G8", "C"), matched_G2_C=M("G2", "C"),
                                 ci_G8_minus_G2_C_lo=c3["lo"], ci_G8_minus_G2_C_hi=c3["hi"],
                                 aurc_G8_ABD=A("G8", "ABD"), aurc_G2_ABD=A("G2", "ABD")),
                      lambda matched_G8_C, matched_G2_C, ci_G8_minus_G2_C_lo, ci_G8_minus_G2_C_hi,
                      aurc_G8_ABD, aurc_G2_ABD: (matched_G8_C < matched_G2_C - 0.02
                                                 and (ci_G8_minus_G2_C_hi < 0 or ci_G8_minus_G2_C_lo > 0)
                                                 and aurc_G8_ABD <= aurc_G2_ABD + 0.02),
                      "on C, matched risk(G8) < matched risk(G2) - 0.02 with the 95% interval of "
                      "(G8 - G2) excluding 0, and on ABD AURC(G8) <= AURC(G2) + 0.02", kind="triggered")
    return out


def analyse(units, n_draws=N_SYS_BOOT):
    missing = [_key(u) for u in units if u.get("geneig_pvalue") is None or u.get("energy_pvalue") is None]
    if missing:
        raise SystemExit(f"[stop] {len(missing)} units lack a geneig or energy p-value (see "
                         f"failed_units.json); the prereg allows no dropped units")
    good = np.array([u["good"] for u in units])
    dec = decisions(units)
    masks = _masks(units)
    cv = curves_for(dec, good, masks)
    sm = summary_of(cv)
    caveat_cv = curves_for(dec, good, masks, scopes=tuple(f"B_level{lv}" for lv in CAVEAT_LEVELS))
    caveat = dict(text=CAVEAT, coverage_q05={
        f"B_level{lv}": {g: E10._knob_point(caveat_cv[g][f"B_level{lv}"]["points"], E10.Q_DEFAULT)["coverage"]
                         for g in ("G2", "G8", "G9")} for lv in CAVEAT_LEVELS})
    ci, ci_diff, bands = bootstrap(dec, good, masks, E10._families(units), n_draws)
    preds = predictions(sm, ci_diff)
    return dict(curves=cv, summary=sm, ci=ci, ci_diff=ci_diff, bands=bands, caveat=caveat,
                predictions=preds)


def _print(res):
    sm, ci, cd = res["summary"], res["ci"], res["ci_diff"]
    fmt = lambda x: "  None" if x is None else f"{x:.4f}"  # noqa: E731
    for s in ("ABD", "C"):
        print(f"\n{s}: target coverage (G2 q=0.05) = {fmt(sm[s]['target_coverage'])}")
        for g in GATES:
            r = sm[s][g]
            cov = r.get("coverage_q05")
            print(f"  {g}  cov_q05={fmt(cov)}  matched={fmt(r['matched_risk'])} "
                  f"[{fmt(ci[s][g]['matched_risk']['lo'])}, {fmt(ci[s][g]['matched_risk']['hi'])}]  "
                  f"aurc={fmt(r['aurc'])} [{fmt(ci[s][g]['aurc']['lo'])}, {fmt(ci[s][g]['aurc']['hi'])}]")
        for k in ("G8-G2_matched", "G9-G2_matched"):
            print(f"  {k}: [{fmt(cd[s][k]['lo'])}, {fmt(cd[s][k]['hi'])}] n={cd[s][k]['n_defined']}")
    print("\ncaveat coverage q=0.05:", json.dumps(res["caveat"]["coverage_q05"]))
    for k, v in res["predictions"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "threshold"})


def _jsonable(cv):
    return {g: {s: dict(points=[{k: v for k, v in p.items()} for p in d["points"]], curve=d["curve"])
                for s, d in sc.items()} for g, sc in cv.items()}


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--time-one", action="store_true")
    args = p.parse_args()

    if args.time_one:
        arm = next(a for a in E2.ARMS if E10.ARM_KEY[a] == "A")
        Y_obs, Y_int, _ = projected(arm, 5, 0)
        t0 = time.time()
        g = _call(Y_int, Y_obs, 0, E10.B_NULL, "energy")
        print(f"energy call arm A level 5 seed 0: {len(Y_int)} environments, B={E10.B_NULL}, "
              f"{time.time() - t0:.1f}s, finite={bool(np.all(np.isfinite(g['thresholds'])))}")
        return

    if args.selftest:
        res = data_pass([0], 2, SELFTEST_B, regression=False, write=False)
        Y_obs = res["first_obs"]
        assert PR.geneig_signal(Y_obs, Y_obs) < 1e-12
        assert PR.energy_signal(Y_obs, Y_obs) == 0.0
        print(f"selftest data pass: {len(res['units'])} units, each with exactly one E10 match "
              f"(E10 has {res['e10_n']}); failed={len(res['failures'])}; precision regression skipped "
              f"(B={SELFTEST_B}); {res['wall']}s")
        out = analyse(res["units"])
        _print(out)
        print("selftest ok (nothing written)")
        return

    if args.analyse:
        doc = json.loads((OUT / "units.json").read_text())
        if doc.get("status") != "complete":
            raise SystemExit("[stop] results/e14/units.json is not complete")
        out = analyse(doc["units"])
        meta = dict(units_sha256=T2.sha256_file(OUT / "units.json"), provenance=provenance(None))
        T2.write_json(OUT / "curves.json", dict(
            experiment="e14_matched_coverage", prereg="notes/e14_prereg.md", gates=GATES, scopes=SCOPES,
            q_grid=E10.GRID["G2"], q_default=E10.Q_DEFAULT,
            bootstrap=dict(draws=N_SYS_BOOT, rng=f"default_rng({BOOT_SEED})", unit="(arm, level, seed) system",
                           decisions="fixed; metrics and matched target recomputed per draw"),
            summary=out["summary"], ci=out["ci"], ci_diff=out["ci_diff"], curves=_jsonable(out["curves"]),
            bands=out["bands"], **meta))
        T2.write_json(OUT / "predictions.json", dict(
            prereg="notes/e14_prereg.md", predictions=out["predictions"], caveat=out["caveat"], **meta))
        _print(out)
        print(f"written {OUT / 'curves.json'} and {OUT / 'predictions.json'}")
        return

    res = data_pass(E2.SEEDS, None, E10.B_NULL, regression=True, write=True)
    units = res["units"]
    keys = [_key(u) for u in units]
    if len(units) != res["e10_n"] or len(set(keys)) != len(keys) or len(units) != 900:
        raise SystemExit(f"[stop] join is not one-to-one over 900 units: {len(units)} units, "
                         f"{len(set(keys))} distinct keys, E10 has {res['e10_n']}")
    T2.write_json(OUT / "units.json", dict(
        experiment="e14_matched_coverage", prereg="notes/e14_prereg.md", status="complete",
        config=res["config"], precision_regression_vs_e10=res["regression"],
        join=dict(n_units=len(units), distinct_keys=len(set(keys)), e10_units=res["e10_n"], one_to_one=True),
        failed_units=len(res["failures"]), counts=E10.unit_counts(units), units=units,
        provenance=provenance(res["wall"])))
    T2.write_json(OUT / "failed_units.json", dict(failed=res["failures"], count=len(res["failures"])))
    print("regression vs E10:", res["regression"])
    print(f"written {OUT / 'units.json'} ({len(units)} units, {len(res['failures'])} failed, {res['wall']}s)")


if __name__ == "__main__":
    main()
