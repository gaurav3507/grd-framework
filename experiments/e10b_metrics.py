"""E10b: E10 re-scored with stricter recovery definitions, common-range AURC and a
seed-level bootstrap. E10's committed outputs (results/e10_risk_coverage, figure10)
are not touched; everything here is written to results/e10b_metrics.

Data pass: E10's own data pass (e10_risk_coverage.data_pass, same seeds, same rng
streams) with three appended fields per unit: corr_all (|corr| of the recovered
column with every latent), target (environment node k targets latent k; arm C
contaminated environments have none) and corr_target. Every pre-existing field must
equal results/e10_risk_coverage/units.json exactly, or the run stops.

Recovery definitions (threshold 0.90, E2.DIR_CORR):
  best_match  best-match |corr| with any latent (E10's definition);
  target      |corr| with the unit's own target latent; no target = not recovered;
  one_to_one  per gate and knob: within each (arm, level, seed) family, a Hungarian
              assignment (maximised) of the certified units to the 5 latents on
              corr_all; a certified unit is recovered iff its assigned |corr| > 0.90.
              Uncertified units are not scored. The G7 oracle for this definition
              certifies the units recovered with the whole family certified.
Gates, knob grids, coverage and the matched-coverage and AURC computations are E10's
(imported). Common-range AURC: per scope and definition, the range is [max of the
non-single-point gates' minimum coverage, min of their maximum coverage]; each gate's
curve is linearly interpolated on that range and integrated by trapezoid.

Seed bootstrap: units within a seed share one causal system, so seeds are resampled
(2,000 draws of the 10 seeds with replacement, rng default_rng(10_101)). Gate decisions
and knob grids (including G4's cutoffs) are fixed from the full data; a draw re-weights
whole families by how often their seed is drawn. Per draw, definition and scope (ABD,
C): GRD's q=0.05 coverage and risk, every gate's risk at that draw's GRD coverage, and
the paired difference gate minus GRD; 2.5% and 97.5% percentiles over the draws where
the value is defined.

Usage:
    python experiments/e10b_metrics.py --selftest
    python experiments/e10b_metrics.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import tier2_common as T2  # noqa: E402

E10 = T2.load_module("grd_e10_for_e10b", "experiments/e10_risk_coverage.py")
E2 = E10.E2
OUT = REPO / "results" / "e10b_metrics"
E10_DIR = REPO / "results" / "e10_risk_coverage"
DEFS = ("best_match", "target", "one_to_one")
THR = E2.DIR_CORR
NEW_FIELDS = ("corr_all", "target", "corr_target")
BOOT_FIELDS = ("boot_angle_median", "boot_angle_p90")
N_SEED_BOOT = 2000
SEED_BOOT_RNG = 10_101
BOOT_SCOPES = ("ABD", "C")
TOL = E10.TOL


# ------------------------------------------------------------------ regression
def regression_vs_e10(units, skip=()):
    """Every pre-existing E10 field must equal results/e10_risk_coverage/units.json."""
    ref = {(u["arm"], u["level"], u["seed"], u["node"]): u
           for u in json.loads((E10_DIR / "units.json").read_text())["units"]}
    diffs = []
    for u in units:
        key = (u["arm"], u["level"], u["seed"], u["node"])
        old = {k: v for k, v in u.items() if k not in NEW_FIELDS and k not in skip}
        want = {k: v for k, v in ref.get(key, {}).items() if k not in skip}
        if old != want:
            diffs.append(str(key))
    consistent = all(abs(max(u["corr_all"]) - u["best_match_corr"]) < 1e-12 for u in units)
    return dict(units_checked=len(units), differing=len(diffs), identical=not diffs,
                examples=diffs[:5], skipped_fields=list(skip),
                max_corr_all_equals_best_match_corr=bool(consistent))


# ------------------------------------------------------------------ recovery
def one_to_one(units, families, cert):
    """Hungarian recovery of the certified units within each family."""
    rec = np.zeros(len(units), dtype=bool)
    for idx in families:
        sel = [i for i in idx if cert[i]]
        if not sel:
            continue
        M = np.array([units[i]["corr_all"] for i in sel])
        rows, cols = linear_sum_assignment(-M)
        for r, c in zip(rows, cols):
            rec[sel[r]] = bool(M[r, c] > THR)
    return rec


def gate_table(units):
    """{definition: {gate: [(knob, certified, recovered)]}}, plus per-definition totals."""
    families = E10._families(units)
    base = E10.gate_decisions(units)
    best = np.array([u["good"] for u in units])
    tgt = np.array([u["corr_target"] is not None and u["corr_target"] > THR for u in units])
    full = one_to_one(units, families, np.ones(len(units), dtype=bool))
    totals = dict(best_match=best, target=tgt, one_to_one=full)
    table = {}
    for d in DEFS:
        table[d] = {}
        for gate, ops in base.items():
            if gate == "G7":
                ops = [(None, totals[d].copy())]
            rows = []
            for knob, cert in ops:
                rec = one_to_one(units, families, cert) if d == "one_to_one" else totals[d]
                rows.append((knob, cert, rec))
            table[d][gate] = rows
    return table, totals


def _metrics(cert, rec, mask, recoverable):
    n = int(mask.sum())
    c = cert & mask
    nc = int(c.sum())
    n_rec = int((recoverable & mask).sum())
    return dict(n_units=n, n_certified=nc, coverage=nc / n if n else None,
                risk=(int((c & ~rec).sum()) / nc) if nc else None,
                recall=(int((c & rec).sum()) / n_rec) if n_rec else None)


def common_range(curves):
    spans = [(cv[0][0], cv[-1][0]) for cv in curves.values() if len(cv) > 1]
    if not spans:
        return None
    lo, hi = max(a for a, _ in spans), min(b for _, b in spans)
    return [float(lo), float(hi)] if hi > lo + TOL else None


def aurc_on(curve, rng):
    if rng is None or len(curve) < 2:
        return None
    lo, hi = rng
    c = np.array([x for x, _ in curve])
    r = np.array([y for _, y in curve])
    if c[0] > lo + TOL or c[-1] < hi - TOL:
        return None
    xs = np.unique(np.concatenate([[lo, hi], c[(c > lo) & (c < hi)]]))
    ys = np.interp(xs, c, r)
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0) / (hi - lo))


def analyse(units, table, totals):
    scopes = {s: E10._scope_mask(units, s) for s in E10.SCOPES}
    out = {}
    for d in DEFS:
        curves = {}
        for gate, rows in table[d].items():
            curves[gate] = {}
            for s, mask in scopes.items():
                pts = [dict(knob=k, **_metrics(cert, rec, mask, totals[d]))
                       for k, cert, rec in rows]
                curves[gate][s] = dict(points=pts, curve=E10._curve(pts))
        grd = {s: E10._knob_point(curves["G2"][s]["points"], E10.Q_DEFAULT) for s in scopes}
        g1 = {s: E10._knob_point(curves["G1"][s]["points"], E10.A_DEFAULT) for s in scopes}
        ranges = {}
        for s in scopes:
            ranges[s] = common_range({g: curves[g][s]["curve"] for g in curves})
            for g in curves:
                cv = curves[g][s]["curve"]
                curves[g][s]["aurc"] = E10._aurc(cv)
                curves[g][s]["matched_risk_at_grd_q05_coverage"] = E10._matched(
                    cv, grd[s]["coverage"])
                curves[g][s]["common_range_aurc"] = aurc_on(cv, ranges[s])
        out[d] = dict(curves=curves, common_range=ranges,
                      summary=dict(grd_default_q05=grd, g1_default_a05=g1))
    return out


# ------------------------------------------------------------------ seed bootstrap
def seed_bootstrap(units, table, totals, n_draws, seed_rng):
    seeds = np.array([u["seed"] for u in units])
    seed_ids = sorted(set(seeds.tolist()))
    # integer indicators: a boolean matmul would return a logical OR, not counts
    S = np.array([seeds == s for s in seed_ids], dtype=np.int64)   # (n_seeds, n_units)
    rng = np.random.default_rng(seed_rng)
    draws = [np.bincount(rng.integers(0, len(seed_ids), len(seed_ids)),
                         minlength=len(seed_ids)) for _ in range(n_draws)]
    identity = np.ones(len(seed_ids), dtype=np.int64)   # must reproduce point estimates
    result = {}
    for d in DEFS:
        result[d] = {}
        for scope in BOOT_SCOPES:
            mask = E10._scope_mask(units, scope)
            n_units = S @ mask.astype(np.int64)              # units per seed
            pre = {g: [(k, S @ (cert & mask).astype(np.int64),
                        S @ (cert & ~rec & mask).astype(np.int64))
                       for k, cert, rec in rows] for g, rows in table[d].items()}
            grd_k = [i for i, (k, _, _) in enumerate(pre["G2"]) if k == E10.Q_DEFAULT][0]
            cov_grd, risk_grd, matched = [], [], {g: [] for g in pre}
            identity_point = None
            for w in [identity] + draws:
                nu = float(w @ n_units)
                curves = {}
                for g, rows in pre.items():
                    pts = []
                    for k, nc, nb in rows:
                        c = float(w @ nc)
                        pts.append(dict(coverage=c / nu, risk=(float(w @ nb) / c) if c else None))
                    curves[g] = (pts, E10._curve(pts))
                gp = curves["G2"][0][grd_k]
                if identity_point is None:
                    identity_point = dict(
                        grd_coverage=gp["coverage"], grd_risk=gp["risk"],
                        matched={g: E10._matched(curves[g][1], gp["coverage"])
                                 for g in pre})
                    continue
                cov_grd.append(gp["coverage"])
                risk_grd.append(gp["risk"])
                for g in pre:
                    matched[g].append(E10._matched(curves[g][1], gp["coverage"]))

            def interval(vals):
                v = np.array([x for x in vals if x is not None], dtype=float)
                if not v.size:
                    return dict(n_valid=0, lo=None, hi=None)
                return dict(n_valid=int(v.size), lo=float(np.percentile(v, 2.5)),
                            hi=float(np.percentile(v, 97.5)))
            gates = {}
            for g in pre:
                diffs = [None if (m is None or r is None) else m - r
                         for m, r in zip(matched[g], risk_grd)]
                gates[g] = dict(matched_risk=interval(matched[g]),
                                diff_vs_grd=interval(diffs))
            result[d][scope] = dict(grd_coverage=interval(cov_grd),
                                    grd_risk=interval(risk_grd), gates=gates,
                                    identity_weights=identity_point)
    return dict(n_draws=n_draws, rng=f"default_rng({seed_rng})", resampled="seeds",
                fixed=("gate decisions and knob grids, including G4 cutoffs, from the "
                       "full data"), results=result)


# ------------------------------------------------------------------ summary
def good_counts(units, totals):
    arms = np.array([u["arm"] for u in units])
    return {d: dict(total=int(v.sum()),
                    by_arm={a: int(v[arms == a].sum()) for a in ("A", "B", "C", "D")})
            for d, v in totals.items()}


def build_summary(units, totals, res, boot):
    def r4(x):
        return None if x is None else round(float(x), 4)
    tables = {}
    for d in DEFS:
        tables[d] = {}
        for s in BOOT_SCOPES:
            rows = {}
            for g in E10.GATES:
                cell = res[d]["curves"][g][s]
                b = boot["results"][d][s]["gates"][g]
                rows[g] = dict(
                    matched_risk=r4(cell["matched_risk_at_grd_q05_coverage"]),
                    matched_risk_interval=[r4(b["matched_risk"]["lo"]),
                                           r4(b["matched_risk"]["hi"])],
                    aurc=r4(cell["aurc"]["value"]),
                    aurc_range=cell["aurc"]["coverage_range"],
                    aurc_single_point=cell["aurc"]["single_point"],
                    common_range_aurc=r4(cell["common_range_aurc"]),
                    diff_vs_grd_interval=[r4(b["diff_vs_grd"]["lo"]),
                                          r4(b["diff_vs_grd"]["hi"])])
            grd = res[d]["summary"]["grd_default_q05"][s]
            bg = boot["results"][d][s]
            tables[d][s] = dict(
                common_range=res[d]["common_range"][s], gates=rows,
                grd_q05=dict(coverage=r4(grd["coverage"]),
                             coverage_interval=[r4(bg["grd_coverage"]["lo"]),
                                                r4(bg["grd_coverage"]["hi"])],
                             risk=r4(grd["risk"]),
                             risk_interval=[r4(bg["grd_risk"]["lo"]),
                                            r4(bg["grd_risk"]["hi"])]))
    return dict(good_counts=good_counts(units, totals), tables=tables)


def best_match_reproduces_e10(res):
    ref = json.loads((E10_DIR / "curves.json").read_text())["curves"]
    bad = []
    for g in E10.GATES:
        for s in E10.SCOPES:
            a, b = res["best_match"]["curves"][g][s], ref[g][s]
            if (a["matched_risk_at_grd_q05_coverage"] != b["matched_risk_at_grd_q05_coverage"]
                    or a["aurc"] != b["aurc"]):
                bad.append(f"{g}/{s}")
    return dict(identical=not bad, differing=bad)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    started = time.time()
    if args.selftest:
        units, wall = E10.data_pass([0], 2, 10)
        reg = regression_vs_e10(units, skip=BOOT_FIELDS)
        table, totals = gate_table(units)
        res = analyse(units, table, totals)
        boot = seed_bootstrap(units, table, totals, 10, SEED_BOOT_RNG)
        summ = build_summary(units, totals, res, boot)
        print("selftest regression (bootstrap-angle fields skipped, n_boot=10):", reg)
        print("selftest good counts:", summ["good_counts"])
        print("selftest GRD q=0.05 ABD:", summ["tables"]["target"]["ABD"]["grd_q05"])
        if not reg["identical"]:
            raise SystemExit("selftest regression FAILED")
        print(f"selftest ok (nothing written, {time.time() - started:.0f}s)")
        return

    units, wall = E10.data_pass(E2.SEEDS, None, E10.N_BOOT)
    reg = regression_vs_e10(units)
    print("regression vs e10 units:", reg, flush=True)
    if not reg["identical"]:
        raise SystemExit("[stop] pre-existing unit fields differ from E10; nothing written")
    prov = E10.provenance(wall)
    prov["sha256"]["experiments/e10b_metrics.py"] = T2.sha256_file(Path(__file__))
    T2.write_json(OUT / "units.json", dict(
        experiment="e10b_metrics", source="e10_risk_coverage.data_pass",
        regression_vs_e10=reg, units=units, provenance=prov))
    table, totals = gate_table(units)
    res = analyse(units, table, totals)
    repro = best_match_reproduces_e10(res)
    print("best_match analysis reproduces E10 curves:", repro, flush=True)
    boot = seed_bootstrap(units, table, totals, N_SEED_BOOT, SEED_BOOT_RNG)
    summ = build_summary(units, totals, res, boot)
    checks = []
    for d in DEFS:
        for sc in BOOT_SCOPES:
            ip = boot["results"][d][sc]["identity_weights"]
            grd = res[d]["summary"]["grd_default_q05"][sc]
            checks.append(ip["grd_coverage"] == grd["coverage"] and ip["grd_risk"] == grd["risk"]
                          and all(ip["matched"][g] == res[d]["curves"][g][sc]
                                  ["matched_risk_at_grd_q05_coverage"] for g in ip["matched"]))
    summ["bootstrap_identity_weights_reproduce_point_estimates"] = bool(all(checks))
    if not all(checks):
        raise SystemExit("[stop] bootstrap with identity weights does not reproduce the "
                         "point estimates")
    summ["best_match_reproduces_e10"] = repro
    summ["regression_vs_e10_units"] = reg
    T2.write_json(OUT / "curves.json", dict(
        experiment="e10b_metrics", definitions=list(DEFS), recovery_threshold=THR,
        grids=E10.GRID, **{d: res[d] for d in DEFS}, provenance=prov))
    T2.write_json(OUT / "uncertainty.json", dict(experiment="e10b_metrics", **boot))
    T2.write_json(OUT / "summary.json", dict(experiment="e10b_metrics", **summ,
                                             wall_seconds=round(time.time() - started, 1)))
    print(f"written {OUT} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
