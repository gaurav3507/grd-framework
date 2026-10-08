"""E10: risk-coverage comparison of GRD's gate against simpler gates (synthetic arms).

Pre-registration: notes/e10_prereg.md, committed before this script was written or run.

One data pass over the four E2 arms (every level, seeds 0-9) records, for every
supplied intervention direction (one unit), every quantity any gate needs; all gates
are then computed from the same records, so every gate sees identical data.

Data pass (default): exactly E2.evaluate's construction (E2 builders, control-fit PCA
to D_PROJ, precision backbone rows via backbone.unmixing_rows). The gate call is
precision_readout.detect_with_pvalues(Y_int, Y_obs, alpha=0.05, B=500,
rng=default_rng(910_000 + seed), q=0.05): the same rng seed and call order as E2 under
rule="bh", so the BH decisions must reproduce results/e2_bh (checked; any difference
stops the run). A unit is GOOD when its recovered column's best-match |corr| with a
true latent exceeds 0.90 (E2._best_match_corr, E2.DIR_CORR). Bootstrap stability: 50
resamples per unit (rng default_rng(77_000 + 1000*seed + node); each draw resamples the
environment rows, then the control rows, with replacement), recomputing
backbone._unmixing_row and recording the sign-invariant angle arccos(min(1, |<a, b>|))
to the original row; the median and 90th percentile are stored.

Analysis (--analyse, reads units.json only): gates G0 none, G1 p <= a, G2 GRD (BH at q
within each arm-level-seed family of supplied environments), G3 n_e >= n*, G4 T_e > c
(c = percentiles of T_e over all units), G5 bootstrap 90th-percentile angle < theta, G7
oracle; coverage, risk, recall per gate, knob and scope (A, B, C, D, ABD pooled); AURC
and risk at GRD's q = 0.05 coverage; predictions P1-P6 and the decision rule evaluated
mechanically. Knob grids are fixed below.

--selftest: one seed, two levels per arm, 10 bootstrap draws, data pass and analysis
in memory, regression on that subset; writes nothing.

Usage:
    python experiments/e10_risk_coverage.py --selftest
    python experiments/e10_risk_coverage.py              # data pass -> units.json
    python experiments/e10_risk_coverage.py --analyse    # curves.json, predictions.json
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

E2 = T2.load_module("grd_e2_for_e10", "experiments/e2_calibration.py")
PR, BK = E2.PR, E2.BK
OUT = REPO / "results" / "e10_risk_coverage"
E2_BH = REPO / "results" / "e2_bh" / "e2_calibration_report.json"

ALPHA = 0.05
B_NULL = 500
Q_DEFAULT = 0.05
A_DEFAULT = 0.05
N_BOOT = 50
ARM_KEY = {"A_environment_starvation": "A", "B_power_starvation": "B",
           "C_measurement_contamination": "C", "D_weak_signal_starvation": "D"}
SCOPES = ("ABD", "A", "B", "C", "D")
GATES = ("G0", "G1", "G2", "G3", "G4", "G5", "G7")
GRID = dict(
    G1=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
    G2=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
    G3=[20, 40, 100, 400, 4000, 4001],
    G4_percentiles=list(range(0, 101, 5)),
    G5=[1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 90],
)
TOL = 1e-12


# ------------------------------------------------------------------ data pass
def _angle_deg(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def unit_records(arm, lvl, seed, n_boot):
    builder = E2.ARMS[arm]["builder"]
    ds, int_envs = builder(seed, lvl)
    basis_X, obs_X = ds.environments["basis"].X, ds.environments["obs"].X
    obs_Z = ds.environments["obs"].Z
    mu, Wp = BK.fit_pca(basis_X, E2.D_PROJ)
    Y_obs = BK.project(obs_X, mu, Wp)
    Y_int = [BK.project(X, mu, Wp) for (_, X) in int_envs]
    nodes = [i for (i, _) in int_envs]

    g = PR.detect_with_pvalues(Y_int, Y_obs, alpha=ALPHA, B=B_NULL,
                               rng=np.random.default_rng(910_000 + seed), q=Q_DEFAULT)
    W = BK.unmixing_rows(Y_obs, Y_int, nodes, E2.D_LATENT, readout="precision")
    Z_hat = Y_obs @ W.T

    rows = []
    n_obs = Y_obs.shape[0]
    for k, node in enumerate(nodes):
        Y_env = Y_int[k]
        n_e = Y_env.shape[0]
        rng = np.random.default_rng(77_000 + 1000 * seed + node)
        angles = np.empty(n_boot)
        for b in range(n_boot):
            ie = rng.integers(0, n_e, n_e)
            io = rng.integers(0, n_obs, n_obs)
            angles[b] = _angle_deg(BK._unmixing_row(Y_obs[io], Y_env[ie]), W[node])
        corr = float(E2._best_match_corr(Z_hat[:, node], obs_Z))
        rows.append(dict(
            arm=ARM_KEY[arm], arm_name=arm, level=lvl, seed=seed, node=int(node),
            n_e=int(n_e), signal=float(g["signals"][k]),
            threshold=float(g["thresholds"][k]), pvalue=float(g["pvalues"][k]),
            raw_detect=bool(g["raw_detect"][k]), bh_detect_q05=bool(g["bh_detect"][k]),
            best_match_corr=corr, good=bool(corr > E2.DIR_CORR),
            contaminated=bool(ARM_KEY[arm] == "C" and node >= E2.D_LATENT - lvl),
            boot_angle_median=float(np.median(angles)),
            boot_angle_p90=float(np.quantile(angles, 0.90)),
        ))
    return rows


def data_pass(seeds, levels_per_arm, n_boot):
    units = []
    started = time.time()
    for arm, spec in E2.ARMS.items():
        levels = spec["levels"][:levels_per_arm] if levels_per_arm else spec["levels"]
        for lvl in levels:
            for seed in seeds:
                units += unit_records(arm, lvl, seed, n_boot)
        print(f"  {ARM_KEY[arm]}: done ({time.time() - started:.0f}s)", flush=True)
    return units, round(time.time() - started, 1)


def regression_vs_e2_bh(units):
    """BH-selected nodes per arm-level-seed must equal e2_bh's certified lists."""
    ref = json.loads(E2_BH.read_text())
    expected = {}
    for arm, a in ref["arms"].items():
        for lv in a["levels"]:
            for j, s in enumerate(lv["per_seed"]):
                expected[(ARM_KEY[arm], lv["level"], E2.SEEDS[j])] = sorted(s["certified"])
    got = {}
    for u in units:
        key = (u["arm"], u["level"], u["seed"])
        got.setdefault(key, [])
        if u["bh_detect_q05"]:
            got[key].append(u["node"])
    diffs = [(k, sorted(v), expected.get(k)) for k, v in got.items()
             if sorted(v) != expected.get(k)]
    return dict(families_checked=len(got), differing=len(diffs),
                identical=not diffs, examples=[str(d) for d in diffs[:5]])


# ------------------------------------------------------------------ gates
def _families(units):
    fam = {}
    for i, u in enumerate(units):
        fam.setdefault((u["arm"], u["level"], u["seed"]), []).append(i)
    return list(fam.values())


def gate_decisions(units):
    """{gate: [(knob, bool array over units)]}, every gate on the same records."""
    p = np.array([u["pvalue"] for u in units])
    n_e = np.array([u["n_e"] for u in units])
    T = np.array([u["signal"] for u in units])
    ang = np.array([u["boot_angle_p90"] for u in units])
    good = np.array([u["good"] for u in units])
    fams = _families(units)
    out = {"G0": [(None, np.ones(len(units), dtype=bool))]}
    out["G1"] = [(a, p <= a) for a in GRID["G1"]]
    g2 = []
    for q in GRID["G2"]:
        dec = np.zeros(len(units), dtype=bool)
        for idx in fams:
            dec[idx] = PR.bh_fdr(p[idx], q=q)
        g2.append((q, dec))
    out["G2"] = g2
    out["G3"] = [(n, n_e >= n) for n in GRID["G3"]]
    cuts = np.percentile(T, GRID["G4_percentiles"])
    out["G4"] = [(dict(percentile=pc, c=float(c)), T > c)
                 for pc, c in zip(GRID["G4_percentiles"], cuts)]
    out["G5"] = [(th, ang < th) for th in GRID["G5"]]
    out["G7"] = [(None, good.copy())]
    return out


def _scope_mask(units, scope):
    arms = np.array([u["arm"] for u in units])
    return np.isin(arms, ["A", "B", "D"]) if scope == "ABD" else arms == scope


def _metrics(dec, good, mask):
    n = int(mask.sum())
    cert = dec & mask
    nc = int(cert.sum())
    n_good = int((good & mask).sum())
    return dict(n_units=n, n_certified=nc, coverage=nc / n if n else None,
                risk=(int((cert & ~good).sum()) / nc) if nc else None,
                recall=(int((cert & good).sum()) / n_good) if n_good else None)


def _curve(points):
    """Collapse equal coverage to mean risk, drop undefined risk, sort by coverage."""
    pts = {}
    for p in points:
        if p["risk"] is None:
            continue
        key = round(p["coverage"], 12)
        pts.setdefault(key, []).append(p["risk"])
    return sorted((c, float(np.mean(r))) for c, r in pts.items())


def _aurc(curve):
    if not curve:
        return dict(value=None, coverage_range=None, single_point=False, n_points=0)
    if len(curve) == 1:
        return dict(value=curve[0][1], coverage_range=[curve[0][0], curve[0][0]],
                    single_point=True, n_points=1)
    c = np.array([x for x, _ in curve])
    r = np.array([y for _, y in curve])
    area = float(np.sum((c[1:] - c[:-1]) * (r[1:] + r[:-1]) / 2.0))
    return dict(value=area / float(c[-1] - c[0]), coverage_range=[float(c[0]), float(c[-1])],
                single_point=False, n_points=len(curve))


def _matched(curve, target):
    if not curve or target is None:
        return None
    c = np.array([x for x, _ in curve])
    r = np.array([y for _, y in curve])
    if target < c[0] - TOL or target > c[-1] + TOL:
        return None
    if len(curve) == 1:
        return float(r[0])
    return float(np.interp(target, c, r))


def _knob_point(points, knob):
    return next(p for p in points if p["knob"] == knob)


def analyse(units):
    good = np.array([u["good"] for u in units])
    decisions = gate_decisions(units)
    curves = {}
    for gate, ops in decisions.items():
        curves[gate] = {}
        for scope in SCOPES:
            mask = _scope_mask(units, scope)
            points = [dict(knob=knob, **_metrics(dec, good, mask)) for knob, dec in ops]
            curves[gate][scope] = dict(points=points, curve=_curve(points))
    grd_default, g1_default = {}, {}
    for scope in SCOPES:
        grd_default[scope] = _knob_point(curves["G2"][scope]["points"], Q_DEFAULT)
        g1_default[scope] = _knob_point(curves["G1"][scope]["points"], A_DEFAULT)
    for gate in GATES:
        for scope in SCOPES:
            cv = curves[gate][scope]["curve"]
            curves[gate][scope]["aurc"] = _aurc(cv)
            curves[gate][scope]["matched_risk_at_grd_q05_coverage"] = _matched(
                cv, grd_default[scope]["coverage"])

    target_abd = grd_default["ABD"]["coverage"]
    g5_ok = [p for p in curves["G5"]["ABD"]["points"]
             if p["risk"] is not None and p["coverage"] >= target_abd - TOL]
    g5_best = (min(g5_ok, key=lambda p: (p["risk"], -p["coverage"])) if g5_ok else None)

    def M(gate, scope):
        return curves[gate][scope]["matched_risk_at_grd_q05_coverage"]

    def A(gate, scope):
        return curves[gate][scope]["aurc"]["value"]

    grd_abd = grd_default["ABD"]["risk"]
    preds = {}
    preds["P1"] = dict(value=grd_abd, threshold="<= 0.10",
                       held=bool(grd_abd is not None and grd_abd <= 0.10))
    g0 = curves["G0"]["ABD"]["points"][0]["risk"]
    g3 = M("G3", "ABD")
    preds["P2"] = dict(grd_risk=grd_abd, g0_risk=g0, g3_matched_risk=g3,
                       threshold="grd < g0 and grd < g3 (matched)",
                       held=(None if g3 is None else bool(grd_abd < g0 and grd_abd < g3)),
                       note=(None if g3 is not None else
                             "G3 has no operating range covering GRD's coverage"))
    p3 = {s: dict(g4_aurc=A("G4", s), grd_aurc=A("G2", s),
                  diff=(None if A("G4", s) is None or A("G2", s) is None
                        else A("G4", s) - A("G2", s))) for s in ("A", "B", "D")}
    held3 = (all(p3[s]["diff"] is not None for s in p3)
             and abs(p3["A"]["diff"]) <= 0.02 and abs(p3["D"]["diff"]) <= 0.02
             and p3["B"]["diff"] > 0.02)
    preds["P3"] = dict(values=p3, threshold="|diff| <= 0.02 on A and D; diff > 0.02 on B",
                       held=bool(held3))
    g5m = M("G5", "ABD")
    preds["P4"] = dict(g5_matched_risk=g5m, grd_risk=grd_abd,
                       threshold="g5 <= grd + 0.02",
                       held=(None if g5m is None else bool(g5m <= grd_abd + 0.02)))
    c_pts = [(gate, p) for gate in ("G0", "G1", "G2", "G3", "G4", "G5")
             for p in curves[gate]["C"]["points"]
             if p["risk"] is not None and p["coverage"] >= 0.5 - TOL]
    worst = min(c_pts, key=lambda gp: gp[1]["risk"]) if c_pts else None
    g5_c = [p["risk"] for g, p in c_pts if g == "G5"]
    preds["P5"] = dict(
        min_risk_non_oracle_cov_ge_0p5=(worst[1]["risk"] if worst else None),
        at=(dict(gate=worst[0], knob=worst[1]["knob"], coverage=worst[1]["coverage"])
            if worst else None),
        g5_min_risk_cov_ge_0p5=(min(g5_c) if g5_c else None),
        threshold="every non-oracle risk > 0.30 at coverage >= 0.5",
        held=(None if worst is None else bool(worst[1]["risk"] > 0.30)))
    grd_b, grd_d = grd_default["B"], grd_default["D"]
    g4_b = curves["G4"]["B"]["points"]
    pick = min(range(len(g4_b)), key=lambda i: abs(g4_b[i]["coverage"] - grd_b["coverage"]))
    g4_d = curves["G4"]["D"]["points"][pick]
    d_g4 = (None if g4_d["risk"] is None or grd_d["risk"] is None
            else abs(g4_d["risk"] - grd_d["risk"]))
    d_grd = (None if grd_b["risk"] is None or grd_d["risk"] is None
             else abs(grd_d["risk"] - grd_b["risk"]))
    preds["P6"] = dict(
        c_from_arm_B=g4_b[pick]["knob"], g4_coverage_B=g4_b[pick]["coverage"],
        grd_coverage_B=grd_b["coverage"], g4_risk_D=g4_d["risk"],
        g4_coverage_D=g4_d["coverage"], grd_risk_D=grd_d["risk"], grd_risk_B=grd_b["risk"],
        d_g4=d_g4, d_grd=d_grd, threshold="d_g4 >= d_grd + 0.05",
        held=(None if d_g4 is None or d_grd is None else bool(d_g4 >= d_grd + 0.05)))
    decision = dict(
        rule=("if G5's risk at GRD's q=0.05 coverage on ABD is lower than GRD's by more "
              "than 0.02, report GRD's advantage as calibration, estimator coupling and "
              "theory, not as a better risk-coverage trade-off"),
        g5_matched_risk=g5m, grd_risk=grd_abd,
        triggered=(None if g5m is None else bool(g5m < grd_abd - 0.02)))
    summary = dict(grd_default_q05=grd_default, g1_default_a05=g1_default,
                   g5_best_on_abd_with_coverage_ge_grd=(
                       dict(theta=g5_best["knob"], coverage=g5_best["coverage"],
                            risk=g5_best["risk"]) if g5_best else None))
    return curves, summary, preds, decision


def unit_counts(units):
    by_arm = {}
    for u in units:
        a = by_arm.setdefault(u["arm"], dict(units=0, good=0))
        a["units"] += 1
        a["good"] += int(u["good"])
    return dict(total=len(units), good=int(sum(u["good"] for u in units)), by_arm=by_arm)


def provenance(wall):
    return dict(code_commit=T2.code_commit(), python=platform.python_version(),
                numpy=np.__version__,
                sha256={f: T2.sha256_file(REPO / f) for f in (
                    "experiments/e2_calibration.py", "src/gate/precision_readout.py",
                    "src/recover/backbone.py", "experiments/e10_risk_coverage.py")},
                wall_seconds=wall)


def _print_tables(curves, summary, preds, decision):
    print("\nmatched-coverage risk (at GRD q=0.05 coverage)")
    print("gate  " + "  ".join(f"{s:>7}" for s in SCOPES))
    for g in GATES:
        vals = [curves[g][s]["matched_risk_at_grd_q05_coverage"] for s in SCOPES]
        print(f"{g:5} " + "  ".join(f"{v:7.4f}" if v is not None else "   None" for v in vals))
    for name in ("grd_default_q05", "g1_default_a05"):
        print(name, {s: (round(p["coverage"], 4), None if p["risk"] is None else
                         round(p["risk"], 4)) for s, p in summary[name].items()})
    for k, v in preds.items():
        print(k, "held:", v["held"])
    print("decision rule triggered:", decision["triggered"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        units, wall = data_pass([0], 2, 10)
        reg = regression_vs_e2_bh(units)
        print("selftest units:", unit_counts(units), f"({wall}s)")
        print("selftest regression vs e2_bh:", reg)
        curves, summary, preds, decision = analyse(units)
        _print_tables(curves, summary, preds, decision)
        if not reg["identical"]:
            raise SystemExit("selftest regression FAILED")
        print("selftest ok (nothing written)")
        return

    if args.analyse:
        doc = json.loads((OUT / "units.json").read_text())
        units = doc["units"]
        curves, summary, preds, decision = analyse(units)
        T2.write_json(OUT / "curves.json", dict(
            experiment="e10_risk_coverage", prereg="notes/e10_prereg.md",
            grids=GRID, q_default=Q_DEFAULT, a_default=A_DEFAULT,
            units=unit_counts(units), summary=summary, curves=curves,
            provenance=provenance(None)))
        T2.write_json(OUT / "predictions.json", dict(
            prereg="notes/e10_prereg.md", predictions=preds, decision_rule=decision))
        _print_tables(curves, summary, preds, decision)
        print(f"written {OUT / 'curves.json'} and {OUT / 'predictions.json'}")
        return

    units, wall = data_pass(E2.SEEDS, None, N_BOOT)
    reg = regression_vs_e2_bh(units)
    print("units:", unit_counts(units), f"({wall}s)")
    print("regression vs e2_bh:", reg)
    if not reg["identical"]:
        raise SystemExit("[stop] BH decisions differ from results/e2_bh; nothing written")
    T2.write_json(OUT / "units.json", dict(
        experiment="e10_risk_coverage", prereg="notes/e10_prereg.md",
        config=dict(alpha=ALPHA, B=B_NULL, q=Q_DEFAULT, n_boot=N_BOOT, seeds=E2.SEEDS,
                    good_threshold=E2.DIR_CORR, null_rng="default_rng(910_000 + seed)",
                    boot_rng="default_rng(77_000 + 1000*seed + node)"),
        regression_vs_e2_bh=reg, counts=unit_counts(units), units=units,
        provenance=provenance(wall)))
    print(f"written {OUT / 'units.json'}")


if __name__ == "__main__":
    main()
