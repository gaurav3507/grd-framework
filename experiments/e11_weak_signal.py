"""E11: weak-signal arm audit (E2 arm D as run) and a relative-scale arm D2.

E2 arm D sets each target's post-intervention noise variance to an ABSOLUTE value s
in {0.1, 0.4, 0.7, 0.9, 1.0}, while observational noise variances are U(0.5, 1.5). For
s >= 0.7 some interventions therefore RAISE the target's noise variance; on a
parentless target that is outside the model of Corollary 8 (which needs
lambda'_i < lambda_i when the target has no parents) and invisible to lambda_max
(Proposition 5).

Part 1, audit (post hoc, no new simulation of units): for every seed the SCM of
E2._base_scm(seed) is asserted equal to the one simulate() draws (B and noise
variances), then every arm-D unit of results/e10b_metrics/units.json is classified:
parentless = is_source[node], raises = (level >= noise_var[node]), out_of_model =
parentless and raises. The numbers must reproduce the pre-stated X1/X2 values computed
from the committed units on 8 Oct; otherwise the run stops and writes nothing.

Part 2, arm D2: like E2.arm_D but each intervention sets the target's noise variance
to s * noise_var[i] (relative), s in {0.1, 0.4, 0.7, 0.9, 0.95}, so every intervention
reduces its target's noise variance. Scored with E2.evaluate under rule="bh", q=0.05,
seeds 0-9, and aggregated by E2.run itself (its ARMS table is replaced, in a private
module instance, by the single D2 arm), so the aggregation is exactly E2's. Per-unit
records (target-aligned |corr|, BH decision) give coverage and risk at q=0.05; their
BH decisions are checked against evaluate's certified lists.

--selftest: audit in full (cheap), D2 on seed 0 and two levels; writes nothing.
"""

import argparse
import json
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

E2 = T2.load_module("grd_e2_for_e11", "experiments/e2_calibration.py")
SIM, PR, BK = E2.SIM, E2.PR, E2.BK
OUT = REPO / "results" / "e11_weak_signal"
UNITS = REPO / "results" / "e10b_metrics" / "units.json"
E2_BH = REPO / "results" / "e2_bh" / "e2_calibration_report.json"
RULE, Q = "bh", 0.05
D_LEVELS = [0.1, 0.4, 0.7, 0.9, 1.0]
D2_LEVELS = [0.1, 0.4, 0.7, 0.9, 0.95]
D2_NAME = "D2_relative_weak_signal"
THR = E2.DIR_CORR

# Pre-stated before the run (Tier 2 brief, 8 Oct 2026).
EXPECTED_AUDIT = dict(
    raises=[0, 0, 10, 23, 27],
    out_of_model=[0, 0, 3, 7, 11],
    out_of_model_certified_total=0, out_of_model_recovered_total=0,
    in_model_certified=[50, 50, 35, 25, 20], in_model_n=[50, 50, 47, 43, 39],
    in_model_recovered=[50, 50, 43, 37, 32],
)
EXPECTATIONS = dict(
    X1_X2=("arm D audit reproduces: raises 0,0,10,23,27 of 50; out-of-model 0,0,3,7,11; "
           "out-of-model certified 0 of 21 and recovered 0 of 21; in-model certified "
           "50/50, 50/50, 35/47, 25/43, 20/39 and recovered 50, 50, 43, 37, 32"),
    X3=("arm D2: the gate first leaves full PROCEED at s = 0.7 or 0.9, and certified MCC "
        "stays >= 0.90 at every level where anything is certified"),
    X4=("arm D2: mean ungated MCC stays above arm D's at matched nominal scale for "
        "s >= 0.7 (the matched levels are 0.7 and 0.9)"),
)


# ------------------------------------------------------------------ part 1: audit
def scm_assert(seeds):
    """_base_scm(seed) must equal the SCM simulate() draws for that seed."""
    out = {}
    for seed in seeds:
        B, nv, is_src = E2._base_scm(seed)
        ds, _ = E2.arm_D(seed, D_LEVELS[0])
        out[seed] = bool(np.array_equal(B, ds.B) and np.array_equal(nv, ds.noise_var))
    return out


def audit(seeds):
    scm = {s: E2._base_scm(s) for s in seeds}
    units = [u for u in json.loads(UNITS.read_text())["units"]
             if u["arm"] == "D" and u["seed"] in scm]
    rows = []
    for lvl in D_LEVELS:
        us = [u for u in units if u["level"] == lvl]
        recs = []
        for u in us:
            B, nv, is_src = scm[u["seed"]]
            parentless = bool(is_src[u["node"]])
            raises = bool(lvl >= nv[u["node"]])
            recs.append(dict(seed=u["seed"], node=u["node"], parentless=parentless,
                             raises=raises, out_of_model=parentless and raises,
                             noise_var=float(nv[u["node"]]),
                             certified=bool(u["bh_detect_q05"]),
                             corr_target=u["corr_target"],
                             recovered=bool(u["corr_target"] > THR)))

        def block(rs):
            ct = [r["corr_target"] for r in rs]
            return dict(n=len(rs), certified=int(sum(r["certified"] for r in rs)),
                        recovered=int(sum(r["recovered"] for r in rs)),
                        median_corr_target=(float(np.median(ct)) if ct else None),
                        frac_recovered=(float(np.mean([r["recovered"] for r in rs]))
                                        if rs else None))
        rows.append(dict(level=lvl, n=len(recs),
                         raises=int(sum(r["raises"] for r in recs)),
                         out_of_model=int(sum(r["out_of_model"] for r in recs)),
                         out_of_model_units=block([r for r in recs if r["out_of_model"]]),
                         in_model_units=block([r for r in recs if not r["out_of_model"]]),
                         units=recs))
    return rows


def audit_reproduces(rows):
    got = dict(
        raises=[r["raises"] for r in rows],
        out_of_model=[r["out_of_model"] for r in rows],
        out_of_model_certified_total=sum(r["out_of_model_units"]["certified"] for r in rows),
        out_of_model_recovered_total=sum(r["out_of_model_units"]["recovered"] for r in rows),
        in_model_certified=[r["in_model_units"]["certified"] for r in rows],
        in_model_n=[r["in_model_units"]["n"] for r in rows],
        in_model_recovered=[r["in_model_units"]["recovered"] for r in rows],
    )
    return got == EXPECTED_AUDIT, got


# ------------------------------------------------------------------ part 2: arm D2
def arm_D2(seed, s):
    _, nv, _ = E2._base_scm(seed)
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(E2.D_LATENT):
        specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), float(s * nv[i])))
    ds = SIM.simulate(E2.D_LATENT, E2.D_OBS, E2.N_BASE, specs, seed, mixing="linear",
                      edge_prob=E2.EDGE_PROB)
    int_envs = [(i, ds.environments[f"iv{i}"].X) for i in range(E2.D_LATENT)]
    return ds, int_envs


def d2_units(seed, s):
    """Per-unit records with E2.evaluate's projection, gate call and backbone."""
    ds, int_envs = arm_D2(seed, s)
    mu, Wp = BK.fit_pca(ds.environments["basis"].X, E2.D_PROJ)
    Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
    Y_int = [BK.project(X, mu, Wp) for (_, X) in int_envs]
    nodes = [i for (i, _) in int_envs]
    g = PR.detect_with_pvalues(Y_int, Y_obs, alpha=E2.ALPHA, B=E2.B_BOOT,
                               rng=np.random.default_rng(910_000 + seed), q=Q)
    W = BK.unmixing_rows(Y_obs, Y_int, nodes, E2.D_LATENT, readout="precision")
    Z_hat = Y_obs @ W.T
    obs_Z = ds.environments["obs"].Z
    rows = []
    for k, node in enumerate(nodes):
        ct = float(E2._abs_corr(Z_hat[:, node], obs_Z[:, node]))
        rows.append(dict(seed=seed, level=s, node=int(node), pvalue=float(g["pvalues"][k]),
                         bh_detect_q05=bool(g["bh_detect"][k]), corr_target=ct,
                         recovered=bool(ct > THR),
                         best_match_corr=float(E2._best_match_corr(Z_hat[:, node], obs_Z))))
    return rows


def run_d2(seeds, levels):
    E2.ARMS = {D2_NAME: dict(
        builder=arm_D2, levels=levels, level_name="s_relative",
        expect="relative weak signal: gate restricts while every intervention reduces "
               "its target's noise variance")}
    E2.SEEDS = list(seeds)
    report = E2.run(rule=RULE)
    arm = report["arms"][D2_NAME]
    units = [u for s in levels for seed in seeds for u in d2_units(seed, s)]
    consistent = True
    for lv in arm["levels"]:
        for j, rec in enumerate(lv["per_seed"]):
            cert = sorted(u["node"] for u in units if u["level"] == lv["level"]
                          and u["seed"] == seeds[j] and u["bh_detect_q05"])
            consistent &= (cert == rec["certified"])
    n = len(units)
    cert = [u for u in units if u["bh_detect_q05"]]
    pooled = dict(n_units=n, n_certified=len(cert),
                  coverage=len(cert) / n if n else None,
                  risk_target=(sum(not u["recovered"] for u in cert) / len(cert)
                               if cert else None),
                  recovered_total=int(sum(u["recovered"] for u in units)))
    return arm, units, pooled, bool(consistent), report["config"]


def evaluate_expectations(audit_ok, audit_got, arm, pooled):
    certified = [lv["certified_recovery_mean"] for lv in arm["levels"]
                 if lv["certified_recovery_mean"] is not None]
    x3 = bool(arm["crossover_gate_leaves_proceed"] in (0.7, 0.9)
              and all(c >= 0.90 for c in certified))
    d = {lv["level"]: lv["naive_mcc_mean"]
         for lv in json.loads(E2_BH.read_text())["arms"]["D_weak_signal_starvation"]["levels"]}
    d2 = {lv["level"]: lv["naive_mcc_mean"] for lv in arm["levels"]}
    matched = {s: dict(d2=d2.get(s), d=d.get(s)) for s in (0.7, 0.9)}
    x4 = bool(all(m["d2"] is not None and m["d"] is not None and m["d2"] > m["d"]
                  for m in matched.values()))
    return dict(
        X1_X2=dict(held=bool(audit_ok), got=audit_got, expected=EXPECTED_AUDIT),
        X3=dict(held=x3, gate_leaves_proceed=arm["crossover_gate_leaves_proceed"],
                certified_recovery_min=(min(certified) if certified else None)),
        X4=dict(held=x4, naive_mcc_matched=matched),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    started = time.time()
    seeds = list(range(10))

    scm = scm_assert(seeds)
    print("scm assert:", "ok" if all(scm.values()) else f"FAILED {scm}", flush=True)
    if not all(scm.values()):
        raise SystemExit("[stop] _base_scm differs from simulate's SCM")
    rows = audit(seeds)
    ok, got = audit_reproduces(rows)
    for r in rows:
        o, i = r["out_of_model_units"], r["in_model_units"]
        print(f"  D level {r['level']}: n {r['n']} raises {r['raises']} out_of_model "
              f"{r['out_of_model']} | out: certified {o['certified']}/{o['n']} recovered "
              f"{o['recovered']} | in: certified {i['certified']}/{i['n']} recovered "
              f"{i['recovered']}", flush=True)
    print("audit reproduces X1/X2:", ok, flush=True)
    if not ok:
        raise SystemExit(f"[stop] audit does not reproduce the pre-stated numbers: {got}")

    d2_seeds = [0] if args.selftest else seeds
    d2_levels = D2_LEVELS[:2] if args.selftest else D2_LEVELS
    arm, units, pooled, consistent, cfg = run_d2(d2_seeds, d2_levels)
    for lv in arm["levels"]:
        vc = lv["verdict_counts"]
        print(f"  D2 s={lv['level']}: gate {lv['gate_n_recoverable_mean']} "
              f"{lv['gate_n_recoverable_spread']} P/C/A {vc['PROCEED']}/"
              f"{vc['PROCEED_CAPPED']}/{vc['ABSTAIN']} naive {lv['naive_mcc_mean']} cert "
              f"{lv['certified_recovery_mean']} ({lv['n_seeds_certifying']} seeds)", flush=True)
    print(f"  D2 crossovers: gate leaves PROCEED at {arm['crossover_gate_leaves_proceed']}, "
          f"naive<0.90 at {arm['crossover_naive_below_0p90']}, {arm['gate_behavior']}; "
          f"pooled {pooled}; unit BH == evaluate certified: {consistent}", flush=True)
    if not consistent:
        raise SystemExit("[stop] per-unit BH decisions differ from evaluate's certified lists")
    if args.selftest:
        print(f"selftest ok (nothing written, {time.time() - started:.0f}s)")
        return

    expect = evaluate_expectations(ok, got, arm, pooled)
    prov = T2.provenance(__file__, extra_files=["experiments/e2_calibration.py",
                                                "sim/simulator.py"])
    T2.write_json(OUT / "arm_d_audit.json", dict(
        experiment="e11_arm_d_audit", source=str(UNITS.relative_to(REPO)),
        source_sha256=T2.sha256_file(UNITS),
        definitions=dict(parentless="is_source[node] from E2._base_scm(seed)",
                         raises="level >= noise_var[node] (absolute post-intervention "
                                "noise variance at or above the observational one)",
                         out_of_model="parentless and raises",
                         recovered="corr_target > 0.90"),
        scm_assert=scm, expected=EXPECTED_AUDIT, reproduces=ok, levels=rows,
        provenance=prov))
    T2.write_json(OUT / "arm_d2_report.json", dict(
        experiment="e11_arm_d2_relative_weak_signal",
        config=dict(e2_config=cfg, arm=D2_NAME, levels=D2_LEVELS, seeds=seeds,
                    intervention="hard; target noise variance s * noise_var[i]",
                    rule=RULE, q=Q, expectations=EXPECTATIONS),
        arm=arm, pooled_q05_target_aligned=pooled,
        unit_bh_matches_evaluate=consistent, expectations=expect, units=units,
        provenance=prov, wall_seconds=round(time.time() - started, 1)))
    print("expectations:", {k: v["held"] for k, v in expect.items()})
    print(f"written {OUT} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
