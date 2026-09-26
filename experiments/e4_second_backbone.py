"""E4 (Tier 2, Experiment A): the Gate-Recover-Discover contract with a second backbone.

Backbone B swaps the precision-difference statistic for a covariance-difference one
and changes nothing else:

    gate statistic  T_e = lambda_max( Cov(Y_0) - Cov(Y_e) )
                    (rank_readout.covariance_difference with the sign flipped, so a
                    variance REDUCTION is the leading eigenvalue)
    recovery rule   w_i = v_max( Cov(Y_0) - Cov(Y_e(i)) ),  Z_i = Y_0 w_i
    null            the same size-matched construction as the precision gate
                    (precision_readout._precision_null_values, readout="covariance")
    BH, d_rec, verdict, Discover   unchanged, reused.

Every run goes through the existing entry points with readout="covariance"; no
experiment logic is copied:

    calibration  the four E2 arms (experiments/e2_calibration.py run()), 10 seeds.
                 The same arms are also run with the precision readout on the same
                 code, so the comparison is not against the historical E2 snapshot
                 (results/e2 was written before the size-matched null fix and
                 differs from current code at a few borderline cells).
    starvation   the E2c random-subset starvation series
                 (experiments/e2c_starvation.py evaluate_seed/aggregate), 10 seeds.
                 Committed results/e2c is reproduced exactly by current code, so it
                 is the precision reference.
    real         the Backbone B gate only (no recovery) on K562 / RPE1 / Norman,
                 corrected disjoint null, BH within family, NMIN=200, d_proj=10,
                 five gate seeds, plus the E3 random pure-control family.

PRE-REGISTERED EXPECTATION (from the Tier 2 brief, written before running): Backbone
B's gate restricts before its own recovery MCC crosses 0.90 under starvation,
abstains under power and weak-signal starvation, and is fooled by measurement
contamination. If Backbone B behaves differently in any arm, that is recorded as a
finding. Nothing is tuned. A contract difference never makes this script exit
non-zero; only a crash or a malformed artifact does.

Usage:
    python experiments/e4_second_backbone.py calibration
    python experiments/e4_second_backbone.py starvation
    python experiments/e4_second_backbone.py real --dataset k562
    python experiments/e4_second_backbone.py real --dataset k562 --selftest --out-dir /tmp/x
"""

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np

import tier2_common as T2

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e4_second_backbone"
READOUT = "covariance"
BACKBONE_B = dict(
    readout=READOUT,
    gate_statistic="lambda_max(Cov(Y_0) - Cov(Y_e))",
    recovery_rule="w_i = v_max(Cov(Y_0) - Cov(Y_e(i))), Z_i = Y_0 w_i",
    null="size-matched; disjoint split when n_e <= n_0/2, else two-resample",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ calibration
def _arm_contract(arm):
    """Did this arm keep the precision-gate contract? Descriptive, from the numbers."""
    return dict(
        gate_behavior=arm["gate_behavior"],
        silent_failure=bool(arm["silent_failure"]),
        certification_trustworthy=bool(arm["certification_trustworthy"]),
        crossover_gate_leaves_proceed=arm["crossover_gate_leaves_proceed"],
        crossover_naive_below_0p90=arm["crossover_naive_below_0p90"],
    )


def population_decomposition(E2):
    """Spec-versus-implementation split for both row rules at the E2 reference point.

    Arm A with all five environments (n=2000, iv_scale=0.1), seeds 0-9. Each rule
    is applied twice: to the EXACT population covariances of the projected data
    (Cov(Y_e) = R Sigma_e R^T + sd_obs^2 I, R = W_pca^T A), and to the sample
    covariances, as the backbone does. Both unmixings are scored with the same E0
    Hungarian MCC on the same observational sample. Population below the 0.90 bar
    means the rule itself falls short (a specification property); sample far below
    population would mean an estimation or implementation problem.
    """
    SIM, BK, E0 = E2.SIM, E2.BK, E2.E0
    d = E2.D_LATENT
    rows = []
    for seed in E2.SEEDS:
        ds, int_envs = E2.arm_A(seed, d)
        mu, Wp = BK.fit_pca(ds.environments["basis"].X, E2.D_PROJ)
        R = Wp.T @ ds.A
        noise = ds.sd_obs ** 2 * np.eye(E2.D_PROJ)
        C0 = R @ SIM.population_latent_cov(ds.B, ds.noise_var) @ R.T + noise
        Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
        Z = ds.environments["obs"].Z
        W = {k: np.zeros((d, d)) for k in ("pop_cov", "pop_prec", "smp_cov", "smp_prec")}
        for i, X in int_envs:
            Ce = R @ SIM.population_latent_cov(
                ds.B, ds.noise_var, "hard", (i,), E2.IV_BASE) @ R.T + noise
            vals, vecs = np.linalg.eigh(C0 - Ce)
            W["pop_cov"][i] = vecs[:, int(np.argmax(vals))]
            vals, vecs = np.linalg.eigh(np.linalg.inv(Ce) - np.linalg.inv(C0))
            W["pop_prec"][i] = vecs[:, int(np.argmax(vals))]
            Y_env = BK.project(X, mu, Wp)
            W["smp_cov"][i] = BK._unmixing_row(Y_obs, Y_env, readout="covariance")
            W["smp_prec"][i] = BK._unmixing_row(Y_obs, Y_env, readout="precision")
        rows.append(dict(seed=seed, **{k: round(float(E0.mcc(Y_obs @ w.T, Z)), 4)
                                       for k, w in W.items()}))

    def stat(key):
        v = [r[key] for r in rows]
        return dict(mean=round(float(np.mean(v)), 4), min=min(v), max=max(v))
    return dict(
        setting="arm A, m=5, n=2000, iv_scale=0.1, seeds 0-9",
        covariance_rule=dict(population=stat("pop_cov"), sample=stat("smp_cov")),
        precision_rule=dict(population=stat("pop_prec"), sample=stat("smp_prec")),
        per_seed=rows,
    )


def run_calibration(out_dir):
    E2 = _load(REPO / "experiments" / "e2_calibration.py", "grd_e2_for_e4")
    started = time.time()
    report = E2.run(readout=READOUT)
    precision = E2.run(readout="precision")
    decomposition = population_decomposition(E2)

    behaviors = {n: a["gate_behavior"] for n, a in report["arms"].items()}
    precision_behaviors = {n: a["gate_behavior"] for n, a in precision["arms"].items()}
    tracking = [n for n, b in behaviors.items()
                if b in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")]
    fooled = [n for n, b in behaviors.items() if b == "FOOLED"]
    silent = [n for n, a in report["arms"].items() if a["silent_failure"]]
    differs = [n for n in behaviors if behaviors[n] != precision_behaviors[n]]

    power = report["arms"]["B_power_starvation"]
    weak = report["arms"]["D_weak_signal_starvation"]
    contam = report["arms"]["C_measurement_contamination"]
    expectation = dict(
        power_starvation_restricts_without_silent_failure=bool(
            power["gate_behavior"] in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")
            and not power["silent_failure"]),
        weak_signal_restricts_without_silent_failure=bool(
            weak["gate_behavior"] in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")
            and not weak["silent_failure"]),
        measurement_contamination_fools_gate=bool(
            contam["gate_behavior"] == "FOOLED"),
    )

    report["milestone"] = "E4"
    report["experiment"] = "e4_second_backbone_calibration"
    report["config"]["readout"] = READOUT
    report["config"]["backbone"] = BACKBONE_B
    # "status" keeps its E2 meaning (PASS unless no arm tracks or an arm fails
    # silently). A FAIL here is a finding about Backbone B, so the script still
    # exits 0; the launcher checks the artifact, not the verdict.
    report["central_claim"] = (
        f"Backbone B (covariance readout) on the E2 arms: behaviors {behaviors}; "
        f"tracking arms {tracking}; fooled arms {fooled}; silent-failure arms "
        f"{silent}. Same-code precision behaviors {precision_behaviors}. Arms "
        f"where Backbone B differs from precision: {differs} (recorded, not tuned).")
    report["preregistered_expectation"] = dict(
        statement=("gate abstains or caps under power and weak-signal starvation "
                   "and is fooled by measurement contamination, as the precision "
                   "gate is"),
        observed=expectation,
        all_held=bool(all(expectation.values())),
    )
    report["precision_same_code"] = dict(
        note=("precision readout run in this job on the same code; use this, not "
              "results/e2, for the precision-vs-covariance comparison"),
        contract_status=precision["status"],
        arms={n: _arm_contract(a) for n, a in precision["arms"].items()},
        level_means={
            n: [dict(level=lv["level"],
                     gate_n_recoverable_mean=lv["gate_n_recoverable_mean"],
                     naive_mcc_mean=lv["naive_mcc_mean"],
                     certified_recovery_mean=lv["certified_recovery_mean"])
                for lv in a["levels"]]
            for n, a in precision["arms"].items()},
    )
    report["comparison"] = {
        n: dict(covariance=_arm_contract(report["arms"][n]),
                precision=_arm_contract(precision["arms"][n]),
                same_behavior=behaviors[n] == precision_behaviors[n])
        for n in behaviors
    }
    report["population_decomposition"] = decomposition
    report["provenance"] = T2.provenance(
        __file__, extra_files=["experiments/e2_calibration.py"])
    report["wall_seconds"] = round(time.time() - started, 1)
    out = Path(out_dir) / "e4_calibration_report.json"
    T2.write_json(out, report)

    print(f"\nE4 CALIBRATION (Backbone B, readout={READOUT})", flush=True)
    for n in behaviors:
        a = report["arms"][n]
        print(f"  {n:30s} covariance={behaviors[n]:20s} precision="
              f"{precision_behaviors[n]:20s} gate leaves PROCEED at "
              f"{a['crossover_gate_leaves_proceed']}, naive<0.90 at "
              f"{a['crossover_naive_below_0p90']}", flush=True)
        for lv in a["levels"]:
            vc = lv["verdict_counts"]
            print(f"      level {str(lv['level']):>6}: gate n_rec {lv['gate_n_recoverable_mean']:>4} "
                  f"{lv['gate_n_recoverable_spread']}  P/C/A {vc['PROCEED']}/"
                  f"{vc['PROCEED_CAPPED']}/{vc['ABSTAIN']}  naive MCC "
                  f"{lv['naive_mcc_mean']:.4f}  cert {lv['certified_recovery_mean']}",
                  flush=True)
    print(f"  expectation held: {expectation}", flush=True)
    for rule in ("covariance_rule", "precision_rule"):
        dd = decomposition[rule]
        print(f"  {rule}: MCC with population covariances {dd['population']['mean']} "
              f"| with sample covariances {dd['sample']['mean']}", flush=True)
    print(f"written {out.relative_to(REPO) if out.is_relative_to(REPO) else out} "
          f"({report['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ starvation
def run_starvation(out_dir, smoke=False):
    E2C = _load(REPO / "experiments" / "e2c_starvation.py", "grd_e2c_for_e4")
    seeds = E2C.SMOKE_SEEDS if smoke else E2C.FULL_SEEDS
    B = E2C.SMOKE_BOOT if smoke else E2C.FULL_BOOT
    started = time.time()
    rows = []
    for seed in seeds:
        rows.append(E2C.evaluate_seed(seed, B, readout=READOUT))
        print(f"[seed {seed}] done ({time.time() - started:.1f}s)", flush=True)
    agg = E2C.aggregate(rows)
    wiring_pass = bool(
        agg["estimator_checks"]["full_rank_all"]
        and agg["estimator_checks"]["zero_or_rank_deficient_output_count"] == 0
        and np.isfinite([r["random_subset_control"]["mcc_mean"]
                         for r in agg["levels"]]).all())

    reference = {}
    ref_path = REPO / "results" / "e2c" / "starvation_report.json"
    if ref_path.exists() and not smoke:
        import json
        ref = json.loads(ref_path.read_text())["aggregate"]
        reference = dict(
            source=str(ref_path.relative_to(REPO)),
            sha256=T2.sha256_file(ref_path),
            note="precision readout; reproduced exactly by current code",
            crossover_random_subset_mcc_below_0p90=ref[
                "crossover_random_subset_mcc_below_0p90"],
            crossover_gate_first_restricts=ref["crossover_gate_first_restricts"],
            gate_restricts_at_or_before_mcc_crossover=ref[
                "gate_restricts_at_or_before_mcc_crossover"],
            random_subset_mcc_mean=[r["random_subset_control"]["mcc_mean"]
                                    for r in ref["levels"]],
        )

    report = dict(
        experiment="e4_second_backbone_starvation",
        mode="smoke" if smoke else "full",
        result_eligible=not smoke,
        status="PASS" if wiring_pass else "FAIL",
        scientific_conclusion=(
            "Reduced-power wiring check only; no paper conclusion." if smoke else
            "See measured crossovers; a difference from the precision backbone is "
            "a finding, not tuned."),
        config=dict(
            seeds=seeds, B=B, alpha=E2C.ALPHA, d_latent=E2C.D_LATENT,
            D_observed=E2C.D_OBS, d_projected=E2C.D_PROJ,
            n_per_environment=E2C.N_BASE, iv_scale=E2C.IV_BASE,
            edge_prob=E2C.EDGE_PROB, starvation_levels=E2C.STARVATION_LEVELS,
            max_random_subsets=E2C.MAX_SUBSETS, mcc_threshold=E2C.MCC_THRESHOLD,
            readout=READOUT, backbone=BACKBONE_B),
        estimator=(
            "Backbone B covariance rows for supplied targets plus the unchanged E2c "
            "absolute-eigenvalue spectral completion of the pooled "
            "interventional-minus-control covariance in their orthogonal complement."),
        primary_metric="Hungarian MCC; permutation and sign safe, axis-sensitive.",
        preregistered_expectation=dict(
            statement="gate restricts no later than the first m where mean "
                      "random-subset MCC falls below 0.90",
            observed=agg["gate_restricts_at_or_before_mcc_crossover"]),
        precision_reference=reference or None,
        aggregate=agg,
        per_seed=rows,
        provenance=T2.provenance(
            __file__, extra_files=["experiments/e2c_starvation.py"]),
        wall_seconds=round(time.time() - started, 1),
    )
    name = "e4_starvation_smoke.json" if smoke else "e4_starvation_report.json"
    out = Path(out_dir) / name
    T2.write_json(out, report)
    print("\nE4 STARVATION (Backbone B)", flush=True)
    print("m  gate-n  MCC-mean  MCC-seed-SD  verdicts(P/C/A)", flush=True)
    for row in agg["levels"]:
        c = row["random_subset_control"]
        vc = c["verdict_counts"]
        print(f"{row['m']:d}  {c['gate_n_recoverable_mean']:6.2f}  {c['mcc_mean']:8.4f}"
              f"  {c['mcc_seed_sd']:11.4f}  {vc['PROCEED']}/{vc['PROCEED_CAPPED']}/"
              f"{vc['ABSTAIN']}", flush=True)
    print(f"crossover: gate first restricts at m={agg['crossover_gate_first_restricts']}; "
          f"MCC first below 0.90 at m={agg['crossover_random_subset_mcc_below_0p90']}; "
          f"restricts at or before: {agg['gate_restricts_at_or_before_mcc_crossover']}",
          flush=True)
    print(f"status: {report['status']}  written {out}", flush=True)
    if not wiring_pass:
        raise SystemExit(1)


# ------------------------------------------------------------------ real panel
def run_real(key, out_dir, seeds, B, selftest=False):
    started = time.time()
    panel = T2.make_selftest_panel(key) if selftest else T2.load_panel(key)
    name = panel["name"]
    Xc, perts = panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]
    sizes = [int(len(Y)) for Y in Yperts]
    print(f"{name}: control {len(Xc)} | powered perts {len(perts)}", flush=True)

    stab = None if selftest else T2.e3_stability_sets(name)
    dec = None if selftest else T2.e3_decision_set(name)

    per_seed = []
    for seed in seeds:
        res = T2.screen(Yperts, Yobs, seed, readout=READOUT, B=B)
        fake_labels, Yfake = T2.random_controls(Xc, proj, sizes, seed, T2.N_FAKE)
        fres = T2.screen(Yfake, Yobs, T2.FAKE_SEED_OFFSET + seed,
                         readout=READOUT, B=B)
        bh_ids = T2.selected(perts, res["bh_detect"])
        raw_ids = T2.selected(perts, res["raw_detect"])
        precision_bh = (stab["per_seed"].get(seed, {}).get("bh")
                        if stab else None)
        per_seed.append(dict(
            seed=seed,
            perturbation_screen=T2.summarize(res),
            negative_control_screen=T2.summarize(fres),
            bh_selected_ids=bh_ids,
            raw_selected_ids=raw_ids,
            jaccard_bh_vs_precision=dict(
                e3_decisions_seed0=(T2.jaccard(bh_ids, dec["bh"]) if dec else None),
                e3_stability_same_seed=(T2.jaccard(bh_ids, precision_bh)
                                        if precision_bh is not None else None),
                precision_bh_count_same_seed=(len(precision_bh)
                                              if precision_bh is not None else None),
            ),
            per_perturbation=T2.design_records(perts, Yperts, {READOUT: res}),
            negative_control_records=T2.design_records(
                fake_labels, Yfake, {READOUT: fres}),
        ))
        print(f"  seed {seed}: raw {res['raw_count']}/{len(perts)}  BH "
              f"{res['bh_count']}  | random controls raw {fres['raw_count']}/"
              f"{T2.N_FAKE} BH {fres['bh_count']}  ({time.time() - started:.0f}s)",
              flush=True)

    bh_sets = [set(r["bh_selected_ids"]) for r in per_seed]
    block = dict(
        dataset=name,
        source=panel["source"],
        n_cells=panel["n_cells"],
        n_genes=panel["n_genes"],
        control_label=panel["control_label"],
        n_control=int(len(Xc)),
        n_environments=len(perts),
        seeds=list(seeds),
        raw_count_per_seed=[r["perturbation_screen"]["raw_count"] for r in per_seed],
        bh_count_per_seed=[r["perturbation_screen"]["bh_count"] for r in per_seed],
        bh_selected_ids_per_seed={str(r["seed"]): r["bh_selected_ids"]
                                  for r in per_seed},
        bh_selected_all_seeds=sorted(set.intersection(*bh_sets)) if bh_sets else [],
        bh_selected_any_seed=sorted(set.union(*bh_sets)) if bh_sets else [],
        raw_fraction=T2.seed_sd([r["perturbation_screen"]["raw_fraction"]
                                 for r in per_seed]),
        bh_fraction=T2.seed_sd([r["perturbation_screen"]["bh_fraction"]
                                for r in per_seed]),
        negative_control_raw_count_per_seed=[
            r["negative_control_screen"]["raw_count"] for r in per_seed],
        negative_control_bh_count_per_seed=[
            r["negative_control_screen"]["bh_count"] for r in per_seed],
        precision_reference=dict(
            e3_decisions=({k: dec[k] for k in ("source", "sha256")}
                          | dict(bh_count=len(dec["bh"]),
                                 labels_match=dec["labels"] == perts)) if dec else None,
            e3_stability=({k: stab[k] for k in ("source", "sha256")}
                          | dict(labels_match=all(
                              v["labels"] == perts for v in stab["per_seed"].values())))
            if stab else None,
        ),
        selftest=bool(selftest),
        per_seed=per_seed,
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    top = dict(
        experiment="e4_second_backbone_real_panel",
        description=("Backbone B gate only (no recovery): covariance readout, "
                     "corrected disjoint null, BH within each family, raw alpha "
                     "decisions kept for audit."),
        config=dict(readout=READOUT, backbone=BACKBONE_B, d_proj=T2.D_PROJ,
                    nmin=T2.NMIN, alpha=T2.ALPHA, q=T2.Q, B=B,
                    null="corrected_disjoint", design="shared_reference",
                    seeds=list(seeds), n_random_controls=T2.N_FAKE,
                    bh_families=("BH applied separately to powered perturbations "
                                 "and random controls, within each seed"),
                    jaccard_convention="None when both BH sets are empty"),
    )
    out = Path(out_dir) / "e4_real_panel.json"
    T2.update_dataset_block(out, top, name, block)
    print(f"written {out} [{name}] ({block['wall_seconds']}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="part", required=True)
    sub.add_parser("calibration").add_argument("--out-dir", default=str(RESULTS))
    st = sub.add_parser("starvation")
    st.add_argument("--out-dir", default=str(RESULTS))
    st.add_argument("--smoke", action="store_true",
                    help="three seeds, B=100; writes e4_starvation_smoke.json")
    rl = sub.add_parser("real")
    rl.add_argument("--dataset", choices=sorted(T2.DATASETS), required=True)
    rl.add_argument("--out-dir", default=str(RESULTS))
    rl.add_argument("--selftest", action="store_true",
                    help="synthetic panel, code path only; needs --out-dir")
    rl.add_argument("--seeds", type=int, nargs="+", default=T2.SEEDS)
    rl.add_argument("--B", type=int, default=T2.B_BOOT)
    return p.parse_args()


def main():
    args = parse_args()
    if getattr(args, "selftest", False) and Path(args.out_dir).resolve() == RESULTS:
        raise SystemExit("--selftest must not write under results/; pass --out-dir")
    if args.part == "calibration":
        run_calibration(args.out_dir)
    elif args.part == "starvation":
        run_starvation(args.out_dir, smoke=args.smoke)
    else:
        run_real(args.dataset, args.out_dir, args.seeds, args.B,
                 selftest=args.selftest)


if __name__ == "__main__":
    main()
