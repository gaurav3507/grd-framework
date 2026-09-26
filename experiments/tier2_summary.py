"""Tier 2 artifact checks and the paste-ready summary table.

    python experiments/tier2_summary.py                  # print the summary table
    python experiments/tier2_summary.py --check e4_real:K562

--check validates one stage's artifact (exists, parses, expected shape) and prints
a one-line artifact count; exit 1 if it is missing or malformed. It judges the
artifact, never the scientific outcome. Used by experiments/run_tier2_a100.sh.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E4 = REPO / "results" / "e4_second_backbone"
E5 = REPO / "results" / "e5_split_control"
N_SEEDS_SYNTH = 10
N_SEEDS_REAL = 5


def _load(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ checks
def check(stage, root_e4=E4, root_e5=E5):
    """Return a one-line artifact description; raise on a bad artifact."""
    name, _, dataset = stage.partition(":")
    if name == "e4_calibration":
        d = _load(root_e4 / "e4_calibration_report.json")
        arms = d["arms"]
        assert len(arms) == 4, f"expected 4 arms, got {len(arms)}"
        for a in arms.values():
            assert len(a["levels"]) == 5
            assert all(len(lv["per_seed"]) == N_SEEDS_SYNTH for lv in a["levels"])
        assert "population_decomposition" in d and "precision_same_code" in d
        return (f"4 arms x 5 levels x {N_SEEDS_SYNTH} seeds; contract status "
                f"{d['status']}")
    if name == "e4_starvation":
        d = _load(root_e4 / "e4_starvation_report.json")
        assert d["result_eligible"] and d["status"] == "PASS"
        assert len(d["per_seed"]) == N_SEEDS_SYNTH
        return (f"{len(d['per_seed'])} seeds x {len(d['aggregate']['levels'])} levels; "
                f"restricts at or before MCC crossover: "
                f"{d['aggregate']['gate_restricts_at_or_before_mcc_crossover']}")
    if name in ("e4_real", "e5_real"):
        root, fname = ((root_e4, "e4_real_panel.json") if name == "e4_real"
                       else (root_e5, "e5_real_panel.json"))
        d = _load(root / fname)
        block = d["datasets"][dataset]
        assert not block["selftest"], "selftest block in a results artifact"
        assert len(block["per_seed"]) == N_SEEDS_REAL
        n = block.get("n_environments", block.get("n_perts"))
        return f"{dataset}: {N_SEEDS_REAL} seeds x {n} environments"
    if name == "e5_confound":
        d = _load(root_e5 / "e5_rpe1_confound.json")
        assert not d["selftest"]
        fams = d["shared_vs_split"]
        assert set(fams) == {"perturbations", "random_controls", "structured_controls"}
        return "3 families: " + ", ".join(
            f"{k} {v['n_environments']}" for k, v in fams.items())
    if name == "e5_poscontrol":
        d = _load(root_e5 / "e5_poscontrol.json")
        assert not d["selftest"] and len(d["dose_response"]) == 5
        return f"{len(d['dose_response']) + 1} signal ratios (dose response + ratio 8)"
    raise SystemExit(f"unknown stage {stage!r}")


# ------------------------------------------------------------------ summary
def _jaccard(a, b):
    a, b = set(a), set(b)
    return None if not (a | b) else len(a & b) / len(a | b)


def _fmt_counts(values):
    mean = sum(values) / len(values)
    return f"{mean:6.1f} [{min(values)}-{max(values)}]"


def summary():
    lines = ["", "TIER 2 SUMMARY (real panel: mean over gate seeds [min-max])", ""]
    header = (f"{'dataset':8} {'readout':11} {'design':17} {'n_env':>5}  "
              f"{'raw':>14}  {'BH':>14}  {'BH Jaccard vs shared precision'}")
    lines += [header, "-" * len(header)]
    e4 = _load(E4 / "e4_real_panel.json")["datasets"] if (E4 / "e4_real_panel.json").exists() else {}
    e5 = _load(E5 / "e5_real_panel.json")["datasets"] if (E5 / "e5_real_panel.json").exists() else {}
    for ds in ("K562", "RPE1", "Norman"):
        if ds in e5:
            per = e5[ds]["shared_vs_split"]["per_seed"]
            n = e5[ds]["n_perts"]
            lines.append(f"{ds:8} {'precision':11} {'shared (E3)':17} {n:>5}  "
                         f"{_fmt_counts([e['shared']['raw_count'] for e in per])}  "
                         f"{_fmt_counts([e['shared']['bh_count'] for e in per])}  (reference)")
            jac = e5[ds]["shared_vs_split"]["summary"]["jaccard_bh_mean"]
            lines.append(f"{'':8} {'precision':11} {'split control':17} {n:>5}  "
                         f"{_fmt_counts([e['split']['raw_count'] for e in per])}  "
                         f"{_fmt_counts([e['split']['bh_count'] for e in per])}  "
                         f"{'both empty' if jac is None else f'{jac:.3f}'}")
        if ds in e4:
            b = e4[ds]
            if ds in e5:
                # Same run, same data and seeds: E5's shared design is the
                # precision gate, so compare against its BH sets.
                ref = {r["seed"]: r["bh_selected_ids"]["corrected_disjoint"]
                       for r in e5[ds]["per_seed"]}
                jac = [_jaccard(r["bh_selected_ids"], ref[r["seed"]])
                       for r in b["per_seed"] if r["seed"] in ref]
            else:
                jac = [r["jaccard_bh_vs_precision"]["e3_stability_same_seed"]
                       for r in b["per_seed"]]
            jtxt = ("both empty" if all(j is None for j in jac) else
                    f"{sum(j for j in jac if j is not None) / sum(j is not None for j in jac):.3f}")
            lines.append(f"{'' if ds in e5 else ds:8} {'covariance':11} "
                         f"{'shared':17} {b['n_environments']:>5}  "
                         f"{_fmt_counts(b['raw_count_per_seed'])}  "
                         f"{_fmt_counts(b['bh_count_per_seed'])}  {jtxt}")
    if not (e4 or e5):
        lines.append("(no real-panel artifacts yet)")

    conf_path = E5 / "e5_rpe1_confound.json"
    if conf_path.exists():
        c = _load(conf_path)["shared_vs_split"]
        lines += ["", "RPE1 control splits (seed 0), raw/BH of n: shared -> split"]
        for fam, e in c.items():
            lines.append(f"  {fam:20} {e['shared']['raw_count']}/{e['shared']['bh_count']}"
                         f" -> {e['split']['raw_count']}/{e['split']['bh_count']} of "
                         f"{e['n_environments']}  (BH Jaccard {e['jaccard_bh']})")

    pc_path = E5 / "e5_poscontrol.json"
    if pc_path.exists():
        p = _load(pc_path)["shared_vs_split"]
        lines += ["", "Positive control, BH-selected planted of 3 (null draws BH of 30): "
                  "shared -> split"]
        for snr in sorted(p, key=float):
            e = p[snr]
            lines.append(f"  ratio {float(snr):>4g}: planted {e['planted']['shared']['bh_count']}"
                         f" -> {e['planted']['split']['bh_count']} | null "
                         f"{e['null_draws']['shared']['bh_count']} -> "
                         f"{e['null_draws']['split']['bh_count']}")

    cal_path = E4 / "e4_calibration_report.json"
    if cal_path.exists():
        cal = _load(cal_path)
        lines += ["", "Backbone B synthetic arms (covariance vs same-code precision)"]
        for arm, cmp in cal["comparison"].items():
            lines.append(f"  {arm:30} {cmp['covariance']['gate_behavior']:20} vs "
                         f"{cmp['precision']['gate_behavior']}")
        dec = cal["population_decomposition"]
        lines.append(f"  covariance rule MCC: population {dec['covariance_rule']['population']['mean']}"
                     f", sample {dec['covariance_rule']['sample']['mean']} | precision rule: "
                     f"population {dec['precision_rule']['population']['mean']}, sample "
                     f"{dec['precision_rule']['sample']['mean']}")
    st_path = E4 / "e4_starvation_report.json"
    if st_path.exists():
        a = _load(st_path)["aggregate"]
        lines.append(f"  starvation: gate first restricts at m={a['crossover_gate_first_restricts']},"
                     f" MCC < 0.90 from m={a['crossover_random_subset_mcc_below_0p90']}, "
                     f"restricts at or before: {a['gate_restricts_at_or_before_mcc_crossover']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", default=None)
    args = p.parse_args()
    if args.check:
        try:
            print(check(args.check))
        except SystemExit:
            raise
        except Exception as exc:  # missing file, bad JSON, wrong shape
            print(f"bad artifact: {type(exc).__name__}: {exc}")
            sys.exit(1)
        return
    print(summary())


if __name__ == "__main__":
    main()
