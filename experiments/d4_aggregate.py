"""D4 aggregate: results/d4_heldout/summary.json (pre-registration: notes/d4_prereg.md).

Panel (E12 T0h per seed and mean; T0i fakes, in-sample, reported once as uncalibrated),
D4-T stability (per-seed BH fractions and pairwise Jaccard, T0h and T1), D4-A with the P2
verdict, D4-C with the P3 verdict, D4-I with the P4 verdict, P5 values, missing_inputs.
A prediction is "undetermined" when any of its inputs is missing.
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d4_common as C  # noqa: E402

T2 = C.T2
ROOT = C.OUT_ROOT
TAGS = ("seed0", "seed1", "seed2", "union")


def _load(path, missing):
    if not path.exists():
        missing.append(str(path.relative_to(C.E.REPO)))
        return None
    return json.loads(path.read_text())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else None


def verdict(ok, determined):
    return "undetermined" if not determined else ("held" if ok else "failed")


def main():
    t0 = time.time()
    missing = []
    summ = json.loads((C.E12_ROOT / "summary.json").read_text())["results"]["perturbseq"]

    panel, stability = {}, {}
    for ds in C.DATASETS:
        tests = summ[ds]["d10"]["tests"]
        per = {str(s): dict(bh_count_real=tests["T0h"]["per_seed"][str(s)]["bh_count_real"],
                            n_real=tests["T0h"]["per_seed"][str(s)]["n_real"],
                            bh_count_fake=tests["T0h"]["per_seed"][str(s)]["bh_count_fake"],
                            T0i_bh_count_fake=tests["T0i"]["per_seed"][str(s)]["bh_count_fake"],
                            T0i_raw_reject_fake=tests["T0i"]["per_seed"][str(s)]["raw_reject_fake"])
               for s in C.SEEDS}
        panel[ds] = dict(per_seed=per, T0h_mean_bh_count_real=float(np.mean(
            [v["bh_count_real"] for v in per.values()])))
        stability[ds] = {}
        for test in ("T0h", "T1"):
            sets = {s: C.bh_sets(ds, s, test)["labels"] for s in C.SEEDS}
            n = tests[test]["per_seed"]["0"]["n_real"]
            stability[ds][test] = dict(
                bh_frac_per_seed={str(s): len(v) / n for s, v in sets.items()},
                pairwise_jaccard={f"{a}-{b}": jaccard(sets[a], sets[b])
                                  for a, b in combinations(C.SEEDS, 2)},
                n_common_all_seeds=len(set.intersection(*(set(v) for v in sets.values()))))

    align = {}
    p2_hits, p2_det = 0, True
    for ds in C.DATASETS:
        align[ds] = {}
        for s in C.SEEDS:
            doc = _load(ROOT / "alignment" / f"{ds}_seed{s}.json", missing)
            if doc is None:
                if ds == "rpe1":
                    p2_det = False
                continue
            sets = doc["results"]["sets"]
            align[ds][str(s)] = {t: {k: sets[t].get(k) for k in (
                "n_detected", "observed_median", "baseline", "p_one_sided", "observed_exceeds_q95", "note")}
                for t in ("T0h", "T1")}
            if ds == "rpe1":
                p2_hits += int(bool(sets["T0h"].get("observed_exceeds_q95")))

    attr = {}
    p3_ok, p3_det = True, True
    for ds in ("rpe1", "norman", "k562"):
        attr[ds] = {}
        for tag in TAGS:
            doc = _load(ROOT / "attribution" / f"{ds}_{tag}.json", missing)
            if doc is None:
                if ds != "k562":
                    p3_det = False
                continue
            r = doc["results"]
            if not r["ran"]:
                attr[ds][tag] = dict(n_upstream=0, ran=False)
                continue
            rep = r["report"]
            up = [e for e in rep["environments"] if e["upstream_corrected_bh"]]
            row = dict(n_upstream=r["n_upstream"], ran=True,
                       n_subspace_raw_rejected=rep["summary"]["n_subspace_raw_rejected"],
                       n_subspace_bh_rejected=rep["summary"]["n_subspace_bh_rejected"],
                       min_subspace_p=min((e["subspace_pvalue"] for e in up), default=None),
                       seconds=doc["meta"]["seconds"])
            attr[ds][tag] = row
            if ds != "k562":
                p3_ok &= row["n_subspace_bh_rejected"] == 0

    # P4 per dataset and seed, three parts. D4 pre-run clarification: the Jaccard part is
    # evaluated only where the union of the calibrated set and the T0h set is non-empty;
    # otherwise jaccard is None and that part is "undetermined".
    ilcs = {}
    p4_parts = []
    for ds in C.DATASETS:
        ilcs[ds] = {}
        for s in C.SEEDS:
            doc = _load(ROOT / "ilcs" / f"{ds}_seed{s}.json", missing)
            if doc is None:
                p4_parts.append("undetermined")
                continue
            r = doc["results"]
            row = {k: r[k] for k in ("naive_fake_rate", "calib_fake_rate", "naive_real_rate",
                                     "calib_real_rate", "jaccard_calib_vs_heldout_T0h",
                                     "jaccard_union_empty", "n_calib_real", "n_gate_T0h",
                                     "ica_nonconverged_frac", "ica_nonconverged_count", "total_ica_fits")}
            jac = row["jaccard_calib_vs_heldout_T0h"]
            row["P4_parts"] = dict(
                naive_fake_eq_1="held" if row["naive_fake_rate"] == 1.0 else "failed",
                calib_fake_le_015="held" if row["calib_fake_rate"] <= 0.15 else "failed",
                jaccard_le_010="undetermined" if jac is None else ("held" if jac <= 0.10 else "failed"))
            ilcs[ds][str(s)] = row
            p4_parts.extend(row["P4_parts"].values())
    p4_verdict = ("failed" if "failed" in p4_parts
                  else "undetermined" if "undetermined" in p4_parts else "held")

    results = dict(
        experiment="D4", prereg="notes/d4_prereg.md", panel=panel, stability=stability,
        alignment=align, attribution=attr, ilcs=ilcs, missing_inputs=missing,
        predictions=dict(
            P2=dict(verdict=verdict(p2_hits >= 2, p2_det), rpe1_seeds_exceeding_q95=p2_hits),
            P3=dict(verdict=verdict(p3_ok, p3_det)),
            P4=dict(verdict=p4_verdict,
                    rule="failed if any determined part fails; else undetermined if any part is "
                         "undetermined (missing input or empty Jaccard union); else held"),
            P5=dict(rpe1_T0h_pairwise_jaccard=stability["rpe1"]["T0h"]["pairwise_jaccard"],
                    rpe1_T0h_n_common_all_seeds=stability["rpe1"]["T0h"]["n_common_all_seeds"])))
    doc = C.document(results, __file__, t0)
    T2.write_json(ROOT / "summary.json", doc)
    print(json.dumps(dict(predictions=results["predictions"], missing=missing), indent=1))


if __name__ == "__main__":
    main()
