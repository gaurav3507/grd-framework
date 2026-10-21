"""D4: export the held-out T0h BH decisions in the schema e3_attribution_panel reads
(pre-registration: notes/d4_prereg.md).

For each dataset in (rpe1, norman, k562) and seed 0 to 2:
  results/d4_heldout/decisions/<ds>_seed<s>.json  with top-level primary_decision and
  per_perturbation = [{environment, n_samples, corrected_disjoint{bh_detect, pvalue}}]
  over the retained reals in E12 file order (p from the stored E12 T0h, bh_detect from
  d4_common.bh_sets, which asserts agreement with summary.json);
  results/d4_heldout/decisions/<ds>_union.json  bh_detect = detected in any seed,
  pvalue = min over seeds;
  results/d4_heldout/decisions_compact/<ds>_seed<s>.json  decisions[] with environment,
  bh_detect, pvalue (reference only).
Each file also carries results, results_sha256 and meta.
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d4_common as C  # noqa: E402

T2 = C.T2
OUT = C.OUT_ROOT / "decisions"
OUT_COMPACT = C.OUT_ROOT / "decisions_compact"
PRIMARY = "held-out T0h BH q=0.05 (E12, D4)"


def write(path, top, t0):
    doc = C.document(top, __file__, t0)
    doc = dict(top, **doc)
    T2.write_json(path, doc)
    print(f"-> {path}", flush=True)


def main():
    t0 = time.time()
    for ds in ("rpe1", "norman", "k562"):
        per_seed = {}
        for seed in C.SEEDS:
            b = C.bh_sets(ds, seed, "T0h")
            per_seed[seed] = b
            rows = [dict(environment=lab, n_samples=b["n_e"][lab],
                         corrected_disjoint=dict(bh_detect=b["decision"][lab], pvalue=b["pvalue"][lab]))
                    for lab in b["order"]]
            write(OUT / f"{ds}_seed{seed}.json",
                  dict(primary_decision=PRIMARY, dataset=ds, seed=seed, test="T0h",
                       n_detected=b["bh_count_real"], per_perturbation=rows), t0)
            write(OUT_COMPACT / f"{ds}_seed{seed}.json",
                  dict(primary_decision=PRIMARY, dataset=ds, seed=seed,
                       decisions=[dict(environment=lab, bh_detect=b["decision"][lab],
                                       pvalue=b["pvalue"][lab]) for lab in b["order"]]), t0)
        order = per_seed[0]["order"]
        for seed in C.SEEDS:
            if per_seed[seed]["order"] != order or per_seed[seed]["n_e"] != per_seed[0]["n_e"]:
                raise SystemExit(f"STOP: {ds}: retained environments or sizes differ across seeds")
        rows = [dict(environment=lab, n_samples=per_seed[0]["n_e"][lab],
                     corrected_disjoint=dict(
                         bh_detect=any(per_seed[s]["decision"][lab] for s in C.SEEDS),
                         pvalue=min(per_seed[s]["pvalue"][lab] for s in C.SEEDS)))
                for lab in order]
        write(OUT / f"{ds}_union.json",
              dict(primary_decision=PRIMARY + ", union over seeds 0-2 (pvalue = min over seeds)",
                   dataset=ds, seed="union", test="T0h",
                   n_detected=sum(r["corrected_disjoint"]["bh_detect"] for r in rows),
                   per_perturbation=rows), t0)


if __name__ == "__main__":
    main()
