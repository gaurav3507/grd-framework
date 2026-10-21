"""D4-C: subspace attribution on the held-out T0h decisions (pre-registration:
notes/d4_prereg.md).

Calls e3_attribution_panel.run_dataset unchanged (largest principal angle between top-10
subspaces in full gene space, disjoint null from all controls, B = 500,
SeedSequence([0, n_env])) with gate_path pointing at
results/d4_heldout/decisions/<ds>_<seed|union>.json (d4_decisions_export.py). Writes
results/d4_heldout/attribution/<ds>_<seed|union>.json; AP.main() is not called.

Threading: OMP/OpenBLAS/MKL threads are set from --threads before numpy is imported,
and neither e12_common nor d4_common is imported (e12_common pins threads to 1).
results_sha256 is computed with the same expression as e12_common.sha256_json.
K562 (no detected environment in any seed): a stub with n_upstream 0, no run.
"""

import os
import sys


def _threads(argv):
    return argv[argv.index("--threads") + 1] if "--threads" in argv else "4"


for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = _threads(sys.argv)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e3_attribution_panel as AP  # noqa: E402
import tier2_common as T2  # noqa: E402
from data_paths import perturbseq_path  # noqa: E402

FILES = {"rpe1": ("RPE1", "causalbench_rpe1.h5ad"), "norman": ("Norman", "Norman2019_raw.h5ad"),
         "k562": ("K562", "causalbench_k562.h5ad")}
DEC = AP.REPO / "results" / "d4_heldout" / "decisions"
OUT = AP.REPO / "results" / "d4_heldout" / "attribution"


def sha256_json(obj):
    """Same expression as e12_common.sha256_json."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(FILES))
    ap.add_argument("--seed", required=True, choices=["0", "1", "2", "union"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    tag = "union" if a.seed == "union" else f"seed{a.seed}"
    name, fname = FILES[a.dataset]
    gate_path = DEC / f"{a.dataset}_{tag}.json"
    if not gate_path.exists():
        raise SystemExit(f"STOP: decisions file missing: {gate_path}")
    rows = json.loads(gate_path.read_text())["per_perturbation"]
    n_up = sum(bool(r["corrected_disjoint"]["bh_detect"]) for r in rows)
    common = dict(experiment="D4-C", dataset=a.dataset, seed=a.seed, d_subspace=AP.D_SUBSPACE,
                  B=AP.B_BOOT, alpha=AP.ALPHA, q=AP.Q, gate_path=str(gate_path.relative_to(AP.REPO)),
                  limitation="full-space statistic without target correction (D4 prereg)")
    if a.dataset == "k562":
        if n_up != 0:
            raise SystemExit(f"STOP: K562 {tag} has {n_up} upstream detections; the stub assumes 0")
        results = dict(common, n_upstream=0, n_environments=len(rows), ran=False,
                       note="no held-out T0h BH-detected environment; reported as 0 upstream, no run")
    else:
        config = dict(name=name, kind="perturbseq", data_path=str(perturbseq_path(fname)),
                      gate_path=gate_path, control="")
        report = AP.run_dataset(config)
        results = dict(common, n_upstream=report["summary"]["n_upstream_bh_detected"],
                       ran=True, report=report)
    results = json.loads(json.dumps(results))
    doc = dict(results=results, results_sha256=sha256_json(results), meta=dict(
        seconds=round(time.time() - t0, 3), threads=a.threads, argv=sys.argv[1:],
        provenance=T2.provenance(__file__, ["experiments/e3_attribution_panel.py"])))
    path = OUT / f"{a.dataset}_{tag}.json"
    T2.write_json(path, doc)
    s = results.get("report", {}).get("summary", {})
    print(f"[{a.dataset} {tag}] upstream {results['n_upstream']} raw {s.get('n_subspace_raw_rejected')} "
          f"bh {s.get('n_subspace_bh_rejected')} seconds {doc['meta']['seconds']} -> {path}", flush=True)


if __name__ == "__main__":
    main()
