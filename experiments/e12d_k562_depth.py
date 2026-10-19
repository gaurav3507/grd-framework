"""E12d: E12b for K562 with depth = log(raw per-cell UMI_count), amendment 2.

Pre-registration: notes/e12b_prereg.md, amended by notes/e12b_amendment_2.md.

Runs experiments/e12b_confound_check.py unchanged except for the depth source of K562:
instead of the X row-sum fallback (the h5ad has no library size), depth is the log of the
raw figshare obs["UMI_count"], aligned row for row to causalbench_k562.h5ad by the data
build (experiments/verify_causalbench_data.py, saved as
$GRD_DATA_ROOT/build/k562_umi_count_aligned.npy). Everything else (seed, d, B, REF draws,
fakes, R1 to R5, BH families, S1 to S3 rules) is e12b_confound_check.py's own code, with
one more change: the reproduction tolerance against the stored E12 T1 statistics is 1e-6
instead of 1e-8, because causalbench_k562.h5ad is the rebuilt file (E13 amendment 1:
statistics differ from the A100 file by about 3e-8, every p-value is unchanged). The
tolerance used is recorded in the output's reproduction block.

Outputs go under results/e12d_k562_depth/ (smoke under smoke/). --aggregate combines K562
from here, Norman from results/e12c_norman_depth/ and writes summary.json here; RPE1 is
reported only and keeps its E12b result. Same flags as e12b_confound_check.py; only
--dataset k562 is accepted for a run.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12b_confound_check as M  # noqa: E402
from data_paths import data_root  # noqa: E402

E = M.E
T2 = M.T2
OUT_ROOT = E.REPO / "results" / "e12d_k562_depth"
E12C_ROOT = E.REPO / "results" / "e12c_norman_depth"
UMI_FILE = "k562_umi_count_aligned.npy"
REPRO_TOL = 1e-6  # rebuilt K562 file, as E13 amendment 1; e12b's own value is 1e-8
SOURCE = "raw obs[UMI_count] (figshare 35773219), aligned external vector (amendment 2)"
_inspect_orig = M.inspect_h5ad


def umi_path():
    return data_root() / "build" / UMI_FILE


def inspect_k562(key, depth_source="auto"):
    """e12b's inspect_h5ad, with K562's depth replaced by the aligned raw UMI_count."""
    info, g, depth, batch, extra = _inspect_orig(key, "auto")
    if key != "k562":
        return info, g, depth, batch, extra
    path = umi_path()
    if not path.exists():
        raise SystemExit(f"STOP: aligned UMI_count vector missing: {path}")
    lib = np.load(path).astype(np.float64)
    if lib.shape != (len(g),):
        raise SystemExit(f"STOP: aligned UMI_count has shape {lib.shape}, h5ad has {len(g)} cells")
    if not np.all(np.isfinite(lib)) or np.min(lib) <= 0:
        raise SystemExit("STOP: aligned UMI_count has nonpositive or non-finite entries")
    info = dict(info, depth_source=SOURCE, depth_fallback=False, depth_transform="log(v)",
                umi_file=str(path), umi_count=dict(n=int(lib.size), min=float(lib.min()),
                                                   median=float(np.median(lib)), max=float(lib.max())))
    extra = dict(lib=lib, umi_obs=None)
    return info, g, np.log(lib), batch, extra


def aggregate_e12d():
    """K562 from E12d, Norman from E12c; RPE1 is missing by design (reported only)."""
    t0 = time.time()
    results = M.aggregate(E12C_ROOT, 0, 10, k562_from=OUT_ROOT)
    results["experiment"] = "E12d"
    results["prereg"] = "notes/e12b_prereg.md + notes/e12b_amendment_2.md"
    k = results["datasets"].get("k562")
    results["prediction_e12d"] = dict(
        k562_R3_retains_ge_70=None if k is None or k["retain_R3"] is None else k["retain_R3"] >= 0.70,
        k562_R4_retains_ge_70=None if k is None or k["retain_R4"] is None else k["retain_R4"] >= 0.70,
        k562_corr_pc1_depth_abs_gt_05=None if k is None else abs(k["corr_pc1_depth_basis"]) > 0.5)
    doc = dict(results=results, results_sha256=E.sha256_json(results), meta=dict(
        seconds=round(time.time() - t0, 3), provenance=T2.provenance(__file__)))
    T2.write_json(OUT_ROOT / "summary.json", doc)
    print(json.dumps(dict(verdict=results["verdict"], prediction_e12d=results["prediction_e12d"],
                          missing=results["missing_inputs"], sources=results.get("dataset_sources")),
                     indent=1))
    return doc


def main():
    argv = sys.argv[1:]
    if "--depth-source" in argv:
        raise SystemExit("STOP: E12d fixes the depth source; do not pass --depth-source")
    if "--oracles" in argv:
        raise SystemExit("STOP: do not run --oracles through E12d (it would overwrite e12b oracles)")
    M.OUT_ROOT = OUT_ROOT
    M.inspect_h5ad = inspect_k562
    M.REPRO_TOL = REPRO_TOL
    if "--aggregate" in argv:
        aggregate_e12d()
        return
    if "--dataset" in argv and argv[argv.index("--dataset") + 1] != "k562":
        raise SystemExit("STOP: E12d runs K562 only")
    M.main()


if __name__ == "__main__":
    main()
