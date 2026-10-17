"""Verify rebuilt causalbench_{k562,rpe1}.h5ad against the A100 fingerprints.

Content targets come from CLAUDE.md "Dataset provenance" and
~/a100/precondition-audit/docs/DATA_SOURCES.md (control mean / RMS computed on
the CausalBench npz control rows, interventions == "non-targeting", float64).
Mean and RMS are computed twice: on the npz control rows and on the h5ad
control rows (guide_ids == ''). Both must agree with the target to 4 decimals.

--umi (K562 only, check-only): align the RAW k562.h5ad obs["UMI_count"] to the
rebuilt rows and save it to $GRD_DATA_ROOT/build/k562_umi_count_aligned.npy.
Not an experiment input.

Usage (GRD_DATA_ROOT must be set):
  python -u experiments/verify_causalbench_data.py --dataset rpe1
  python -u experiments/verify_causalbench_data.py --dataset k562 --umi --norman
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_paths import data_root, perturbseq_path  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = "non-targeting"

TARGETS = {
    "k562": {"shape": (240148, 1158), "n_controls": 10691, "n_unique_guides": 1159,
             "ctrl_mean": 0.8234, "ctrl_rms": 1.2599,
             "sha256": "006a1c1892f42d1edb8d3e8bcbdbb2f7d58d1611c0f22426768b20c60632ee4e",
             "bytes": 1319095768},
    "rpe1": {"shape": (139825, 651), "n_controls": 11485, "n_unique_guides": None,
             "ctrl_mean": 0.6142, "ctrl_rms": 1.0135,
             "sha256": "b14b0db0d755ad787e41b46e04695407db856d2cc55864716c4832801e495831",
             "bytes": 370700866},
}
NORMAN_SHA = "298871ec3fc8d34ed8bbc9fddd2f2c916f847a9d71195e0b63b7c3f89f570de7"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def mean_rms(M):
    M = np.asarray(M, dtype=np.float64)
    return float(M.mean()), float(np.sqrt((M * M).mean()))


def agree4(x, target):
    return round(x, 4) == round(target, 4)


def verify(ds, D):
    t = TARGETS[ds]
    path = str(perturbseq_path(f"causalbench_{ds}.h5ad"))
    a = ad.read_h5ad(path)
    g = np.asarray(a.obs["guide_ids"]).astype(str)
    ctrl = g == ""
    Xc = a.X[ctrl]
    Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
    h_mean, h_rms = mean_rms(Xc)
    X = a.X
    data = X.data if hasattr(X, "data") else np.asarray(X)
    x_min = float(data.min()) if data.size else 0.0
    if hasattr(X, "nnz") and X.nnz < X.shape[0] * X.shape[1]:
        x_min = min(x_min, 0.0)
    any_nan = bool(np.isnan(data).any())

    d = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    iv = np.asarray(d["interventions"]).astype(str)
    E = d["expression_matrix"]
    n_mean, n_rms = mean_rms(E[iv == CTRL])
    npz_ctrl = int((iv == CTRL).sum())
    del E

    e12 = json.load(open(os.path.join(
        REPO, "results", "e12_standard_tests", ds, "seed0_d10.json")))["results"]

    sha = sha256(path)
    nbytes = os.path.getsize(path)
    r = {
        "dataset": ds, "path": path,
        "shape": list(a.shape), "shape_target": list(t["shape"]),
        "shape_match": tuple(a.shape) == t["shape"],
        "n_controls": int(ctrl.sum()), "n_controls_target": t["n_controls"],
        "n_controls_match": int(ctrl.sum()) == t["n_controls"],
        "npz_n_controls": npz_ctrl,
        "n_unique_guides": int(len(np.unique(g))),
        "n_unique_guides_target": t["n_unique_guides"],
        "n_unique_guides_match": (t["n_unique_guides"] is None
                                  or int(len(np.unique(g))) == t["n_unique_guides"]),
        "ctrl_mean_npz": n_mean, "ctrl_rms_npz": n_rms,
        "ctrl_mean_h5ad": h_mean, "ctrl_rms_h5ad": h_rms,
        "ctrl_mean_target": t["ctrl_mean"], "ctrl_rms_target": t["ctrl_rms"],
        "mean_rms_match_4dp": all([agree4(n_mean, t["ctrl_mean"]),
                                   agree4(h_mean, t["ctrl_mean"]),
                                   agree4(n_rms, t["ctrl_rms"]),
                                   agree4(h_rms, t["ctrl_rms"])]),
        "X_dtype": str(X.dtype), "X_format": type(X).__name__,
        "X_min": x_min, "X_min_ge_0": x_min >= 0, "X_any_nan": any_nan,
        "sha256": sha, "sha256_target": t["sha256"], "sha256_match": sha == t["sha256"],
        "bytes": nbytes, "bytes_target": t["bytes"], "bytes_match": nbytes == t["bytes"],
        "e12_n_cells": e12["n_cells"], "e12_n_genes": e12["n_genes"],
        "e12_n_controls": e12["n_controls"],
        "e12_crosscheck": (e12["n_cells"] == a.shape[0] and e12["n_genes"] == a.shape[1]
                           and e12["n_controls"] == int(ctrl.sum())),
    }
    r["verdict"] = ("CONTENT MATCH" if (r["shape_match"] and r["n_controls_match"]
                                        and r["n_unique_guides_match"]
                                        and r["mean_rms_match_4dp"])
                    else "MISMATCH")
    return r


def umi_check(D, B, chunk=20000):
    raw = os.path.join(D, "k562.h5ad")
    a = ad.read_h5ad(raw, backed="r")
    has_umi = "UMI_count" in a.obs.columns
    n = a.shape[0]
    rs = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        blk = a.X[i:min(i + chunk, n)]
        rs[i:i + blk.shape[0]] = np.asarray(blk.sum(axis=1), dtype=np.float64).ravel()
        print(f"  row sums {min(i + chunk, n)}/{n}", flush=True)
    nz = rs >= 1
    n_zero = int((~nz).sum())

    d = np.load(os.path.join(D, "dataset_k562.npz"), allow_pickle=True)
    iv = np.asarray(d["interventions"]).astype(str)
    assert int(nz.sum()) == len(iv), (int(nz.sum()), len(iv))
    keep = iv != "excluded"

    gene_id = np.asarray(a.obs["gene_id"]).astype(str)[nz][keep]
    gene_id = np.where(gene_id == CTRL, "", gene_id)
    reb = ad.read_h5ad(str(perturbseq_path("causalbench_k562.h5ad")), backed="r")
    guide = np.asarray(reb.obs["guide_ids"]).astype(str)
    n_mis = int((gene_id != guide).sum()) if len(gene_id) == len(guide) else None
    aligned = (n_mis == 0)
    assert aligned, f"alignment failed: mismatches {n_mis}, lengths {len(gene_id)} {len(guide)}"

    r = {"UMI_count_exists": has_umi, "raw_rows": n, "zero_sum_rows_dropped": n_zero,
         "npz_rows": len(iv), "kept_rows": int(keep.sum()),
         "alignment_exact": aligned, "alignment_mismatches": n_mis}
    if has_umi:
        umi = np.asarray(a.obs["UMI_count"], dtype=np.float64)[nz][keep]
        rsk = rs[nz][keep]
        r.update({"UMI_count_min": float(umi.min()),
                  "UMI_count_median": float(np.median(umi)),
                  "UMI_count_max": float(umi.max()),
                  "corr_UMI_count_vs_raw_rowsum": float(np.corrcoef(umi, rsk)[0, 1])})
        out = os.path.join(B, "k562_umi_count_aligned.npy")
        np.save(out, umi)
        r["saved"] = out
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["k562", "rpe1"])
    ap.add_argument("--umi", action="store_true", help="K562 raw UMI_count check")
    ap.add_argument("--norman", action="store_true", help="Norman SHA-256 check")
    a = ap.parse_args()
    if "GRD_DATA_ROOT" not in os.environ:
        sys.exit("STOP: GRD_DATA_ROOT is not set")
    if a.umi and a.dataset != "k562":
        sys.exit("--umi is K562 only")
    root = data_root()
    D = str(root / "raw")
    B = str(root / "build")
    os.makedirs(B, exist_ok=True)

    import scipy, h5py, pandas, scanpy
    out = {"provenance": {"python": sys.version.split()[0], "numpy": np.__version__,
                          "scipy": scipy.__version__, "pandas": pandas.__version__,
                          "anndata": ad.__version__, "scanpy": scanpy.__version__,
                          "h5py": h5py.__version__}}
    out["verify"] = verify(a.dataset, D)
    print(json.dumps(out["verify"], indent=1), flush=True)
    if out["verify"]["verdict"] != "CONTENT MATCH":
        json.dump(out, open(os.path.join(B, f"verify_{a.dataset}.json"), "w"), indent=1)
        sys.exit(f"MISMATCH for {a.dataset}: stop")
    if a.norman:
        p = str(perturbseq_path("Norman2019_raw.h5ad"))
        s = sha256(p)
        out["norman"] = {"path": p, "sha256": s, "sha256_target": NORMAN_SHA,
                         "sha256_match": s == NORMAN_SHA}
        print(json.dumps(out["norman"], indent=1), flush=True)
    if a.umi:
        out["umi"] = umi_check(D, B)
        print(json.dumps(out["umi"], indent=1), flush=True)
    json.dump(out, open(os.path.join(B, f"verify_{a.dataset}.json"), "w"), indent=1)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
