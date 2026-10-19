"""Element-wise comparison of two CausalBench rebuilds (diagnostic for E13 verify).

Compares, per dataset, root A vs root B:
  h5ad: <root>/discrepancy_vae/datasets/causalbench_<ds>.h5ad (X csr, obs guide_ids, var_names)
  npz:  <root>/raw/dataset_<ds>.npz (expression_matrix, interventions, var_names)
Reports identical yes/no, number of differing entries, max abs diff, max float32 ULP diff.

Usage:
  python -u experiments/diag_compare_rebuilds.py --a ~/grd_data --b ~/grd_data_py310 --dataset rpe1 --out f.json
"""
import argparse
import json
import os

import numpy as np
import anndata as ad


def ulp_diff32(x, y):
    """Max ULP distance between two float32 arrays (sign-magnitude to ordered ints)."""
    a = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.ascontiguousarray(y, dtype=np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(a & 0x7FFFFFFF), a)
    b = np.where(b < 0, -(b & 0x7FFFFFFF), b)
    return int(np.abs(a - b).max()) if a.size else 0


def dense_compare(A, B, chunk=20000):
    n_diff, max_abs, max_ulp = 0, 0.0, 0
    for i in range(0, A.shape[0], chunk):
        a = A[i:i + chunk]
        b = B[i:i + chunk]
        a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        ne = a != b
        n_diff += int(ne.sum())
        if ne.any():
            max_abs = max(max_abs, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
            if a.dtype == np.float32 and b.dtype == np.float32:
                max_ulp = max(max_ulp, ulp_diff32(a[ne], b[ne]))
    return dict(identical=n_diff == 0, n_differing=n_diff, max_abs_diff=max_abs,
                max_ulp_diff_float32=max_ulp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--dataset", required=True, choices=["k562", "rpe1"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ra, rb = os.path.expanduser(a.a), os.path.expanduser(a.b)
    ds = a.dataset
    out = {"dataset": ds, "a": ra, "b": rb}

    ha = ad.read_h5ad(os.path.join(ra, "discrepancy_vae", "datasets", f"causalbench_{ds}.h5ad"))
    hb = ad.read_h5ad(os.path.join(rb, "discrepancy_vae", "datasets", f"causalbench_{ds}.h5ad"))
    h = dict(shape_a=list(ha.shape), shape_b=list(hb.shape),
             dtype_a=str(ha.X.dtype), dtype_b=str(hb.X.dtype))
    if ha.shape == hb.shape:
        h.update(dense_compare(ha.X, hb.X))
        if hasattr(ha.X, "indptr") and hasattr(hb.X, "indptr"):
            h["csr_structure_identical"] = bool(np.array_equal(ha.X.indptr, hb.X.indptr)
                                                and np.array_equal(ha.X.indices, hb.X.indices))
    h["guide_ids_identical"] = bool(np.array_equal(np.asarray(ha.obs["guide_ids"]).astype(str),
                                                   np.asarray(hb.obs["guide_ids"]).astype(str)))
    h["obs_index_identical"] = bool(np.array_equal(ha.obs_names, hb.obs_names))
    h["var_names_identical"] = bool(np.array_equal(ha.var_names, hb.var_names))
    out["h5ad"] = h
    del ha, hb

    na = np.load(os.path.join(ra, "raw", f"dataset_{ds}.npz"), allow_pickle=True)
    nb = np.load(os.path.join(rb, "raw", f"dataset_{ds}.npz"), allow_pickle=True)
    Ea, Eb = na["expression_matrix"], nb["expression_matrix"]
    n = dict(shape_a=list(Ea.shape), shape_b=list(Eb.shape), dtype_a=str(Ea.dtype), dtype_b=str(Eb.dtype))
    if Ea.shape == Eb.shape:
        n.update(dense_compare(Ea, Eb))
    n["interventions_identical"] = bool(np.array_equal(np.asarray(na["interventions"]).astype(str),
                                                       np.asarray(nb["interventions"]).astype(str)))
    n["var_names_identical"] = bool(np.array_equal(np.asarray(na["var_names"]).astype(str),
                                                   np.asarray(nb["var_names"]).astype(str)))
    out["npz"] = n
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
