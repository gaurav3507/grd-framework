"""Rebuild causalbench_{k562,rpe1}.h5ad from the figshare raw h5ad files.

Two stages, one dataset per process:
  a. CausalBench commit 1a2143cf, unfiltered: CreateDataset(RAW, filter=False)
     .preprocess_and_save(path, None, "dataset_<ds>") -> RAW/dataset_<ds>.npz.
     Same code path as CreateDataset.load() (A100 02_preprocess.py, filter=False).
  b. The logic of the A100 meridian-identifiability/framework/scripts/
     20_to_anndata.py for that dataset, line for line, with only the paths changed.

Paths resolve from GRD_DATA_ROOT (must be set):
  RAW = $GRD_DATA_ROOT/raw, OUT = $GRD_DATA_ROOT/discrepancy_vae/datasets,
  SPL = $GRD_DATA_ROOT/build/splits.
CausalBench is imported from PYTHONPATH (the pinned checkout).

Usage:
  PYTHONPATH=$GRD_DATA_ROOT/build/causalbench \
      python -u experiments/build_causalbench_data.py --dataset rpe1
"""
import argparse
import json
import os
import sys

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_paths import data_root, perturbseq_path  # noqa: E402

CB_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"


def build_npz(ds, D):
    from causalscbench.data_access.create_dataset import CreateDataset
    from causalscbench.data_access.datasets import download_weissmann

    if not hasattr(sc.pp, "normalize_per_cell"):
        sys.exit("STOP: sc.pp.normalize_per_cell missing in this scanpy")
    raw = os.path.join(D, f"{ds}.h5ad")
    if not os.path.exists(raw):
        sys.exit(f"STOP: raw file missing: {raw} (would trigger a download)")
    path = getattr(download_weissmann, f"download_weissmann_{ds}")(D)
    print(f"raw h5ad: {path}", flush=True)
    out = CreateDataset(D, filter=False).preprocess_and_save(
        path, None, f"dataset_{ds}")
    print(f"npz: {out}", flush=True)
    return out


def to_anndata(ds, D, OUT, SPL):
    # 20_to_anndata.py logic for one dataset; only D, OUT, SPL differ.
    CTRL = "non-targeting"
    NMIN = 200
    SEED = 0
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SPL, exist_ok=True)

    d  = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float32)
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]

    keep = iv != "excluded"
    X, iv = X[keep], iv[keep]
    guide = np.where(iv == CTRL, "", iv)

    vs = set(vn)
    tg = sorted(set(guide) - {""})
    counts = {g: int((guide == g).sum()) for g in tg}
    usable = sorted([g for g in tg if counts[g] >= NMIN])
    missing = [g for g in tg if g not in vs]

    print(f"\n===== {ds} =====", flush=True)
    print(f"  cells {X.shape[0]}  genes {len(vn)}  control {(guide=='').sum()}", flush=True)
    print(f"  perturbations {len(tg)}  usable (>={NMIN} cells) {len(usable)}", flush=True)
    print(f"  targets missing from feature columns: {len(missing)}  (expect 0)", flush=True)
    print(f"  X: min {X.min():.3f} max {X.max():.3f} mean {X.mean():.3f}", flush=True)
    assert X.min() >= 0, "expected non-negative log1p values"
    assert not missing, f"{len(missing)} targets absent from features"

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(usable))
    n_te = int(0.2 * len(usable))
    held  = sorted(usable[i] for i in perm[:n_te])
    train = sorted(usable[i] for i in perm[n_te:])

    sp = os.path.join(SPL, f"{ds}_zeroshot_split.json")
    json.dump({"dataset": ds, "nmin": NMIN, "seed": SEED,
               "n_usable": len(usable), "n_train": len(train), "n_heldout": len(held),
               "cells_per_perturbation": counts,
               "train_perturbations": train,
               "heldout_perturbations": held},
              open(sp, "w"), indent=2)
    print(f"  split: {len(train)} train / {len(held)} held out", flush=True)
    print(f"  -> {sp}", flush=True)

    dst = os.path.join(OUT, f"causalbench_{ds}.h5ad")
    if os.path.exists(dst):
        chk = ad.read_h5ad(dst, backed="r")
        if chk.shape == X.shape:
            print(f"  SKIP {dst} (exists, shape matches)", flush=True)
            return
        print(f"  STALE {dst} {chk.shape} != {X.shape}, rewriting", flush=True)

    ad.AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame({"guide_ids": pd.Categorical(guide)},
                         index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=vn)
    ).write_h5ad(dst)
    print(f"  wrote {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["k562", "rpe1"])
    a = ap.parse_args()
    if "GRD_DATA_ROOT" not in os.environ:
        sys.exit("STOP: GRD_DATA_ROOT is not set")

    root = data_root()
    D = str(root / "raw")
    OUT = str(perturbseq_path("causalbench_x.h5ad").parent)
    SPL = str(root / "build" / "splits")

    import causalscbench
    import h5py
    import scipy
    print(json.dumps({
        "dataset": a.dataset,
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scanpy": sc.__version__, "anndata": ad.__version__,
        "scipy": scipy.__version__, "h5py": h5py.__version__,
        "causalbench_path": os.path.dirname(causalscbench.__file__),
        "causalbench_commit_expected": CB_COMMIT,
        "D": D, "OUT": OUT, "SPL": SPL}, indent=1), flush=True)

    build_npz(a.dataset, D)
    to_anndata(a.dataset, D, OUT, SPL)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
