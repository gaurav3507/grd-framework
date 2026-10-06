"""E8 (Tier 2): the centred pooled-permutation null (Theorem 2) on the Perturb-seq panels.

For each environment the null is drawn by centring the environment and the control
at their own column means, pooling them, and splitting uniform permutations of the
pool into n_e and n_0 rows (precision_readout._pooled_null_values, the null of
Theorem 2 in notes/theory/grd_theory_note.pdf). The observed statistic, threshold,
raw decision and p-value are the gate's usual ones; only the null draws change.
Decisions are reported under BH and under Benjamini-Yekutieli (BY, valid under
arbitrary dependence), both at q=0.05.

Inputs are built exactly as E5 builds them (tier2_common: load_panel, control-fit
projection, the 50 E3 random pure-control environments per seed), shared-reference
design only. Each seed is compared with the E6 null-resolution rerun at the same
seed when its file exists: Jaccard of the BH sets against
results/e6_resolution/<key>[_s<seed>]/e5_real_panel.json, design corrected_disjoint.

--validity-check needs no data and writes nothing: it reproduces the theory-note
validity check (notes/theory/checks/thm_checks.py, checks C and E) with this code.
Covariance: the note's random linear-SEM construction at seed 0 (d=10). 120
replicates of 10 null environments (n_e=300) sharing n_0=1500 controls, B=99; then
150 environments with a Mahalanobis mean shift of 1 and no covariance change. PASS
needs null fraction(p<=0.05) <= 0.07, fraction(p<=0.20) <= 0.24, and mean-shift
fraction(p<=0.05) <= 0.10.

Usage:
    python experiments/e8_pooled_null.py --dataset rpe1
    python experiments/e8_pooled_null.py --dataset norman --selftest --seeds 0 --B 99 --out-dir /tmp/x
    python experiments/e8_pooled_null.py --validity-check
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import tier2_common as T2

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
PR = T2.PR
NULL_NAME = "pooled_centred"


# ------------------------------------------------------------------ real panel
def e6_reference(key, name, seed):
    """BH set of the E6 shared (corrected_disjoint) design at this seed, if present."""
    sub = key if seed == 0 else f"{key}_s{seed}"
    path = RESULTS / "e6_resolution" / sub / "e5_real_panel.json"
    if not path.exists():
        return None
    block = json.loads(path.read_text()).get("datasets", {}).get(name)
    if block is None:
        return None
    for row in block["per_seed"]:
        if int(row["seed"]) == int(seed):
            bh = list(row["bh_selected_ids"]["corrected_disjoint"])
            return dict(source=str(path.relative_to(REPO)), sha256=T2.sha256_file(path),
                        bh=bh, bh_count=len(bh))
    return None


def _records(labels, Ys, result, by):
    return [dict(label=str(label), n=int(len(Y)),
                 signal=float(result["signals"][i]),
                 threshold=float(result["thresholds"][i]),
                 pvalue=float(result["pvalues"][i]),
                 bh=bool(result["bh_detect"][i]), by=bool(by[i]))
            for i, (label, Y) in enumerate(zip(labels, Ys))]


def _n_at_floor(pvalues, B):
    floor = 1.0 / (B + 1)
    return int(sum(p <= floor * (1.0 + 1e-12) for p in pvalues))


def run_real(key, seeds, B, out_dir, selftest=False):
    started = time.time()
    panel = T2.make_selftest_panel(key) if selftest else T2.load_panel(key)
    name, Xc, perts = panel["name"], panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]
    sizes = [int(len(Y)) for Y in Yperts]
    print(f"{name}: control {len(Xc)} | powered perts {len(perts)} | null {NULL_NAME}, "
          f"B={B}", flush=True)

    rows = []
    for seed in seeds:
        t0 = time.time()
        res = T2.screen(Yperts, Yobs, seed, B=B, null="pooled")
        fake_labels, Yfake = T2.random_controls(Xc, proj, sizes, seed, T2.N_FAKE)
        fres = T2.screen(Yfake, Yobs, T2.FAKE_SEED_OFFSET + seed, B=B, null="pooled")
        by = PR.by_fdr(res["pvalues"], q=T2.Q)
        fby = PR.by_fdr(fres["pvalues"], q=T2.Q)
        bh_ids = T2.selected(perts, res["bh_detect"])
        ref = None if selftest else e6_reference(key, name, seed)
        jac = T2.jaccard(bh_ids, ref["bh"]) if ref else None
        rows.append(dict(
            seed=seed,
            raw_count=int(res["raw_count"]),
            bh_count=int(res["bh_count"]),
            by_count=int(by.sum()),
            bh_selected=bh_ids,
            by_selected=T2.selected(perts, by),
            random_raw_count=int(fres["raw_count"]),
            random_bh_count=int(fres["bh_count"]),
            random_by_count=int(fby.sum()),
            n_pvalues_at_floor=_n_at_floor(res["pvalues"], B),
            random_n_pvalues_at_floor=_n_at_floor(fres["pvalues"], B),
            jaccard_bh_vs_e6_shared=jac,
            e6_reference=({k: ref[k] for k in ("source", "sha256", "bh_count")}
                          if ref else None),
            per_environment=_records(perts, Yperts, res, by),
            random_control_records=_records(fake_labels, Yfake, fres, fby),
            wall_seconds=round(time.time() - t0, 1),
        ))
        r = rows[-1]
        jtxt = ("null" if ref is None else
                "both empty" if jac is None else f"{jac:.3f}")
        print(f"{name} seed {seed}: pooled BH/BY {r['bh_count']}/{r['by_count']} | "
              f"random BH/BY {r['random_bh_count']}/{r['random_by_count']} | floor "
              f"{r['n_pvalues_at_floor']} | Jaccard vs e6 {jtxt} "
              f"({time.time() - t0:.0f}s)", flush=True)

    block = dict(
        dataset=name,
        source=panel["source"],
        n_control=int(len(Xc)),
        n_perts=len(perts),
        seeds=list(seeds),
        null=NULL_NAME,
        B=int(B),
        p_floor=1.0 / (B + 1),
        bh_count_per_seed=[r["bh_count"] for r in rows],
        by_count_per_seed=[r["by_count"] for r in rows],
        random_bh_count_per_seed=[r["random_bh_count"] for r in rows],
        random_by_count_per_seed=[r["random_by_count"] for r in rows],
        n_pvalues_at_floor_per_seed=[r["n_pvalues_at_floor"] for r in rows],
        jaccard_bh_vs_e6_shared_per_seed=[r["jaccard_bh_vs_e6_shared"] for r in rows],
        selftest=bool(selftest),
        per_seed=rows,
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    top = dict(
        experiment="e8_pooled_null",
        description=("Centred pooled-permutation null (Theorem 2) on the Perturb-seq "
                     "panel, shared-reference design; BH and BY at q; E3 random "
                     "pure-control family per seed; Jaccard of BH sets against E6."),
        config=dict(null=NULL_NAME, B=int(B), alpha=T2.ALPHA, q=T2.Q,
                    seeds=list(seeds), d_proj=T2.D_PROJ, nmin=T2.NMIN),
    )
    out = Path(out_dir) / "e8_pooled_null.json"
    T2.update_dataset_block(out, top, name, block)
    print(f"written {out} [{name}] ({block['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ validity check
VALIDITY_RULE = dict(null_p05_at_most=0.07, null_p20_at_most=0.24,
                     meanshift_p05_at_most=0.10)


def _note_covariance(seed, d=10):
    """The theory note's random linear-SEM covariance (thm_checks.model/cov)."""
    rng = np.random.default_rng(seed)
    Bm = np.tril(rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1, 1], (d, d)), -1)
    Bm *= rng.random((d, d)) < 0.4
    lam = rng.uniform(0.5, 1.5, d)
    R = rng.standard_normal((d, d))
    A = np.linalg.inv(np.eye(d) - Bm)
    return R @ A @ np.diag(lam) @ A.T @ R.T


def validity_check():
    started = time.time()
    d, m, n_e, n_0, B, reps, n_shift = 10, 10, 300, 1500, 99, 120, 150
    S0 = _note_covariance(0, d)
    L = np.linalg.cholesky(S0)

    pvals, any_by, any_bh = [], 0, 0
    for rep in range(reps):
        rng = np.random.default_rng([8, rep])
        Y0 = rng.standard_normal((n_0, d)) @ L.T
        envs = [rng.standard_normal((n_e, d)) @ L.T for _ in range(m)]
        res = PR.detect_with_pvalues(envs, Y0, B=B, rng=np.random.default_rng([8, rep, 1]),
                                     null="pooled")
        p = res["pvalues"]
        pvals += p
        any_by += int(PR.by_fdr(p, q=0.2).any())
        any_bh += int(PR.bh_fdr(p, q=0.2).any())
    pvals = np.asarray(pvals)

    shift_p = []
    for rep in range(n_shift):
        rng = np.random.default_rng([9, rep])
        Y0 = rng.standard_normal((n_0, d)) @ L.T
        s = rng.standard_normal(d)
        s /= np.sqrt(s @ np.linalg.solve(S0, s))           # Mahalanobis length 1
        Ye = rng.standard_normal((n_e, d)) @ L.T + s
        res = PR.detect_with_pvalues([Ye], Y0, B=B, rng=np.random.default_rng([9, rep, 1]),
                                     null="pooled")
        shift_p.append(res["pvalues"][0])
    shift_p = np.asarray(shift_p)

    out = dict(
        null_frac_p_le_0_05=round(float(np.mean(pvals <= 0.05)), 4),
        null_frac_p_le_0_20=round(float(np.mean(pvals <= 0.20)), 4),
        null_any_by_q0_2=round(any_by / reps, 4),
        null_any_bh_q0_2=round(any_bh / reps, 4),
        meanshift_frac_p_le_0_05=round(float(np.mean(shift_p <= 0.05)), 4),
    )
    passed = (out["null_frac_p_le_0_05"] <= VALIDITY_RULE["null_p05_at_most"]
              and out["null_frac_p_le_0_20"] <= VALIDITY_RULE["null_p20_at_most"]
              and out["meanshift_frac_p_le_0_05"] <= VALIDITY_RULE["meanshift_p05_at_most"])
    print(f"validity check (d={d}, m={m}, n_e={n_e}, n_0={n_0}, B={B}, reps={reps}; "
          f"mean shift {n_shift} envs)", flush=True)
    print(f"  null p<=0.05 = {out['null_frac_p_le_0_05']}, p<=0.20 = "
          f"{out['null_frac_p_le_0_20']}, any-BY(q=0.2) = {out['null_any_by_q0_2']} "
          f"(any-BH {out['null_any_bh_q0_2']}); mean-shift p<=0.05 = "
          f"{out['meanshift_frac_p_le_0_05']}", flush=True)
    print(f"  rule {VALIDITY_RULE}: {'PASS' if passed else 'FAIL'} "
          f"({time.time() - started:.0f}s)", flush=True)
    return passed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(T2.DATASETS))
    p.add_argument("--seeds", type=int, nargs="+", default=T2.SEEDS)
    p.add_argument("--B", type=int, default=9999)
    p.add_argument("--out-dir", default=None,
                   help="default results/e8_pooled_null/<dataset>")
    p.add_argument("--selftest", action="store_true",
                   help="synthetic panel, code path only; refuses results/")
    p.add_argument("--validity-check", action="store_true",
                   help="synthetic pooled-null validity check; no data, no writes")
    args = p.parse_args()
    if not args.validity_check and args.dataset is None:
        p.error("--dataset is required unless --validity-check is given")
    return args


def main():
    args = parse_args()
    if args.validity_check:
        sys.exit(0 if validity_check() else 1)
    out_dir = Path(args.out_dir or RESULTS / "e8_pooled_null" / args.dataset)
    if args.selftest and out_dir.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("--selftest must not write under results/; pass --out-dir")
    run_real(args.dataset, args.seeds, args.B, out_dir, selftest=args.selftest)


if __name__ == "__main__":
    main()
