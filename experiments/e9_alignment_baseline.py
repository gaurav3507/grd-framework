"""E9: null baseline for the top-2 control-PC shift alignment.

shift_alignment (e3_gate_compare) measures the share of an environment's mean-shift
energy in the first two coordinates of the control-fit PCA projection. Those
coordinates are not whitened, so a pure-noise mean shift of a control subsample has
covariance proportional to the control covariance and puts, in expectation, about
the top-2 variance share of its energy there, not 2/10. This script measures the
correct reference for each dataset:

  1. top2_variance_share: (l1 + l2) / sum(l1..l10) of the projected controls;
  2. random control subsets, sizes drawn from the powered perturbation sizes:
     alignment distribution (no detection filter);
  3. all powered perturbations: alignment distribution (no detection filter);
  4. for the seed-0 E3 BH set (RPE1: 48 perturbations), the observed median and a
     size-matched permutation reference: n_rep draws of random control subsets with
     exactly the selected sizes, median alignment per draw, one-sided p-value;
  5. selection-effect check: the non-selected powered perturbations' alignment and a
     selected-versus-non-selected comparison (median difference, two-sided
     Mann-Whitney U, and a two-sided label-permutation p-value with its own seeded rng,
     so items 1-4 are unchanged).

Uses e3_gate_compare.shift_alignment unchanged. Check: the RPE1 observed median
must reproduce the published 0.728772.
"""

import argparse
from pathlib import Path

import numpy as np

import tier2_common as T2

OUT = T2.REPO / "results" / "e9_alignment_baseline"


def align_all(Ys, Yobs):
    vals = []
    for Y in Ys:
        r = T2.shift_alignment([Y], Yobs, [True])
        vals.append(r["top2_energy_median"])
    return np.asarray(vals, float)


def summ(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=round(float(v.mean()), 6),
                median=round(float(np.median(v)), 6),
                q05=round(float(np.quantile(v, 0.05)), 6),
                q95=round(float(np.quantile(v, 0.95)), 6))


def run(key, n_random, n_rep, selftest):
    panel = T2.make_selftest_panel(key) if selftest else T2.load_panel(key)
    Xc = panel["Xc"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    lam = np.sort(np.linalg.eigvalsh(np.cov(Yobs, rowvar=False)))[::-1]
    share = float(lam[:2].sum() / lam.sum())
    Yp = [proj(X) for X in panel["Xperts"]]
    sizes = [len(Y) for Y in Yp]

    rng = np.random.default_rng(9_000)
    rand = []
    for _ in range(n_random):
        n = int(rng.choice(sizes))
        idx = rng.choice(len(Xc), n, replace=False)
        rand.append(Yobs[idx])
    out = dict(
        dataset=panel["name"], n_controls=int(len(Xc)), n_perturbations=len(Yp),
        top2_variance_share=round(share, 6),
        eigenvalues_top10=[round(float(x), 6) for x in lam],
        isotropic_reference=0.2,
        random_controls=summ(align_all(rand, Yobs)),
        all_perturbations=summ(align_all(Yp, Yobs)),
    )

    dec = None if selftest else T2.e3_decision_set(panel["name"])
    if dec and dec["bh"]:
        lab = list(panel["perts"])
        sel = [lab.index(p) for p in dec["bh"]]
        obs = align_all([Yp[i] for i in sel], Yobs)
        sel_sizes = [sizes[i] for i in sel]
        meds = np.empty(n_rep)
        for r in range(n_rep):
            draw = [Yobs[rng.choice(len(Xc), n, replace=False)] for n in sel_sizes]
            meds[r] = np.median(align_all(draw, Yobs))
        obs_med = float(np.median(obs))
        out["selected"] = dict(
            source=dec["source"], sha256=dec["sha256"], n=len(sel),
            observed=summ(obs),
            size_matched_null_median=summ(meds),
            p_value_one_sided=round(float((1 + (meds >= obs_med).sum()) / (n_rep + 1)), 6),
        )
        sel_set = set(sel)
        non = align_all([Yp[i] for i in range(len(Yp)) if i not in sel_set], Yobs)
        if non.size:
            from scipy.stats import mannwhitneyu
            md = float(np.median(obs) - np.median(non))
            pooled = np.concatenate([obs, non])
            k = obs.size
            prng = np.random.default_rng(9_100)
            n_perm = 10_000
            hits = 0
            for _ in range(n_perm):
                perm = prng.permutation(pooled.size)
                d = abs(np.median(pooled[perm[:k]]) - np.median(pooled[perm[k:]]))
                hits += int(d >= abs(md) - 1e-12)
            out["selected"]["non_selected"] = summ(non)
            out["selected"]["selected_vs_non_selected"] = dict(
                n_selected=int(k), n_non_selected=int(non.size),
                median_diff=round(md, 6),
                mannwhitney_u_p_two_sided=float(
                    mannwhitneyu(obs, non, alternative="two-sided").pvalue),
                permutation_p_two_sided=round(float((1 + hits) / (n_perm + 1)), 6),
                n_permutations=n_perm,
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["rpe1", "k562", "norman"])
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--n-rep", type=int, default=2000)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    doc = dict(provenance=T2.provenance(__file__), n_random=a.n_random,
               n_rep=a.n_rep, results={})
    for key in a.datasets:
        res = run(key, a.n_random, a.n_rep, a.selftest)
        doc["results"][key] = res
        print(key, "share", res["top2_variance_share"],
              "random_med", res["random_controls"]["median"],
              "perts_med", res["all_perturbations"]["median"],
              "selected", res.get("selected", {}).get("observed", {}).get("median"),
              "null_med", res.get("selected", {}).get("size_matched_null_median", {}).get("median"),
              "non_selected_med", res.get("selected", {}).get("non_selected", {}).get("median"),
              "p", res.get("selected", {}).get("p_value_one_sided"), flush=True)
    if a.selftest:
        print("selftest ok (no file written)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    T2.write_json(OUT / "e9_alignment_baseline.json", doc)
    print("DONE", OUT / "e9_alignment_baseline.json")


if __name__ == "__main__":
    main()
