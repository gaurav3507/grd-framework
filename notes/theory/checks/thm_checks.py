"""Numerical checks for the draft Theorems 2 and 3 (GRD theory note).

Model: projected linear SEM, Y = R Z, Z = B Z + eps, d = 10, hard interventions
(row of B zeroed, noise variance -> 0.1) on k targets, remaining environments
unintervened. Gate statistic and disjoint null come from the repo's
src/gate/precision_readout.py; the pooled-permutation null is implemented here.
"""
import importlib.util
import json
import sys
import time

import numpy as np

REPO = "/home/claude/gaurav3507/grd-framework"
spec = importlib.util.spec_from_file_location("pr", f"{REPO}/src/gate/precision_readout.py")
PR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PR)

D = 10
OUT = {}


def model(seed):
    rng = np.random.default_rng(seed)
    B = np.tril(rng.uniform(0.5, 1.0, (D, D)) * rng.choice([-1, 1], (D, D)), -1)
    B *= rng.random((D, D)) < 0.4
    lam = rng.uniform(0.5, 1.5, D)
    R = rng.standard_normal((D, D))
    return B, lam, R


def cov(B, lam, R, target=None, lam_new=0.1):
    B = B.copy(); lam = lam.copy()
    if target is not None:
        B[target] = 0.0; lam[target] = lam_new
    A = np.linalg.inv(np.eye(D) - B)
    return R @ A @ np.diag(lam) @ A.T @ R.T


def draw(rng, S, n):
    return rng.multivariate_normal(np.zeros(D), S, size=n)


def pooled_pvalue(Ye, Y0, B, rng, stat=PR.precision_signal):
    """Pooled-permutation p-value: exchangeable under H0 (Ye, Y0 i.i.d. same law)."""
    obs = stat(Ye, Y0)
    pool = np.vstack([Ye, Y0]); ne = len(Ye); cnt = 0
    for _ in range(B):
        idx = rng.permutation(len(pool))
        cnt += stat(pool[idx[:ne]], pool[idx[ne:]]) >= obs
    return (1 + cnt) / (B + 1), obs


def bh(p, q):
    return PR.bh_fdr(np.asarray(p), q=q)


def by(p, q):
    m = len(p); c = sum(1 / i for i in range(1, m + 1))
    return PR.bh_fdr(np.asarray(p), q=q / c)


# ---------------------------------------------------------------- check A: rates
def check_rates():
    Bm, lam, R = model(0)
    S0 = cov(Bm, lam, R); tgt = int(np.argmax(np.abs(Bm).sum(1)))  # a node with parents
    Se = cov(Bm, lam, R, target=tgt)
    true = float(np.linalg.eigvalsh(np.linalg.inv(Se) - np.linalg.inv(S0))[-1])
    rows = []
    for ne in (250, 500, 1000, 2000, 4000, 8000):
        errs, thr = [], []
        for rep in range(20):
            rng = np.random.default_rng([1, ne, rep])
            Y0 = draw(rng, S0, 20000); Ye = draw(rng, Se, ne)
            errs.append(abs(PR.precision_signal(Ye, Y0) - true))
            null = PR._precision_null_values(Y0, B=200, rng=rng, n_env=ne)
            thr.append(np.quantile(null, 0.95))
        rows.append(dict(n_e=ne, median_abs_err=float(np.median(errs)),
                         median_null_q95=float(np.median(thr))))
        print("A", rows[-1], flush=True)
    x = np.log([r["n_e"] for r in rows])
    slope_err = np.polyfit(x, np.log([r["median_abs_err"] for r in rows]), 1)[0]
    slope_thr = np.polyfit(x, np.log([r["median_null_q95"] for r in rows]), 1)[0]
    OUT["A_rates"] = dict(true_lambda_max=true, target_has_parents=bool(np.any(Bm[tgt])),
                          rows=rows, slope_err=float(slope_err), slope_thr=float(slope_thr))
    print("A slopes", slope_err, slope_thr, flush=True)


# ---------------------------------------------------------------- check B: resolution threshold
def check_resolution():
    Bm, lam, R = model(1)
    S0 = cov(Bm, lam, R); Se = cov(Bm, lam, R, target=0, lam_new=0.05)
    m, k, q, ne = 21, 1, 0.05, 600       # sure screening needs B+1 >= m/(k q) = 420
    res = {}
    for Bb in (399, 499):
        sel_true, sel_null = 0, 0
        for rep in range(20):
            rng = np.random.default_rng([2, Bb, rep])
            Y0 = draw(rng, S0, 12000)
            envs = [draw(rng, Se, ne)] + [draw(rng, S0, ne) for _ in range(m - 1)]
            p = []
            for i, Y in enumerate(envs):
                r = np.random.default_rng([3, Bb, rep, i])
                p.append(PR.detect_with_pvalues([Y], Y0, B=Bb, rng=r)["pvalues"][0])
            d = bh(p, q)
            sel_true += int(d[0]); sel_null += int(d[1:].sum())
        res[Bb] = dict(true_env_selected=sel_true, reps=20, null_selections=sel_null)
        print("B", Bb, res[Bb], flush=True)
    OUT["B_resolution"] = dict(m=m, k=k, q=q, threshold_B_plus_1=m / (k * q), results=res)


# ---------------------------------------------------------------- check C: pooled-permutation validity
def check_validity():
    Bm, lam, R = model(2)
    S0 = cov(Bm, lam, R)
    m, ne, n0, Bb, reps = 10, 300, 1500, 99, 120
    P = []
    fdr_by, fdr_bh, any_by, any_bh = [], [], 0, 0
    for rep in range(reps):
        rng = np.random.default_rng([4, rep])
        Y0 = draw(rng, S0, n0)                       # shared controls
        p = [pooled_pvalue(draw(rng, S0, ne), Y0, Bb, rng)[0] for _ in range(m)]
        P += p
        any_bh += int(bh(p, 0.2).any()); any_by += int(by(p, 0.2).any())
    P = np.array(P)
    OUT["C_validity"] = dict(m=m, n_e=ne, n0=n0, B=Bb, reps=reps,
                             frac_p_le_0_05=float(np.mean(P <= 0.05)),
                             frac_p_le_0_10=float(np.mean(P <= 0.10)),
                             frac_p_le_0_20=float(np.mean(P <= 0.20)),
                             all_null_FWER_BH_q0_2=any_bh / reps,
                             all_null_FWER_BY_q0_2=any_by / reps)
    print("C", OUT["C_validity"], flush=True)


if __name__ == "__main__":
    t = time.time()
    which = sys.argv[1:] or ["A", "B", "C"]
    if "A" in which: check_rates()
    if "B" in which: check_resolution()
    if "C" in which: check_validity()
    OUT["wall_seconds"] = round(time.time() - t, 1)
    json.dump(OUT, open(f"/tmp/claude-0/-home-claude/a1c41ab2-3442-593f-8516-47756ec8bc4e/scratchpad/thm_checks_{''.join(which)}.json", "w"), indent=2)


def check_meanshift():
    """Pooled null when an environment shifts only its mean (tau_e = 0)."""
    Bm, lam, R = model(3)
    S0 = cov(Bm, lam, R)
    P = []
    for rep in range(150):
        rng = np.random.default_rng([5, rep])
        Y0 = draw(rng, S0, 1500)
        shift = rng.standard_normal(D); shift *= 1.0 / np.sqrt(shift @ np.linalg.solve(S0, shift))
        Ye = draw(rng, S0, 300) + 1.0 * shift          # Mahalanobis mean shift of 1
        P.append(pooled_pvalue(Ye, Y0, 99, rng)[0])
    P = np.array(P)
    OUT["D_meanshift"] = dict(reps=150, mahalanobis_shift=1.0,
                              frac_p_le_0_05=float(np.mean(P <= 0.05)),
                              frac_p_le_0_20=float(np.mean(P <= 0.20)),
                              median_p=float(np.median(P)))
    print("D", OUT["D_meanshift"], flush=True)


if __name__ == "__main__" and "D" in sys.argv[1:]:
    check_meanshift()
    json.dump(OUT, open("/tmp/claude-0/-home-claude/a1c41ab2-3442-593f-8516-47756ec8bc4e/scratchpad/thm_checks_D2.json", "w"), indent=2)
