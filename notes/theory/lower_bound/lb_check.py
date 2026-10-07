"""Numerical check of the lower-bound theorem: detection and direction-recovery transitions
collapse on x = gamma * sqrt(n_e / d), for an oracle test (Sigma_0 = I known) and the GRD gate
(Sigma_0 estimated from n_0 = 4 n_e controls, disjoint null, B = 199).
Model: Omega_e = I + gamma v v^T (parentless hard intervention, whitened), v uniform on the sphere."""
import importlib.util, json, sys
import numpy as np

spec = importlib.util.spec_from_file_location("pr", "/home/claude/gaurav3507/grd-framework/src/gate/precision_readout.py")
PR = importlib.util.module_from_spec(spec); spec.loader.exec_module(PR)
D = 10
XS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
out = {}


def env(rng, n, gamma):
    v = rng.standard_normal(D); v /= np.linalg.norm(v)
    S = np.eye(D) - (gamma / (1 + gamma)) * np.outer(v, v)
    return rng.multivariate_normal(np.zeros(D), S, size=n), v


def oracle_stat(Y):
    return float(np.linalg.eigvalsh(np.linalg.inv(np.cov(Y, rowvar=False)) - np.eye(D))[-1])


for n in (200, 800, 3200):
    rng = np.random.default_rng([11, n])
    null = np.array([oracle_stat(rng.standard_normal((n, D))) for _ in range(2000)])
    thr = np.quantile(null, 0.95)
    rows = []
    for x in XS:
        g = x * np.sqrt(D / n)
        pw_or, pw_gate, sin_gate = [], [], []
        for rep in range(60):
            r = np.random.default_rng([12, n, int(100 * x), rep])
            Y, v = env(r, n, g)
            pw_or.append(oracle_stat(Y) > thr)
            if rep < 30:
                Y0 = r.standard_normal((4 * n, D))
                res = PR.detect_with_pvalues([Y], Y0, B=199, rng=r)
                pw_gate.append(res["pvalues"][0] <= 0.05)
                w = np.linalg.eigh(np.linalg.inv(np.cov(Y, rowvar=False)) - np.linalg.inv(np.cov(Y0, rowvar=False)))[1][:, -1]
                sin_gate.append(float(np.sqrt(max(0.0, 1 - (w @ v) ** 2))))
        rows.append(dict(x=x, gamma=round(g, 4), power_oracle=float(np.mean(pw_or)),
                         power_gate=float(np.mean(pw_gate)), median_sin_gate=float(np.median(sin_gate))))
        print(n, rows[-1], flush=True)
    out[str(n)] = rows
json.dump(out, open("/tmp/claude-0/-home-claude/a1c41ab2-3442-593f-8516-47756ec8bc4e/scratchpad/lb_check.json", "w"), indent=2)
