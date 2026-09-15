"""iLCS ungated baseline on the E3 projected inputs, with a size-matched null.

iLCS (interventional Latent Causal Shift detector) is reimplemented from Chen et al.
2024 (NeurIPS, arXiv 2410.24059), Algorithm 1 and the empirical L variant of their
Section 5.1, because the original repository github.com/TianyuCodings/iLCS is 404.

Purpose: an ungated latent mechanism-shift detector, run on the SAME D=10 control-fit
projected space the E3 precision gate uses, so it is apples-to-apples with the gate.
It shows whether such a detector, WITHOUT a precondition (size-matched) gate, fires on
sampling noise (pure-control fakes), on systematic control heterogeneity (structured
splits), and on real perturbations. Two arms per environment:
  NAIVE       fixed alpha=0.2 (the paper default for d <= 10);
  CALIBRATED  alpha_env = 95th percentile of max_i L over B size-matched control-vs-
              control null draws (same Lesson-2 size-matching as pr.null_threshold).

Reuses, verbatim, the E3 panel's pr-loader and control-projection recipe and the
e3_rpe1_confound_check structured-split construction. Modifies no existing e3 script.

Data lives on the A100 (/workspace/external/...), not the Mac; run --smoke only where
the K562 h5ad is reachable. anndata is imported lazily inside the loader so the iLCS
functions can be exercised without it.
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

# --- panel importlib block, verbatim (loads the precision-gate readout module) ---
spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)

D = 10; NMIN = 200; SEED = 0

# The three dataset invocations, taken verbatim from experiments/e3_stability_perturbseq.py
# run(name, path, ctrl, single). ctrl is the control guide label ('' = control).
DATASETS = [
    ("K562",   "/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad", "", False),
    ("RPE1",   "/workspace/external/discrepancy_vae/datasets/causalbench_rpe1.h5ad", "", False),
    ("Norman", "/workspace/external/discrepancy_vae/datasets/Norman2019_raw.h5ad",   "", True),
]
GATE_JSON = {
    "K562":   "results/e3/e3_k562_gate_fixed.json",
    "RPE1":   "results/e3/e3_rpe1_gate_fixed.json",
    "Norman": "results/e3/e3_Norman_CRISPRa_singlegene.json",
}
OUT = Path("results/ilcs_baseline")
NAIVE_ALPHA = 0.2
B_NULL = 50


# ------------------------------------------------------------ iLCS (Chen 2024 Alg 1)
ICA_MAX_ITER = 2000


def _ica_rows(Y, seed):
    """FastICA on one environment (already in D=10). Returns the psi-sorted unmixing
    rows M (D x D), a unit-variance-ok flag, and a converged flag (n_iter_ did not hit
    ICA_MAX_ITER).
    """
    ica = FastICA(n_components=D, whiten='unit-variance', max_iter=ICA_MAX_ITER, tol=1e-4,
                  random_state=seed)
    S = ica.fit_transform(Y)
    M = ica.components_                       # D x D here (sources by projected dims)
    unit_ok = bool(np.allclose(S.var(0), 1.0, atol=0.1))
    converged = bool(int(getattr(ica, "n_iter_", ICA_MAX_ITER)) < ICA_MAX_ITER)
    psi = np.array([np.mean(np.abs(S[:, i]) <= 1.0) for i in range(S.shape[1])])
    order = np.argsort(psi)                   # ascending psi: least-Gaussian-ish first
    return M[order], unit_ok, converged


def ilcs_L(Y_ctrl, Y_env, seed):
    """Length-D iLCS shift vector L between control and one environment (K=2).
    L_i = sum|abs(M_ctrl[i]) - abs(M_env[i])| / (sum|M_ctrl[i]| + sum|M_env[i]|).
    Returns (L, unit_var_ok, converged). The L statistic and psi-sort are unchanged.
    """
    Mc, okc, cc = _ica_rows(np.asarray(Y_ctrl), seed)
    Me, oke, ce = _ica_rows(np.asarray(Y_env), seed)
    num = np.abs(np.abs(Mc) - np.abs(Me)).sum(1)
    den = np.abs(Mc).sum(1) + np.abs(Me).sum(1) + 1e-12
    return num / den, bool(okc and oke), bool(cc and ce)


def ilcs_fires(L, alpha):
    return bool(np.any(np.asarray(L) > alpha))


# ------------------------------------------------------------ calibrated alpha_env
def calib_alpha_env(proj, Xc, Yobs_full, n_env, B, rng):
    """95th percentile of max_i L over B size-matched null draws that mirror the REAL
    test geometry: the first arm is fixed at the full control projection Yobs_full (the
    same first arm as ilcs_L(Yobs, Yp)), the second arm is a control subsample of size
    n_env. Each draw uses a per-draw ICA seed SEED+b so the null carries ICA-init
    variation. Same size-matching principle as pr.null_threshold (Lesson 2).
    """
    N = len(Xc)
    maxL = np.empty(B)
    for b in range(B):
        idx = rng.choice(N, n_env if n_env < N else N, replace=False)   # size-matched to the env
        Lb, _, _ = ilcs_L(Yobs_full, proj(Xc[idx]), SEED + b)
        maxL[b] = float(Lb.max())
    return float(np.percentile(maxL, 95))


def _eval_env(Y_env, Yobs, proj, Xc, n_cells, rng, B, alpha_cache, env_id, kind):
    L, unit_ok, converged = ilcs_L(Yobs, Y_env, SEED)
    naive = ilcs_fires(L, NAIVE_ALPHA)
    if n_cells not in alpha_cache:
        alpha_cache[n_cells] = calib_alpha_env(proj, Xc, Yobs, n_cells, B, rng)
    alpha_env = alpha_cache[n_cells]
    calibrated = bool(float(L.max()) > alpha_env)
    return dict(kind=kind, id=env_id, n_cells=int(n_cells),
                L=[round(float(x), 5) for x in L],
                L_max=round(float(L.max()), 5),
                naive_fires=naive, alpha_env=round(alpha_env, 5),
                calibrated_fires=calibrated, ica_unit_var_ok=unit_ok,
                ica_converged=converged)


# ------------------------------------------------------------------ per dataset run
def run_dataset(name, path, ctrl, single, smoke):
    import anndata as ad                      # lazy: only needed for the real run
    rng = np.random.default_rng(SEED)
    A = ad.read_h5ad(path)
    X = A.X.toarray().astype(np.float64) if hasattr(A.X, "toarray") else np.asarray(A.X, np.float64)
    g = A.obs['guide_ids'].astype(str).values
    del A
    if single:
        is_single = lambda s: s != ctrl and s != "" and "," not in s and "+" not in s and "_" not in s
        cand = sorted({s for s in np.unique(g) if is_single(s)})
    else:
        cand = [p for p in np.unique(g) if p != ctrl]

    Xc = X[g == ctrl]
    mu = Xc.mean(0); _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False); Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Xc); Zc = Yobs
    perts = [p for p in cand if (g == p).sum() >= NMIN]
    sizes = [int((g == p).sum()) for p in perts]

    n_fake, n_rand, n_pert, B = (50, 20, len(perts), B_NULL)
    if smoke:
        n_fake, n_rand, n_pert, B = (5, 5, 20, 10)

    # control-projection kurtosis (ICA needs at most one near-Gaussian component)
    kurt = kurtosis(Zc, axis=0, fisher=True)
    n_near_gaussian = int(np.sum(np.abs(kurt) < 0.5))

    alpha_cache = {}
    records = []

    # (i) size-matched pure-control fakes (size drawn from powered-pert sizes)
    for f in range(n_fake):
        n = int(rng.choice(sizes))
        sub = proj(Xc[rng.choice(len(Xc), n, replace=False)])
        records.append(_eval_env(sub, Yobs, proj, Xc, n, rng, B, alpha_cache, f, "fake"))

    # (ii) random control splits
    for r in range(n_rand):
        n = int(rng.choice(sizes))
        idx = rng.choice(len(Xc), n, replace=False)
        records.append(_eval_env(proj(Xc[idx]), Yobs, proj, Xc, n, rng, B, alpha_cache, r, "random"))

    # (iii) structured control splits along top control PCs (verbatim from confound check)
    pc = Zc - Zc.mean(0); msz = int(np.median(sizes)); struct = []
    for j in range(D):
        o = np.argsort(pc[:, j]); struct += [o[:msz], o[-msz:]]
    if smoke:
        struct = struct[:5]
    for si, sub_idx in enumerate(struct):
        records.append(_eval_env(proj(Xc[sub_idx]), Yobs, proj, Xc, len(sub_idx), rng, B,
                                 alpha_cache, si, "struct"))

    # (iv) powered real perturbations
    pert_list = perts[:n_pert] if smoke else perts
    for p in pert_list:
        Yp = proj(X[g == p]); n = len(Yp)
        records.append(_eval_env(Yp, Yobs, proj, Xc, n, rng, B, alpha_cache, p, "pert"))

    def rate(kind, key):
        rows = [r for r in records if r["kind"] == kind]
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None

    calib_pert_ids = {r["id"] for r in records if r["kind"] == "pert" and r["calibrated_fires"]}
    jaccard, jreason = _jaccard_vs_gate(name, calib_pert_ids)

    unit_warn = int(sum(1 for r in records if not r["ica_unit_var_ok"]))
    nonconv = int(sum(1 for r in records if not r["ica_converged"]))
    per_dataset = dict(dataset=name, control_label=repr(ctrl), single_gene_only=single,
                       d_proj=D, nmin=NMIN, n_control=int(len(Xc)),
                       n_powered_perts=len(perts), naive_alpha=NAIVE_ALPHA, B_null=B,
                       kurtosis=[round(float(x), 4) for x in kurt],
                       n_dims_near_gaussian=n_near_gaussian,
                       ica_unit_var_warn=unit_warn, ica_nonconverged_count=nonconv,
                       records=records)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(per_dataset, indent=2))

    summary = dict(
        dataset=name,
        naive_fake_rate=rate("fake", "naive_fires"), calib_fake_rate=rate("fake", "calibrated_fires"),
        naive_random_rate=rate("random", "naive_fires"), calib_random_rate=rate("random", "calibrated_fires"),
        naive_struct_rate=rate("struct", "naive_fires"), calib_struct_rate=rate("struct", "calibrated_fires"),
        naive_pert_rate=rate("pert", "naive_fires"), calib_pert_rate=rate("pert", "calibrated_fires"),
        jaccard_calib_vs_gate=jaccard, jaccard_note=jreason,
        n_dims_near_gaussian=n_near_gaussian, ica_unit_var_warn=unit_warn,
        ica_nonconverged_count=nonconv)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _jaccard_vs_gate(name, calib_ids):
    """Jaccard of calibrated-iLCS detected perts with the gate's detected perts. The
    committed gate JSONs carry only aggregate counts, no per-pert detect dict, so this
    is not computable from them; returns (None, reason) in that case.
    """
    path = Path(GATE_JSON[name])
    if not path.exists():
        return None, f"gate JSON {path} not found"
    gate = json.loads(path.read_text())
    detect = None
    for k in ("detect", "detections", "per_pert_detect", "detected_perts"):
        if isinstance(gate.get(k), (dict, list)):
            detect = gate[k]; break
    if detect is None:
        return None, ("gate JSON carries only aggregate counts (no per-pert detect dict); "
                      "Jaccard not computable from committed gate files")
    if isinstance(detect, dict):
        gate_ids = {p for p, v in detect.items() if v}
    else:
        gate_ids = set(detect)
    union = calib_ids | gate_ids
    return (round(len(calib_ids & gate_ids) / len(union), 4) if union else 0.0), "per-pert detect dict"


def main():
    smoke = "--smoke" in sys.argv
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    datasets = [DATASETS[0]] if smoke else DATASETS
    summaries = {}
    for (name, path, ctrl, single) in datasets:
        summaries[name] = run_dataset(name, path, ctrl, single, smoke)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(dict(
        mode="smoke" if smoke else "full", seed=SEED,
        imported_from="experiments/e3_perturbseq_panel.py (pr-loader, proj recipe), "
                      "experiments/e3_rpe1_confound_check.py (structured splits), "
                      "experiments/e3_stability_perturbseq.py (dataset tuples)",
        ilcs="reimplemented from Chen et al. 2024 (arXiv 2410.24059) Algorithm 1; "
             "original repo github.com/TianyuCodings/iLCS is unavailable (404)",
        per_dataset=summaries), indent=2))


if __name__ == "__main__":
    main()
