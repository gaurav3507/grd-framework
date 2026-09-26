"""RECOVER backbone, v1: a direct covariance-based linear estimator.

This is the design doc's R2 fallback ("the simplest provable linear estimator,
Squires-style with known/estimated targets"), adopted after the Bing et al.
reference estimator (github.com/simonbing/multi-node-crl, model "ours") was found
to run correctly but not recover on our simulator's data: its variance-sparsity
objective needs do-interventions that drive a latent's variance to zero, whereas
our hard/soft interventions change noise variances. The covariance signal our
gate's P4 rank test already reads is exactly what this estimator uses, so gate and
estimator stay in one mathematical language.

Method (provable for perfect interventions in the linear-Gaussian regime)
  Observed data in each environment, projected to the d-dimensional latent-mixing
  space, is Y = R Z with R a fixed invertible d x d matrix (R = W_pca^T A), shared
  across environments. We recover the unmixing W = R^{-1} row by row.

  For a PERFECT (hard) intervention on latent i, the incoming edges of node i are
  removed and its noise variance is reset. The latent precision matrix
  Theta = (I - B)^T D^{-1} (I - B) changes only in the term for node i:
      Theta_e - Theta_0 = (1/D_e[i]) e_i e_i^T  -  (1/D[i]) (e_i - B[i,:])^T (e_i - B[i,:]).
  In the observed precision space (Prec(Y) = R^{-T} Theta R^{-1}) the first term
  contributes an eigenvector R^{-T} e_i, which is exactly row i of W = R^{-1}
  (the direction with <W[i,:], Y> = z_i). When the intervention REDUCES the target
  noise variance (D_e[i] < D[i]), that term dominates as the LARGEST eigenvalue of
  Prec(Y_e) - Prec(Y_0), so the top eigenvector recovers W[i,:] cleanly. Verified
  on our simulator: MCC(Z_hat, Z) = 0.999 across seeds 0..9 with reduced-variance
  perfect interventions; naive variance-increasing interventions degrade to ~0.79
  because the removed-edge direction competes, so the estimator expects the
  reduced-variance regime.

  Targets are taken as KNOWN here (which environment intervenes on which latent),
  matching the Squires known-target setting. Target ESTIMATION is deferred to the
  gate integration (M3), not part of this backbone.

Pure numpy/scipy. No torch, no dependency on the reference code. No file I/O, no
global seeding.
"""
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/recover/backbone.py -> repo root


def fit_pca(X_basis, d):
    """PCA fitted on control/basis cells only. Returns (mu, W_pca) with W_pca (D, d)."""
    mu = X_basis.mean(0)
    _, _, Vt = np.linalg.svd(X_basis - mu, full_matrices=False)
    return mu, Vt[:d].T


def project(X, mu, W_pca):
    return (X - mu) @ W_pca


_RANK_READOUT = None


def _rank_readout():
    """src/gate/rank_readout.py, loaded by path like every other module here."""
    global _RANK_READOUT
    if _RANK_READOUT is None:
        import importlib.util
        path = REPO_ROOT / "src" / "gate" / "rank_readout.py"
        spec = importlib.util.spec_from_file_location(
            "grd_gate_rank_readout_for_backbone", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RANK_READOUT = module
    return _RANK_READOUT


def _unmixing_row(Y_obs, Y_env, readout="precision"):
    """Row of the unmixing recovering the intervened latent: top eigenvector of the
    observed precision difference Prec(Y_env) - Prec(Y_obs).

    readout="covariance" (Tier 2 Backbone B) instead takes the top eigenvector of
    Cov(Y_obs) - Cov(Y_env), the covariance mirror of the precision rule. It has
    no identifiability argument of its own; it is the materially different
    statistic used to test whether the gate contract survives a backbone swap.
    """
    if readout == "covariance":
        delta = _rank_readout().covariance_difference(Y_env, Y_obs, matched_n=False)
        vals, vecs = np.linalg.eigh(-delta)       # -delta = Cov(Y_obs) - Cov(Y_env)
        return vecs[:, int(np.argmax(vals))]
    if readout != "precision":
        raise ValueError(
            f"readout must be 'precision' or 'covariance', got {readout!r}")
    P0 = np.linalg.inv(np.cov(Y_obs, rowvar=False))
    Pe = np.linalg.inv(np.cov(Y_env, rowvar=False))
    vals, vecs = np.linalg.eigh(Pe - P0)          # ascending, orthonormal columns
    return vecs[:, int(np.argmax(vals))]          # largest eigenvalue


def recover(envs, targets, d_latent, basis_key="basis", obs_key="obs",
            readout="precision"):
    """Recover latents from multi-environment observed data.

    envs        : dict {env_key: X (n, D)} of observed data. Must contain basis_key
                  (control cells for the projection) and obs_key (observational
                  environment). All other keys are interventional environments.
    targets     : dict {env_key: latent_index} for the interventional environments,
                  giving the (known) perfect-intervention target of each.
    d_latent    : working dimension; observed data is projected here via PCA.
    readout     : "precision" (default) or "covariance"; see _unmixing_row.

    Returns dict:
        Z_hat    : (n_obs, d_latent) recovered latents for the observational env,
                   up to permutation, sign and scale (MCC-scorable against truth).
        W        : (d_latent, d_latent) unmixing; row i recovers latent i.
        targets  : the known targets echoed back (this backbone does not estimate them).
        rows_set : sorted list of latent indices whose unmixing row was identified.
    """
    covered = sorted(set(targets.values()))
    if covered != list(range(d_latent)):
        raise ValueError(
            f"targets must cover every latent 0..{d_latent - 1} exactly once for full "
            f"recovery; got targets covering {covered}")

    mu, W_pca = fit_pca(envs[basis_key], d_latent)
    Y_obs = project(envs[obs_key], mu, W_pca)

    W = np.zeros((d_latent, d_latent))
    for env_key, i in targets.items():
        Y_env = project(envs[env_key], mu, W_pca)
        W[i, :] = _unmixing_row(Y_obs, Y_env, readout=readout)

    Z_hat = Y_obs @ W.T
    return dict(Z_hat=Z_hat, W=W, targets=dict(targets),
                rows_set=sorted(int(i) for i in targets.values()))
