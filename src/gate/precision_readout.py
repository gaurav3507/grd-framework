"""Per-node precision gate readout (Path A).

The pooled-covariance P4 readout (rank_readout.py) and the RECOVER backbone read
DIFFERENT signals: the gate read the rank of one pooled covariance difference, while
the backbone (src/recover/backbone.py, _unmixing_row) isolates each latent as the
TOP-eigenvalue eigenvector of a PER-NODE precision difference
    inv(cov(Y_env)) - inv(cov(Y_obs)),
one direction per intervention environment. On an anisotropic seed the pooled reading
masked a weak-but-recoverable direction (M2.6, seed 7: gate said 4, backbone recovered
5). This readout reads the SAME quantity the backbone uses, just counting how many
directions carry detectable signal instead of extracting them.

Per intervention environment the discriminant is the largest eigenvalue of the
precision difference (exactly the value backbone._unmixing_row takes the argmax over).
A direction is detectable when that discriminant exceeds a control-vs-control null:
the (1 - alpha) quantile of the same discriminant computed from observational data
(no interventional effect). Smaller real-data environments use a disjoint pseudo-
environment/reference split that matches the tested sample-size geometry; equal-size
experiments retain the historical two-bootstrap construction. The count of detectable
environments is the recoverable dimension n_recoverable.

Reused settings, no new knob: alpha and B are the existing LFC test's (alpha=0.05,
B=500). The statistic itself has no tuning constant. Because each environment's
precision difference is localized to its intervened node, the masking that the pooled
covariance suffered under an anisotropic spectrum does not occur here.

Pure readout (numpy/scipy). Operates on data already projected to the working space
(as the backbone does internally); no file I/O, no global seeding.
"""
import numpy as np


def precision_signal(Y_env, Y_obs):
    """Backbone's per-node discriminant: the largest eigenvalue of the precision
    difference inv(cov(Y_env)) - inv(cov(Y_obs)). Y_env, Y_obs are (n, d).
    """
    Y_env = np.asarray(Y_env)
    Y_obs = np.asarray(Y_obs)
    Pe = np.linalg.inv(np.cov(Y_env, rowvar=False))
    P0 = np.linalg.inv(np.cov(Y_obs, rowvar=False))
    return float(np.max(np.linalg.eigvalsh(Pe - P0)))


def _use_disjoint_null(n_obs, n_env, d, disjoint):
    """Whether a disjoint pseudo-environment/reference split is well posed."""
    return bool(disjoint and n_env <= n_obs // 2 and (n_obs - n_env) > d)


def _precision_null_values(Y_obs, B, rng, n_env=None, disjoint=True):
    """Draw precision statistics under the requested null geometry.

    For a smaller environment, the disjoint path matches the tested comparison as
    closely as possible without sharing rows: n_env control rows form the pseudo-
    environment and all remaining rows form its reference. Equal-size comparisons
    retain the historical two-bootstrap path exactly.
    """
    Y_obs = np.asarray(Y_obs)
    n, d = Y_obs.shape
    m = n if n_env is None else int(n_env)
    if m <= d:
        raise ValueError(
            f"precision null needs n_env > d for invertible covariance; got {m} <= {d}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")

    vals = np.empty(B)
    if _use_disjoint_null(n, m, d, disjoint):
        for b in range(B):
            idx = rng.permutation(n)
            vals[b] = precision_signal(Y_obs[idx[:m]], Y_obs[idx[m:]])
    else:
        # Historical path. Keep the draw order unchanged so all equal-size synthetic
        # experiments are numerically identical to the pre-geometry-fix code.
        for b in range(B):
            i1 = rng.integers(0, n, m)
            i2 = rng.integers(0, n, m)
            vals[b] = precision_signal(Y_obs[i1], Y_obs[i2])
    return vals


def null_threshold(Y_obs, alpha=0.05, B=500, rng=None, n_env=None,
                   disjoint=True):
    """(1 - alpha) quantile of the precision signal under no intervention.

    When n_env is a smaller environment size, the default null uses a disjoint
    control split: n_env rows form a pseudo-environment and the remaining rows form
    the reference. This matches the small-environment-versus-large-reference geometry
    while avoiding overlapping samples. If a disjoint split is not well posed
    (including equal-size experiments), the historical two-bootstrap null is used.

    Set disjoint=False to reproduce the historical null explicitly.
    """
    if rng is None:
        raise ValueError("null_threshold needs an explicit rng (no global seeding)")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    vals = _precision_null_values(
        Y_obs, B=B, rng=rng, n_env=n_env, disjoint=disjoint)
    return float(np.quantile(vals, 1.0 - alpha))


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg FDR decisions, aligned to the input order."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("pvals must be one-dimensional")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if not np.all(np.isfinite(pvals)) or np.any((pvals < 0.0) | (pvals > 1.0)):
        raise ValueError("pvals must contain finite values in [0, 1]")
    reject = np.zeros(pvals.shape, dtype=bool)
    n = pvals.size
    if n == 0:
        return reject
    order = np.argsort(pvals, kind="stable")
    ranked = pvals[order]
    passing = np.flatnonzero(ranked <= q * np.arange(1, n + 1) / n)
    if passing.size:
        cutoff = ranked[passing[-1]]
        reject = pvals <= cutoff
    return reject


def detect_with_pvalues(Y_int_list, Y_obs, alpha=0.05, B=500, rng=None,
                        q=0.05, disjoint=True):
    """Detect precision changes and control environment-wise FDR.

    Each environment is compared with a null at its own sample size. The raw
    threshold decision and empirical p-value use the same null draws. Empirical
    p-values use the finite-resampling correction
        (1 + number of null statistics >= observed) / (B + 1).
    Benjamini-Hochberg is then applied across the supplied environments.
    """
    if rng is None:
        raise ValueError("detect_with_pvalues needs an explicit rng")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    signals, thresholds, pvalues, raw_detect = [], [], [], []
    for Y in Y_int_list:
        Y = np.asarray(Y)
        null = _precision_null_values(
            Y_obs, B=B, rng=rng, n_env=Y.shape[0], disjoint=disjoint)
        signal = precision_signal(Y, Y_obs)
        threshold = float(np.quantile(null, 1.0 - alpha))
        pvalue = float((1 + np.count_nonzero(null >= signal)) / (B + 1))
        signals.append(float(signal))
        thresholds.append(threshold)
        pvalues.append(pvalue)
        raw_detect.append(bool(signal > threshold))

    bh_detect = bh_fdr(pvalues, q=q)
    raw_detect = np.asarray(raw_detect, dtype=bool)
    return dict(
        signals=signals,
        thresholds=thresholds,
        raw_detect=raw_detect.tolist(),
        pvalues=pvalues,
        bh_detect=bh_detect.tolist(),
        raw_count=int(raw_detect.sum()),
        bh_count=int(bh_detect.sum()),
        alpha=float(alpha),
        q=float(q),
        B=int(B),
        disjoint=bool(disjoint),
    )


def count_recoverable(Y_int_list, Y_obs, alpha=0.05, B=500, rng=None):
    """Count intervention environments whose per-node precision signal exceeds the
    control-vs-control null threshold.

    Y_int_list : list of (n, d) projected interventional environments, one per
                 intervened latent (matching how the backbone isolates directions).
    Y_obs      : (n, d) projected observational environment.

    Returns dict: count (n_recoverable), threshold, per-environment signals, detect
    flags, and signal/threshold ratios.
    """
    if rng is None:
        raise ValueError("count_recoverable needs an explicit rng")
    signals, thresholds, detect, ratios = [], [], [], []
    for Y in Y_int_list:
        Y = np.asarray(Y)
        # Size-matched null (Lesson 2): the null resamples at THIS environment's size,
        # so a smaller perturbation environment is not scored against a tighter
        # full-size null. Equal-size environments reduce to the prior behaviour.
        crit = null_threshold(Y_obs, alpha=alpha, B=B, rng=rng, n_env=Y.shape[0])
        s = precision_signal(Y, Y_obs)
        signals.append(float(s))
        thresholds.append(float(crit))
        detect.append(bool(s > crit))
        ratios.append(float(s / crit) if crit > 0 else float("inf"))
    # Backward-compat scalar 'threshold' = the MAX per-environment threshold (the most
    # conservative one); per-environment thresholds are in 'thresholds'.
    scalar_threshold = float(max(thresholds)) if thresholds else 0.0
    return dict(count=int(sum(detect)), threshold=scalar_threshold,
                thresholds=thresholds,
                signals=signals, detect=detect,
                ratios=[round(r, 2) for r in ratios],
                alpha=float(alpha), B=int(B))


# ------------------------------------------------------------------ Subspace attribution (P3)
# Additive readout for the mechanism-vs-measurement verdict (Section X.5, Proposition 3).
# Unlike the precision statistic above, these operate on the FULL observed space (not the
# control-fit projection): a per-environment measurement gain rotates the signal subspace
# OUT of the projected basis, so the rotation is only visible before projection. Same
# size-matched-null principle as null_threshold (Lesson 2).

def _top_subspace(X, d):
    """Orthonormal basis (D, d) of the top-d principal subspace of centered X."""
    X = np.asarray(X)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d].T


def subspace_angle(X_env, X_obs, d):
    """Largest principal angle (radians) between the top-d signal subspaces of the
    environment and the control, in the full observed space. 0 means the environment
    preserved the control's signal subspace (consistent with shared mixing and a
    mechanism shift, Prop 3(i)); a large angle means the subspace rotated (the
    signature of a diagonal measurement gain, Prop 3(ii)).
    """
    U0 = _top_subspace(X_obs, d)
    Ue = _top_subspace(X_env, d)
    s = np.linalg.svd(U0.T @ Ue, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


def subspace_null(X_obs, d, alpha=0.05, B=500, rng=None, n_env=None,
                  disjoint=True):
    """Null quantile for the largest top-d principal angle.

    Smaller environments use the same disjoint pseudo-environment/reference geometry
    as null_threshold. Equal-size and otherwise infeasible disjoint comparisons retain
    the historical two-bootstrap path.
    """
    if rng is None:
        raise ValueError("subspace_null needs an explicit rng")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    X_obs = np.asarray(X_obs)
    n = X_obs.shape[0]
    m = n if n_env is None else int(n_env)
    if m <= d:
        raise ValueError(
            f"subspace null needs n_env > d; got {m} <= {d}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    vals = np.empty(B)
    if _use_disjoint_null(n, m, d, disjoint):
        for b in range(B):
            idx = rng.permutation(n)
            vals[b] = subspace_angle(X_obs[idx[:m]], X_obs[idx[m:]], d)
    else:
        for b in range(B):
            i1 = rng.integers(0, n, m)
            i2 = rng.integers(0, n, m)
            vals[b] = subspace_angle(X_obs[i1], X_obs[i2], d)
    return float(np.quantile(vals, 1.0 - alpha))


def attribute_environment(X_env, X_obs, d, detected, alpha=0.05, B=500, rng=None,
                          subspace_crit=None):
    """Three-way verdict for one environment (Section X.6).

    detected : bool, whether the precision screen fired on this environment (computed
               separately in the projected space via precision_signal/null_threshold).
    subspace_crit : optional precomputed size-matched subspace null (radians) for this
               environment's sample size, e.g. cached when many environments share one
               control and one sample size. When None it is computed here. This is a
               precomputed value of the same statistic, not a tuning constant.
    Returns dict with the verdict label, the measured subspace angle, its size-matched
    null, and the raw fired flag. Requires the FULL-D X_env / X_obs.

      NO_DETECTABLE_SHIFT         precision screen did not fire.
      MECHANISM_SUPPORTED         fired AND subspace angle within its null (subspace
                                  preserved: consistent with a mechanism shift under
                                  shared mixing).
      DETECTABLE_BUT_UNATTRIBUTED fired AND subspace angle exceeds its null (subspace
                                  rotated: the change could be a measurement gain, so
                                  detectability does not license a mechanism claim).
    """
    if rng is None and subspace_crit is None:
        raise ValueError("attribute_environment needs an explicit rng (or a precomputed subspace_crit)")
    if not detected:
        return dict(verdict="NO_DETECTABLE_SHIFT", detected=False,
                    subspace_angle_deg=None, subspace_null_deg=None,
                    subspace_rotated=None)
    n_env = np.asarray(X_env).shape[0]
    ang = subspace_angle(X_env, X_obs, d)
    crit = subspace_crit if subspace_crit is not None else subspace_null(
        X_obs, d, alpha=alpha, B=B, rng=rng, n_env=n_env)
    rotated = bool(ang > crit)
    verdict = "DETECTABLE_BUT_UNATTRIBUTED" if rotated else "MECHANISM_SUPPORTED"
    return dict(verdict=verdict, detected=True,
                subspace_angle_deg=round(float(np.degrees(ang)), 3),
                subspace_null_deg=round(float(np.degrees(crit)), 3),
                subspace_rotated=rotated)
