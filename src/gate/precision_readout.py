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
the (1 - alpha) quantile of the same discriminant computed between two independent
bootstrap resamples of the observational data (no interventional effect). The count of
detectable environments is the recoverable dimension n_recoverable.

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


def null_threshold(Y_obs, alpha=0.05, B=500, rng=None, n_env=None):
    """(1 - alpha) quantile of the precision signal under the null of no
    interventional effect, estimated by comparing two independent bootstrap
    resamples of the observational data. alpha and B are the LFC test's settings.

    n_env sets the resample size (Lesson 2): the null's per-sample covariance noise
    must match the sample size of the environment being tested, because
    precision_signal grows as the covariance is estimated from fewer rows. When
    n_env is None the resample size is len(Y_obs), reproducing the earlier behaviour
    exactly (used when the environment and Y_obs are the same size). n_env is a
    sample size, not a tuning knob.
    """
    if rng is None:
        raise ValueError("null_threshold needs an explicit rng (no global seeding)")
    Y_obs = np.asarray(Y_obs)
    n = Y_obs.shape[0]
    m = n if n_env is None else int(n_env)
    vals = np.empty(B)
    for b in range(B):
        i1 = rng.integers(0, n, m)
        i2 = rng.integers(0, n, m)
        vals[b] = precision_signal(Y_obs[i1], Y_obs[i2])
    return float(np.quantile(vals, 1.0 - alpha))


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


def subspace_null(X_obs, d, alpha=0.05, B=500, rng=None, n_env=None):
    """(1 - alpha) quantile of the largest principal angle between two independent
    size-n_env bootstrap resamples of the control, i.e. the angle attributable to
    sampling alone at this environment's sample size. Same Lesson-2 size matching as
    null_threshold: a smaller environment tolerates a larger sampling angle.
    """
    if rng is None:
        raise ValueError("subspace_null needs an explicit rng")
    X_obs = np.asarray(X_obs)
    n = X_obs.shape[0]
    m = n if n_env is None else int(n_env)
    vals = np.empty(B)
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
