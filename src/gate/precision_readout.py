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


def null_threshold(Y_obs, alpha=0.05, B=500, rng=None):
    """(1 - alpha) quantile of the precision signal under the null of no
    interventional effect, estimated by comparing two independent bootstrap
    resamples of the observational data. alpha and B are the LFC test's settings.
    """
    if rng is None:
        raise ValueError("null_threshold needs an explicit rng (no global seeding)")
    Y_obs = np.asarray(Y_obs)
    n = Y_obs.shape[0]
    vals = np.empty(B)
    for b in range(B):
        i1 = rng.integers(0, n, n)
        i2 = rng.integers(0, n, n)
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
    crit = null_threshold(Y_obs, alpha=alpha, B=B, rng=rng)
    signals = [precision_signal(Y, Y_obs) for Y in Y_int_list]
    detect = [bool(s > crit) for s in signals]
    ratios = [float(s / crit) if crit > 0 else float("inf") for s in signals]
    return dict(count=int(sum(detect)), threshold=float(crit),
                signals=[float(s) for s in signals], detect=detect,
                ratios=[round(r, 2) for r in ratios],
                alpha=float(alpha), B=int(B))
