"""E12 shared code (pre-registration: notes/e12_prereg.md, committed before this file).

- load_panel_full: tier2_common.load_panel's data path (same candidates, same NMIN, same
  environment order) plus feature names, string var columns, and the map from each
  perturbation label to a feature column. The namespace (var_names or one var column)
  is the one matching the most perturbation labels; all match counts are recorded.
- load_hcp_subjects: {task: {subject_id: connectivity vector}} with E3's file pattern
  ({subject}_{task}_{LR|RL}.npy, subject = text before the first underscore) and E3's
  connectivity (upper triangle of the correlation of the concatenated runs).
- Control split, PCA projection, rank-1 target-gene correction.
- Standard two-sample tests: T1 Hotelling T2, T2 Roy two-sided, T3 energy distance (all
  by one permutation helper), T4 Box's M (analytic). GRD statistics and nulls are never
  reimplemented: T0h, T0i and T0m call precision_readout.detect_with_pvalues.
- RNG: SeedSequence([seed, i, k]); k = 0 GRD null (identical to E3's [seed, i] stream,
  since numpy ignores a trailing zero), 1 size-matched REF draw, 2 T0m, 3/4/5 T1/T2/T3,
  6 fake row selection (not named in the pre-registration; recorded here). Fakes use
  i = 100000 + f. The control split uses SeedSequence([seed, 7777]).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import linalg, stats  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tier2_common as T2  # noqa: E402
from data_paths import hcp_ts_root  # noqa: E402

PR = T2.PR
ALPHA = 0.05
Q = 0.05
N_FAKE = 50
FAKE_OFFSET = 100_000
SPLIT_SALT = 7777
SPLIT_FRACTIONS = (0.3, 0.4, 0.3)
K_NULL, K_REFDRAW, K_T0M, K_T1, K_T2, K_T3, K_FAKE_ROWS = 0, 1, 2, 3, 4, 5, 6
TESTS = ("T0h", "T0i", "T0m", "T1", "T2", "T3")
HCP_TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
CHUNK = 250


def rng_for(seed, i, k):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(i), int(k)]))


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ loaders
def _string_columns(var):
    out = {}
    for c in var.columns:
        col = var[c]
        kind = str(col.dtype)
        if col.dtype == object or kind in ("category", "string"):
            out[str(c)] = np.asarray(col.astype(str).values)
    return out


def target_map(perts, namespaces):
    """Choose the namespace matching the most labels; map label -> first column index."""
    counts = {}
    for name, v in namespaces.items():
        names = set(v.tolist())
        counts[name] = int(sum(p in names for p in perts))
    best = max(counts, key=counts.get) if counts else None
    columns = {}
    duplicates = 0
    if best is not None and counts[best] > 0:
        index = {}
        for j, name in enumerate(namespaces[best].tolist()):
            if name in index:
                duplicates += 1
            index.setdefault(name, j)
        columns = {p: int(index[p]) for p in perts if p in index}
    return dict(namespace=best, match_counts=counts, n_mapped=len(columns),
                duplicate_names_in_namespace=duplicates,
                column={p: columns.get(p) for p in perts})


def load_panel_full(key):
    """tier2_common.load_panel plus feature names and the target-column map."""
    import anndata as ad

    spec = T2.DATASETS[key]
    path = T2.perturbseq_path(spec["file"])
    A = ad.read_h5ad(path)
    X = A.X
    g = A.obs["guide_ids"].astype(str).values
    n_cells, n_genes = (int(s) for s in A.shape)
    var_names = np.asarray(A.var_names).astype(str)
    var_columns = _string_columns(A.var)
    del A
    ctrl = spec["ctrl"]
    cand = T2.candidates(g, ctrl, spec["single"])
    perts = [p for p in cand if (g == p).sum() >= T2.NMIN]
    namespaces = {"var_names": var_names, **{f"var[{c}]": v for c, v in var_columns.items()}}
    return dict(
        name=spec["name"], key=key, source=str(path), n_cells=n_cells, n_genes=n_genes,
        control_label=repr(ctrl), Xc=T2._dense_rows(X, g == ctrl), perts=perts,
        Xperts=[T2._dense_rows(X, g == p) for p in perts],
        var_names_head=var_names[:5].tolist(), var_columns=list(var_columns),
        mapping=target_map(perts, namespaces))


def conn(ts):
    """E3's connectivity: upper triangle of the correlation matrix."""
    C = np.corrcoef(ts, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def load_hcp_subjects():
    """{task: {subject_id: connectivity vector}} with E3's file pattern."""
    ts = str(hcp_ts_root())
    subjects = sorted({os.path.basename(f).split("_")[0]
                       for f in glob.glob(f"{ts}/*.npy")})
    out = {}
    for task in HCP_TASKS:
        per = {}
        for s in subjects:
            arrays = [np.load(f"{ts}/{s}_{task}_{enc}.npy").astype(float)
                      for enc in ("LR", "RL") if os.path.exists(f"{ts}/{s}_{task}_{enc}.npy")]
            if arrays:
                per[s] = conn(np.concatenate(arrays, 0))
        out[task] = per
    return out


# ------------------------------------------------------------------ geometry
def control_split(n, seed):
    """BASIS 30% / REF 40% / FAKE 30% of the control rows, disjoint, per seed."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SPLIT_SALT]))
    perm = rng.permutation(n)
    nb, nr = int(SPLIT_FRACTIONS[0] * n), int(SPLIT_FRACTIONS[1] * n)
    return np.sort(perm[:nb]), np.sort(perm[nb:nb + nr]), np.sort(perm[nb + nr:])


def fit_projection(X, d):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:d].T


def project(X, mu, P):
    return (X - mu) @ P


def correct(Y, x_j, mu_j, P_j):
    """Projection after setting feature j to the centring mean: rank-1 form."""
    return Y - np.outer(x_j - mu_j, P_j)


# ------------------------------------------------------------------ tests
def hotelling_direct(Ya, Yb):
    na, nb = len(Ya), len(Yb)
    diff = Ya.mean(0) - Yb.mean(0)
    Sp = ((na - 1) * np.cov(Ya, rowvar=False) + (nb - 1) * np.cov(Yb, rowvar=False)) / (na + nb - 2)
    return float(na * nb / (na + nb) * diff @ np.linalg.solve(Sp, diff))


def hotelling_F_p(t2, na, nb, d):
    N = na + nb
    F = (N - d - 1) / (d * (N - 2)) * t2
    return float(stats.f.sf(F, d, N - d - 1))


def roy_stat(Ya, Yb):
    lam = linalg.eigh(np.cov(Ya, rowvar=False), np.cov(Yb, rowvar=False), eigvals_only=True)
    return float(np.max(np.abs(np.log(lam))))


def box_m(Ya, Yb):
    d = Ya.shape[1]
    n1, n2 = len(Ya), len(Yb)
    N, g = n1 + n2, 2
    S1, S2 = np.cov(Ya, rowvar=False), np.cov(Yb, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (N - g)
    ld = lambda S: np.linalg.slogdet(S)[1]  # noqa: E731
    M = (N - g) * ld(Sp) - (n1 - 1) * ld(S1) - (n2 - 1) * ld(S2)
    c1 = (1 / (n1 - 1) + 1 / (n2 - 1) - 1 / (N - g)) * (2 * d * d + 3 * d - 1) / (6 * (d + 1) * (g - 1))
    stat = (1 - c1) * M
    return float(stat), float(stats.chi2.sf(stat, d * (d + 1) * (g - 1) / 2))


def _hotelling_batch(Pc, ST_inv, S, n_a):
    """Exact two-sample T2 for each membership row of S (Sherman-Morrison on the fixed
    total scatter): T2 = c (N - 2) q / (1 - c q), q = diff' ST^-1 diff, c = n_a n_b / N."""
    N = Pc.shape[0]
    n_b = N - n_a
    c = n_a * n_b / N
    diff = (S.astype(float) @ Pc) * (1.0 / n_a + 1.0 / n_b)
    q = np.einsum("kd,de,ke->k", diff, ST_inv, diff)
    return c * (N - 2) * q / (1.0 - c * q)


def _energy_batch(D, rowsum, total, S, n_a):
    """V-statistic energy distance 2 E|a-b| - E|a-a'| - E|b-b'| per membership row."""
    N = D.shape[0]
    n_b = N - n_a
    Sf = S.astype(float)
    s_aa = np.einsum("kn,kn->k", Sf @ D, Sf)
    s_a1 = Sf @ rowsum
    s_ab = s_a1 - s_aa
    s_bb = total - 2.0 * s_a1 + s_aa
    return 2.0 * s_ab / (n_a * n_b) - s_aa / n_a ** 2 - s_bb / n_b ** 2


def _memberships(N, n_a, B, rng):
    done = 0
    while done < B:
        c = min(CHUNK, B - done)
        S = np.zeros((c, N), dtype=bool)
        for r in range(c):
            S[r, rng.permutation(N)[:n_a]] = True
        done += c
        yield S


def permutation_test(Ya, Yb, B, rng, kind):
    """Permutation p = (1 + #null >= obs) / (B + 1) over the pooled rows, for T1, T2, T3.
    The observed value uses the same code path (identity membership) as the null."""
    P = np.vstack([Ya, Yb])
    N, n_a = len(P), len(Ya)
    if kind == "T1":
        Pc = P - P.mean(0)
        ST_inv = np.linalg.inv(Pc.T @ Pc)
        f = lambda S: _hotelling_batch(Pc, ST_inv, S, n_a)  # noqa: E731
    elif kind == "T2":
        f = lambda S: np.array([roy_stat(P[s], P[~s]) for s in S])  # noqa: E731
    elif kind == "T3":
        D = cdist(P, P)
        rowsum = D.sum(1)
        total = float(rowsum.sum())
        f = lambda S: _energy_batch(D, rowsum, total, S, n_a)  # noqa: E731
    else:
        raise ValueError(kind)
    ident = np.zeros((1, N), dtype=bool)
    ident[0, :n_a] = True
    obs = float(f(ident)[0])
    exceed = 0
    for S in _memberships(N, n_a, B, rng):
        exceed += int(np.count_nonzero(f(S) >= obs))
    return obs, (1 + exceed) / (B + 1)


def grd(Y, Yref, B, rng, null="disjoint"):
    r = PR.detect_with_pvalues([Y], Yref, alpha=ALPHA, B=B, rng=rng, q=Q, null=null)
    return float(r["signals"][0]), float(r["pvalues"][0])


# ------------------------------------------------------------------ worker
_SHARED = {}


def init_worker(shared):
    _SHARED.clear()
    _SHARED.update(shared)


def run_environment(task):
    """Every test for one environment. Shared references come from init_worker;
    the environment's reference copy is target-corrected here when needed."""
    seed, i, B = task["seed"], task["index"], task["B"]
    Yh, Yi = task["Y_heldout"], task["Y_insample"]
    Rh, Ri = _SHARED["ref_heldout"], _SHARED["ref_insample"]
    j = task["target_column"]
    if j is not None:
        xr = _SHARED["ref_target_columns"][j]
        Rh = correct(Rh, xr, task["mu_h_j"], task["P_h_j"])
        Ri = correct(Ri, xr, task["mu_i_j"], task["P_i_j"])
    n_e, d = Yh.shape
    rec = dict(label=task["label"], index=i, n_e=int(n_e), is_fake=task["is_fake"],
               excluded=False, target_corrected=j is not None, target_column=j,
               disjoint_applied=dict(
                   T0h=bool(PR._use_disjoint_null(len(Rh), n_e, d, True)),
                   T0i=bool(PR._use_disjoint_null(len(Ri), n_e, d, True))))
    s, p = grd(Yh, Rh, B, rng_for(seed, i, K_NULL))
    rec["T0h"] = dict(stat=s, p=p)
    s, p = grd(Yi, Ri, B, rng_for(seed, i, K_NULL))
    rec["T0i"] = dict(stat=s, p=p)
    rows = rng_for(seed, i, K_REFDRAW).choice(len(Rh), n_e, replace=False)
    Rm = Rh[rows]
    s, p = grd(Yh, Rm, B, rng_for(seed, i, K_T0M), null="pooled")
    rec["T0m"] = dict(stat=s, p=p)
    t1, p1 = permutation_test(Yh, Rm, B, rng_for(seed, i, K_T1), "T1")
    rec["T1"] = dict(stat=t1, p=p1, p_analytic_F=hotelling_F_p(t1, n_e, n_e, d))
    t2, p2 = permutation_test(Yh, Rm, B, rng_for(seed, i, K_T2), "T2")
    rec["T2"] = dict(stat=t2, p=p2)
    t3, p3 = permutation_test(Yh, Rm, B, rng_for(seed, i, K_T3), "T3")
    rec["T3"] = dict(stat=t3, p=p3)
    m, pm = box_m(Yh, Rm)
    rec["T4"] = dict(stat=m, p_analytic=pm)
    return rec


# ------------------------------------------------------------------ selftest inputs
def make_selftest_panel_full(key, seed=0):
    """tier2_common.make_selftest_panel plus a synthetic feature namespace in which
    every second perturbation label names a feature column, so the target-gene
    correction path runs. Code-path check only; never written under results/."""
    panel = T2.make_selftest_panel(key, seed=seed)
    n_genes = panel["n_genes"]
    var_names = np.array([f"G{j:03d}" for j in range(n_genes)])
    symbols = var_names.copy().astype(object)
    for k, label in enumerate(panel["perts"]):
        if k % 2 == 0:
            symbols[3 * k + 1] = label
    namespaces = {"var_names": var_names, "var[symbol]": symbols.astype(str)}
    panel.update(key=key, var_names_head=var_names[:5].tolist(), var_columns=["symbol"],
                 mapping=target_map(panel["perts"], namespaces))
    return panel


def make_selftest_hcp(seed=0, p=300):
    """Synthetic {task: {subject_id: vector}}: 99 WM subjects, 40 of them with all 7
    tasks, the rest with a random subset. Code-path check only."""
    rng = np.random.default_rng(91_000 + seed)
    W = rng.standard_normal((12, p))
    subjects = [f"S{k:03d}" for k in range(99)]
    out = {task: {} for task in HCP_TASKS}
    for k, s in enumerate(subjects):
        tasks = HCP_TASKS if k < 40 else ["WM"] + [t for t in HCP_TASKS[1:] if rng.random() < 0.5]
        for t, task in enumerate(HCP_TASKS):
            if task in tasks:
                z = rng.standard_normal(12)
                z[t] *= 0.4 if t % 2 else 1.0
                out[task][s] = z @ W + rng.standard_normal(p)
    return out
