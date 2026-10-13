"""E12b: do the E12 T1 (Hotelling) detections survive removing depth and the leading PC?
Pre-registration: notes/e12b_prereg.md (committed before this code).

Reuses E12 unchanged: e12_standard_tests.build_tasks for the split, held-out basis,
target correction, environments and fakes; the REF-draw and T1 lines of
e12_common.run_environment (rng_for k=1 and k=3) are copied verbatim, and the
reproduced T1 must equal the stored E12 value to 1e-8 (first 5 retained environments
checked before any other work, then every environment), else STOP.

Depth source, first available (recorded per dataset):
  1. an obs total-count column (named list first, then "similar" names);
  2. the row sum of a full-gene counts matrix: an AnnData layer shares X's genes, so the
     only place a full-gene matrix can live is .raw with more genes than X;
  3. last resort, the row sum of X (not library size; the dataset is then S3).
Depth = log(library size); if any value is <= 0 (possible only for an X row-sum
fallback), log(v - min + 1) is used and recorded.

Readouts per environment and fake: R1 depth_diff, R3 T1_depthreg (k=10), R4 T1_noPC1
(k=11), R2 pc1_share and the marginal diff_1^2 / S_11 (pooled), R5 batch TVD of the
environment's cells against the full REF composition (null when there is no batch column).

E12c (notes/e12c_deviation.md, post hoc correction for Norman): --depth-source
counts_layer_rowsum takes depth = log(row sum of layers["counts"]) instead of the
automatic choice, stops if the median BASIS row sum is below 500, records the BASIS
row-sum distribution and corr(depth, log obs["UMI_count"]) in BASIS cells, and writes
under results/e12c_norman_depth/. --depth-source auto (default) is the E12b behaviour.

Modes: --dataset (one run), --smoke (10 envs + 10 fakes, B=199, results/e12b_confound/
smoke/), --aggregate (summary.json, verdict computed), --inspect (Step 0 h5ad report
only), --selftest (synthetic, code path only), --oracles (Step 3 self-checks).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import sparse, stats  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402
import e12_standard_tests as S  # noqa: E402

T2 = E.T2
OUT_ROOT = E.REPO / "results" / "e12b_confound"
E12C_ROOT = E.REPO / "results" / "e12c_norman_depth"
DEPTH_SOURCES = ("auto", "counts_layer_rowsum")
COUNTS_MIN_MEDIAN = 500
E12_ROOT = E.REPO / "results" / "e12_standard_tests"
DATASETS = ("k562", "norman", "rpe1")
DECISION = ("k562", "norman")
K_R3, K_R4 = 10, 11
N_REPRO, REPRO_TOL = 5, 1e-8
SMOKE_B = 199
COUNT_COLUMNS = ("n_counts", "total_counts", "umi_count", "ncounts", "ncount_rna", "n_umi",
                 "numi", "n_umis", "total_umis", "umi_counts", "library_size", "total_count")
COUNT_EXCLUDE = re.compile(r"gene|pct|log|frac|ratio|(^|_)mt(_|$)")
BATCH_COLUMNS = ("gemgroup", "gem_group", "gem.group", "batch", "batch_id", "lane", "gem",
                 "library_id", "plate", "channel", "sample", "sample_id")
SCRIPT = "experiments/e12b_confound_check.py"


# ------------------------------------------------------------------ Step 0 inspection
def _integer_check(M, rng, n=1000):
    """Integer test on up to n random nonzero stored entries; never densifies M."""
    if sparse.issparse(M):
        data = np.asarray(M.data)
    else:
        rows = np.sort(rng.choice(M.shape[0], min(M.shape[0], 2000), replace=False))
        data = np.asarray(M[rows]).ravel()
    data = data[data != 0]
    if data.size == 0:
        return dict(n_checked=0, all_integer=None)
    vals = data[rng.choice(data.size, min(n, data.size), replace=False)].astype(np.float64)
    return dict(n_checked=int(vals.size), all_integer=bool(np.all(vals == np.round(vals))),
                min=float(vals.min()), max=float(vals.max()))


def _row_sums(M):
    return np.asarray(M.sum(1), dtype=np.float64).ravel()


def inspect_h5ad(key, depth_source="auto"):
    """Step 0(b) report plus per-cell depth and batch, aligned to obs order.

    depth_source="counts_layer_rowsum" (E12c) also returns the per-cell counts-layer row
    sums and obs["UMI_count"] as a fifth value; with "auto" that value is None."""
    if depth_source not in DEPTH_SOURCES:
        raise ValueError(f"depth_source must be one of {DEPTH_SOURCES}, got {depth_source!r}")
    import anndata as ad
    from pandas.api.types import is_numeric_dtype

    spec = T2.DATASETS[key]
    A = ad.read_h5ad(T2.perturbseq_path(spec["file"]))
    rng = np.random.default_rng(np.random.SeedSequence([0, 8888]))
    obs = A.obs
    cols = [str(c) for c in obs.columns]
    X = A.X
    rows200 = np.sort(rng.choice(X.shape[0], 200, replace=False))
    rs = _row_sums(X[rows200])
    xint = _integer_check(X, rng)
    info = dict(obs_columns=cols, layers=list(A.layers.keys()), X_type=type(X).__name__,
                X_dtype=str(X.dtype), X_shape=[int(s) for s in X.shape], X_integer_check=xint,
                X_verdict=("raw counts (integer entries)" if xint.get("all_integer")
                           else "not raw counts (non-integer entries)"),
                X_rowsum_200=dict(min=float(rs.min()), median=float(np.median(rs)), max=float(rs.max())))
    info["layer_checks"] = {str(k): dict(dtype=str(A.layers[k].dtype), full_gene=False,
                                         integer_check=_integer_check(A.layers[k], rng))
                            for k in A.layers.keys()}
    raw = A.raw
    raw_full = raw is not None and raw.X.shape[1] > X.shape[1]
    info["raw"] = None if raw is None else dict(
        shape=[int(s) for s in raw.X.shape], full_gene=bool(raw_full),
        integer_check=_integer_check(raw.X, rng))
    lower = {c.lower(): c for c in cols}
    named = [lower[n] for n in COUNT_COLUMNS if n in lower]
    similar = [c for c in cols if re.search("count|umi", c.lower())
               and not COUNT_EXCLUDE.search(c.lower()) and c not in named]
    usable = []
    for c in named + similar:
        v = np.asarray(obs[c], dtype=np.float64) if is_numeric_dtype(obs[c]) else None
        if v is not None and np.all(np.isfinite(v)) and np.all(v > 0):
            usable.append(c)
    info.update(count_columns_named=named, count_columns_similar=similar, count_columns_usable=usable)
    batch_col = next((lower[n] for n in BATCH_COLUMNS if n in lower), None)
    info["batch_column"] = batch_col
    extra = None
    if depth_source == "counts_layer_rowsum":
        if "counts" not in A.layers:
            raise SystemExit(f"STOP: {key} has no layers['counts'] (layers: {list(A.layers.keys())})")
        counts = A.layers["counts"]
        lib = _row_sums(counts)
        if np.min(lib) <= 0:
            raise SystemExit(f"STOP: {key} layers['counts'] has {int(np.sum(lib <= 0))} cells "
                             f"with row sum <= 0")
        source, fallback = f"layers[counts] row sum ({counts.shape[1]} genes)", False
        umi = np.asarray(obs["UMI_count"], dtype=np.float64) if "UMI_count" in obs.columns else None
        extra = dict(lib=lib, umi_obs=umi)
    elif usable:
        lib, source, fallback = np.asarray(obs[usable[0]], dtype=np.float64), f"obs[{usable[0]}]", False
    elif raw_full:
        lib, source, fallback = _row_sums(raw.X), "raw.X row sum (full-gene)", False
    else:
        lib, source, fallback = _row_sums(X), "X row sum (fallback, not library size)", True
    if np.min(lib) > 0:
        depth, transform = np.log(lib), "log(v)"
    else:
        depth, transform = np.log(lib - np.min(lib) + 1.0), "log(v - min + 1): nonpositive values"
    g = obs["guide_ids"].astype(str).values
    batch = obs[batch_col].astype(str).values if batch_col else None
    info.update(depth_source=source, depth_fallback=fallback, depth_transform=transform)
    del A
    return info, g, depth, batch, extra


def depth_inputs(panel, info, g, depth, batch, ctrl, extra=None):
    """Per-cell depth and batch in the panel's row order (load_panel's masks)."""
    mask = g == ctrl
    assert mask.sum() == len(panel["Xc"]), "control count differs from load_panel"
    out = dict(depth_ctrl=depth[mask], depth_env={}, batch_ctrl=None if batch is None else batch[mask],
               batch_env=None if batch is None else {}, info=info)
    if extra is not None:
        out["lib_ctrl"] = extra["lib"][mask]
        out["umi_obs_ctrl"] = None if extra["umi_obs"] is None else extra["umi_obs"][mask]
    for label, X in zip(panel["perts"], panel["Xperts"]):
        m = g == label
        assert m.sum() == len(X), f"{label}: cell count differs from load_panel"
        out["depth_env"][label] = depth[m]
        if batch is not None:
            out["batch_env"][label] = batch[m]
    return out


def selftest_depth(panel, seed=0):
    """Synthetic depth and batch for the selftest panel (code path only)."""
    rng = np.random.default_rng(92_000 + seed)
    labs = np.array(["g1", "g2", "g3"])
    info = dict(depth_source="synthetic", depth_fallback=False, depth_transform="log(v)",
                batch_column="synthetic")
    return dict(depth_ctrl=rng.normal(8, 0.4, len(panel["Xc"])),
                depth_env={p: rng.normal(8 - 0.1 * (k % 3), 0.4, len(X))
                           for k, (p, X) in enumerate(zip(panel["perts"], panel["Xperts"]))},
                batch_ctrl=labs[rng.integers(0, 3, len(panel["Xc"]))],
                batch_env={p: labs[rng.integers(0, 3, len(X))] for p, X in zip(panel["perts"], panel["Xperts"])},
                info=info)


# ------------------------------------------------------------------ readouts
def fit_depth_beta(Y_basis, d_basis):
    """Least squares Y = a + beta*depth per coordinate on BASIS cells."""
    dc = d_basis - d_basis.mean()
    return (dc @ (Y_basis - Y_basis.mean(0))) / (dc @ dc), float(d_basis.mean())


def readouts(Y, R, dep_e, dep_r, beta, dbar, B, rng, t1):
    """R1 to R4 for one environment (Y) against its size-matched REF draw (R)."""
    Ya = Y - np.outer(dep_e - dbar, beta)
    Ra = R - np.outer(dep_r - dbar, beta)
    t_reg, p_reg = E.permutation_test(Ya, Ra, B, rng(K_R3), "T1")
    t_no, p_no = E.permutation_test(Y[:, 1:], R[:, 1:], B, rng(K_R4), "T1")
    na, nb = len(Y), len(R)
    s11 = ((na - 1) * np.var(Y[:, 0], ddof=1) + (nb - 1) * np.var(R[:, 0], ddof=1)) / (na + nb - 2)
    diff1 = Y[:, 0].mean() - R[:, 0].mean()
    return dict(depth_diff=float(dep_e.mean() - dep_r.mean()),
                T1_depthreg=dict(stat=float(t_reg), p=float(p_reg)),
                T1_noPC1=dict(stat=float(t_no), p=float(p_no)),
                pc1_share=float((t1 - t_no) / t1), pc1_marginal=float(diff1 ** 2 / s11))


def tvd(a, ref_comp):
    labels, counts = np.unique(a, return_counts=True)
    pa = dict(zip(labels.tolist(), (counts / counts.sum()).tolist()))
    keys = set(pa) | set(ref_comp)
    return float(0.5 * sum(abs(pa.get(k, 0.0) - ref_comp.get(k, 0.0)) for k in keys))


def run_env(task):
    sh = E._SHARED
    seed, i, B = task["seed"], task["index"], sh["B"]
    Yh = task["Y_heldout"]
    n_e = len(Yh)
    # Verbatim from e12_common.run_environment (REF correction, REF draw, T1).
    Rh = sh["ref_heldout"]
    j = task["target_column"]
    if j is not None:
        xr = sh["ref_target_columns"][j]
        Rh = E.correct(Rh, xr, task["mu_h_j"], task["P_h_j"])
    rows = E.rng_for(seed, i, E.K_REFDRAW).choice(len(Rh), n_e, replace=False)
    t1 = E.permutation_test(Yh, Rh[rows], B, E.rng_for(seed, i, E.K_T1), "T1")[0]
    rec = dict(label=task["label"], index=i, is_fake=task["is_fake"], n_e=int(n_e),
               target_corrected=j is not None, T1_e12=float(t1), depth_source=sh["depth_source"])
    rec.update(readouts(Yh, Rh[rows], task["depth_env"], sh["depth_ref"][rows], sh["beta"],
                        sh["dbar"], B, lambda k: E.rng_for(seed, i, k), t1))
    rec["batch_tvd"] = None if task["batch_env"] is None else tvd(task["batch_env"], sh["batch_ref"])
    return rec


def prepare(panel, dep, seed, B, d, smoke):
    tasks, records, shared, design = S.build_tasks(panel, seed, B, d, smoke)
    Xc = panel["Xc"]
    basis, ref, fake_pool = E.control_split(len(Xc), seed)
    mu_h, P_h = E.fit_projection(Xc[basis], d)
    assert np.array_equal(E.project(Xc[ref], mu_h, P_h), shared["ref_heldout"])
    Y_basis = E.project(Xc[basis], mu_h, P_h)
    d_basis = dep["depth_ctrl"][basis]
    beta, dbar = fit_depth_beta(Y_basis, d_basis)
    corr = float(np.corrcoef(Y_basis[:, 0], d_basis)[0, 1])
    override = {}
    if "lib_ctrl" in dep:
        lib_b = dep["lib_ctrl"][basis]
        dist = dict(min=float(lib_b.min()), median=float(np.median(lib_b)), max=float(lib_b.max()))
        print(f"[depth] counts-layer row sum in BASIS cells: min={dist['min']:.0f} "
              f"median={dist['median']:.0f} max={dist['max']:.0f}", flush=True)
        if dist["median"] < COUNTS_MIN_MEDIAN:
            raise SystemExit(f"STOP: median BASIS counts-layer row sum {dist['median']:.1f} < "
                             f"{COUNTS_MIN_MEDIAN} (min {dist['min']:.1f}, max {dist['max']:.1f})")
        umi = dep["umi_obs_ctrl"]
        umi_b = None if umi is None else umi[basis]
        corr_umi = (float(np.corrcoef(d_basis, np.log(umi_b))[0, 1])
                    if umi_b is not None and np.all(umi_b > 0) else None)
        print(f"[depth] corr(depth_counts, log UMI_count_obs) in BASIS cells: {corr_umi}", flush=True)
        override = dict(depth_counts_basis=dist, corr_depth_counts_log_umi_obs_basis=corr_umi)
    d_pool = dep["depth_ctrl"][fake_pool]
    b_ref = None
    if dep["batch_ctrl"] is not None:
        labels, counts = np.unique(dep["batch_ctrl"][ref], return_counts=True)
        b_ref = dict(zip(labels.tolist(), (counts / counts.sum()).tolist()))
    for t in tasks:
        if t["is_fake"]:
            rows = E.rng_for(seed, t["index"], E.K_FAKE_ROWS).choice(
                len(fake_pool), design["fake_n"], replace=False)
            assert np.array_equal(E.project(Xc[fake_pool][rows], mu_h, P_h), t["Y_heldout"])
            t["depth_env"] = d_pool[rows]
            t["batch_env"] = None if b_ref is None else dep["batch_ctrl"][fake_pool][rows]
        else:
            t["depth_env"] = dep["depth_env"][t["label"]]
            assert len(t["depth_env"]) == len(t["Y_heldout"])
            t["batch_env"] = None if b_ref is None else dep["batch_env"][t["label"]]
    shared = dict(shared, B=B, depth_ref=dep["depth_ctrl"][ref], beta=beta, dbar=dbar,
                  batch_ref=b_ref, depth_source=dep["info"]["depth_source"])
    basis_info = dict(corr_pc1_depth_basis=corr, beta=beta.tolist(), basis_depth_mean=dbar,
                      n_basis=int(len(basis)), **override)
    return tasks, shared, design, basis_info


def run(panel, dep, dataset, seed, B, d, smoke, workers, e12):
    t0 = time.time()
    tasks, shared, design, basis_info = prepare(panel, dep, seed, B, d, smoke)
    assert design["excluded"] == e12["excluded"] and design["fake_n"] == e12["fake_n"], "E12 design differs"
    stored = {e["label"]: e for e in e12["environments"]}
    E.init_worker(shared)
    gate = [t for t in tasks if not t["is_fake"]][:N_REPRO]
    gate_diff = [abs(run_env(t)["T1_e12"] - stored[t["label"]]["T1"]["stat"]) for t in gate]
    if not gate_diff or max(gate_diff) > REPRO_TOL:
        raise SystemExit(f"STOP: reproduced T1 differs from stored E12 (diffs {gate_diff})")
    if workers <= 1:
        done = [run_env(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=E.init_worker,
                                 initargs=(shared,)) as pool:
            done = list(pool.map(run_env, tasks, chunksize=1))
    for rec in done:
        s = stored[rec["label"]]
        rec["T1_e12_stored"] = s["T1"]["stat"]
        rec["T1_e12_p"] = s["T1"]["p"]
        rec["T1_repro_absdiff"] = abs(rec["T1_e12"] - s["T1"]["stat"])
    worst = max(r["T1_repro_absdiff"] for r in done)
    if worst > REPRO_TOL:
        raise SystemExit(f"STOP: reproduced T1 differs from stored E12 (max abs diff {worst})")
    info = dep["info"]
    results = dict(
        experiment="E12b", prereg="notes/e12b_prereg.md", dataset=dataset, seed=int(seed), d=int(d),
        B=int(B), smoke=bool(smoke), alpha=E.ALPHA, q=E.Q, inspection=info,
        depth_source=info["depth_source"], depth_fallback=bool(info["depth_fallback"]),
        depth_transform=info["depth_transform"], batch_column=info["batch_column"],
        excluded=design["excluded"], fake_n=design["fake_n"], n_ref=design["n_ref"], **basis_info,
        r4_decision_relevant=bool(abs(basis_info["corr_pc1_depth_basis"]) > 0.5),
        reproduction=dict(gate_n=len(gate), gate_max_abs_diff=max(gate_diff),
                          all_n=len(done), all_max_abs_diff=worst, tol=REPRO_TOL),
        rng="SeedSequence([seed, i, k]); k 1 REF draw and 3 T1 (E12), 10 R3, 11 R4; fakes i = 100000 + f",
        environments=sorted(done, key=lambda r: r["index"]))
    results = json.loads(json.dumps(results))
    meta = dict(workers=int(workers), argv=sys.argv[1:], seconds=round(time.time() - t0, 3),
                provenance=T2.provenance(__file__, ["experiments/e12_common.py",
                                                    "experiments/e12_standard_tests.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ aggregate
def partial_spearman(x, y, z):
    if len(x) < 4:
        return None
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    r = np.corrcoef([rx, ry, rz])
    den = np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))
    return None if not np.isfinite(den) or den == 0 else float((r[0, 1] - r[0, 2] * r[1, 2]) / den)


def dataset_block(r):
    envs = r["environments"]
    reals = [e for e in envs if not e["is_fake"]]
    fakes = [e for e in envs if e["is_fake"]]
    n = len(reals)

    def bh(get):
        sel = E.PR.bh_fdr([get(e) for e in reals + fakes], q=E.Q)
        return sel[:n], sel[n:]

    e12_r, e12_f = bh(lambda e: e["T1_e12_p"])
    r3_r, r3_f = bh(lambda e: e["T1_depthreg"]["p"])
    r4_r, r4_f = bh(lambda e: e["T1_noPC1"]["p"])
    base = int(e12_r.sum())
    ratio = lambda k: (float(k) / base) if base else None  # noqa: E731
    shares = [e["pc1_share"] for e, s in zip(reals, e12_r) if s]
    tvds = [e["batch_tvd"] for e in reals if e["batch_tvd"] is not None]
    med_tvd = float(np.median(tvds)) if tvds else None
    return dict(
        depth_source=r["depth_source"], depth_fallback=r["depth_fallback"],
        batch_column=r["batch_column"], n_real=n, n_fake=len(fakes),
        corr_pc1_depth_basis=r["corr_pc1_depth_basis"], r4_decision_relevant=r["r4_decision_relevant"],
        e12_T1_bh_count_real=base, e12_T1_bh_count_fake=int(e12_f.sum()),
        R3_bh_count_real=int(r3_r.sum()), R3_bh_count_fake=int(r3_f.sum()),
        R4_bh_count_real=int(r4_r.sum()), R4_bh_count_fake=int(r4_f.sum()),
        retain_R3=ratio(r3_r.sum()), retain_R4=ratio(r4_r.sum()),
        partial_spearman_T1_per_n_vs_absdepth_given_n=partial_spearman(
            [e["T1_e12"] / e["n_e"] for e in reals], [abs(e["depth_diff"]) for e in reals],
            [e["n_e"] for e in reals]),
        median_pc1_share_e12_bh_detected=float(np.median(shares)) if shares else None,
        median_abs_depth_diff_real=float(np.median([abs(e["depth_diff"]) for e in reals])) if reals else None,
        median_abs_depth_diff_fake=float(np.median([abs(e["depth_diff"]) for e in fakes])) if fakes else None,
        median_batch_tvd_real=med_tvd,
        batch_exposed=None if r["batch_column"] is None or med_tvd is None else bool(med_tvd > 0.3),
        reproduction=r["reproduction"])


def verdict(blocks):
    dec = {k: blocks.get(k) for k in DECISION}
    if any(v is None for v in dec.values()):
        return dict(verdict=None, rule=None, text="not evaluable (missing inputs)")

    def low(v):
        return ((v["retain_R3"] is not None and v["retain_R3"] < 0.30)
                or (v["r4_decision_relevant"] and v["retain_R4"] is not None and v["retain_R4"] < 0.30))

    def high(v):
        return (v["retain_R3"] is not None and v["retain_R3"] >= 0.70
                and (not v["r4_decision_relevant"] or (v["retain_R4"] is not None and v["retain_R4"] >= 0.70))
                and v["R3_bh_count_fake"] <= 3)

    per = {k: dict(fallback=v["depth_fallback"], meets_ii=low(v), meets_iii=high(v)) for k, v in dec.items()}
    if any(v["depth_fallback"] for v in dec.values()):
        out = dict(verdict="S3", rule="(i)", text="S3, mixed: a decision dataset used the X row-sum depth fallback")
    elif any(low(v) for v in dec.values()):
        out = dict(verdict="S2", rule="(ii)", text="S2, depth-aligned: detections are depth-aligned; "
                   "confound vs genuine depth phenotype unresolved; D1 suspended pending discussion")
    elif all(high(v) for v in dec.values()):
        out = dict(verdict="S1", rule="(iii)", text="S1, survives: the mean-shift headline stands; D1 confirmed")
    else:
        out = dict(verdict="S3", rule="(iv)", text="S3, mixed: report per dataset and decide after discussion")
    out["per_dataset"] = per
    return out


def aggregate(root, seed=0, d=10, k562_from=None):
    """k562_from (E12c): load K562 from that folder and the other datasets from root."""
    root = Path(root)
    blocks, missing, inputs, sources = {}, [], {}, {}
    for ds in DATASETS:
        src = Path(k562_from) if (k562_from and ds == "k562") else root
        path = src / f"{ds}_seed{seed}_d{d}.json"
        sources[ds] = str(src)
        if not path.exists():
            missing.append(path.name)
            continue
        doc = json.loads(path.read_text())
        inputs[path.name] = doc["results_sha256"]
        blocks[ds] = dataset_block(doc["results"])
        e12_sum = E12_ROOT / "summary.json"
        if e12_sum.exists() and src.resolve() in (OUT_ROOT.resolve(), E12C_ROOT.resolve()):
            stored = json.loads(e12_sum.read_text())["results"]["perturbseq"][ds][f"d{d}"]["tests"]["T1"]
            blocks[ds]["e12_summary_T1_bh_count_real"] = stored["per_seed"][str(seed)]["bh_count_real"]
    k, n = blocks.get("k562"), blocks.get("norman")
    pred = dict(
        k562_R3_retains_ge_70=None if k is None or k["retain_R3"] is None else k["retain_R3"] >= 0.70,
        norman_R3_retains_ge_30=None if n is None or n["retain_R3"] is None else n["retain_R3"] >= 0.30,
        k562_median_pc1_share_lt_05=None if k is None or k["median_pc1_share_e12_bh_detected"] is None
        else k["median_pc1_share_e12_bh_detected"] < 0.5)
    results = dict(experiment="E12b", prereg="notes/e12b_prereg.md", seed=seed, d=d, datasets=blocks,
                   batch_flag={ds: b["batch_exposed"] for ds, b in blocks.items()},
                   verdict=verdict(blocks), prediction=pred, missing_inputs=missing,
                   input_results_sha256=inputs)
    if k562_from:
        results["dataset_sources"] = sources
        results["prediction_e12c"] = dict(
            norman_R3_retains_ge_70=None if n is None or n["retain_R3"] is None else n["retain_R3"] >= 0.70)
    return json.loads(json.dumps(results))


# ------------------------------------------------------------------ Step 3 self-checks
ORACLE_SALT, ORACLE_REPS, ORACLE_B = 12_100, 200, 199


def _oracle(o, depth_shift, coord_shift, reps=ORACLE_REPS, B=ORACLE_B, n=200, d=10):
    hits = dict(T1=0, T1_depthreg=0, T1_noPC1=0)
    for rep in range(reps):
        g = np.random.default_rng(np.random.SeedSequence([ORACLE_SALT, o, rep]))

        def cells(shift_depth, shift_coord):
            dep = g.standard_normal(n) + shift_depth
            Y = g.standard_normal((n, d))
            Y[:, 0] = dep + 0.5 * g.standard_normal(n)
            if shift_coord:
                Y[:, 2] += shift_coord
            return Y, dep

        Yb, db = cells(0.0, 0.0)
        Ye, de = cells(depth_shift, coord_shift)
        Yr, dr = cells(0.0, 0.0)
        beta, dbar = fit_depth_beta(Yb, db)
        rng = lambda k: E.rng_for(ORACLE_SALT + o, rep, k)  # noqa: E731
        t1, p1 = E.permutation_test(Ye, Yr, B, rng(E.K_T1), "T1")
        r = readouts(Ye, Yr, de, dr, beta, dbar, B, rng, t1)
        assert -1e-9 <= r["pc1_share"] <= 1 + 1e-9
        hits["T1"] += int(p1 <= E.ALPHA)
        hits["T1_depthreg"] += int(r["T1_depthreg"]["p"] <= E.ALPHA)
        hits["T1_noPC1"] += int(r["T1_noPC1"]["p"] <= E.ALPHA)
    return {k: v / reps for k, v in hits.items()}


def _synthetic_runs(workers, datasets=DATASETS, smoke=True):
    out = {}
    for ds in datasets:
        panel = E.make_selftest_panel_full(ds, seed=0)
        e12 = S.run(panel, "selftest", 0, SMOKE_B, 10, smoke, workers)["results"]
        out[ds] = run(panel, selftest_depth(panel), ds, 0, SMOKE_B, 10, smoke, workers, e12)
    return out


def oracles():
    import py_compile
    import tempfile

    t0 = time.time()
    dep = _oracle(1, 0.6, 0.0)
    real = _oracle(2, 0.0, 0.5)
    a = _synthetic_runs(1)
    b = _synthetic_runs(4)
    with tempfile.TemporaryDirectory() as tmp:
        for ds, doc in a.items():
            T2.write_json(Path(tmp) / f"{ds}_seed0_d10.json", doc)
        s1, s2 = aggregate(tmp), aggregate(tmp)
    repro = {ds: doc["results"]["reproduction"] for ds, doc in a.items()}
    shares = [e["pc1_share"] for doc in a.values() for e in doc["results"]["environments"]]
    g = np.random.default_rng(np.random.SeedSequence([ORACLE_SALT, 9]))
    x, y, z = g.standard_normal(80), g.standard_normal(80), g.standard_normal(80)
    rx, ry, rz = (stats.rankdata(v) for v in (x + z, y + z, z))
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
    ps_direct = float(np.corrcoef(res(rx), res(ry))[0, 1])
    ps_formula = partial_spearman(x + z, y + z, z)
    def blk(fallback=False, r3=0.8, r4=0.8, relevant=False, fake=0):
        return dict(depth_fallback=fallback, retain_R3=r3, retain_R4=r4,
                    r4_decision_relevant=relevant, R3_bh_count_fake=fake)
    cases = {
        "(i)": ({"k562": blk(), "norman": blk(fallback=True)}, "S3"),
        "(ii) R3": ({"k562": blk(r3=0.29), "norman": blk()}, "S2"),
        "(ii) R4 relevant": ({"k562": blk(), "norman": blk(r4=0.2, relevant=True)}, "S2"),
        "(iii)": ({"k562": blk(r4=0.1), "norman": blk(r3=0.7, fake=3)}, "S1"),
        "(iv) fakes": ({"k562": blk(), "norman": blk(fake=4)}, "S3"),
        "(iv) middle": ({"k562": blk(r3=0.5), "norman": blk()}, "S3"),
    }
    rule_out = {k: verdict(b)["verdict"] for k, (b, _) in cases.items()}
    dash = re.compile("[\u2014\u2013\u2012\u2015\u2e3a\u2e3b]")
    files = {}
    for f in (SCRIPT, "notes/e12b_prereg.md"):
        text = (E.REPO / f).read_text(encoding="utf-8")
        files[f] = dict(sha256=T2.sha256_file(E.REPO / f), lines=text.count("\n"),
                        dash_hits=[k for k, line in enumerate(text.splitlines(), 1) if dash.search(line)])
    py_compile.compile(str(E.REPO / SCRIPT), doraise=True)
    files[SCRIPT].update(py_compile="ok", grep_back="def readouts" in (E.REPO / SCRIPT).read_text())
    checks = dict(
        depth_oracle=dict(design="n_env = n_REF = 200, d=10, depth ~ N(0,1), Y_1 = depth + 0.5*noise, "
                                 "env depth +0.6, separate BASIS of 200", rates=dep,
                          criterion="T1 >= 0.85, T1_depthreg <= 0.10, T1_noPC1 <= 0.10",
                          passed=dep["T1"] >= 0.85 and dep["T1_depthreg"] <= 0.10 and dep["T1_noPC1"] <= 0.10),
        real_shift_oracle=dict(design="same sizes, own BASIS, +0.5 SD on coordinate 3, no depth difference",
                               rates=real, criterion="T1_depthreg >= 0.85 and T1_noPC1 >= 0.85",
                               passed=real["T1_depthreg"] >= 0.85 and real["T1_noPC1"] >= 0.85),
        reproduction_synthetic=dict(
            note="real-data reproduction runs as a STOP gate inside every A100 run (data not on Mac)",
            per_dataset=repro, passed=all(v["all_max_abs_diff"] <= REPRO_TOL for v in repro.values())),
        pc1_share_range=dict(min=min(shares), max=max(shares),
                             passed=min(shares) >= -1e-9 and max(shares) <= 1 + 1e-9),
        partial_spearman_identity=dict(formula=ps_formula, residual_route=ps_direct,
                                       passed=abs(ps_formula - ps_direct) < 1e-10),
        verdict_rules=dict(observed=rule_out, expected={k: v for k, (_, v) in cases.items()},
                           passed=all(rule_out[k] == v for k, (_, v) in cases.items())),
        determinism=dict(workers1={k: v["results_sha256"] for k, v in a.items()},
                         workers4={k: v["results_sha256"] for k, v in b.items()},
                         aggregate=[E.sha256_json(s1), E.sha256_json(s2)],
                         passed=all(a[k]["results_sha256"] == b[k]["results_sha256"] for k in a)
                         and s1 == s2),
        files=dict(files=files, passed=all(not v["dash_hits"] for v in files.values())
                   and files[SCRIPT]["grep_back"]))
    results = dict(experiment="E12b self-checks", reps=ORACLE_REPS, B=ORACLE_B, alpha=E.ALPHA,
                   rng=f"data SeedSequence([{ORACLE_SALT}, o, rep]); tests SeedSequence([{ORACLE_SALT}+o, rep, k])",
                   checks=checks, all_pass=all(v["passed"] for v in checks.values()))
    results = json.loads(json.dumps(results))
    meta = dict(seconds=round(time.time() - t0, 1), provenance=T2.provenance(
        __file__, ["experiments/e12_common.py", "experiments/e12_standard_tests.py"]))
    return dict(results=results, results_sha256=E.sha256_json(results), meta=meta)


# ------------------------------------------------------------------ main
def out_path(dataset, seed, d, smoke, depth_source="auto"):
    """auto keeps the E12b paths; the E12c override writes under results/e12c_norman_depth/."""
    base = OUT_ROOT if depth_source == "auto" else E12C_ROOT
    return (base / "smoke" if smoke else base) / f"{dataset}_seed{seed}_d{d}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--depth-source", choices=DEPTH_SOURCES, default="auto",
                    help="auto = E12b behaviour; counts_layer_rowsum = E12c Norman override")
    ap.add_argument("--root", default=str(OUT_ROOT), help="--aggregate input and output folder")
    ap.add_argument("--k562-from", default=None, help="--aggregate: load K562 from this folder")
    ap.add_argument("--inspect", action="store_true", help="print the Step 0 h5ad report only")
    ap.add_argument("--selftest", action="store_true", help="synthetic panel, code path only")
    ap.add_argument("--oracles", action="store_true", help="Step 3 self-checks")
    args = ap.parse_args()
    if args.oracles:
        doc = oracles()
        T2.write_json(OUT_ROOT / "oracles.json", doc)
        c = doc["results"]["checks"]
        print(json.dumps({k: dict(passed=v["passed"], **({"rates": v["rates"]} if "rates" in v else {}))
                          for k, v in c.items()}, indent=1))
        print(f"all_pass={doc['results']['all_pass']} -> {OUT_ROOT / 'oracles.json'}")
        if not doc["results"]["all_pass"]:
            raise SystemExit("STOP: a self-check failed")
        return
    if args.aggregate:
        t0 = time.time()
        results = aggregate(args.root, args.seed, args.d, k562_from=args.k562_from)
        doc = dict(results=results, results_sha256=E.sha256_json(results), meta=dict(
            seconds=round(time.time() - t0, 3), provenance=T2.provenance(__file__)))
        T2.write_json(Path(args.root) / "summary.json", doc)
        print(json.dumps(dict(verdict=results["verdict"], batch_flag=results["batch_flag"],
                              prediction=results["prediction"], missing=results["missing_inputs"]), indent=1))
        nb = results["datasets"].get("norman")
        if nb is not None:
            nv = results["verdict"].get("per_dataset", {}).get("norman", {})
            print(f"norman: depth={nb['depth_source']} meets_ii={nv.get('meets_ii')} "
                  f"meets_iii={nv.get('meets_iii')} retain_R3={nb['retain_R3']}")
        return
    if args.selftest:
        for ds, doc in _synthetic_runs(args.workers, datasets=(args.dataset or "k562",),
                                       smoke=args.smoke).items():
            print(ds, doc["results_sha256"], json.dumps(doc["results"]["reproduction"]))
        return
    datasets = [args.dataset] if args.dataset else (list(DATASETS) if args.inspect else [])
    if not datasets:
        ap.error("--dataset is required (or use --aggregate)")
    if args.inspect:
        for ds in datasets:
            info = inspect_h5ad(ds, args.depth_source)[0]
            print(json.dumps({ds: info}, indent=1), flush=True)
        return
    ds = args.dataset
    info, g, depth, batch, extra = inspect_h5ad(ds, args.depth_source)
    print(json.dumps({ds: info}), flush=True)
    panel = E.load_panel_full(ds)
    dep = depth_inputs(panel, info, g, depth, batch, T2.DATASETS[ds]["ctrl"], extra)
    del g, depth, batch, extra
    e12 = json.loads((E12_ROOT / ds / f"seed{args.seed}_d{args.d}.json").read_text())["results"]
    B = SMOKE_B if args.smoke else args.B
    doc = run(panel, dep, ds, args.seed, B, args.d, args.smoke, args.workers, e12)
    path = out_path(ds, args.seed, args.d, args.smoke, args.depth_source)
    T2.write_json(path, doc)
    r = doc["results"]
    print(f"[{ds}] depth={r['depth_source']} batch={r['batch_column']} corr(PC1,depth)="
          f"{r['corr_pc1_depth_basis']:.3f} repro={r['reproduction']} -> {path}")
    print(f"results_sha256={doc['results_sha256']} seconds={doc['meta']['seconds']}")


if __name__ == "__main__":
    main()
