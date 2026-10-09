"""E12 aggregate: BH families, predictions P1 to P6 and decisions D1 to D7, all computed
from the per-run JSON files (pre-registration: notes/e12_prereg.md).

Perturb-seq: one BH family (precision_readout.bh_fdr, q=0.05) per dataset, test, d and
seed: retained real perturbations plus the fakes. Raw rejection is p <= 0.05. HCP: per
seed, direction and test, BH separately over the 6 task environments and the fakes.
Over seeds: mean, min and max. Spearman(stat, n_e) over retained real perturbations.
Predictions and decisions use d=10 and seed means; the d=30 table is reported only.
T1_analytic_F and T4_analytic are descriptive and enter no prediction or decision.

Output: <root>/summary.json with {"results", "results_sha256", "meta"}; meta (timings,
host, provenance) is excluded from the hash.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e12_common as E  # noqa: E402

T2 = E.T2
DATASETS = ("k562", "norman", "rpe1")
DIMS = (10, 30)
SEEDS = (0, 1, 2)
PRIMARY_D = 10
ALL_TESTS = E.TESTS + ("T1_analytic_F", "T4_analytic")
P2_TESTS = ("T0h", "T0m", "T1", "T2", "T3")
EXPECTED_EXCLUDED = {"k562": [], "norman": ["KLF1"], "rpe1": ["ENSG00000108064"]}
KEYS = ("bh_count_real", "bh_frac_real", "bh_count_fake", "raw_reject_real", "raw_reject_fake")


def p_of(env, test):
    if test == "T1_analytic_F":
        return env["T1"]["p_analytic_F"]
    if test == "T4_analytic":
        return env["T4"]["p_analytic"]
    return env[test]["p"]


def stat_of(env, test):
    return env[{"T1_analytic_F": "T1", "T4_analytic": "T4"}.get(test, test)]["stat"]


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(stats.spearmanr(x, y)[0])


def over_seeds(per_seed, keys):
    out = {}
    for k in keys:
        vals = [v[k] for v in per_seed.values() if v.get(k) is not None]
        out[k] = (dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)),
                       n_seeds=len(vals)) if vals else None)
    return out


def family(reals, fakes, test):
    p_real = [p_of(e, test) for e in reals]
    p_fake = [p_of(e, test) for e in fakes]
    bh = E.PR.bh_fdr(p_real + p_fake, q=E.Q)
    n = len(reals)
    return dict(
        n_real=n, n_fake=len(fakes),
        bh_count_real=int(bh[:n].sum()), bh_frac_real=float(bh[:n].sum() / n) if n else None,
        bh_count_fake=int(bh[n:].sum()),
        raw_reject_real=float(np.mean(np.asarray(p_real) <= E.ALPHA)) if n else None,
        raw_reject_fake=float(np.mean(np.asarray(p_fake) <= E.ALPHA)) if fakes else None,
        spearman_stat_n_e=spearman([stat_of(e, test) for e in reals], [e["n_e"] for e in reals]),
        bh_selected_real=[e["label"] for e, s in zip(reals, bh[:n]) if s])


def load(path):
    return json.loads(Path(path).read_text())


def perturbseq_block(root, missing, inputs):
    out, fakes_raw = {}, {}
    for ds in DATASETS:
        out[ds] = {}
        for d in DIMS:
            runs = {}
            for s in SEEDS:
                path = root / ds / f"seed{s}_d{d}.json"
                if not path.exists():
                    missing.append(str(path.relative_to(root)))
                    continue
                doc = load(path)
                inputs[str(path.relative_to(root))] = doc["results_sha256"]
                runs[s] = doc["results"]
            if not runs:
                continue
            block = dict(seeds=sorted(runs), tests={})
            for test in ALL_TESTS:
                per_seed = {}
                for s, r in runs.items():
                    reals = [e for e in r["environments"] if not e["is_fake"] and not e["excluded"]]
                    fakes = [e for e in r["environments"] if e["is_fake"]]
                    per_seed[str(s)] = family(reals, fakes, test)
                block["tests"][test] = dict(per_seed=per_seed, over_seeds=over_seeds(
                    per_seed, KEYS + ("spearman_stat_n_e",)))
            first = runs[min(runs)]
            block["design"] = {str(s): dict(
                excluded=r["excluded"], n_ref=r["n_ref"], fake_n=r["fake_n"], B=r["B"],
                n_target_corrected_retained=r["n_target_corrected_retained"],
                target_mapping=r["target_mapping"]) for s, r in runs.items()}
            block["excluded_matches_prereg_expectation"] = all(
                sorted(r["excluded"]) == sorted(EXPECTED_EXCLUDED[ds]) for r in runs.values())
            block["n_powered"] = first["n_powered"]
            out[ds][f"d{d}"] = block
            fakes_raw[(ds, d)] = {s: [(e["T0i"]["p"] <= E.ALPHA, e["T0h"]["p"] <= E.ALPHA)
                                      for e in r["environments"] if e["is_fake"]]
                                  for s, r in runs.items()}
    return out, fakes_raw


def hcp_block(root, missing, inputs):
    runs = {}
    for s in SEEDS:
        path = root / "hcp" / f"seed{s}.json"
        if not path.exists():
            missing.append(str(path.relative_to(root)))
            continue
        doc = load(path)
        inputs[str(path.relative_to(root))] = doc["results_sha256"]
        runs[s] = doc["results"]
    if not runs:
        return None
    out = dict(seeds=sorted(runs), directions={})
    for direction in ("AB", "BA"):
        tests = {}
        for test in ("T0h", "T0i"):
            per_seed = {}
            for s, r in runs.items():
                block = r["directions"][direction]
                p_env = [e[test]["p"] for e in block["environments"]]
                p_fake = [e[test]["p"] for e in block["fakes"]]
                per_seed[str(s)] = dict(
                    n_env=len(p_env), n_fake=len(p_fake),
                    hcp_bh_count=int(E.PR.bh_fdr(p_env, q=E.Q).sum()),
                    bh_count_fake=int(E.PR.bh_fdr(p_fake, q=E.Q).sum()),
                    raw_reject_real=float(np.mean(np.asarray(p_env) <= E.ALPHA)),
                    raw_reject_fake=float(np.mean(np.asarray(p_fake) <= E.ALPHA)),
                    bh_selected=[e["label"] for e, k in zip(
                        block["environments"], E.PR.bh_fdr(p_env, q=E.Q)) if k])
            tests[test] = dict(per_seed=per_seed, over_seeds=over_seeds(
                per_seed, ("hcp_bh_count", "bh_count_fake", "raw_reject_real", "raw_reject_fake")))
        out["directions"][direction] = tests
    return out


def mean_of(ps, ds, d, test, key):
    try:
        v = ps[ds][f"d{d}"]["tests"][test]["over_seeds"][key]
    except KeyError:
        return None
    return None if v is None else v["mean"]


def all_present(*values):
    return all(v is not None for v in values)


def mcnemar_one_sided(pairs):
    """Exact one-sided McNemar: H1 T0i rejects more often than T0h on paired fakes."""
    b = sum(1 for i, h in pairs if i and not h)
    c = sum(1 for i, h in pairs if h and not i)
    p = float(stats.binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0
    return dict(n=len(pairs), t0i_only=b, t0h_only=c, p=p,
                rate_T0i=float(np.mean([i for i, _ in pairs])) if pairs else None,
                rate_T0h=float(np.mean([h for _, h in pairs])) if pairs else None)


def predictions(ps, hcp, fakes_raw, d):
    m = lambda ds, t, k: mean_of(ps, ds, d, t, k)  # noqa: E731
    out = {}
    v = dict(T1=m("k562", "T1", "bh_frac_real"), T3=m("k562", "T3", "bh_frac_real"),
             T0h=m("k562", "T0h", "bh_frac_real"))
    out["P1"] = dict(text="K562: T1 and T3 >= 30%, and T0h <= 2%", key="bh_frac_real", values=v,
                     holds=(v["T1"] >= 0.30 and v["T3"] >= 0.30 and v["T0h"] <= 0.02)
                     if all_present(*v.values()) else None)
    fails, seen = [], 0
    for ds in DATASETS:
        for t in P2_TESTS:
            try:
                per_seed = ps[ds][f"d{d}"]["tests"][t]["per_seed"]
            except KeyError:
                continue
            for s, row in per_seed.items():
                seen += 1
                if row["bh_count_fake"] > 3:
                    fails.append(dict(dataset=ds, test=t, seed=int(s), bh_count_fake=row["bh_count_fake"]))
    complete = seen == len(DATASETS) * len(P2_TESTS) * len(SEEDS)
    out["P2"] = dict(text="Fakes BH-selected <= 3 of 50 for T0h, T0m, T1, T2, T3, every dataset and seed",
                     key="bh_count_fake", failures=fails,
                     uncalibrated_tests=sorted({f"{f['dataset']}:{f['test']}" for f in fails}),
                     holds=(not fails) if complete else (False if fails else None))
    if hcp is None:
        out["P3"] = dict(text="HCP: T0i >= 4/6 and T0h <= 1/6, both directions", key="hcp_bh_count",
                         values=None, holds=None)
    else:
        v = {k: {t: hcp["directions"][k][t]["over_seeds"]["hcp_bh_count"]["mean"] for t in ("T0h", "T0i")}
             for k in ("AB", "BA")}
        out["P3"] = dict(text="HCP: T0i >= 4/6 and T0h <= 1/6, both directions", key="hcp_bh_count",
                         values=v, holds=all(v[k]["T0i"] >= 4 and v[k]["T0h"] <= 1 for k in v))
    h, i = m("rpe1", "T0h", "bh_count_real"), m("rpe1", "T0i", "bh_count_real")
    ratio = (h / i) if all_present(h, i) and i > 0 else None
    out["P4"] = dict(text="RPE1: T0h / T0i <= 0.60 (seed-mean BH counts)", key="bh_count_real",
                     values=dict(T0h=h, T0i=i, ratio=ratio),
                     undefined=bool(all_present(h, i) and i == 0),
                     holds=(ratio <= 0.60) if ratio is not None else None)
    a, b = m("k562", "T2", "bh_count_real"), m("k562", "T0h", "bh_count_real")
    out["P5"] = dict(text="K562: T2 > T0h (seed-mean BH counts)", key="bh_count_real",
                     values=dict(T2=a, T0h=b), holds=(a > b) if all_present(a, b) else None)
    tests6 = {}
    for ds in DATASETS:
        runs = fakes_raw.get((ds, d))
        if not runs or len(runs) < len(SEEDS):
            tests6[ds] = None
            continue
        r = mcnemar_one_sided([pair for s in sorted(runs) for pair in runs[s]])
        r["holds"] = bool(r["rate_T0i"] > r["rate_T0h"] and r["p"] < 0.05)
        tests6[ds] = r
    out["P6"] = dict(text="Fakes pooled over 3 seeds: T0i raw rejection > T0h, one-sided McNemar p < 0.05, "
                          "every Perturb-seq dataset", key="raw_reject_fake", values=tests6,
                     holds=all(v["holds"] for v in tests6.values()) if all(tests6.values()) else None)
    return out


def decisions(ps, hcp, preds):
    m = lambda ds, t: mean_of(ps, ds, PRIMARY_D, t, "bh_frac_real")  # noqa: E731
    out = {}
    k = {t: m("k562", t) for t in ("T0h", "T0m", "T1", "T2", "T3")}
    n = {t: m("norman", t) for t in ("T1", "T2", "T3")}
    d1 = d2 = None
    if all_present(*k.values()):
        d1 = bool((k["T1"] >= 0.20 or k["T3"] >= 0.20) and k["T0h"] <= 0.02 and k["T0m"] <= 0.02)
    if all_present(*k.values(), *n.values()):
        d2 = bool(all(k[t] <= 0.05 for t in ("T1", "T2", "T3")) and all(v <= 0.05 for v in n.values()))
    if d1 is None or d2 is None:
        q1 = "not evaluable (missing inputs)"
    elif d1:
        q1 = "D1: withdraw the thin-signal claim; pivot to a mean-aware screen"
    elif d2:
        q1 = "D2: thin signal stands; GRD abstention corroborated by T1 to T3; keep GRD"
    else:
        q1 = "D3: mixed; report all tests and decide after discussion"
    out["Q1"] = dict(inputs=dict(k562_bh_frac_real=k, norman_bh_frac_real=n), D1=d1, D2=d2,
                     D3=(not d1 and not d2) if d1 is not None and d2 is not None else None, branch=q1)
    p4 = preds["P4"]["values"]
    if p4["ratio"] is not None:
        d4 = bool(p4["ratio"] < 0.5)
        q4 = ("D4 triggered: rework every E3 real-data number on the held-out basis" if d4
              else "D4 not triggered")
    else:
        d4 = None
        q4 = ("D4 not triggered: ratio undefined (T0i = 0)" if preds["P4"]["undefined"]
              else "not evaluable (missing inputs)")
    out["D4"] = dict(inputs=p4, triggered=d4, branch=q4)
    if hcp is None:
        out["HCP"] = dict(inputs=None, branch="not evaluable (missing inputs)")
    else:
        v = preds["P3"]["values"]
        both = lambda f: all(f(v[x]) for x in ("AB", "BA"))  # noqa: E731
        d5 = both(lambda r: r["T0i"] >= 4 and r["T0h"] <= 1)
        d6 = any(v[x]["T0i"] < 4 for x in ("AB", "BA"))
        d7 = both(lambda r: r["T0h"] >= 2 and r["T0i"] >= 4)
        if d5:
            br = "D5: artefact confirmed; the fMRI section is removed"
        elif d6:
            br = "D6: inconclusive at this sample size; the fMRI section is removed"
        elif d7:
            br = "D7: detections survive the held-out basis; keep fMRI on held-out numbers only"
        else:
            br = "mixed: not covered by D5 to D7; the fMRI section is removed"
        out["HCP"] = dict(inputs=v, D5=d5, D6=d6, D7=d7, branch=br)
    return out


def aggregate(root):
    root = Path(root)
    missing, inputs = [], {}
    ps, fakes_raw = perturbseq_block(root, missing, inputs)
    hcp = hcp_block(root, missing, inputs)
    preds = {f"d{d}": predictions(ps, hcp, fakes_raw, d) for d in DIMS}
    results = dict(
        experiment="E12", prereg="notes/e12_prereg.md", alpha=E.ALPHA, q=E.Q,
        primary_d=PRIMARY_D, seeds=list(SEEDS), tests=list(ALL_TESTS),
        descriptive_tests=["T1_analytic_F", "T4_analytic"],
        perturbseq=ps, hcp=hcp,
        hcp_bh_count=None if hcp is None else {
            k: {t: hcp["directions"][k][t]["over_seeds"]["hcp_bh_count"] for t in ("T0h", "T0i")}
            for k in ("AB", "BA")},
        predictions=preds[f"d{PRIMARY_D}"],
        predictions_d30_reported_only=preds["d30"],
        decisions=decisions(ps, hcp, preds[f"d{PRIMARY_D}"]),
        missing_inputs=sorted(missing), input_results_sha256=dict(sorted(inputs.items())))
    return json.loads(json.dumps(results))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(E.REPO / "results" / "e12_standard_tests"))
    args = ap.parse_args()
    t0 = time.time()
    results = aggregate(args.root)
    doc = dict(results=results, results_sha256=E.sha256_json(results), meta=dict(
        seconds=round(time.time() - t0, 3), root=str(args.root),
        provenance=T2.provenance(__file__, ["experiments/e12_common.py"])))
    out = Path(args.root) / "summary.json"
    T2.write_json(out, doc)
    print(f"-> {out}  results_sha256={doc['results_sha256']}")
    if results["missing_inputs"]:
        print(f"missing inputs ({len(results['missing_inputs'])}): {results['missing_inputs']}")
    for pid, p in results["predictions"].items():
        print(f"{pid} holds={p['holds']}  {p['text']}")
    for name, dec in results["decisions"].items():
        print(f"{name}: {dec['branch']}")


if __name__ == "__main__":
    main()
