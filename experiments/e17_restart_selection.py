import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

"""E17: restart selection in front of the gate, on fresh seeds 10 to 19.

Pre-registration: notes/e17_prereg.md (committed before this script).

E16's poly arm (experiments/e16_neural_backbone.py at b29139e, imported, not edited) with:
an 80/20 split of the basis environment (standardisation on the training rows), five
restarts with model seeds 170_000 + 100_000 r + 1000 seed + 100 arm_index + level_index,
selection by full-batch validation reconstruction MSE (ties to the lower r), then E16's
downstream unchanged on the selected encoder. E14, E10 and E2 come through E16 (one
module instance each). Secondary rule G2a: G2 at q=0.05 restricted to systems whose
selected validation loss is <= 0.02 (point metrics only).

Modes: --selftest (writes nothing), default data pass, --analyse.
"""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e16_neural_backbone as E16  # noqa: E402

E14, E10, E2 = E16.E14, E16.E10, E16.E2
T2 = E16.T2
REPO = E16.REPO
OUT = REPO / "results" / "e17"
MIXING = "poly"
SEEDS = list(range(10, 20))
N_RESTARTS = 5
SPLIT_SEED, MODEL_SEED = 175_000, 170_000
VAL_FRAC = 0.2
G2A_LOSS = 0.02
P3_MIN_BASIS = 2000
BANDS = ((None, 0.90), (0.90, 0.99), (0.99, None))
SELFTEST_STEPS, SELFTEST_B, SELFTEST_RESTARTS, SELFTEST_NBOOT = 300, 20, 2, 10


def _idx(arm, lvl):
    return 100 * E16.ARM_INDEX[E10.ARM_KEY[arm]] + E2.ARMS[arm]["levels"].index(lvl)


def split_basis(Xb, arm, lvl, seed):
    n_basis = len(Xb)
    perm = np.random.default_rng(SPLIT_SEED + 1000 * seed + _idx(arm, lvl)).permutation(n_basis)
    n_val = max(1, int(np.floor(VAL_FRAC * n_basis)))
    return Xb[np.sort(perm[n_val:])], Xb[np.sort(perm[:n_val])]


def restart_seed(r, arm, lvl, seed):
    return MODEL_SEED + 100_000 * r + 1000 * seed + _idx(arm, lvl)


def run_system(job):
    """One system: build, split, restarts, selection, E16's downstream on the selected encoder.
    job = (arm, lvl, seed, steps, B, n_boot, n_restarts, split, seed_override)."""
    arm, lvl, seed, steps, B, n_boot, n_restarts, split, seed_override = job
    key = dict(arm=E10.ARM_KEY[arm], level=lvl, seed=seed)
    t0 = time.time()
    data = E16.build(arm, lvl, seed, MIXING)
    out = dict(key=key, mixing=MIXING, z_regression_max_abs=E16.z_regression(arm, lvl, seed, data),
               contam_replay_max_abs=E16.replay_contam(arm, lvl, seed, data) if data["contam"] else None,
               n_nodes=len(data["nodes"]))
    nodes = data["nodes"]
    obs_Z = data["envs"]["obs"]["Z"]
    Xb = data["envs"]["basis"]["X"]
    tr, va = split_basis(Xb, arm, lvl, seed) if split else (Xb, None)
    mu, sdv = tr.mean(0), tr.std(0)
    stdz = lambda X: ((X - mu) / sdv).astype(np.float32)  # noqa: E731
    Xobs_s = stdz(data["envs"]["obs"]["X"])
    restarts, models = [], {}
    for r in range(n_restarts):
        S = seed_override if seed_override is not None else restart_seed(r, arm, lvl, seed)
        rec = dict(r=r, model_seed=S)
        try:
            enc, dec = E16.train_autoencoder(stdz(tr), MIXING, S, steps)
        except FloatingPointError as exc:  # non-finite training loss: restart excluded
            rec.update(failed=f"{type(exc).__name__}: {exc}")
            restarts.append(rec)
            continue
        _, train_loss = E16._encode(enc, dec, stdz(tr))
        val_loss = E16._encode(enc, dec, stdz(va))[1] if va is not None else None
        if not np.isfinite(train_loss) or (val_loss is not None and not np.isfinite(val_loss)):
            rec.update(failed="non-finite training or validation loss", train_loss=train_loss, val_loss=val_loss)
            restarts.append(rec)
            continue
        Yo, _ = E16._encode(enc, dec, Xobs_s)
        rec.update(train_loss=train_loss, val_loss=val_loss,
                   affine_r2_obs=E16.affine_r2(Yo, obs_Z) if np.all(np.isfinite(Yo)) else None)
        restarts.append(rec)
        models[r] = (enc, dec)
    ok = [x for x in restarts if "failed" not in x]
    out.update(n_basis=int(len(Xb)), n_train=int(len(tr)), n_val=(int(len(va)) if va is not None else None),
               restarts=restarts)
    try:
        if not ok:
            raise RuntimeError("all restarts failed")
        sel = min(ok, key=lambda x: ((x["val_loss"] if x["val_loss"] is not None else 0.0), x["r"]))
        out.update(selected_r=sel["r"], selected_val_loss=sel["val_loss"], selected_model_seed=sel["model_seed"])
        enc, dec = models[sel["r"]]
        Y_obs, loss_obs = E16._encode(enc, dec, Xobs_s)
        Y_int = [E16._encode(enc, dec, stdz(data["envs"][f"iv{i}"]["X"]))[0] for i in nodes]
        if not (np.all(np.isfinite(Y_obs)) and all(np.all(np.isfinite(Y)) for Y in Y_int)):
            raise FloatingPointError("non-finite codes")
        r2_int = {str(i): E16.affine_r2(Y, data["envs"][f"iv{i}"]["Z"])
                  for i, Y in zip(nodes, Y_int) if i not in data["contam"]}
        out["diagnostics"] = dict(affine_r2_obs=E16.affine_r2(Y_obs, obs_Z), affine_r2_int=r2_int,
                                  recon_loss_train=sel["train_loss"], recon_loss_obs=loss_obs,
                                  n_basis=int(len(Xb)), n_train=int(len(tr)),
                                  n_val=(int(len(va)) if va is not None else None))
        g = E14._call(Y_int, Y_obs, seed, B, "precision")
        rows = E16.unit_rows(arm, lvl, seed, Y_obs, Y_int, nodes, obs_Z, g, n_boot)
        out["readout_failures"] = E16._readouts(rows, Y_int, Y_obs, seed, B)
        out["units"] = rows
    except Exception as exc:  # no fallback to another restart; units count as G0 only
        out["failed"] = f"{type(exc).__name__}: {exc}"
        out["units"] = None
    out["seconds"] = round(time.time() - t0, 2)
    return out


def versions():
    return E16.versions()


def data_pass(workers, seeds=SEEDS, levels_per_arm=None, steps=E16.STEPS, B=E16.B_GATE, n_boot=E16.N_BOOT,
              n_restarts=N_RESTARTS, write=True):
    t0 = time.time()
    syst = list(E14.systems(seeds, levels_per_arm))
    jobs = [(arm, lvl, seed, steps, B, n_boot, n_restarts, True, None) for arm, lvl, seed in syst]
    units, failed, diags, readout_failures = [], [], [], []
    config = dict(mixing=MIXING, seeds=list(seeds), steps=steps, B=B, n_boot=n_boot, n_restarts=n_restarts,
                  split="perm = default_rng(175000 + 1000 seed + 100 arm + level_index); first max(1, floor(0.2 n)) "
                        "validation", model_seed="170000 + 100000 r + 1000 seed + 100 arm + level_index",
                  selection="lowest validation MSE, ties to lower r", null_rng="default_rng(910_000 + seed)")
    for (arm, lvl, seed), res in zip(syst, E16.pool_map(run_system, jobs, workers)):
        if res["z_regression_max_abs"] != 0.0 or res["contam_replay_max_abs"] not in (None, 0.0):
            raise SystemExit(f"STOP: Z or contamination regression failed for {res['key']}; run void")
        if res.get("failed"):
            failed.append(dict(res["key"], reason=res["failed"]))
            units += E16.failed_units(res, arm)
        else:
            units += res["units"]
            readout_failures += [dict(res["key"], **f) for f in res["readout_failures"]]
        diags.append(dict(res["key"], seconds=res["seconds"], failed=res.get("failed"), n_basis=res["n_basis"],
                          n_train=res["n_train"], n_val=res["n_val"], restarts=res["restarts"],
                          selected_r=res.get("selected_r"), selected_val_loss=res.get("selected_val_loss"),
                          **{k: v for k, v in (res.get("diagnostics") or {}).items()
                             if k not in ("n_basis", "n_train", "n_val")}))
        if write:
            T2.write_json(OUT / "units.json", dict(
                experiment="E17", prereg="notes/e17_prereg.md", status="partial", config=config, units=units,
                diagnostics=diags, failed_systems=failed, readout_failures=readout_failures, versions=versions()))
            T2.write_json(OUT / "failed_units.json", dict(failed_systems=failed, readout_failures=readout_failures))
        print(f"  {E10.ARM_KEY[arm]} level={lvl} seed={seed} units={len(units)} failed={len(failed)} "
              f"sel_r={res.get('selected_r')} r2={(res.get('diagnostics') or {}).get('affine_r2_obs')} "
              f"({time.time() - t0:.0f}s)", flush=True)
    doc = dict(experiment="E17", prereg="notes/e17_prereg.md", status="complete", config=config,
               n_systems=len(syst), n_units=len(units), units=units, diagnostics=diags, failed_systems=failed,
               readout_failures=readout_failures, wall_seconds=round(time.time() - t0, 1), versions=versions(),
               provenance=T2.provenance(__file__, ["experiments/e16_neural_backbone.py",
                                                   "experiments/e14_matched_coverage.py",
                                                   "experiments/e10_risk_coverage.py",
                                                   "experiments/e2_calibration.py", "sim/simulator.py"]))
    if write:
        T2.write_json(OUT / "units.json", doc)
    return doc


# ------------------------------------------------------------------ analysis
def _sys_key(u):
    return (u["arm"], u["level"], u["seed"])


def _point(dec_mask, good, mask):
    m = E10._metrics(dec_mask, good, mask)
    return dict(coverage=m["coverage"], risk=m["risk"], n_certified=m["n_certified"], n_units=m["n_units"])


def analyse(units_doc=None, n_draws=E16.N_DRAWS, write=True):
    t0 = time.time()
    doc = units_doc or json.loads((OUT / "units.json").read_text())
    if doc.get("status") != "complete":
        raise SystemExit("STOP: results/e17/units.json is not complete")
    units = doc["units"]
    diags = doc["diagnostics"]
    n_sys, n_failed = len(diags), len(doc["failed_systems"])
    too_many = n_sys > 0 and n_failed / n_sys > E16.FAIL_FRAC
    readout_missing = [u for u in units if not u.get("failed_system")
                       and (u.get("geneig_pvalue") is None or u.get("energy_pvalue") is None)]
    for u in readout_missing:  # E16's readout placeholder fill
        u["geneig_pvalue"] = 1.0 if u.get("geneig_pvalue") is None else u["geneig_pvalue"]
        u["energy_pvalue"] = 1.0 if u.get("energy_pvalue") is None else u["energy_pvalue"]
    good = np.array([u["good"] for u in units])
    dec = E14.decisions(units)
    failed_mask = np.array([bool(u.get("failed_system")) for u in units])
    if failed_mask.any():  # E16's failed-system mask
        for gate, ops in dec.items():
            if gate != "G0":
                dec[gate] = [(k, d & ~failed_mask) for k, d in ops]
    masks = E14._masks(units)
    cv = E14.curves_for(dec, good, masks)
    sm = E14.summary_of(cv)
    pts = E16._point_curves(dec, good, masks, E16.POINT_ONLY, ("ABD", "C"))
    point_only = {g: {s: dict(aurc=E10._aurc(pts[g][s]["curve"])["value"],
                              matched_risk=E10._matched(pts[g][s]["curve"], sm[s]["target_coverage"]))
                      for s in ("ABD", "C")} for g in E16.POINT_ONLY}
    ci, ci_diff = E16.bootstrap16(dec, good, masks, E10._families(units), n_draws)

    # G2a: G2 at q=0.05 restricted to systems with selected validation loss <= 0.02 (point only)
    by_sys = {(d["arm"], d["level"], d["seed"]): d for d in diags}
    g2 = dict(dec["G2"])[0.05]
    val = [by_sys[_sys_key(u)].get("selected_val_loss") for u in units]
    obsl = [by_sys[_sys_key(u)].get("recon_loss_obs") for u in units]
    ok_val = np.array([v is not None and v <= G2A_LOSS for v in val])
    ok_obs = np.array([v is not None and v <= G2A_LOSS for v in obsl])
    g2a = {s: _point(g2 & ok_val, good, masks[s]) for s in ("ABD", "C")}
    g2a_obs = {s: _point(g2 & ok_obs, good, masks[s]) for s in ("ABD", "C")}

    # P3: 160 systems with n_basis >= 2000; failed systems count below 0.90
    p3_sys = [d for d in diags if d["n_basis"] >= P3_MIN_BASIS]
    p3_hits = [bool(d.get("failed") is None and d.get("affine_r2_obs") is not None and d["affine_r2_obs"] >= 0.90)
               for d in p3_sys]
    p3_share = float(np.mean(p3_hits)) if p3_hits else None

    # restart selection diagnostics
    hist = {str(r): 0 for r in range(N_RESTARTS)}
    best_match, n_best = 0, 0
    for d in diags:
        if d.get("selected_r") is not None:
            hist[str(d["selected_r"])] += 1
            okr = [x for x in d["restarts"] if "failed" not in x and x.get("affine_r2_obs") is not None]
            if okr:
                n_best += 1
                best = max(okr, key=lambda x: (x["affine_r2_obs"], -x["r"]))
                best_match += int(best["r"] == d["selected_r"])

    # 3-band affine R^2 table on ABD units (selected encoder's obs R^2; failed systems in the lowest band)
    r2u = np.array([by_sys[_sys_key(u)].get("affine_r2_obs") if by_sys[_sys_key(u)].get("affine_r2_obs")
                    is not None else -np.inf for u in units], dtype=float)
    abd = masks["ABD"]
    bands = {}
    for lo, hi in BANDS:
        name = f"{'-inf' if lo is None else lo} to {'inf' if hi is None else hi}"
        bm = abd & (r2u >= (-np.inf if lo is None else lo)) & (r2u < (np.inf if hi is None else hi))
        n = int(bm.sum())
        cert = g2 & bm
        bands[name] = dict(abd_units=n, recoverable=(float(good[bm].mean()) if n else None),
                           G2_coverage_q05=(float(cert.sum() / n) if n else None),
                           G2_risk_q05=(float((cert & ~good).sum() / cert.sum()) if cert.sum() else None))

    G0 = {s: E16._g0(cv, s) for s in ("ABD", "C")}
    r2 = sm["ABD"]["G2"]["risk_q05"]
    c2 = ci_diff["ABD"]["G0-G2_risk_q05"]
    preds = dict(
        P1=dict(verdict=E16._v(None if r2 is None else r2 <= 0.05), risk_G2_q05=r2, threshold=0.05),
        P2=dict(verdict=E16._v(None if not E16._interval_ok(c2) else c2["lo"] > 0), interval=c2,
                risk_G0=G0["ABD"]["risk"], risk_G2_q05=r2),
        P3=dict(verdict=E16._v(None if p3_share is None else p3_share >= 0.90), share=p3_share,
                n_systems=len(p3_sys), n_at_least_090=int(sum(p3_hits)), threshold=0.90, e16_single_run=0.71),
        P4=dict(verdict=E16._v(None if sm["ABD"]["G2"]["coverage_q05"] is None
                               else sm["ABD"]["G2"]["coverage_q05"] >= 0.60),
                coverage_G2_q05=sm["ABD"]["G2"]["coverage_q05"], threshold=0.60),
        P5=dict(verdict=E16._v(None if g2a["ABD"]["risk"] is None or g2a["ABD"]["coverage"] is None
                               else (g2a["ABD"]["risk"] <= 0.05 and g2a["ABD"]["coverage"] >= 0.30)),
                risk_G2a=g2a["ABD"]["risk"], coverage_G2a=g2a["ABD"]["coverage"]),
        P6=dict(verdict=E16._v(None if sm["C"]["G2"]["risk_q05"] is None else sm["C"]["G2"]["risk_q05"] >= 0.10),
                risk_G2_q05_C=sm["C"]["G2"]["risk_q05"], threshold=0.10))
    if too_many:
        for p in preds.values():
            p["verdict"] = "undetermined"
            p["note_failed"] = "more than 5% of systems failed"
    table = {s: dict(target_coverage=sm[s]["target_coverage"], G0=G0[s], G2a=g2a[s], G2a_obs_loss_descriptive=g2a_obs[s],
                     gates={g: dict(aurc=sm[s][g]["aurc"], aurc_ci=ci[s][g]["aurc"],
                                    matched_risk=sm[s][g]["matched_risk"], matched_ci=ci[s][g]["matched_risk"],
                                    **({k: sm[s][g][k] for k in ("coverage_q05", "risk_q05")}
                                       if g in ("G2", "G8", "G9") else {}))
                            for g in E16.GATES_E16}) for s in ("ABD", "C")}
    caveat_cv = E14.curves_for(dec, good, masks, scopes=tuple(f"B_level{lv}" for lv in E14.CAVEAT_LEVELS))
    caveat = {f"B_level{lv}": {g: E10._knob_point(caveat_cv[g][f"B_level{lv}"]["points"], E10.Q_DEFAULT)
                               for g in ("G2", "G8", "G9")} for lv in E14.CAVEAT_LEVELS}
    r2o = [d["affine_r2_obs"] for d in diags if d.get("affine_r2_obs") is not None]
    r2i = [v for d in diags for v in (d.get("affine_r2_int") or {}).values()]
    results = dict(
        experiment="E17", prereg="notes/e17_prereg.md", n_units=len(units), n_systems=n_sys,
        n_failed_systems=n_failed, failed_fraction=(n_failed / n_sys if n_sys else None), too_many_failed=too_many,
        n_readout_placeholder_units=len(readout_missing), e17=table, point_only_G1_G3_G4=point_only,
        caveat_arm_B=caveat, affine_r2=dict(obs_median=float(np.median(r2o)) if r2o else None, obs_n=len(r2o),
                                             int_median=float(np.median(r2i)) if r2i else None, int_n=len(r2i)),
        restart_selection=dict(histogram=hist, selected_equals_best_by_affine_r2=best_match, n_compared=n_best,
                               n_failed_restarts=sum(1 for d in diags for x in d["restarts"] if "failed" in x)),
        affine_r2_bands_abd=bands, ci_diff={s: ci_diff[s] for s in ("ABD", "C")}, predictions=preds,
        bootstrap=dict(draws=n_draws, rng=f"default_rng({E14.BOOT_SEED})", min_defined=E16.MIN_DEFINED))
    results = json.loads(json.dumps(results))
    if write:
        meta = dict(seconds=round(time.time() - t0, 1), units_sha256=T2.sha256_file(OUT / "units.json"),
                    versions=versions(), provenance=T2.provenance(__file__, ["experiments/e16_neural_backbone.py"]))
        T2.write_json(OUT / "curves.json", dict(results=dict(summary=sm, ci=ci, ci_diff=ci_diff,
                                                             curves=E14._jsonable(cv)), meta=meta))
        T2.write_json(OUT / "predictions.json", dict(results=results, meta=meta))
    return results


# ------------------------------------------------------------------ selftest
def selftest():
    t0 = time.time()
    syst = list(E14.systems([10], 2))
    zr = cr = 0.0
    for arm, lvl, seed in syst:
        d = E16.build(arm, lvl, seed, MIXING)
        zr = max(zr, E16.z_regression(arm, lvl, seed, d))
        if d["contam"]:
            cr = max(cr, E16.replay_contam(arm, lvl, seed, d))
    print(f"[selftest] {len(syst)} systems seed 10: poly Z vs linear Z max |dZ| {zr}; arm C Zc and gain vs replay "
          f"max diff {cr}", flush=True)
    if zr != 0.0 or cr != 0.0:
        raise SystemExit("STOP: Z or contamination regression failed")

    E16._init_worker()
    mism = []
    for arm, lvl, seed in syst:
        e16 = E16.run_system((MIXING, arm, lvl, seed, SELFTEST_STEPS, SELFTEST_B, SELFTEST_NBOOT, True))
        e17 = run_system((arm, lvl, seed, SELFTEST_STEPS, SELFTEST_B, SELFTEST_NBOOT, 1, False,
                          E16._model_seed(arm, lvl, seed)))
        same_units = json.dumps(e16["units"], sort_keys=True) == json.dumps(e17["units"], sort_keys=True)
        d16, d17 = e16["diagnostics"], e17["diagnostics"]
        same_diag = (d16["affine_r2_obs"] == d17["affine_r2_obs"] and d16["affine_r2_int"] == d17["affine_r2_int"]
                     and d16["recon_loss_basis"] == d17["recon_loss_train"]
                     and d16["recon_loss_obs"] == d17["recon_loss_obs"] and d16["n_basis"] == d17["n_basis"])
        if not (same_units and same_diag and e17["selected_r"] == 0 and e17["selected_val_loss"] is None):
            mism.append(dict(key=e17["key"], units=same_units, diagnostics=same_diag))
    print(f"[selftest] equivalence with E16.run_system (1 restart, no split, E16 model seed, n_boot=10): "
          f"{len(syst)} systems, {len(mism)} mismatches", flush=True)
    if mism:
        raise SystemExit(f"STOP: E17 path does not reproduce E16: {mism}")

    runs = {}
    for w in (1, 4):
        runs[w] = data_pass(w, seeds=[10], levels_per_arm=2, steps=SELFTEST_STEPS, B=SELFTEST_B,
                            n_boot=SELFTEST_NBOOT, n_restarts=SELFTEST_RESTARTS, write=False)
    strip = lambda ds: [{k: v for k, v in d.items() if k != "seconds"} for d in ds]  # noqa: E731
    same = (json.dumps(runs[1]["units"], sort_keys=True) == json.dumps(runs[4]["units"], sort_keys=True)
            and json.dumps(strip(runs[1]["diagnostics"]), sort_keys=True)
            == json.dumps(strip(runs[4]["diagnostics"]), sort_keys=True))
    sel = [d["selected_r"] for d in runs[1]["diagnostics"]]
    print(f"[selftest] data pass {len(runs[1]['units'])} units, {SELFTEST_RESTARTS} restarts: workers 1 vs 4 "
          f"identical {same}; selected r {sel}; failed systems {len(runs[1]['failed_systems'])}", flush=True)
    if not same:
        raise SystemExit("STOP: results differ between 1 and 4 workers")
    res = analyse(runs[1], n_draws=50, write=False)
    print(f"[selftest] analysis end to end: units {res['n_units']}, predictions "
          f"{ {k: v['verdict'] for k, v in res['predictions'].items()} }, G2a ABD {res['e17']['ABD']['G2a']}",
          flush=True)
    print(f"[selftest] passed (nothing written) {time.time() - t0:.1f}s; versions {versions()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E17: restart selection in front of the gate")
    ap.add_argument("--workers", type=int, default=E16.MAX_WORKERS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.workers > E16.MAX_WORKERS:
        ap.error(f"at most {E16.MAX_WORKERS} workers")
    if a.selftest:
        selftest()
    elif a.analyse:
        r = analyse()
        print(json.dumps(dict(predictions={k: v["verdict"] for k, v in r["predictions"].items()},
                              failed=r["n_failed_systems"], restart_selection=r["restart_selection"]), indent=1),
              flush=True)
    else:
        doc = data_pass(a.workers)
        print(f"[data pass] {doc['n_units']} units, {len(doc['failed_systems'])} failed systems, "
              f"{doc['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
