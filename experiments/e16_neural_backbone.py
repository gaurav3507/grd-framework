"""E16: the gate in front of a neural backbone (autoencoder on nonlinear mixing).

Pre-registration: notes/e16_prereg.md, committed before this script.

The E10 benchmark (four E2 arms, every level, seeds 0-9, one unit per intervened
direction) with the deterministic mixing step replaced (poly primary, mlp exploratory)
and the PCA projection replaced by an autoencoder trained on the basis environment.
Everything downstream is E10's and E14's code applied to the codes. E14 is loaded once
with tier2_common.load_module; E10, E2, BK, PR and SIM are taken from it (one instance).

Builders (build): repeat SIM.simulate's random-number order (make_scm, A, then per
environment sample_latent, mixing, observation noise) and arm_C's contamination stream
default_rng(seed + 7); only the mixing changes. Mixing parameters come from
default_rng(160_000 + seed). mixing="linear" is the regression path: X bit-identical to
E2.ARMS[arm]["builder"](seed, level) in every environment.

Unit record (unit_rows): E10.unit_records' record logic, taking projections and the
precision readout call as inputs (E10.unit_records builds its own PCA). The selftest runs
it on E14.projected() output and reproduces results/e10_risk_coverage/units.json.

Modes: --selftest (writes nothing), --preflight, default data pass (--mixing poly|mlp),
--analyse --mixing poly|mlp. At most 16 workers, one per system; torch single-threaded
and deterministic in every worker.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tier2_common as T2  # noqa: E402

warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

E14 = T2.load_module("grd_e14_for_e16", "experiments/e14_matched_coverage.py")
E10, E2, BK, PR = E14.E10, E14.E2, E14.BK, E14.PR
SIM = E2.SIM

OUT = REPO / "results" / "e16"
E10_UNITS = REPO / "results" / "e10_risk_coverage" / "units.json"
MIXINGS = ("poly", "mlp")
D_LATENT, D_OBS = E2.D_LATENT, E2.D_OBS
KAPPA = 0.5
MIX_SEED = 160_000
STEPS = 6000
LR = 1e-3
BATCH = 256
HIDDEN = 128
SLOPE = 0.1
B_GATE = E10.B_NULL  # 500
N_BOOT = E10.N_BOOT  # 50
ARM_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
MAX_WORKERS = 16
FAIL_FRAC = 0.05
MIN_DEFINED = 900
N_DRAWS = E14.N_SYS_BOOT
GATES_E16 = ("G0", "G2", "G5", "G8", "G9")
POINT_ONLY = ("G1", "G3", "G4")
SELFTEST_STEPS, SELFTEST_B = 300, 20
REGRESSION_FLOAT = ("signal", "threshold", "best_match_corr", "boot_angle_median", "boot_angle_p90")
REGRESSION_FLOAT_TOL = 1e-9
REGRESSION_EXCLUDED = ("corr_all", "target", "corr_target")


# ------------------------------------------------------------------ builders
def _mixer(mixing, seed, A, s):
    if mixing == "linear":
        return lambda Z: SIM.mix_linear(Z, A)
    rngm = np.random.default_rng(MIX_SEED + seed)
    if mixing == "poly":
        G = rngm.standard_normal((D_OBS, 20))
        iu = np.triu_indices(D_LATENT)

        def poly(Z):
            u = Z / s
            return np.hstack([u, KAPPA * (u[:, iu[0]] * u[:, iu[1]])]) @ G.T
        return poly
    if mixing == "mlp":
        A1 = rngm.standard_normal((64, D_LATENT)) / np.sqrt(D_LATENT)
        A2 = rngm.standard_normal((D_OBS, 64)) / np.sqrt(64)
        return lambda Z: SIM.mix_mlp(Z / s, A1, A2, s=1.0, slope=SLOPE)
    raise ValueError(mixing)


def _arm_specs(arm_key, lvl):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    if arm_key == "A":
        specs += [SIM.EnvSpec(f"iv{i}", "hard", (i,), E2.IV_BASE) for i in range(lvl)]
    elif arm_key == "D":
        specs += [SIM.EnvSpec(f"iv{i}", "hard", (i,), lvl) for i in range(D_LATENT)]
    else:
        specs += [SIM.EnvSpec(f"iv{i}", "hard", (i,), E2.IV_BASE) for i in range(D_LATENT)]
    return specs


def build(arm, lvl, seed, mixing):
    """SIM.simulate's random-number order with the mixing step swapped; arm_C's stream."""
    key = E10.ARM_KEY[arm]
    n = lvl if key == "B" else E2.N_BASE
    rng = np.random.default_rng(seed)
    B, nv, _is_src = SIM.make_scm(D_LATENT, rng, edge_prob=E2.EDGE_PROB)
    A = rng.standard_normal((D_OBS, D_LATENT))
    s = np.sqrt(np.diag(SIM.population_latent_cov(B, nv)))
    mix = _mixer(mixing, seed, A, s)
    envs, sd = {}, None
    for spec in _arm_specs(key, lvl):
        Z = SIM.sample_latent(B, nv, n, rng, kind=spec.kind, nodes=spec.nodes, iv_scale=spec.iv_scale)
        Xs = mix(Z)
        if sd is None:
            sd = 0.1 * float(np.mean(Xs.std(0)))  # simulate's obs_noise_frac (E2 default 0.1)
        X = SIM.add_obs_noise(Xs, sd, rng)
        if not (np.isfinite(Z).all() and np.isfinite(X).all()):
            raise FloatingPointError(f"non-finite samples in {spec.env_id} (seed {seed})")
        envs[spec.env_id] = dict(X=X, Z=Z)
    contam = {}
    if key == "C":
        g = np.random.default_rng(seed + 7)
        for i in range(D_LATENT - lvl, D_LATENT):
            Zc = SIM.sample_latent(B, nv, E2.N_BASE, g)
            gain = g.uniform(0.5, 1.5, D_OBS)
            Xc = mix(Zc) * gain
            envs[f"iv{i}"]["X"] = SIM.add_obs_noise(Xc, sd, g)
            contam[i] = dict(Zc=Zc, gain=gain)
    m_int = lvl if key == "A" else D_LATENT
    nodes = list(range(m_int))
    return dict(envs=envs, nodes=nodes, contam=contam, B=B, nv=nv, sd=sd, s=s)


def replay_contam(arm, lvl, seed, ref):
    """default_rng(seed + 7) replayed independently: Zc and gain, with the noise draw kept."""
    g = np.random.default_rng(seed + 7)
    worst = 0.0
    for i in range(D_LATENT - lvl, D_LATENT):
        Zc = SIM.sample_latent(ref["B"], ref["nv"], E2.N_BASE, g)
        gain = g.uniform(0.5, 1.5, D_OBS)
        g.standard_normal((E2.N_BASE, D_OBS))
        worst = max(worst, float(np.max(np.abs(Zc - ref["contam"][i]["Zc"]))),
                    float(np.max(np.abs(gain - ref["contam"][i]["gain"]))))
    return worst


def z_regression(arm, lvl, seed, data):
    """max |Z - Z_linear| over every environment (must be 0)."""
    ds, _ = E2.ARMS[arm]["builder"](seed, lvl)
    return max(float(np.max(np.abs(data["envs"][k]["Z"] - ds.environments[k].Z))) for k in data["envs"])


# ------------------------------------------------------------------ autoencoder
def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _model_seed(arm, lvl, seed):
    return MIX_SEED + 1000 * seed + 100 * ARM_INDEX[E10.ARM_KEY[arm]] + E2.ARMS[arm]["levels"].index(lvl)


def train_autoencoder(Xb, mixing, S, steps):
    import torch
    from torch import nn

    class PolyDecoder(nn.Module):
        """Linear map over [1, z, z_i z_j for i <= j] (21 features) to 200; the constant
        feature carries the intercept, so the Linear has no separate bias."""

        def __init__(self):
            super().__init__()
            iu = torch.triu_indices(D_LATENT, D_LATENT)
            self.register_buffer("i0", iu[0])
            self.register_buffer("i1", iu[1])
            self.lin = nn.Linear(1 + D_LATENT + 15, D_OBS, bias=False)

        def forward(self, z):
            ones = torch.ones(z.shape[0], 1, dtype=z.dtype)
            return self.lin(torch.cat([ones, z, z[:, self.i0] * z[:, self.i1]], dim=1))

    torch.manual_seed(S)
    gen = torch.Generator().manual_seed(S)
    enc = nn.Sequential(nn.Linear(D_OBS, HIDDEN), nn.LeakyReLU(SLOPE), nn.Linear(HIDDEN, HIDDEN),
                        nn.LeakyReLU(SLOPE), nn.Linear(HIDDEN, D_LATENT))
    if mixing == "poly":
        dec = PolyDecoder()
    else:
        dec = nn.Sequential(nn.Linear(D_LATENT, HIDDEN), nn.LeakyReLU(SLOPE), nn.Linear(HIDDEN, HIDDEN),
                            nn.LeakyReLU(SLOPE), nn.Linear(HIDDEN, D_OBS))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    Xt = torch.from_numpy(Xb)
    n = Xt.shape[0]
    bs = min(BATCH, n)
    step = 0
    while step < steps:
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, bs):
            if step >= steps:
                break
            xb = Xt[perm[start:start + bs]]
            loss = nn.functional.mse_loss(dec(enc(xb)), xb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return enc, dec


def _encode(enc, dec, Xs):
    import torch
    with torch.no_grad():
        Xt = torch.from_numpy(Xs)
        Yt = enc(Xt)
        loss = float(torch.nn.functional.mse_loss(dec(Yt), Xt))
    return Yt.numpy().astype(np.float64), loss


def affine_r2(Y, Z):
    """min over j of R^2 of the OLS regression of z_j on [1, Y]."""
    Xd = np.hstack([np.ones((len(Y), 1)), Y])
    best = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        coef, *_ = np.linalg.lstsq(Xd, z, rcond=None)
        res = z - Xd @ coef
        tot = z - z.mean()
        best.append(1.0 - float(res @ res) / float(tot @ tot))
    return float(min(best))


# ------------------------------------------------------------------ unit record (E10 logic)
def unit_rows(arm, lvl, seed, Y_obs, Y_int, nodes, obs_Z, g, n_boot):
    """E10.unit_records' record logic, on supplied projections and readout call g."""
    W = BK.unmixing_rows(Y_obs, Y_int, nodes, E2.D_LATENT, readout="precision")
    Z_hat = Y_obs @ W.T
    rows = []
    n_obs = Y_obs.shape[0]
    for k, node in enumerate(nodes):
        Y_env = Y_int[k]
        n_e = Y_env.shape[0]
        rng = np.random.default_rng(77_000 + 1000 * seed + node)
        angles = np.empty(n_boot)
        for b in range(n_boot):
            ie = rng.integers(0, n_e, n_e)
            io = rng.integers(0, n_obs, n_obs)
            angles[b] = E10._angle_deg(BK._unmixing_row(Y_obs[io], Y_env[ie]), W[node])
        corr = float(E2._best_match_corr(Z_hat[:, node], obs_Z))
        rows.append(dict(
            arm=E10.ARM_KEY[arm], arm_name=arm, level=lvl, seed=seed, node=int(node),
            n_e=int(n_e), signal=float(g["signals"][k]),
            threshold=float(g["thresholds"][k]), pvalue=float(g["pvalues"][k]),
            raw_detect=bool(g["raw_detect"][k]), bh_detect_q05=bool(g["bh_detect"][k]),
            best_match_corr=corr, good=bool(corr > E2.DIR_CORR),
            contaminated=bool(E10.ARM_KEY[arm] == "C" and node >= E2.D_LATENT - lvl),
            boot_angle_median=float(np.median(angles)),
            boot_angle_p90=float(np.quantile(angles, 0.90)),
        ))
        corr_all = [float(E2._abs_corr(Z_hat[:, node], obs_Z[:, j])) for j in range(E2.D_LATENT)]
        target = None if rows[-1]["contaminated"] else int(node)
        rows[-1].update(corr_all=corr_all, target=target,
                        corr_target=(corr_all[node] if target is not None else None))
    return rows


def _readouts(rows, Y_int, Y_obs, seed, B):
    """E14.system_records' readout fields (geneig, energy) with its failure handling."""
    failures = []
    for readout in E14.NEW_READOUTS:
        try:
            g = E14._call(Y_int, Y_obs, seed, B, readout)
        except Exception as exc:
            failures += [dict(node=r["node"], readout=readout, reason=f"{type(exc).__name__}: {exc}") for r in rows]
            for r in rows:
                r[f"{readout}_signal"] = r[f"{readout}_pvalue"] = None
            continue
        for k, r in enumerate(rows):
            sig, thr = g["signals"][k], g["thresholds"][k]
            if not (np.isfinite(sig) and np.isfinite(thr)):
                failures.append(dict(node=r["node"], readout=readout, reason=(
                    "non-finite signal" if not np.isfinite(sig) else "non-finite threshold (NaN null draw)")))
                r[f"{readout}_signal"] = r[f"{readout}_pvalue"] = None
            else:
                r[f"{readout}_signal"] = float(sig)
                r[f"{readout}_pvalue"] = float(g["pvalues"][k])
    return failures


# ------------------------------------------------------------------ one system
def run_system(job):
    mixing, arm, lvl, seed, steps, B, n_boot, gates = job
    key = dict(arm=E10.ARM_KEY[arm], level=lvl, seed=seed)
    t0 = time.time()
    data = build(arm, lvl, seed, mixing)
    zdiff = z_regression(arm, lvl, seed, data)
    cdiff = replay_contam(arm, lvl, seed, data) if data["contam"] else None
    nodes = data["nodes"]
    out = dict(key=key, mixing=mixing, model_seed=_model_seed(arm, lvl, seed), z_regression_max_abs=zdiff,
               contam_replay_max_abs=cdiff, n_nodes=len(nodes))
    try:
        Xb = data["envs"]["basis"]["X"]
        mu, sdv = Xb.mean(0), Xb.std(0)
        stdz = lambda X: ((X - mu) / sdv).astype(np.float32)  # noqa: E731
        enc, dec = train_autoencoder(stdz(Xb), mixing, out["model_seed"], steps)
        _, loss_basis = _encode(enc, dec, stdz(Xb))
        Y_obs, loss_obs = _encode(enc, dec, stdz(data["envs"]["obs"]["X"]))
        Y_int = [_encode(enc, dec, stdz(data["envs"][f"iv{i}"]["X"]))[0] for i in nodes]
        if not (np.all(np.isfinite(Y_obs)) and all(np.all(np.isfinite(Y)) for Y in Y_int)):
            raise FloatingPointError("non-finite codes")
        obs_Z = data["envs"]["obs"]["Z"]
        r2_int = {str(i): affine_r2(Y, data["envs"][f"iv{i}"]["Z"])
                  for i, Y in zip(nodes, Y_int) if i not in data["contam"]}
        out["diagnostics"] = dict(affine_r2_obs=affine_r2(Y_obs, obs_Z), affine_r2_int=r2_int,
                                  recon_loss_basis=loss_basis, recon_loss_obs=loss_obs, n_basis=int(len(Xb)))
        if gates:
            g = E14._call(Y_int, Y_obs, seed, B, "precision")
            rows = unit_rows(arm, lvl, seed, Y_obs, Y_int, nodes, obs_Z, g, n_boot)
            out["readout_failures"] = _readouts(rows, Y_int, Y_obs, seed, B)
            out["units"] = rows
    except Exception as exc:  # recorded; its units count as not recoverable, G0 only
        out["failed"] = f"{type(exc).__name__}: {exc}"
        out["units"] = None
    out["seconds"] = round(time.time() - t0, 2)
    return out


def failed_units(res, arm_name):
    """Placeholders for a failed system: not recoverable, certified by G0 only."""
    k = res["key"]
    rows = []
    for node in range(res["n_nodes"]):
        rows.append(dict(arm=k["arm"], arm_name=arm_name, level=k["level"], seed=k["seed"], node=node,
                         n_e=int(k["level"] if k["arm"] == "B" else E2.N_BASE), signal=float("-inf"), threshold=None, pvalue=1.0, raw_detect=False,
                         bh_detect_q05=False, best_match_corr=None, good=False,
                         contaminated=bool(k["arm"] == "C" and node >= D_LATENT - k["level"]),
                         boot_angle_median=None, boot_angle_p90=float("inf"), geneig_signal=None,
                         geneig_pvalue=1.0, energy_signal=None, energy_pvalue=1.0, failed_system=True))
    return rows


def pool_map(fn, jobs, workers):
    if workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers")
    if workers <= 1:
        _init_worker()
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def versions():
    import scipy
    import torch
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, venv=sys.prefix)


def systems_list(seeds, levels_per_arm):
    return list(E14.systems(seeds, levels_per_arm))


def data_pass(mixing, workers, seeds=None, levels_per_arm=None, steps=STEPS, B=B_GATE, n_boot=N_BOOT,
              write=True):
    t0 = time.time()
    seeds = E2.SEEDS if seeds is None else seeds
    syst = systems_list(seeds, levels_per_arm)
    jobs = [(mixing, arm, lvl, seed, steps, B, n_boot, True) for arm, lvl, seed in syst]
    units, failed, diags, readout_failures = [], [], [], []
    root = OUT / mixing
    config = dict(mixing=mixing, steps=steps, B=B, n_boot=n_boot, lr=LR, batch=BATCH, kappa=KAPPA,
                  null_rng="default_rng(910_000 + seed)", model_seed="160000 + 1000 seed + 100 arm + level_index")
    for (arm, lvl, seed), res in zip(syst, pool_map(run_system, jobs, workers)):
        if res["z_regression_max_abs"] != 0.0 or (res["contam_replay_max_abs"] not in (None, 0.0)):
            raise SystemExit(f"STOP: Z or contamination regression failed for {res['key']}: "
                             f"{res['z_regression_max_abs']}, {res['contam_replay_max_abs']}; run void")
        if res.get("failed"):
            failed.append(dict(res["key"], reason=res["failed"]))
            units += failed_units(res, arm)
        else:
            units += res["units"]
            readout_failures += [dict(res["key"], **f) for f in res["readout_failures"]]
        diags.append(dict(res["key"], model_seed=res["model_seed"], seconds=res["seconds"],
                          failed=res.get("failed"), **(res.get("diagnostics") or {})))
        if write:
            T2.write_json(root / "units.json", dict(
                experiment="E16", prereg="notes/e16_prereg.md", status="partial", config=config,
                units=units, diagnostics=diags, failed_systems=failed, readout_failures=readout_failures,
                versions=versions()))
            T2.write_json(root / "failed_units.json", dict(failed_systems=failed, readout_failures=readout_failures))
        print(f"  {mixing} {E10.ARM_KEY[arm]} level={lvl} seed={seed} units={len(units)} failed={len(failed)} "
              f"r2_obs={(res.get('diagnostics') or {}).get('affine_r2_obs')} ({time.time() - t0:.0f}s)", flush=True)
    doc = dict(experiment="E16", prereg="notes/e16_prereg.md", status="complete", config=config,
               n_systems=len(syst), n_units=len(units), units=units, diagnostics=diags, failed_systems=failed,
               readout_failures=readout_failures, wall_seconds=round(time.time() - t0, 1), versions=versions(),
               provenance=T2.provenance(__file__, ["experiments/e14_matched_coverage.py",
                                                   "experiments/e10_risk_coverage.py",
                                                   "experiments/e2_calibration.py", "sim/simulator.py"]))
    if write:
        T2.write_json(root / "units.json", doc)
    return doc


# ------------------------------------------------------------------ analysis
def _point_curves(dec, good, masks, gates, scopes):
    out = {}
    for gate in gates:
        out[gate] = {}
        for scope in scopes:
            pts = [dict(knob=k, **E10._metrics(d, good, masks[scope])) for k, d in dec[gate]]
            out[gate][scope] = dict(points=pts, curve=E10._curve(pts))
    return out


def _g0(cv, scope):
    p = cv["G0"][scope]["points"][0]
    return dict(coverage=p["coverage"], risk=p["risk"])


def bootstrap16(dec, good, masks, fams, n_draws=N_DRAWS):
    """E14.bootstrap's draw loop (default_rng(120000), same picks), plus H1's per-draw
    risk(G0) - risk(G2 at q=0.05) on ABD."""
    rng = np.random.default_rng(E14.BOOT_SEED)
    fams = [np.asarray(f) for f in fams]
    draws = {s: {g: dict(aurc=[], matched=[]) for g in GATES_E16} for s in E14.SCOPES}
    keys = ("G8-G2_matched", "G9-G2_matched", "G8-G2_aurc", "G9-G2_aurc", "G0-G2_aurc", "G0-G5_aurc",
            "G0-G8_aurc", "G0-G9_aurc", "G0-G2_risk_q05")
    diffs = {s: {k: [] for k in keys} for s in E14.SCOPES}
    for _ in range(n_draws):
        pick = rng.integers(0, len(fams), len(fams))
        idx = np.concatenate([fams[k] for k in pick])
        cv = E14.curves_for(dec, good, masks, idx)
        sm = E14.summary_of(cv)
        for s in E14.SCOPES:
            for g in GATES_E16:
                draws[s][g]["aurc"].append(sm[s][g]["aurc"])
                draws[s][g]["matched"].append(sm[s][g]["matched_risk"])
            for g in ("G8", "G9"):
                diffs[s][f"{g}-G2_matched"].append(E14._diff(sm[s][g]["matched_risk"], sm[s]["G2"]["matched_risk"]))
                diffs[s][f"{g}-G2_aurc"].append(E14._diff(sm[s][g]["aurc"], sm[s]["G2"]["aurc"]))
            for g in ("G2", "G5", "G8", "G9"):
                diffs[s][f"G0-{g}_aurc"].append(E14._diff(sm[s]["G0"]["aurc"], sm[s][g]["aurc"]))
            diffs[s]["G0-G2_risk_q05"].append(E14._diff(cv["G0"][s]["points"][0]["risk"], sm[s]["G2"]["risk_q05"]))
    ci = {s: {g: dict(aurc=E14._pct(draws[s][g]["aurc"]), matched_risk=E14._pct(draws[s][g]["matched"]))
              for g in GATES_E16} for s in E14.SCOPES}
    ci_diff = {s: {k: E14._pct(v) for k, v in diffs[s].items()} for s in E14.SCOPES}
    return ci, ci_diff


def _interval_ok(c):
    return c["n_defined"] >= MIN_DEFINED and c["lo"] is not None


def _v(cond):
    return "undetermined" if cond is None else ("held" if cond else "failed")


def predictions_for(mixing, sm, cv, ci_diff, too_many_failed):
    out = {}
    G2 = lambda s: sm[s]["G2"]  # noqa: E731
    g0 = {s: _g0(cv, s) for s in ("ABD", "C")}
    if mixing == "poly":
        r2, r0 = G2("ABD")["risk_q05"], g0["ABD"]["risk"]
        c1 = ci_diff["ABD"]["G0-G2_risk_q05"]
        h1 = None if r2 is None or r0 is None or not _interval_ok(c1) else (r2 < r0 and c1["lo"] > 0)
        out["H1"] = dict(verdict=_v(h1), risk_G2_q05=r2, risk_G0=r0, interval=c1)
        out["H2"] = dict(verdict=_v(None if r2 is None else r2 <= 0.05), risk_G2_q05=r2, threshold=0.05)
        a2, a0 = sm["ABD"]["G2"]["aurc"], sm["ABD"]["G0"]["aurc"]
        c3 = ci_diff["ABD"]["G0-G2_aurc"]
        h3 = None if a2 is None or a0 is None or not _interval_ok(c3) else (a2 < a0 and c3["lo"] > 0)
        out["H3"] = dict(verdict=_v(h3), aurc_G2=a2, aurc_G0=a0, interval=c3)
        m9, m2 = sm["ABD"]["G9"]["matched_risk"], sm["ABD"]["G2"]["matched_risk"]
        c4 = ci_diff["ABD"]["G9-G2_matched"]
        h4 = None if m9 is None or m2 is None or not _interval_ok(c4) else (m9 > m2 and c4["lo"] > 0)
        out["H4"] = dict(verdict=_v(h4), matched_G9=m9, matched_G2=m2, interval=c4)
        cov = G2("ABD")["coverage_q05"]
        out["H5"] = dict(verdict=_v(None if cov is None else cov >= 0.60), coverage_G2_q05=cov, threshold=0.60)
        rc = G2("C")["risk_q05"]
        out["H6"] = dict(verdict=_v(None if rc is None else rc >= 0.10), risk_G2_q05_C=rc, threshold=0.10)
    else:
        r2, r0 = G2("ABD")["risk_q05"], g0["ABD"]["risk"]
        out["H7"] = dict(verdict=_v(None if r2 is None or r0 is None else r2 < r0), risk_G2_q05=r2, risk_G0=r0,
                         note="exploratory, outside the theory's assumptions; point estimates only")
    if too_many_failed:
        for h in out.values():
            h["verdict"] = "undetermined"
            h["note_failed"] = f"more than {int(FAIL_FRAC * 100)}% of systems failed"
    return out


def linear_reference():
    c14 = json.loads((REPO / "results" / "e14" / "curves.json").read_text())
    ref = {}
    for s in ("ABD", "C"):
        ref[s] = dict(target_coverage=c14["summary"][s]["target_coverage"], gates={})
        for g in ("G0", "G2", "G5", "G8", "G9"):
            row = dict(c14["summary"][s][g])
            row["aurc_ci"] = c14["ci"][s][g]["aurc"]
            row["matched_ci"] = c14["ci"][s][g]["matched_risk"]
            ref[s]["gates"][g] = row
        p0 = c14["curves"]["G0"][s]["points"][0]
        ref[s]["gates"]["G0"].update(coverage=p0["coverage"], risk=p0["risk"])
    return dict(source="results/e14/curves.json (linear backbone, E10 units)", scopes=ref)


def analyse(mixing, units_doc=None, n_draws=N_DRAWS, write=True):
    t0 = time.time()
    doc = units_doc or json.loads((OUT / mixing / "units.json").read_text())
    if doc.get("status") != "complete":
        raise SystemExit(f"STOP: results/e16/{mixing}/units.json is not complete")
    units = doc["units"]
    n_sys = len(doc["diagnostics"])
    n_failed = len(doc["failed_systems"])
    too_many = n_sys > 0 and n_failed / n_sys > FAIL_FRAC
    readout_missing = [u for u in units if not u.get("failed_system")
                       and (u.get("geneig_pvalue") is None or u.get("energy_pvalue") is None)]
    for u in readout_missing:  # readout failure within a successful system: same placeholder rule
        u["geneig_pvalue"] = 1.0 if u.get("geneig_pvalue") is None else u["geneig_pvalue"]
        u["energy_pvalue"] = 1.0 if u.get("energy_pvalue") is None else u["energy_pvalue"]
    good = np.array([u["good"] for u in units])
    dec = E14.decisions(units)
    failed_mask = np.array([bool(u.get("failed_system")) for u in units])
    if failed_mask.any():
        for gate, ops in dec.items():
            if gate == "G0":
                continue
            dec[gate] = [(k, d & ~failed_mask) for k, d in ops]
    masks = E14._masks(units)
    cv = E14.curves_for(dec, good, masks)
    sm = E14.summary_of(cv)
    pts = _point_curves(dec, good, masks, POINT_ONLY, ("ABD", "C"))
    point_only = {g: {s: dict(aurc=E10._aurc(pts[g][s]["curve"])["value"],
                              matched_risk=E10._matched(pts[g][s]["curve"], sm[s]["target_coverage"]))
                      for s in ("ABD", "C")} for g in POINT_ONLY}
    caveat_cv = E14.curves_for(dec, good, masks, scopes=tuple(f"B_level{lv}" for lv in E14.CAVEAT_LEVELS))
    caveat = {f"B_level{lv}": {g: E10._knob_point(caveat_cv[g][f"B_level{lv}"]["points"], E10.Q_DEFAULT)
                               for g in ("G2", "G8", "G9")} for lv in E14.CAVEAT_LEVELS}
    ci, ci_diff = bootstrap16(dec, good, masks, E10._families(units), n_draws)
    preds = predictions_for(mixing, sm, cv, ci_diff, too_many)
    diags = doc["diagnostics"]
    r2o = [d["affine_r2_obs"] for d in diags if d.get("affine_r2_obs") is not None]
    r2i = [v for d in diags for v in (d.get("affine_r2_int") or {}).values()]
    table = {s: dict(target_coverage=sm[s]["target_coverage"], G0=_g0(cv, s), gates={
        g: dict(aurc=sm[s][g]["aurc"], aurc_ci=ci[s][g]["aurc"], matched_risk=sm[s][g]["matched_risk"],
                matched_ci=ci[s][g]["matched_risk"],
                **({k: sm[s][g][k] for k in ("coverage_q05", "risk_q05")} if g in ("G2", "G8", "G9") else {}))
        for g in GATES_E16}) for s in ("ABD", "C")}
    results = dict(
        experiment="E16", mixing=mixing, prereg="notes/e16_prereg.md", n_units=len(units), n_systems=n_sys,
        n_failed_systems=n_failed, failed_fraction=(n_failed / n_sys if n_sys else None),
        too_many_failed=too_many, n_readout_placeholder_units=len(readout_missing),
        e16=table, point_only_G1_G3_G4=point_only, caveat_arm_B=caveat,
        affine_r2=dict(obs_median=float(np.median(r2o)) if r2o else None, obs_n=len(r2o),
                       int_median=float(np.median(r2i)) if r2i else None, int_n=len(r2i),
                       recon_basis_median=float(np.median([d["recon_loss_basis"] for d in diags if "recon_loss_basis" in d])),
                       recon_obs_median=float(np.median([d["recon_loss_obs"] for d in diags if "recon_loss_obs" in d]))),
        ci_diff={s: ci_diff[s] for s in ("ABD", "C")}, predictions=preds, linear_backbone=linear_reference(),
        bootstrap=dict(draws=n_draws, rng=f"default_rng({E14.BOOT_SEED})", min_defined=MIN_DEFINED))
    results = json.loads(json.dumps(results))
    if write:
        meta = dict(seconds=round(time.time() - t0, 1), units_sha256=T2.sha256_file(OUT / mixing / "units.json"),
                    versions=versions(), provenance=T2.provenance(__file__, ["experiments/e14_matched_coverage.py",
                                                                            "experiments/e10_risk_coverage.py"]))
        T2.write_json(OUT / mixing / "curves.json", dict(results=dict(summary=sm, ci=ci, ci_diff=ci_diff,
                                                                      curves=E14._jsonable(cv)), meta=meta))
        T2.write_json(OUT / mixing / "predictions.json", dict(results=results, meta=meta))
    return results


# ------------------------------------------------------------------ preflight
def preflight(workers):
    t0 = time.time()
    arm = next(a for a in E2.ARMS if E10.ARM_KEY[a] == "A")
    jobs = [("poly", arm, 5, seed, STEPS, B_GATE, N_BOOT, False) for seed in E2.SEEDS]
    res = list(pool_map(run_system, jobs, min(workers, len(jobs))))
    for r in res:
        if r["z_regression_max_abs"] != 0.0:
            raise SystemExit(f"STOP: Z regression failed in preflight {r['key']}")
    per = {str(r["key"]["seed"]): (r.get("diagnostics") or {}).get("affine_r2_obs") for r in res}
    vals = [v for v in per.values() if v is not None]
    med = float(np.median(vals)) if vals else None
    k0 = med is None or med < 0.90
    doc = dict(experiment="E16", part="preflight", mixing="poly", arm="A", level=5, seeds=E2.SEEDS,
               steps=STEPS, affine_r2_obs_per_seed=per, median_affine_r2=med, K0_threshold=0.90,
               K0_triggered=bool(k0), failed=[dict(r["key"], reason=r["failed"]) for r in res if r.get("failed")],
               diagnostics=[dict(r["key"], **(r.get("diagnostics") or {}), seconds=r["seconds"]) for r in res],
               wall_seconds=round(time.time() - t0, 1), versions=versions(),
               provenance=T2.provenance(__file__))
    T2.write_json(OUT / "preflight.json", doc)
    print(f"[preflight] median affine R^2 {med}; per seed {per}; K0 {'TRIGGERED' if k0 else 'not triggered'}",
          flush=True)


# ------------------------------------------------------------------ selftest
def selftest():
    t0 = time.time()
    syst = systems_list([0], 2)
    worst_x = worst_z = 0.0
    for arm, lvl, seed in syst:
        ds, int_envs = E2.ARMS[arm]["builder"](seed, lvl)
        lin = build(arm, lvl, seed, "linear")
        for k, env in ds.environments.items():
            worst_x = max(worst_x, float(np.max(np.abs(lin["envs"][k]["X"] - env.X))))
            worst_z = max(worst_z, float(np.max(np.abs(lin["envs"][k]["Z"] - env.Z))))
            if not np.array_equal(lin["envs"][k]["X"], env.X):
                raise SystemExit(f"STOP: linear builder X differs for {arm} {lvl} {k}")
        for (i, X) in int_envs:
            if not np.array_equal(lin["envs"][f"iv{i}"]["X"], X):
                raise SystemExit(f"STOP: linear builder int X differs for {arm} {lvl} iv{i}")
    print(f"[selftest] linear builder vs E2 builders ({len(syst)} systems, every environment): bit-identical "
          f"(max |dX| {worst_x}, max |dZ| {worst_z})", flush=True)

    zr, cr = 0.0, 0.0
    for mixing in MIXINGS:
        for arm, lvl, seed in syst:
            d = build(arm, lvl, seed, mixing)
            zr = max(zr, z_regression(arm, lvl, seed, d))
            if d["contam"]:
                cr = max(cr, replay_contam(arm, lvl, seed, d))
    print(f"[selftest] poly and mlp Z vs linear Z: max |dZ| {zr}; arm C Zc and gain vs replay of "
          f"default_rng(seed + 7): max diff {cr}", flush=True)
    if zr != 0.0 or cr != 0.0:
        raise SystemExit("STOP: Z or contamination regression failed")

    e10 = {}
    for u in json.loads(E10_UNITS.read_text())["units"]:
        e10[(u["arm"], u["level"], u["seed"], u["node"])] = u
    # Regression criterion (approved 24 Oct, after exact float equality failed at 1e-12):
    # decision fields and pvalue exactly equal; float fields within 1e-9; the E10b fields
    # corr_all, target, corr_target excluded (the stored units.json predates them).
    n_cmp, mism, worst = 0, [], {}
    for arm, lvl, seed in syst:
        Y_obs, Y_int, nodes = E14.projected(arm, lvl, seed)
        ds, _ = E2.ARMS[arm]["builder"](seed, lvl)
        g = E14._call(Y_int, Y_obs, seed, E10.B_NULL, "precision")
        for r in unit_rows(arm, lvl, seed, Y_obs, Y_int, nodes, ds.environments["obs"].Z, g, E10.N_BOOT):
            ref = e10[(r["arm"], r["level"], r["seed"], r["node"])]
            n_cmp += 1
            bad = [k for k in set(ref) - set(r)]
            for k, v in r.items():
                if k in REGRESSION_EXCLUDED:
                    continue
                if k not in ref:
                    bad.append(k)
                elif k in REGRESSION_FLOAT:
                    d = abs(v - ref[k])
                    worst[k] = max(worst.get(k, 0.0), d)
                    if d > REGRESSION_FLOAT_TOL:
                        bad.append(k)
                elif v != ref[k]:
                    bad.append(k)
            if bad:
                mism.append(dict(key=[r["arm"], r["level"], r["seed"], r["node"]], fields=sorted(set(bad))))
    print(f"[selftest] unit_rows on E14.projected() vs results/e10_risk_coverage/units.json: {n_cmp} units, "
          f"{len(mism)} mismatches (decision fields and pvalue exact; floats within {REGRESSION_FLOAT_TOL}; "
          f"excluded {list(REGRESSION_EXCLUDED)}); max abs diff per float field "
          f"{ {k: float(v) for k, v in sorted(worst.items())} }", flush=True)
    if mism:
        raise SystemExit(f"STOP: unit record regression failed: {mism[:3]}")

    runs = {}
    for w in (1, 4):
        runs[w] = data_pass("poly", w, seeds=[0], levels_per_arm=2, steps=SELFTEST_STEPS, B=SELFTEST_B,
                            n_boot=10, write=False)
    a, b = (json.dumps(runs[w]["units"], sort_keys=True) for w in (1, 4))
    da, db = (json.dumps([{k: v for k, v in d.items() if k != "seconds"} for d in runs[w]["diagnostics"]],
                         sort_keys=True) for w in (1, 4))
    print(f"[selftest] poly data pass {len(runs[1]['units'])} units: workers 1 vs 4 units identical {a == b}, "
          f"diagnostics identical {da == db}; failed systems {len(runs[1]['failed_systems'])}", flush=True)
    if a != b or da != db:
        raise SystemExit("STOP: results differ between 1 and 4 workers")
    res_poly = analyse("poly", runs[1], n_draws=50, write=False)
    mlp = data_pass("mlp", 4, seeds=[0], levels_per_arm=2, steps=SELFTEST_STEPS, B=SELFTEST_B, n_boot=10,
                    write=False)
    res_mlp = analyse("mlp", mlp, n_draws=50, write=False)
    for name, r in (("poly", res_poly), ("mlp", res_mlp)):
        print(f"[selftest] {name} end to end: units {r['n_units']}, failed {r['n_failed_systems']}, "
              f"affine R^2 obs median {r['affine_r2']['obs_median']:.4f} (300 steps), "
              f"G2 ABD cov {r['e16']['ABD']['gates']['G2']['coverage_q05']}, "
              f"predictions {{{', '.join(f'{k}: {v['verdict']}' for k, v in r['predictions'].items())}}}",
              flush=True)
    print(f"[selftest] passed (nothing written) {time.time() - t0:.1f}s; versions {versions()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mixing", choices=MIXINGS)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.workers > MAX_WORKERS:
        ap.error(f"at most {MAX_WORKERS} workers")
    if a.selftest:
        selftest()
    elif a.preflight:
        preflight(a.workers)
    elif a.analyse:
        if a.mixing is None:
            ap.error("--analyse needs --mixing")
        r = analyse(a.mixing)
        print(json.dumps(dict(predictions={k: v["verdict"] for k, v in r["predictions"].items()},
                              failed=r["n_failed_systems"], affine_r2=r["affine_r2"]), indent=1), flush=True)
    else:
        if a.mixing is None:
            ap.error("the data pass needs --mixing")
        doc = data_pass(a.mixing, a.workers)
        print(f"[data pass {a.mixing}] {doc['n_units']} units, {len(doc['failed_systems'])} failed systems, "
              f"{doc['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
