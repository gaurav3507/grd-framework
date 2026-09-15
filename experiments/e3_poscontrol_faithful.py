import numpy as np, anndata as ad, importlib.util, json
from pathlib import Path
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pr=load("pr","src/gate/precision_readout.py")
SIM=load("sim","sim/simulator.py")
BK=load("bk","src/recover/backbone.py")

DLAT=10; DPROJ=10; R_PLANT=3; NPER=200; IV_SCALE=0.1; EDGE_PROB=0.4; SEED=0
rng=np.random.default_rng(SEED)

# real K562 control cells = the observation-space noise pool (GENE space, 1158-dim)
A=ad.read_h5ad('/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad')
X=A.X.toarray().astype(np.float64) if hasattr(A.X,"toarray") else np.asarray(A.X,np.float64)
g=A.obs['guide_ids'].astype(str).values; Xc=X[g=='']; del A,X
DGENE=Xc.shape[1]
Xc_c=Xc-Xc.mean(0)                                  # centered real noise pool
bg_total=float(Xc_c.var(0).sum())
print(f"real noise pool: {Xc_c.shape}, total var {bg_total:.3f}")

# shared SCM latents (basis, obs, 3 reduced-variance interventions); we use Z only
env_specs=[SIM.EnvSpec("basis",None,()), SIM.EnvSpec("obs",None,())]
for i in range(R_PLANT): env_specs.append(SIM.EnvSpec(f"iv{i}","hard",(i,),IV_SCALE))
ds=SIM.simulate(DLAT, DGENE, NPER, env_specs, SEED, edge_prob=EDGE_PROB)
E=ds.environments; print("envs:",list(E.keys()))
for i in range(R_PLANT):
    print(f"  iv{i} constructed rank:", SIM.constructed_rank(ds.B, ds.noise_var,"hard",(i,),iv_scale=IV_SCALE))

# fixed mixing latent -> GENE space, one scalar scale from obs
Amix=rng.standard_normal((DGENE,DLAT)); Amix/=np.linalg.norm(Amix,axis=0,keepdims=True)
sig_total=float((E["obs"].Z@Amix.T).var(0).sum())+1e-12

def make_env(Z, snr, seed):
    r=np.random.default_rng(seed)
    Sg=(Z@Amix.T)*np.sqrt(snr*bg_total/sig_total) if snr>0 else np.zeros((len(Z),DGENE))
    return Sg + Xc_c[r.integers(0,len(Xc_c),len(Z))]   # real K562 noise, gene space

def run_gate(snr, seeds):
    # E1 PIPELINE: build gene-space envs, fit PCA on BASIS, project all to DPROJ, gate
    Xb=make_env(E["basis"].Z, snr, seeds["basis"])
    Xo=make_env(E["obs"].Z,   snr, seeds["obs"])
    Xi=[make_env(E[f"iv{k}"].Z, snr, seeds["iv"]+k) for k in range(R_PLANT)]
    mu,W=BK.fit_pca(Xb, DPROJ)
    Yo=BK.project(Xo,mu,W); Yi=[BK.project(x,mu,W) for x in Xi]
    dets=0; ratios=[]
    for y in Yi:
        crit=pr.null_threshold(Yo,alpha=0.05,B=500,rng=rng,n_env=NPER)
        s=pr.precision_signal(y,Yo); ratios.append(round(s/crit,2))
        if s>crit: dets+=1
    # null: basis vs obs (independent un-intervened draws), projected identically
    Yb=BK.project(Xb,mu,W)
    nfake=30; fd=0
    for j in range(nfake):
        Xf=make_env(E["basis"].Z, snr, 700000+j)
        Yf=BK.project(Xf,mu,W)
        crit=pr.null_threshold(Yo,alpha=0.05,B=500,rng=rng,n_env=NPER)
        if pr.precision_signal(Yf,Yo)>crit: fd+=1
    return dets, ratios, fd, nfake

# smoke: strong signal must detect all 3
d,rt,fd,nf=run_gate(8.0, dict(basis=9000,obs=9001,iv=9100))
print(f"\nsmoke: {d}/{R_PLANT} at snr=8 | ratios {rt} | neg {fd}/{nf}")
if d<R_PLANT:
    raise SystemExit(f"[stop] construction broken: strong planted intervention not detected ({d}/{R_PLANT})")

results=[]
for snr in [0.0,0.5,1.0,2.0,4.0]:
    d,rt,fd,nf=run_gate(snr, dict(basis=5000,obs=5001,iv=5100))
    results.append(dict(snr=snr,certified=d,planted=R_PLANT,ratios=rt,neg=f"{fd}/{nf}",neg_rate=round(fd/nf,3)))
    print(f"snr {snr}: certified {d}/{R_PLANT} | ratios {rt} | neg {fd}/{nf}")

rep=dict(experiment="FAITHFUL positive control: shared-SCM reduced-variance interventions mixed into 1158-dim gene space + real K562 noise, full E1 pipeline (fit PCA on basis, project, gate)",
         d_latent=DLAT,d_gene=DGENE,d_proj=DPROJ,planted_rank=R_PLANT,cells_per_env=NPER,iv_scale=IV_SCALE,
         scale="single scalar (signal total var = snr x real bg total var, gene space)",
         null="basis vs obs (independent un-intervened draws), projected identically, size-matched",
         note="snr=0 is a null sanity check, not a reproduction of the E3 real-data verdict",
         real_bg_var_total=round(bg_total,3),dose_response=results)
Path("results/e3_poscontrol/poscontrol_final.json").write_text(json.dumps(rep,indent=2))
print("\n"+json.dumps(rep,indent=2))
