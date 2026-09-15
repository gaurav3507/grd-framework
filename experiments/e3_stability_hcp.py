import numpy as np, importlib.util, json, glob, os
from pathlib import Path
spec=importlib.util.spec_from_file_location("pr","src/gate/precision_readout.py")
pr=importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
D=10; TS="/workspace/meridian-identifiability/hcp/ts"
TASKS=["WM","GAMBLING","MOTOR","LANGUAGE","SOCIAL","RELATIONAL","EMOTION"]
subs=sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
def conn(ts):
    C=np.corrcoef(ts,rowvar=False); iu=np.triu_indices(C.shape[0],k=1); return C[iu]
groups={}
for t in TASKS:
    vs=[]
    for s in subs:
        arrs=[np.load(f"{TS}/{s}_{t}_{e}.npy").astype(float) for e in ("LR","RL") if os.path.exists(f"{TS}/{s}_{t}_{e}.npy")]
        if arrs: vs.append(conn(np.concatenate(arrs,0)))
    if vs: groups[t]=np.array(vs)
base=max(groups,key=lambda k:len(groups[k]))
def once(seed):
    rng=np.random.default_rng(seed)
    Yb=groups[base]; mu=Yb.mean(0); _,_,Vt=np.linalg.svd(Yb-mu,full_matrices=False); Bp=Vt[:D].T
    proj=lambda M:(M-mu)@Bp; Yobs=proj(Yb)
    envs={k:v for k,v in groups.items() if k!=base}; det=0; shifts=[]
    for k,v in envs.items():
        Yp=proj(v); n=len(Yp)
        crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
        s=pr.precision_signal(Yp,Yobs)
        if s>crit: det+=1; sh=Yp.mean(0)-Yobs.mean(0); shifts.append(sh/(np.linalg.norm(sh)+1e-12))
    nfake=50; fd=0; ez=[len(v) for v in envs.values()]
    for _ in range(nfake):
        n=min(int(rng.choice(ez)),len(Yb)//2); sub=proj(Yb[rng.choice(len(Yb),n,replace=False)])
        crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
        if pr.precision_signal(sub,Yobs)>crit: fd+=1
    align=float(np.median((np.array(shifts)[:,:2]**2).sum(1))) if shifts else float('nan')
    return det/len(envs), fd/nfake, align
R=np.array([once(s) for s in range(5)])
agg=dict(dataset="fMRI_HCP_task",
  frac_detectable_mean=round(float(R[:,0].mean()),4), frac_detectable_sd=round(float(R[:,0].std()),4),
  negative_control_mean=round(float(R[:,1].mean()),4), negative_control_sd=round(float(R[:,1].std()),4),
  confound_align_mean=round(float(np.nanmean(R[:,2])),4), confound_align_sd=round(float(np.nanstd(R[:,2])),4),
  per_seed_detect=[round(x,4) for x in R[:,0]])
Path("results/e3_stability/HCP.json").write_text(json.dumps(agg,indent=2))
print("HCP done"); print(json.dumps(agg,indent=2))
