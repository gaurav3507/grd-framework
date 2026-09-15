import sys, numpy as np, importlib.util, json, glob, os
from pathlib import Path
spec=importlib.util.spec_from_file_location("pr","src/gate/precision_readout.py")
pr=importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
mode=sys.argv[1]; SEED=0; rng=np.random.default_rng(SEED)

def subject_connectivity(ts):
    # ts: (T, R) time series. Return upper-triangle of the RxR correlation matrix.
    C=np.corrcoef(ts, rowvar=False)          # (R,R) functional connectivity
    iu=np.triu_indices(C.shape[0], k=1)
    return C[iu]                               # length R*(R-1)/2

def build_subject_vectors():
    """Return dict {group_label: (n_subjects, F) connectivity matrix} and the baseline label."""
    if mode=="abide":
        z=np.load("/workspace/ranktest-diagnostics/data/abide_harmonized.npz", allow_pickle=True)
        X=z['X'].astype(float); grp=z['site_ids']            # X:(subj,T,R)
        feats={}
        for i in range(len(X)):
            feats.setdefault(str(grp[i]), []).append(subject_connectivity(X[i]))
        groups={k:np.array(v) for k,v in feats.items()}
        baseline=max(groups, key=lambda k: len(groups[k]))   # largest site
        gtype="site (MEASUREMENT shift)"
    else:  # hcp
        TS="/workspace/meridian-identifiability/hcp/ts"
        TASKS=["WM","GAMBLING","MOTOR","LANGUAGE","SOCIAL","RELATIONAL","EMOTION"]
        subs=sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
        groups={}
        for t in TASKS:
            vs=[]
            for s in subs:
                # one connectivity per subject per task: concat LR+RL timepoints, then corr
                arrs=[np.load(f"{TS}/{s}_{t}_{e}.npy").astype(float) for e in ("LR","RL")
                      if os.path.exists(f"{TS}/{s}_{t}_{e}.npy")]
                if arrs: vs.append(subject_connectivity(np.concatenate(arrs,0)))
            if vs: groups[t]=np.array(vs)
        baseline=max(groups, key=lambda k: len(groups[k]))   # largest task as reference
        gtype="task (MECHANISM shift)"
    return groups, baseline, gtype

groups, baseline, gtype = build_subject_vectors()
sizes={k:len(v) for k,v in groups.items()}
print(f"{mode}: {len(groups)} groups, baseline={baseline}")
print("subjects per group:", sizes)

# D must be well below the smallest group; connectivity is high-dim so project on baseline
D=min(10, min(sizes.values())-2)
if D < 3: raise SystemExit(f"[stop] smallest group {min(sizes.values())} too small for a gate")
print(f"projection D={D} (min group size {min(sizes.values())})")

Yb=groups[baseline]
mu=Yb.mean(0); _,_,Vt=np.linalg.svd(Yb-mu, full_matrices=False); Bp=Vt[:D].T
proj=lambda M:(M-mu)@Bp
Yobs=proj(Yb); Zc=Yobs
envs={k:v for k,v in groups.items() if k!=baseline}
env_sizes=[len(v) for v in envs.values()]

detect={};ratios={};shifts=[]
for k,v in envs.items():
    Yp=proj(v); n=len(Yp)
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
    s=pr.precision_signal(Yp,Yobs); detect[k]=bool(s>crit); ratios[k]=float(s/crit)
    if s>crit:
        sh=Yp.mean(0)-Yobs.mean(0); shifts.append(sh/(np.linalg.norm(sh)+1e-12))
n_det=sum(detect.values()); rr=np.array(list(ratios.values()))

# negative control: random splits of the BASELINE subjects, size-matched
nfake=50; fdet=0
for _ in range(nfake):
    n=int(rng.choice(env_sizes)); 
    if n>len(Yb): n=len(Yb)//2
    sub=proj(Yb[rng.choice(len(Yb),n,replace=False)])
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
    if pr.precision_signal(sub,Yobs)>crit: fdet+=1

# structured-split confound control on baseline subjects
m=int(np.median(env_sizes)); m=min(m,len(Yb)//2); struct=[]
for j in range(D):
    o=np.argsort(Zc[:,j]); struct+=[o[:m],o[-m:]]
sdet=0
for sub in struct:
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=len(sub))
    if pr.precision_signal(proj(Yb[sub]),Yobs)>crit: sdet+=1

if shifts:
    M=np.array(shifts); t2=(M[:,:2]**2).sum(1); am,amd=float(t2.mean()),float(np.median(t2))
else: am=amd=None

rep=dict(dataset=("fMRI_ABIDE_site" if mode=="abide" else "fMRI_HCP_task"),
  grouping=gtype, feature="per-subject functional connectivity (upper-tri corr), projected",
  sample_unit="subject", d_proj=D, baseline=str(baseline),
  n_environments=len(envs), env_subjects=env_sizes,
  negative_control=f"{fdet}/{nfake}", negative_control_rate=round(fdet/nfake,4),
  structured_split_rate=round(sdet/len(struct),4),
  n_detectable=n_det, frac_detectable=round(n_det/len(envs),4),
  ratio_median=round(float(np.median(rr)),3), ratio_p90=round(float(np.quantile(rr,.9)),3),
  confound_align_top2_mean=None if am is None else round(am,3),
  confound_align_top2_median=None if amd is None else round(amd,3))
Path(f"results/e3/e3_{rep['dataset']}.json").write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
