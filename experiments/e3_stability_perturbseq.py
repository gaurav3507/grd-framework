import sys, numpy as np, anndata as ad, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location("pr","src/gate/precision_readout.py")
pr=importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
D=10; NMIN=200; SEEDS=range(5)
kind=sys.argv[1]

def screen_once(X, g, ctrl, cand, seed):
    rng=np.random.default_rng(seed)
    Xc=X[g==ctrl]
    mu=Xc.mean(0); _,_,Vt=np.linalg.svd(Xc-mu,full_matrices=False); Bp=Vt[:D].T
    proj=lambda M:(M-mu)@Bp
    Yobs=proj(Xc); Zc=Yobs
    perts=[p for p in cand if (g==p).sum()>=NMIN]
    sizes=[int((g==p).sum()) for p in perts]
    det=0; ratios=[]; shifts=[]
    for p in perts:
        Yp=proj(X[g==p]); n=len(Yp)
        crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
        s=pr.precision_signal(Yp,Yobs); ratios.append(s/crit)
        if s>crit:
            det+=1; sh=Yp.mean(0)-Yobs.mean(0); shifts.append(sh/(np.linalg.norm(sh)+1e-12))
    nfake=50; fd=0
    for _ in range(nfake):
        n=int(rng.choice(sizes)); sub=proj(Xc[rng.choice(len(Xc),n,replace=False)])
        crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
        if pr.precision_signal(sub,Yobs)>crit: fd+=1
    align=float(np.median((np.array(shifts)[:,:2]**2).sum(1))) if shifts else float('nan')
    return dict(frac_detectable=det/len(perts), negative_control_rate=fd/nfake,
                confound_align_median=align, n_perts=len(perts))

def run(name, path, ctrl, single):
    A=ad.read_h5ad(path)
    X=A.X.toarray().astype(np.float64) if hasattr(A.X,"toarray") else np.asarray(A.X,np.float64)
    g=A.obs['guide_ids'].astype(str).values; del A
    if single:
        isS=lambda s: s!=ctrl and s!="" and ","not in s and "+"not in s and "_"not in s
        cand=sorted({s for s in np.unique(g) if isS(s)})
    else:
        cand=[p for p in np.unique(g) if p!=ctrl]
    rows=[screen_once(X,g,ctrl,cand,s) for s in SEEDS]
    agg={}
    for k in ["frac_detectable","negative_control_rate","confound_align_median"]:
        vals=np.array([r[k] for r in rows])
        agg[k+"_mean"]=round(float(np.nanmean(vals)),4)
        agg[k+"_sd"]=round(float(np.nanstd(vals)),4)
    agg["n_perts"]=rows[0]["n_perts"]; agg["seeds"]=list(SEEDS); agg["dataset"]=name
    agg["per_seed_detect"]=[round(r["frac_detectable"],4) for r in rows]
    Path(f"results/e3_stability/{name}.json").write_text(json.dumps(agg,indent=2))
    print(json.dumps(agg,indent=2)); return agg

if kind=="k562": run("K562",  "/workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad","",False)
if kind=="rpe1": run("RPE1",  "/workspace/external/discrepancy_vae/datasets/causalbench_rpe1.h5ad","",False)
if kind=="norman": run("Norman","/workspace/external/discrepancy_vae/datasets/Norman2019_raw.h5ad","",True)
