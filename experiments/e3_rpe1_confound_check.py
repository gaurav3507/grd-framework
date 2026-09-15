"""RPE1 confound double-check: is RPE1's higher detection systematic variation, not mechanism?
Check 1: random control splits (expect ~alpha) vs STRUCTURED splits along control PCs
         (if high, the gate fires on control heterogeneity = P3 confound).
Check 2: energy of detected perts' shifts in the top-2 control PCs (high = re-reads heterogeneity).
"""
import numpy as np, anndata as ad, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location("pr","src/gate/precision_readout.py")
pr=importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
H5='/workspace/external/discrepancy_vae/datasets/causalbench_rpe1.h5ad'
CTRL=''; NMIN=200; D=10; SEED=0; rng=np.random.default_rng(SEED)
A=ad.read_h5ad(H5)
X=A.X.toarray().astype(np.float64) if hasattr(A.X,"toarray") else np.asarray(A.X,np.float64)
g=A.obs['guide_ids'].astype(str).values; del A
Xc=X[g==CTRL]; mu=Xc.mean(0); _,_,Vt=np.linalg.svd(Xc-mu,full_matrices=False); Bp=Vt[:D].T
proj=lambda M:(M-mu)@Bp; Yobs=proj(Xc); Zc=Yobs
perts=[p for p in np.unique(g) if p!=CTRL and (g==p).sum()>=NMIN]; sizes=[int((g==p).sum()) for p in perts]
def detect_rate(subsets):
    d=0
    for sub in subsets:
        crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=len(sub))
        if pr.precision_signal(proj(Xc[sub]),Yobs)>crit: d+=1
    return d,len(subsets)
pc=Zc-Zc.mean(0); m=int(np.median(sizes)); struct=[]
for j in range(D):
    o=np.argsort(pc[:,j]); struct+=[o[:m],o[-m:]]
rand=[rng.choice(len(Xc),int(rng.choice(sizes)),replace=False) for _ in range(20)]
dr,nr=detect_rate(rand); ds,ns=detect_rate(struct)
det_shifts=[]
for p in perts:
    Yp=proj(X[g==p]); crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=len(Yp))
    if pr.precision_signal(Yp,Yobs)>crit:
        s=Yp.mean(0)-Yobs.mean(0); det_shifts.append(s/np.linalg.norm(s))
M=np.array(det_shifts); top2=(M[:,:2]**2).sum(1)
rep=dict(dataset="causalbench_rpe1",random_split_detect=f"{dr}/{nr}",structured_split_detect=f"{ds}/{ns}",
         n_detected_perts=len(det_shifts),top2_energy_mean=round(float(top2.mean()),3),
         top2_energy_median=round(float(np.median(top2)),3))
Path("results/e3").mkdir(parents=True,exist_ok=True)
Path("results/e3/e3_rpe1_confound_check.json").write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
