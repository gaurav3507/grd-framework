import sys, numpy as np, anndata as ad, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location("pr","src/gate/precision_readout.py")
pr=importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
D=10; NMIN=200; SEED=0
name,path,ctrl,single=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]=="1"
if ctrl=="EMPTY": ctrl=""
rng=np.random.default_rng(SEED)
A=ad.read_h5ad(path)
X=A.X.toarray().astype(np.float64) if hasattr(A.X,"toarray") else np.asarray(A.X,np.float64)
g=A.obs['guide_ids'].astype(str).values
del A
if single:
    is_single=lambda s: s!=ctrl and s!="" and ","not in s and "+"not in s and "_"not in s
    cand=sorted({s for s in np.unique(g) if is_single(s)})
    dropped=sorted({s for s in np.unique(g) if s!=ctrl and s not in cand})
    print("KEPT examples:",cand[:6]); print("DROPPED examples:",dropped[:6])
    print(f"single-gene labels {len(cand)} | dropped {len(dropped)}")
else:
    cand=[p for p in np.unique(g) if p!=ctrl]
Xc=X[g==ctrl]
mu=Xc.mean(0); _,_,Vt=np.linalg.svd(Xc-mu,full_matrices=False); Bp=Vt[:D].T
proj=lambda M:(M-mu)@Bp
Yobs=proj(Xc); Zc=Yobs
perts=[p for p in cand if (g==p).sum()>=NMIN]
print(f"{name}: X {X.shape} | control {len(Xc)} | powered perts {len(perts)}")
sizes=[int((g==p).sum()) for p in perts]
detect={};ratios={};shifts=[]
for p in perts:
    Yp=proj(X[g==p]); n=len(Yp)
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
    s=pr.precision_signal(Yp,Yobs); detect[p]=bool(s>crit); ratios[p]=float(s/crit)
    if s>crit:
        sh=Yp.mean(0)-Yobs.mean(0); shifts.append(sh/(np.linalg.norm(sh)+1e-12))
n_det=sum(detect.values()); rr=np.array(list(ratios.values()))
nfake=50; fdet=0
for _ in range(nfake):
    n=int(rng.choice(sizes)); sub=proj(Xc[rng.choice(len(Xc),n,replace=False)])
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=n)
    if pr.precision_signal(sub,Yobs)>crit: fdet+=1
m=int(np.median(sizes)); struct=[]
for j in range(D):
    o=np.argsort(Zc[:,j]); struct+=[o[:m],o[-m:]]
sdet=0
for sub in struct:
    crit=pr.null_threshold(Yobs,alpha=0.05,B=500,rng=rng,n_env=len(sub))
    if pr.precision_signal(proj(Xc[sub]),Yobs)>crit: sdet+=1
if shifts:
    M=np.array(shifts); t2=(M[:,:2]**2).sum(1); am,amd=float(t2.mean()),float(np.median(t2))
else: am=amd=None
rep=dict(dataset=name,n_cells=int(X.shape[0]),n_genes=int(X.shape[1]),control_label=repr(ctrl),
 n_control=int(len(Xc)),single_gene_only=single,d_proj=D,nmin=NMIN,n_powered_perts=len(perts),
 negative_control=f"{fdet}/{nfake}",negative_control_rate=round(fdet/nfake,4),
 structured_split_detect=f"{sdet}/{len(struct)}",structured_split_rate=round(sdet/len(struct),4),
 n_detectable=n_det,frac_detectable=round(n_det/len(perts),4),
 ratio_median=round(float(np.median(rr)),3),ratio_p90=round(float(np.quantile(rr,.9)),3),
 frac_gt_2=round(float((rr>2).mean()),4),
 confound_align_top2_mean=None if am is None else round(am,3),
 confound_align_top2_median=None if amd is None else round(amd,3))
Path(f"results/e3/e3_{name}.json").write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
