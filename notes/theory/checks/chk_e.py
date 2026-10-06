import numpy as np, json, sys
sys.argv=["x"]; exec(open("thm_checks.py").read().split('if __name__ == "__main__":')[0])
def centered_pooled(Ye, Y0, B, rng):
    Ye = Ye - Ye.mean(0); Y0 = Y0 - Y0.mean(0)
    return pooled_pvalue(Ye, Y0, B, rng)[0]
res = {}
for label, shift_size in (("null_same_law", 0.0), ("mean_shift_1", 1.0), ("mean_shift_3", 3.0)):
    Bm, lam, R = model(3); S0 = cov(Bm, lam, R); P = []
    for rep in range(150):
        rng = np.random.default_rng([6, rep, int(shift_size*10)])
        Y0 = draw(rng, S0, 1500)
        s = rng.standard_normal(D); s /= np.sqrt(s @ np.linalg.solve(S0, s))
        Ye = draw(rng, S0, 300) + shift_size * s
        P.append(centered_pooled(Ye, Y0, 99, rng))
    P = np.array(P); res[label] = dict(frac_p_le_0_05=float(np.mean(P<=0.05)), frac_p_le_0_20=float(np.mean(P<=0.2)))
    print(label, res[label], flush=True)
json.dump(res, open("thm_checks_E.json","w"), indent=2)
