import numpy as np, json, sys
Bb=int(sys.argv[1]); sys.argv=["x"]
exec(open("thm_checks.py").read().split('if __name__ == "__main__":')[0])
Bm, lam, R = model(1)
S0 = cov(Bm, lam, R); Se = cov(Bm, lam, R, target=0, lam_new=0.05)
m, q, ne = 21, 0.05, 2000
st, sn, pt = 0, 0, []
for rep in range(20):
    rng = np.random.default_rng([7, Bb, rep])
    Y0 = draw(rng, S0, 12000)
    envs = [draw(rng, Se, ne)] + [draw(rng, S0, ne) for _ in range(m - 1)]
    p = [PR.detect_with_pvalues([Y], Y0, B=Bb, rng=np.random.default_rng([8, Bb, rep, i]))["pvalues"][0] for i, Y in enumerate(envs)]
    d = bh(p, q); st += int(d[0]); sn += int(d[1:].sum()); pt.append(p[0])
r = dict(B=Bb, n_e=ne, true_selected=st, reps=20, null_selections=sn, true_p_at_floor=int(sum(np.isclose(pt, 1/(Bb+1)))))
print(r, flush=True); json.dump(r, open(f"thm_checks_B2_{Bb}.json","w"))
