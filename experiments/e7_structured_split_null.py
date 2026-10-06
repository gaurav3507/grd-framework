"""E7: structured control splits on homogeneous Gaussian controls (no heterogeneity).

Question: do the leading-PC-tail "structured control splits" of the real-data
diagnostics fire only when the controls are heterogeneous, or by construction?

Design. For each Perturb-seq geometry (control count, median powered-environment
size, d = 10) and each of two homogeneous Gaussian spectra, draw i.i.d. Gaussian
controls with NO subpopulation structure. Build the same 20 structured splits as
tier2_common.structured_controls (lower and upper m-tail of each coordinate, with
m the median environment size; the coordinates are already the PCs of a diagonal
Gaussian) and 20 random size-m splits. Screen both families with the unchanged E3
gate (e3_gate_compare._detect_family, corrected disjoint null, B=500, BH q=0.05),
five seeds. Nothing here touches real data.

Pre-registered reading: if structured splits fire near 20/20 on homogeneous
controls, the structured-split diagnostic demonstrates sensitivity to selection of
a control subpopulation and carries no information about control heterogeneity.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import tier2_common as T2  # noqa: E402
from e3_gate_compare import _detect_family  # noqa: E402

# Control count and median powered-environment size (NMIN=200), from
# results/e3_decisions/*.json and the e6 logs. d matches D_PROJ.
GEOMETRIES = {
    "K562": dict(n_control=10691, m=270),
    "RPE1": dict(n_control=11485, m=262),
    "Norman": dict(n_control=8907, m=516),
}
SPECTRA = {
    "isotropic": lambda d: np.ones(d),
    "decaying": lambda d: np.linspace(5.0, 0.5, d),
}
DATA_SALT = 7_070


def run(B, seeds, out):
    started = time.time()
    rows = []
    for geo, g in GEOMETRIES.items():
        for spec_name, spec in SPECTRA.items():
            for seed in seeds:
                rng = np.random.default_rng(
                    np.random.SeedSequence([DATA_SALT, seed,
                                            list(GEOMETRIES).index(geo),
                                            list(SPECTRA).index(spec_name)]))
                d, n, m = T2.D_PROJ, g["n_control"], g["m"]
                Y = rng.standard_normal((n, d)) * np.sqrt(spec(d))
                struct = []
                for j in range(d):
                    order = np.argsort(Y[:, j])
                    struct += [Y[order[:m]], Y[order[-m:]]]
                rand = [Y[rng.choice(n, m, replace=False)] for _ in range(20)]
                s = _detect_family(T2.PR, struct, Y, T2.STRUCT_SEED_OFFSET + seed,
                                   T2.ALPHA, B, T2.Q, disjoint=True)
                r = _detect_family(T2.PR, rand, Y, T2.FAKE_SEED_OFFSET + seed,
                                   T2.ALPHA, B, T2.Q, disjoint=True)
                row = dict(geometry=geo, spectrum=spec_name, seed=seed, n_control=n,
                           m=m, d=d, structured_raw=s["raw_count"],
                           structured_bh=s["bh_count"], random_raw=r["raw_count"],
                           random_bh=r["bh_count"], n_structured=len(struct),
                           n_random=len(rand))
                rows.append(row)
                print(f"{geo:6s} {spec_name:9s} s{seed}: structured raw/BH "
                      f"{s['raw_count']}/{s['bh_count']} of 20 | random raw/BH "
                      f"{r['raw_count']}/{r['bh_count']} of 20 "
                      f"({time.time() - started:.0f}s)", flush=True)
    report = dict(
        experiment="e7_structured_split_null",
        design=__doc__.strip().splitlines()[0],
        config=dict(B=B, alpha=T2.ALPHA, q=T2.Q, seeds=list(seeds), d=T2.D_PROJ,
                    geometries=GEOMETRIES, spectra=list(SPECTRA),
                    data_salt=DATA_SALT, null="corrected disjoint (E3)"),
        rows=rows,
        summary=dict(
            structured_bh_total=int(sum(r["structured_bh"] for r in rows)),
            structured_total=int(sum(r["n_structured"] for r in rows)),
            random_bh_total=int(sum(r["random_bh"] for r in rows)),
            random_total=int(sum(r["n_random"] for r in rows)),
            random_raw_total=int(sum(r["random_raw"] for r in rows)),
        ),
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    T2.write_json(out, report)
    print("summary", report["summary"])
    print(f"written {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--B", type=int, default=T2.B_BOOT)
    p.add_argument("--seeds", type=int, nargs="+", default=T2.SEEDS)
    p.add_argument("--out", default=str(REPO / "results" / "e7_structured_null"
                                        / "e7_structured_split_null.json"))
    a = p.parse_args()
    run(a.B, a.seeds, a.out)


if __name__ == "__main__":
    main()
