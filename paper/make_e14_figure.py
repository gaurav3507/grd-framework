"""E14 figure: risk against coverage on the clean arms (ABD) and the contaminated arm (C).

Reads results/e14/curves.json only (written by experiments/e14_matched_coverage.py
--analyse). Gates: G0 certify all, G2 GRD precision, G5 bootstrap stability, G8
geneig, G9 energy. Shaded bands are the 2.5 to 97.5 percentiles of the bootstrap
curves (1000 resamples of the 200 (arm, level, seed) systems, decisions fixed)
interpolated on a common coverage grid, for G2, G8 and G9. Open circles mark each
BH gate's q = 0.05 operating point; the dotted line is G2's q = 0.05 coverage, where
matched risk is read. Each panel has its own risk scale.

Output: paper/figures/figure_e14_risk_coverage.pdf (matplotlib only).
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CURVES = REPO / "results" / "e14" / "curves.json"
OUTPUT = REPO / "paper" / "figures" / "figure_e14_risk_coverage.pdf"

STYLE = {
    "G0": dict(label="G0 certify all", color="#7f7f7f", marker="s", ls="none", ms=6.5, z=5),
    "G2": dict(label="G2 GRD precision", color="#000000", marker="o", ls="-"),
    "G5": dict(label="G5 bootstrap stability", color="#E69F00", marker="^", ls="--"),
    "G8": dict(label="G8 geneig", color="#D55E00", marker="D", ls="-"),
    "G9": dict(label="G9 energy", color="#009E73", marker="v", ls="-"),
}
BANDS = ("G2", "G8", "G9")
PANELS = (("ABD", "clean arms (A, B, D)"), ("C", "contaminated arm (C)"))


def configure_style():
    # Same Computer Modern / serif settings as paper/make_all_figures.py.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _nan(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def main():
    doc = json.loads(CURVES.read_text(encoding="utf-8"))
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, (scope, title) in zip(axes, PANELS):
        ymax = 0.0
        for gate in BANDS:
            b = doc["bands"][scope][gate]
            lo, hi = _nan(b["p2_5"]), _nan(b["p97_5"])
            ax.fill_between(b["coverage"], lo, hi, color=STYLE[gate]["color"], alpha=0.14, lw=0)
            if np.isfinite(hi).any():
                ymax = max(ymax, float(np.nanmax(hi)))
        for gate, st in STYLE.items():
            curve = doc["curves"][gate][scope]["curve"]
            if not curve:
                continue
            c, r = zip(*curve)
            ax.plot(c, r, ls=st["ls"], marker=st["marker"], ms=st.get("ms", 3.2), lw=1.1,
                    color=st["color"], label=st["label"], zorder=st.get("z", 3))
            ymax = max(ymax, max(r))
            if gate in BANDS:
                p = next(pt for pt in doc["curves"][gate][scope]["points"] if pt["knob"] == doc["q_default"])
                if p["risk"] is not None:
                    ax.plot([p["coverage"]], [p["risk"]], marker="o", ms=7, mfc="none",
                            mec=st["color"], mew=1.0, ls="none", zorder=4)
        target = doc["summary"][scope]["target_coverage"]
        if target is not None:
            ax.axvline(target, color="#555555", ls=":", lw=0.8, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("coverage (fraction of units certified)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005 * max(1.0, ymax / 0.1), max(0.05, ymax * 1.08))
    axes[0].set_ylabel("risk (bad units among certified)")
    axes[0].legend(loc="upper left", frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
