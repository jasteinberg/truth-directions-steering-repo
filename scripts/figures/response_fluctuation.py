"""
Figures for 'Response against fluctuation'.

The response spectrum against the fluctuation spectrum, part 2, "Steering Vectors and the Limits of Linear Response".
Each writes truth_<name>.png to $FIGURE_DIR (default figures/).

Subcommands:

    response-spectrum      Response against fluctuation at layer 28, pythia-2.8b
    all                    every figure above

Run from the repo root:  python scripts/figures/response_fluctuation.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import provenance  # noqa: E402

# =============================================================================
# response-spectrum
# =============================================================================
VEC = REPO / "artifacts" / "response_spectrum_vectors.npz"


RSPEC_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_response_spectrum.png"


LAYER = 28


PANELS = [("counterfact", "`counterfact`"), ("cities", "`cities`")]


C_PT, C_BIN, C_V1 = "#7f8c8d", "#c0392b", "#d68910"


def run_response_spectrum(argv=None):
    """Response against fluctuation at layer 28, pythia-2.8b: the share of the score
    gradient on each within-class eigendirection, r_i = |g.v_i|^2/||g||^2, against
    that direction's variance lambda_i, log-log, one panel per dataset. Only the
    directions with nonzero sample variance are drawn (N < d). Reference lines: the
    random-direction expectation 1/d, and the mean per-axis noise floor in g-hat,
    tr(Sigma_G)/(n d), and the random median 0.455/d (r_i ~ chi^2_1/d). Binned medians show the trend, or its absence.

    Reads artifacts/response_spectrum_vectors.npz (from `response_fluctuation.py spectrum`).
    Writes $FIGURE_DIR/truth_response_spectrum.png (default REPO/figures).
    """
    argparse.ArgumentParser(description=run_response_spectrum.__doc__).parse_args(argv)
    rcParams.update({
        "font.family": "serif", "axes.grid": True, "grid.linestyle": ":",
        "grid.linewidth": 0.6, "grid.alpha": 0.55, "font.size": 11,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    })
    z = np.load(VEC)

    meta = json.load(open(REPO / "artifacts" / "response_spectrum.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (short, label) in zip(axes, PANELS):
        lam = z[f"{short}_L{LAYER}_lam"]; r = z[f"{short}_L{LAYER}_r"]
        d = len(lam)
        keep = lam > 1e-8 * lam[0]
        lam_k, r_k = lam[keep], r[keep]
        ax.scatter(lam_k[1:], r_k[1:], s=6, color=C_PT, alpha=0.45, linewidths=0, label="eigendirections")
        ax.scatter([lam_k[0]], [r_k[0]], s=90, marker="*", color=C_V1, zorder=5, label=r"$\hat v_1$")
        # binned medians over log-lambda deciles
        edges = np.quantile(np.log(lam_k), np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(np.log(lam_k), edges[1:-1]), 0, 9)
        med = [np.median(r_k[idx == b]) for b in range(10)]
        ctr = [np.exp(np.median(np.log(lam_k[idx == b]))) for b in range(10)]
        ax.plot(ctr, med, "-o", color=C_BIN, ms=4, lw=1.4, label="median per $\\lambda$ decile")
        ax.axhline(1 / d, color="k", lw=1.0, ls="--", label="random direction, mean $1/d$")
        ax.axhline(0.4549 / d, color="k", lw=1.0, ls="-.", label="random direction, median $0.455/d$")
        floor = meta["datasets"][short][str(LAYER)]["noise_share_of_g2"] / d
        ax.axhline(floor, color="k", lw=0.8, ls=":", label="mean noise floor per axis")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\hat\lambda_i$ (within-class variance)")
        ax.set_title(f"{label}, layer {LAYER}")
        ax.text(0.02, 0.04, f"$d={d}$, nonzero $={keep.sum()}$\nshare on $\\hat v_1$: {r_k[0]:.1e}",
                transform=ax.transAxes, fontsize=9, va="bottom")
    axes[0].set_ylabel(r"$r_i = |g\cdot\hat v_i|^2 / \|g\|^2$")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    RSPEC_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RSPEC_OUT, bbox_inches="tight")
    print(f"wrote {RSPEC_OUT}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_response_spectrum,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "response-spectrum": run_response_spectrum,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
