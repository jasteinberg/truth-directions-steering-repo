"""
Figures for 'Depth and dataset profile'.

The bound by depth and the seven-point window curves (with the refusal regimes), part 2, "Steering Vectors and the Limits of Linear Response".
Each writes truth_<name>.png to $FIGURE_DIR (default figures/).

Subcommands:

    depth-bound            Two panels for the linear-bound section
    window-curves          Two panels for the linear-window result
    all                    every figure above

Run from the repo root:  python scripts/figures/depth_profile.py <subcommand> [-h]

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
# depth-bound
# =============================================================================
A = json.load(open(REPO / "artifacts" / "depth_arms_analysis.json"))


BOUND_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_depth_bound.png"


LAYERS = [8, 12, 16, 20, 24, 28]


COL = {"counterfact": "#c0392b", "cities": "#2471a3"}


BOUND_S = A["cities_seeds"]


def run_depth_bound(argv=None):
    """Two panels for the linear-bound section. Left: measured small-alpha
    susceptibility ratio chi_0.5(g_hat)/(c||g_hat||) at each swept layer against
    the noise-floor prediction ||g||^2/(||g||^2 + tr Sigma_G/n), both datasets,
    seed 0 (cities L28 shows the five-seed mean with SE). Right: the width of the
    linear window, A(alpha)/alpha relative to the alpha=0.5 slope, per layer.

    Reads artifacts/depth_arms_analysis.json. Writes $FIGURE_DIR/truth_depth_bound.png.
    """
    argparse.ArgumentParser(description=run_depth_bound.__doc__).parse_args(argv)
    rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.6, "grid.alpha": 0.55, "font.size": 11,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ds in ("counterfact", "cities"):
        L = A["layers"][ds]
        pred = [L[str(l)]["pred"] for l in LAYERS]
        meas = [L[str(l)]["a05_over_pred"] * L[str(l)]["pred"] for l in LAYERS]
        fit = [L[str(l)]["fit_over_pred"] * L[str(l)]["pred"] for l in LAYERS]
        ax1.plot(LAYERS, pred, "-", color=COL[ds], lw=2, label=f"`{ds}` prediction")
        ax1.plot(LAYERS, meas, "o", color=COL[ds], ms=6, label=f"`{ds}` measured, $\\alpha=0.5$ slope")
        ax1.plot(LAYERS, fit, "x", color=COL[ds], ms=6, alpha=0.6, label=f"`{ds}` through-origin fit")
        for a, mk in zip(("1", "2", "4"), ("s", "^", "D")):
            ax2.plot(LAYERS, [L[str(l)]["lin_rel"][{"1": 0, "2": 1, "4": 2}[a]] for l in LAYERS], "-" + mk,
                     color=COL[ds], ms=5, lw=1, alpha={"1": 1.0, "2": 0.75, "4": 0.5}[a], label=f"`{ds}` $\\alpha={a}$")
    q = np.array([BOUND_S[k]["a05_over_pred"] * BOUND_S[k]["pred"] for k in BOUND_S])
    ax1.errorbar([28.6], [q.mean()], yerr=[q.std(ddof=1) / np.sqrt(len(q))], fmt="o", color=COL["cities"],
                 mfc="white", ms=6, capsize=3, label="`cities` L28, five seeds")
    ax1.set_xlabel("layer"); ax1.set_ylabel(r"$\chi(\hat g)\,/\,(c\,\|\hat g\|)$"); ax1.set_ylim(0, 1.05)
    ax1.set_xticks(LAYERS)
    # below the axes: inside, the lower-left legend covers the counterfact L8 through-origin point
    ax1.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax1.set_title("fraction of the linear bound reached vs. prediction")
    ax2.axhline(1, color="k", lw=0.8, ls="--"); ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("layer"); ax2.set_ylabel(r"$[A(\alpha)/\alpha]\,/\,[A(0.5)/0.5]$"); ax2.set_xticks(LAYERS)
    ax2.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)
    ax2.set_title("width of the linear window")
    fig.tight_layout(); BOUND_OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(BOUND_OUT, bbox_inches="tight")
    print("wrote", BOUND_OUT)


# =============================================================================
# window-curves
# =============================================================================
R = json.load(open(REPO / "artifacts" / "overnight_analysis.json"))


WINDOW_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_window_curves.png"


AL = np.array(R["alphas"])


def run_window_curves(argv=None):
    """Two panels for the linear-window result. Left: A(alpha)/alpha along g_hat,
    relative to its smallest resolved alpha, over alpha in {1/16 .. 4} at the six
    swept layers on counterfact (seed 0, complement split) and cities L28, with
    the standard alpha = 0.5 marked. Right: the refusal model at L12 and L15 along
    r_hat and g_hat, log-odds response (relative) with P(refusal | harmless) on a
    twin axis, showing the linear, saturated and destroyed regimes.

    Reads artifacts/overnight_analysis.json (`depth_profile.py windows`).
    Writes $FIGURE_DIR/truth_window_curves.png.
    """
    argparse.ArgumentParser(description=run_window_curves.__doc__).parse_args(argv)
    rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.6, "grid.alpha": 0.55, "font.size": 11,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    reds = plt.cm.Reds(np.linspace(0.35, 0.95, 6))
    for col, (L, w) in zip(reds, R["window"]["counterfact"].items()):
        ax1.plot(AL, w["rel"], "-o", color=col, ms=4, lw=1.4, label=f"`counterfact` L{L}")
    w = R["window"]["cities"]["28"]
    ax1.plot(AL[2:], np.array(w["rel"][2:]), "-s", color="#2471a3", ms=4, lw=1.4, label="`cities` L28 (from $\\alpha=1/4$)")
    ax1.axvline(0.5, color="k", lw=0.8, ls="--"); ax1.text(0.52, 0.06, "standard $\\alpha=0.5$", fontsize=8, rotation=90, va="bottom")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_xscale("log", base=2); ax1.set_xticks(AL); ax1.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    ax1.set_ylim(0, 1.15); ax1.set_xlabel(r"$\alpha$ (class-gap units)"); ax1.set_ylabel(r"$[A(\alpha)/\alpha]\,/\,[A(\alpha_0)/\alpha_0]$")
    ax1.set_title("Pythia-2.8b: window along $\\hat g$, seed 0", fontsize=10); ax1.legend(fontsize=7.5, frameon=False, loc="lower left")

    styles = {"12": "-", "15": "--"}
    for L, v in R["refusal_P"].items():
        for arm, col, lab in (("r", "#2471a3", "$\\hat r$"), ("g", "#c0392b", "$\\hat g$")):
            ax2.plot(AL, v[arm]["rel"], styles[L] + "o", color=col, ms=4, lw=1.4, label=f"{lab} L{L}, log-odds rel.")
    ax2b = ax2.twinx()
    for L, v in R["refusal_P"].items():
        for arm, col in (("r", "#2471a3"), ("g", "#c0392b")):
            ax2b.plot(AL, v[arm]["P_plus_harmless"], styles[L], color=col, lw=0.9, alpha=0.45, marker="x", ms=4)
    ax2b.set_ylim(0, 1.02); ax2b.set_ylabel(r"$P(\mathrm{refusal}\mid\mathrm{harmless},+\alpha)$  (faint, x)", fontsize=9)
    ax2b.grid(False); ax2b.spines["top"].set_visible(False)
    ax2.axvline(1, color="k", lw=0.8, ls="--"); ax2.text(1.04, 0.06, "Arditi $\\alpha=1$", fontsize=8, rotation=90, va="bottom")
    ax2.axhline(1, color="k", lw=0.6)
    ax2.set_xscale("log", base=2); ax2.set_xticks(AL); ax2.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    ax2.set_ylim(0, 1.15); ax2.set_xlabel(r"$\alpha$ (class-gap units)"); ax2.set_ylabel(r"$[A(\alpha)/\alpha]\,/\,[A(1/16)/(1/16)]$")
    ax2.set_title("Qwen1.5-1.8B-Chat refusal: linear, saturated, destroyed", fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="center left")
    fig.tight_layout(); WINDOW_OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(WINDOW_OUT, bbox_inches="tight")
    print("wrote", WINDOW_OUT)


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_depth_bound, run_window_curves,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "depth-bound": run_depth_bound,
    "window-curves": run_window_curves,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
