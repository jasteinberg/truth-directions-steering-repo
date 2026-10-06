"""
Figures for 'Dimensional slack and sample size'.

The shuffled-label law and its PR collapse, from snr_sweep.json and
cover_by_dataset.json (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).

Subcommands:

    shuffled-law           The pooled shuffled-label law (main text)
    pr-collapse            The effective-dimension collapse (appendix)
    dropper-gaussian       The statements dropping massive activations and the Gaussian check (appendix)
    all                    every figure above

Run from the repo root:  python scripts/figures/dimensional_slack.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import math
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
# shuffled-law, pr-collapse
# =============================================================================
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))


COVER = json.load(open(REPO / "artifacts" / "cover_by_dataset.json"))


FIG_DIR = Path(os.environ.get("FIGURE_DIR", REPO / "figures"))


LAW_OUT = FIG_DIR / "truth_shuffled_law.png"


COLLAPSE_OUT = FIG_DIR / "truth_pr_collapse.png"


# counterfact with the dropper statements excluded (followup/cover_no_droppers.py); optional
NO_DROP = REPO / "artifacts" / "cover_no_droppers.json"


MODEL_STYLE = {
    "EleutherAI/pythia-70m":  ("70m",  "o", "#95a5a6"),
    "EleutherAI/pythia-410m": ("410m", "s", "#7f8c8d"),
    "EleutherAI/pythia-1.4b": ("1.4b", "^", "#34495e"),
    "EleutherAI/pythia-2.8b": ("2.8b", "D", "#c0392b"),
}


DS_STYLE = {
    "counterfact_true_false": ("counterfact", "D", "#c0392b"),
    "cities":                 ("cities",      "o", "#2E6DA4"),
    "larger_than":            ("larger_than", "s", "#d68910"),
    "sp_en_trans":            ("sp_en_trans", "^", "#7d3c98"),
}


C0 = 1.0 / math.sqrt(math.pi)


def _style():
    rcParams.update({
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def run_shuffled_law(argv=None):
    """The pooled shuffled-label law (main text, *Dimensional slack and sample size*).

    In-sample excess AUROC of the mass-mean direction under shuffled labels on
    counterfact, four Pythia widths, against N/2d, with the pooled power-law fit.
    Points with non-positive excess (three at pythia-70m) are dropped from the log fit
    and not drawn. Writes truth_shuffled_law.png.
    """
    argparse.ArgumentParser(description=run_shuffled_law.__doc__).parse_args(argv)
    _style()
    fig, ax1 = plt.subplots(figsize=(5.8, 4.3))
    xs, ys = [], []
    for m, (lab, mk, col) in MODEL_STYLE.items():
        rows = SWEEP["models"][m]["cover"]
        x = np.array([r["N_over_2d"] for r in rows])
        y = np.array([r["shuffled"]["auroc"] - 0.5 for r in rows])
        keep = y > 0
        ax1.scatter(x[keep], y[keep], marker=mk, s=34, color=col, label=f"pythia-{lab}", zorder=3)
        xs.append(x[keep]); ys.append(y[keep])
    x = np.concatenate(xs); y = np.concatenate(ys)
    b, la = np.polyfit(np.log(x), np.log(y), 1)
    xx = np.geomspace(x.min() * 0.7, x.max() * 1.4, 100)
    ax1.plot(xx, math.exp(la) * xx ** b, color="k", lw=1.1,
             label=rf"pooled fit ${math.exp(la):.3f}\,(N/2d)^{{{b:.2f}}}$")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"$N/2d$")
    ax1.set_ylabel(r"in-sample $\mathrm{AUROC}_{\mathrm{shuffled}} - 1/2$")
    ax1.set_title("Shuffled labels still separate, and it scales as $N^{-1/2}$", fontsize=11)
    ax1.axvline(1.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax1.text(1.05, 0.3, "Cover's capacity\n$N = 2d$", fontsize=8.5, va="top")
    ax1.legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.tight_layout()
    LAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LAW_OUT, bbox_inches="tight")
    print(f"wrote {LAW_OUT}")
    print(f"pooled fit: a={math.exp(la):.4f} b={b:.3f}")


def _collapse_series(ax, v, lab, mk, col, filled=True, alpha=1.0):
    PR = v["spectrum"]["participation_ratio"]
    N = np.array([r["N"] for r in v["curve"]])
    exc = np.array([r["excess_mean"] for r in v["curve"]])
    sd = np.array([r["excess_sd"] for r in v["curve"]]) / np.sqrt(v["curve"][0]["n_rep"])
    ax.errorbar(N / PR, exc, yerr=sd, fmt=mk, ms=6, color=col, capsize=2, lw=0.8, alpha=alpha,
                mfc=col if filled else "none", label=rf"{lab}  (PR = {PR:.1f})", zorder=3)


def run_pr_collapse(argv=None):
    """The effective-dimension collapse (appendix *The effective-dimension prefactor*).

    The shuffled-label excess on pythia-2.8b for four datasets at their own best layer,
    against N/PR, with the parameter-free prediction (1/sqrt(pi)) sqrt(PR/N).
    counterfact is drawn twice when cover_no_droppers.json exists: hollow with all
    statements, filled with the statements that lack the massive activation excluded.
    Writes truth_pr_collapse.png.
    """
    argparse.ArgumentParser(description=run_pr_collapse.__doc__).parse_args(argv)
    _style()
    fig, ax2 = plt.subplots(figsize=(5.8, 4.3))
    nodrop = json.load(open(NO_DROP))
    for ds, (lab, mk, col) in DS_STYLE.items():
        v = COVER["results"][ds]
        if ds == "counterfact_true_false":
            _collapse_series(ax2, v, f"{lab}, all statements", mk, col, filled=False, alpha=0.55)
            _collapse_series(ax2, nodrop, f"{lab}, without the statements\n    dropping massive activations", mk, col)
        else:
            _collapse_series(ax2, v, lab, mk, col)
    xx = np.geomspace(2.0, 400.0, 100)
    ax2.plot(xx, C0 * xx ** -0.5, color="k", lw=1.1,
             label=r"$\pi^{-1/2}\,\sqrt{\mathrm{PR}/N}$, no free parameter")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"$N/\mathrm{PR}$")
    ax2.set_ylabel(r"in-sample $\mathrm{AUROC}_{\mathrm{shuffled}} - 1/2$")
    ax2.set_title("Effective dimension sets the amplitude (pythia-2.8b)", fontsize=11)
    ax2.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    COLLAPSE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COLLAPSE_OUT, bbox_inches="tight")
    print(f"wrote {COLLAPSE_OUT}")



# =============================================================================
# dropper-gaussian
# =============================================================================
KURT = REPO / "artifacts" / "kurtosis_droppers.json"


PROJ = REPO / "artifacts" / "dropper_projection.npz"


DROPPER_OUT = FIG_DIR / "truth_dropper_gaussian.png"


def run_dropper_gaussian(argv=None):
    """The statements dropping massive activations break the Gaussian check (appendix).

    pythia-2.8b counterfact, 3000 statements, of which the massive_mask rule at layer 28
    (label-free) flags those dropping the massive activation.
      top     layer 31: class-centred projections onto the leading within-class axis v1,
              in bulk standard deviations, bulk against the flagged statements (log counts)
      left    excess kurtosis of the shuffled-label projections against N, layers 30-32,
              all statements (hollow) and with the flagged statements excluded (filled);
              the two-point value 1/p - 6 for the flagged fraction p is marked
      right   step B, measured (AUROC - 1/2) over Phi(d'/sqrt2) - 1/2, same encoding,
              with the +-3% band around 1

    Reads artifacts/kurtosis_droppers.json and artifacts/dropper_projection.npz.
    """
    argparse.ArgumentParser(description=run_dropper_gaussian.__doc__).parse_args(argv)
    _style()
    K = json.load(open(KURT))
    P = np.load(PROJ)
    drop = P["drop"].astype(bool)
    p = drop.mean()
    col = DS_STYLE["counterfact_true_false"][2]
    grey = "#7f8c8d"

    fig = plt.figure(figsize=(8.6, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax0 = fig.add_subplot(gs[0, :])
    axk = fig.add_subplot(gs[1, 0])
    axb = fig.add_subplot(gs[1, 1], sharex=axk)

    z = P["z_L31"]
    bins = np.linspace(min(z.min(), -10), z.max() + 5, 120)
    ax0.hist(z[~drop], bins=bins, color=grey, alpha=0.85, label=f"bulk ({(~drop).sum()} statements)")
    ax0.hist(z[drop], bins=bins, color=col, alpha=0.95,
             label=f"dropping massive activations ({drop.sum()} statements)")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"projection onto $\hat v_1$ at layer 31 (bulk standard deviations)")
    ax0.set_ylabel("statements")
    ax0.legend(frameon=False, fontsize=9, loc="upper center")

    for L, ls, lw in [("31", "-", 2.0), ("30", ":", 1.2), ("32", "--", 1.2)]:
        rows = K["layers"][L]
        for key, filled in [("all", False), ("no_droppers", True)]:
            c = rows[key]["curve"]
            N = [r["N"] for r in c]
            fc = col if filled else "white"
            axk.plot(N, [r["kurtosis"] for r in c], ls=ls, lw=lw, color=col, marker="D",
                     ms=5, mfc=fc, mec=col)
            axb.plot(N, [r["step_B"] for r in c], ls=ls, lw=lw, color=col, marker="D",
                     ms=5, mfc=fc, mec=col)
    axk.set_yscale("symlog", linthresh=1.0)
    axk.axhline(0, color="k", lw=0.9)
    axk.axhline(1 / p - 6, color=grey, lw=1.0, ls="--")
    axk.text(105, (1 / p - 6) * 1.15, r"two-point value $1/p - 6$", fontsize=8.5, color="#566573")
    axk.set_ylabel("excess kurtosis")
    axb.axhspan(0.97, 1.03, color=grey, alpha=0.15, lw=0)
    axb.axhline(1.0, color="k", lw=0.9)
    axb.set_ylabel(r"step B: measured / $\Phi(d'/\sqrt{2})$ prediction")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    Ns = [r["N"] for r in K["layers"]["31"]["all"]["curve"]]
    for ax in (axk, axb):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(Ns))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_xlabel(r"$N$")
    axk.yaxis.set_major_locator(FixedLocator([-1, 0, 1, 10, 100]))
    axk.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col, marker="D", mfc="white", mec=col, ls="none", label="all statements"),
          Line2D([], [], color=col, marker="D", mfc=col, mec=col, ls="none", label="excluded"),
          Line2D([], [], color=col, lw=2.0, ls="-", label="layer 31"),
          Line2D([], [], color=col, lw=1.2, ls=":", label="layer 30"),
          Line2D([], [], color=col, lw=1.2, ls="--", label="layer 32")]
    axk.legend(handles=hs, frameon=False, fontsize=8, loc="center right", ncol=2)
    fig.tight_layout()
    DROPPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DROPPER_OUT, bbox_inches="tight")
    print(f"wrote {DROPPER_OUT}")
    for L in ("30", "31", "32"):
        zz = P[f"z_L{L}"]
        print(f"  L{L}: median z of flagged {np.median(zz[drop]):.1f}, bulk |z|>5: {(np.abs(zz[~drop]) > 5).sum()}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_shuffled_law, run_pr_collapse, run_dropper_gaussian):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "shuffled-law": run_shuffled_law,
    "pr-collapse": run_pr_collapse,
    "dropper-gaussian": run_dropper_gaussian,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
