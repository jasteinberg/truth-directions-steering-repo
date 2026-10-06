"""
Figures for the framework sections of the post.

"Separability, capacity and readout", "Drawing a random direction" and "The scale
of an intervention" (https://jasteinberg.github.io/blog/2026/truth-directions-snr/): the whitening schematic, the decoding null drawn on
real activations, and the behavioral-score schematic. Each writes truth_<name>.png
to $FIGURE_DIR (default figures/).

Subcommands:

    whitening-schematic    Schematic
    null-distribution      The decoding null drawn, not asserted
    behavioral-score       Schematic
    all                    every figure above

Run from the repo root:  python scripts/figures/framework.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import math as _m
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import provenance  # noqa: E402

# ---- shared by several subcommands ------------------------------------------
CACHE = f"{REPO}/artifacts/act_cache"
LAYER = 28
def grid(ax):
    ax.grid(True, ls=":", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)


# =============================================================================
# whitening-schematic
# =============================================================================
WHITEN_OUT = str(Path(os.environ.get("FIGURE_DIR", REPO / "figures")))


WHITEN_C_TRUE, WHITEN_C_FALSE = "#1f6f8b", "#c1553b"


def fig_whitening_schematic():
    rng = np.random.default_rng(0)
    n = 500
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, (title, S) in zip(axes, [
            ("isotropic noise:  $\\hat\\theta$ is already optimal", np.diag([1.0, 1.0])),
            ("anisotropic noise:  $\\hat\\theta$ is not", np.diag([9.0, 0.35]))]):
        d = np.array([1.4, 1.4])                       # class-mean gap
        A = np.linalg.cholesky(S)
        X0 = rng.standard_normal((n, 2)) @ A.T - d / 2
        X1 = rng.standard_normal((n, 2)) @ A.T + d / 2
        ax.scatter(X0[:, 0], X0[:, 1], s=6, alpha=0.35, c=WHITEN_C_FALSE, label="false")
        ax.scatter(X1[:, 0], X1[:, 1], s=6, alpha=0.35, c=WHITEN_C_TRUE, label="true")

        th = d / np.linalg.norm(d)                     # mass-mean
        thf = np.linalg.solve(S, d); thf /= np.linalg.norm(thf)   # Fisher

        for v, c, lab in ((th, "#222", r"$\hat\theta \propto \delta$"),
                          (thf, "#e0a106", r"$\hat\theta_F \propto \Sigma^{-1}\delta$")):
            ax.annotate("", xy=3.2 * v, xytext=-3.2 * v,
                        arrowprops=dict(arrowstyle="-", lw=2.2, color=c))
            ax.plot([], [], color=c, lw=2.2, label=lab)

        def dp(u):
            p1, p0 = X1 @ u, X0 @ u
            return abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var() + p0.var()))
        ax.set_title(f"{title}\n" + rf"$d'(\hat\theta)={dp(th):.2f}$,  $d'(\hat\theta_F)={dp(thf):.2f}$")
        ax.set_xlim(-7, 7); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="upper left", markerscale=2); grid(ax)

    fig.tight_layout()
    fig.savefig(f"{WHITEN_OUT}/truth_whitening_schematic.png", dpi=200)
    plt.close(fig)
    print("saved truth_whitening_schematic.png")


def run_whitening_schematic(argv=None):
    """Schematic: the mass-mean and Fisher directions under isotropic and anisotropic
    noise (truth_whitening_schematic.png).

    Reads the act cache; writes to $FIGURE_DIR (default figures/).
    """
    argparse.ArgumentParser(description=run_whitening_schematic.__doc__).parse_args(argv)
    os.makedirs(WHITEN_OUT, exist_ok=True)
    fig_whitening_schematic()


# =============================================================================
# null-distribution
# =============================================================================
NULLDIST_OUT = str(Path(os.environ.get("FIGURE_DIR", REPO / "figures")))


NULLDIST_C_TRUE, NULLDIST_C_FALSE = "#1f6f8b", "#c1553b"


def geometry(X, y):
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] -= X[y == lab].mean(0)
    C = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    p = X @ th
    p1, p0 = p[y == 1], p[y == 0]
    dprime = abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1)))
    return th, w, V, dprime


def fig_null_schematic():
    z = np.load(f"{CACHE}/pythia-2.8b__cities.npz")
    y = z["y"]; X = z[f"L{LAYER}"].astype(np.float64)
    th, w, V, dp_true = geometry(X, y)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(400):
        u = rng.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
        p = X @ u; p1, p0 = p[y == 1], p[y == 0]
        ds.append(abs(p1.mean() - p0.mean()) / np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1))))
    ds = np.array(ds); p95 = np.percentile(ds, 95)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(ds, bins=40, color="#8a8a8a", alpha=0.75, label="400 random directions")
    ax.axvline(p95, color="#444", ls="--", lw=1.6, label=rf"null $p_{{95}}$ = {p95:.2f}")
    ax.axvline(dp_true, color=NULLDIST_C_TRUE, lw=2.4, label=rf"$\hat\theta$: $d'$ = {dp_true:.2f}")
    ax.set_xlabel(r"$d'(u)$"); ax.set_ylabel("count")
    ax.set_title(f"The decoding null: cities, pythia-2.8b layer {LAYER}")
    ax.legend(fontsize=9); grid(ax)
    fig.tight_layout()
    fig.savefig(f"{NULLDIST_OUT}/truth_null_distribution.png", dpi=200)
    plt.close(fig)
    print("saved truth_null_distribution.png")


def run_null_distribution(argv=None):
    """The decoding null drawn, not asserted: d' of 400 random directions against the
    mass-mean direction, cities, pythia-2.8b layer 28 (truth_null_distribution.png).

    Reads the act cache; writes to $FIGURE_DIR (default figures/).
    """
    argparse.ArgumentParser(description=run_null_distribution.__doc__).parse_args(argv)
    os.makedirs(NULLDIST_OUT, exist_ok=True)
    fig_null_schematic()


# =============================================================================
# behavioral-score
# =============================================================================
BEHAV_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_behavioral_score.png"


BEHAV_C_TRUE = "#2c6fbb"


BEHAV_C_FALSE = "#c0392b"


C_BOX = "#f2f0ea"


C_EDGE = "#8a8578"


C_STEER = "#7a5aa0"


def box(ax, xy, w, h, text, fc=C_BOX, ec=C_EDGE, fs=10, mono=False, tc="black"):
    bx = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                        fc=fc, ec=ec, lw=0.9)
    ax.add_patch(bx)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc,
            family="monospace" if mono else "serif")
    return xy[0] + w / 2, xy[1] + h / 2


def arrow(ax, p0, p1, color=C_EDGE, lw=1.2, style="-|>", shrink=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink,
                                 mutation_scale=12))


def run_behavioral_score(argv=None):
    """Schematic: the behavioral score ell(x) and the odd/even split of the steering
    response.

    Left panel: one prompt, two completions, ell as the log-probability difference.
    Right panel: the +h and -h steered passes and the antisymmetric/symmetric parts
    A and S of Delta(h).

    Pure schematic -- no data dependencies. Example pair is the counterfact
    '.NET Framework' pair used in the post's text.
    """
    argparse.ArgumentParser(description=run_behavioral_score.__doc__).parse_args(argv)
    rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 150,
        "mathtext.fontset": "dejavuserif",
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    for ax in (axL, axR):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---------------- left: the score --------------------------------------------
    axL.set_title(r"The behavioral score $\ell(x)$", fontsize=12, pad=8)

    pc = box(axL, (0.06, 0.42), 0.42, 0.16,
             '".NET Framework is\ncreated by"', mono=True, fs=9)
    axL.text(0.27, 0.63, "prompt", ha="center", fontsize=9, color=C_EDGE)

    tc_ = box(axL, (0.62, 0.66), 0.30, 0.13, '" Microsoft"', mono=True, fs=9,
              ec=BEHAV_C_TRUE, tc=BEHAV_C_TRUE)
    fc_ = box(axL, (0.62, 0.21), 0.30, 0.13, '" Google"', mono=True, fs=9,
              ec=BEHAV_C_FALSE, tc=BEHAV_C_FALSE)
    axL.text(0.77, 0.82, "true completion", ha="center", fontsize=9, color=BEHAV_C_TRUE)
    axL.text(0.77, 0.145, "false completion", ha="center", fontsize=9, color=BEHAV_C_FALSE)

    arrow(axL, (0.48, 0.54), (0.62, 0.72), color=BEHAV_C_TRUE)
    arrow(axL, (0.48, 0.46), (0.62, 0.28), color=BEHAV_C_FALSE)


    _ang = _m.degrees(_m.atan2(0.72 - 0.54, 0.62 - 0.48))
    axL.text(0.525, 0.652, r"$\log P_{\mathrm{true}}$", fontsize=10, color=BEHAV_C_TRUE,
             ha="center", va="center", rotation=_ang, rotation_mode="anchor",
             transform_rotates_text=True)
    axL.text(0.512, 0.332, r"$\log P_{\mathrm{false}}$", fontsize=10, color=BEHAV_C_FALSE,
             ha="center", va="center", rotation=-_ang, rotation_mode="anchor",
             transform_rotates_text=True)

    axL.text(0.5, 0.02,
             r"$\ell(x) = \log P_{\mathrm{true}} - \log P_{\mathrm{false}}$"
             "\n(summed over completion tokens; same prompt, so its likelihood cancels)",
             ha="center", va="bottom", fontsize=10)

    # ---------------- right: the odd/even split ----------------------------------
    axR.set_title(r"Steering response: odd and even parts", fontsize=12, pad=8)

    x0 = box(axR, (0.05, 0.42), 0.26, 0.16, "$x$\nclean pass", fs=10)
    xp = box(axR, (0.55, 0.68), 0.34, 0.14, r"$x + h\,c\,w$", fs=10, ec=C_STEER)
    xm = box(axR, (0.55, 0.28), 0.34, 0.14, r"$x - h\,c\,w$", fs=10, ec=C_STEER)

    arrow(axR, (0.31, 0.54), (0.55, 0.75), color=C_STEER)
    arrow(axR, (0.31, 0.46), (0.55, 0.35), color=C_STEER)
    axR.text(0.42, 0.70, r"$+h$", fontsize=10, color=C_STEER)
    axR.text(0.42, 0.33, r"$-h$", fontsize=10, color=C_STEER)

    axR.text(0.72, 0.86, r"$\Delta(+h)=\langle\ell\rangle_{+h}-\langle\ell\rangle_0$",
             ha="center", fontsize=10)
    axR.text(0.72, 0.21, r"$\Delta(-h)=\langle\ell\rangle_{-h}-\langle\ell\rangle_0$",
             ha="center", fontsize=10)

    axR.text(0.5, 0.02,
             r"$A=\frac{1}{2}\left[\Delta(+h)-\Delta(-h)\right]$: odd, a signed direction"
             "\n"
             r"$S=\frac{1}{2}\left[\Delta(+h)+\Delta(-h)\right]$: even, generic disruption",
             ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    BEHAV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BEHAV_OUT, bbox_inches="tight")
    print(f"wrote {BEHAV_OUT}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_whitening_schematic, run_null_distribution, run_behavioral_score,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "whitening-schematic": run_whitening_schematic,
    "null-distribution": run_null_distribution,
    "behavioral-score": run_behavioral_score,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
