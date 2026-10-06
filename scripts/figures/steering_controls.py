"""
Figures for 'Steering controls and the exact response'.

Measured chi against the exact functional c (w . g_eval) over every steered arm, part 2, "Steering Vectors and the Limits of Linear Response".
Each writes truth_<name>.png to $FIGURE_DIR (default figures/).

Subcommands:

    exact-response         Three panels
    all                    every figure above

Run from the repo root:  python scripts/figures/steering_controls.py <subcommand> [-h]

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
# exact-response
# =============================================================================
C = json.load(open(REPO / "artifacts" / "controls_analysis.json"))


EXACT_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_exact_response.png"


KIND = [("g", "$\\hat g$", "#c0392b", "*", 11), ("target", "targets $\\hat\\theta,\\hat\\theta_F,\\hat e_2$", "#2471a3", "o", 6),
        ("perp", "projected-out", "#1e8449", "s", 5), ("match", "matched-overlap", "#b7950b", "^", 5),
        ("shuf", "shuffled labels", "#7d3c98", "D", 4), ("rand", "random", "#7f8c8d", ".", 4)]


def kind(name):
    if name == "g":
        return "g"
    if name.startswith("rand"):
        return "rand"
    if name.startswith("match"):
        return "match"
    if "perp" in name:
        return "perp"
    if "shuf" in name:
        return "shuf"
    return "target"


def panel(ax, rows, title, unit, target_label=None):
    kinds = [(k, target_label if k == "target" else lab, c, m, s) for k, lab, c, m, s in KIND] if target_label else KIND
    m = np.array([r["measured"] for r in rows]); e = np.array([r["exact"] for r in rows])
    lo, hi = min(e.min(), m.min()), max(e.max(), m.max()); pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8, zorder=0)
    for k, lab, col, mk, ms in kinds:
        idx = [i for i, r in enumerate(rows) if kind(r["arm"]) == k]
        if idx:
            ax.plot(e[idx], m[idx], mk, color=col, ms=ms, mec="k" if k == "g" else col, mew=0.4, alpha=0.85, label=f"{lab} ({len(idx)})")
    s = float((m * e).sum() / (e * e).sum()); r = float(np.corrcoef(m, e)[0, 1])
    ax.set_title(f"{title}\n{len(rows)} arms, slope {s:.3f}, corr {r:.4f}", fontsize=10)
    ax.set_xlabel(f"exact functional $c\\,(w\\cdot\\bar g_{{\\rm eval}})$ [{unit}]")
    ax.set_ylabel(f"measured $\\chi(w)$ [{unit}]")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def run_exact_response(argv=None):
    """Three panels: measured small-alpha susceptibility chi(w) against the exact
    linear functional c (w . g_eval), g_eval the mean per-example gradient over
    the held-out examples, for every steered direction. Left counterfact L28
    (ten seeds, all arms), middle cities L28 (five seeds), right refusal
    (Qwen1.5-1.8B-Chat, four layers, two prompt draws). The line is y = x.

    Reads artifacts/controls_analysis.json (regenerate with
    `steering_controls.py controls`, which must have stored the per-arm rows under
    refusal_exact[*]['arms']). Writes $FIGURE_DIR/truth_exact_response.png.
    """
    argparse.ArgumentParser(description=run_exact_response.__doc__).parse_args(argv)
    rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.6, "grid.alpha": 0.55, "font.size": 11,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panel(axes[0], C["exact_linear_response"]["counterfact"]["arms"], "`counterfact` L28, ten seeds", "log-odds / class gap")
    panel(axes[1], C["exact_linear_response"]["cities"]["arms"], "`cities` L28, five seeds", "log-odds / class gap")
    ref = [dict(r) for v in C["refusal_exact"].values() for r in v["arms"]]  # _P reruns excluded upstream, in controls_analysis
    for r in ref:
        if r["arm"] == "r":
            r["arm"] = "target"
    panel(axes[2], ref, "refusal, Qwen1.5-1.8B-Chat, L9–L18, three draws", "log-odds / class gap", target_label="$\\hat r$ (Arditi)")
    fig.tight_layout(); EXACT_OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(EXACT_OUT, bbox_inches="tight")
    print("wrote", EXACT_OUT)


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_exact_response,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "exact-response": run_exact_response,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
