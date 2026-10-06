"""
Figures for 'Regime of recoverability'.

Emergence across the Pythia ladder, the plausibility-vs-truth depth profile, and the
cross-dataset transfer matrix, all read from artifacts/snr_sweep.json (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).

Subcommands:

    emergence              Figure
    plausibility-depth     Figure
    transfer               Figure
    all                    every figure above

Run from the repo root:  python scripts/figures/recoverability.py <subcommand> [-h]

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
from matplotlib.lines import Line2D

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import provenance  # noqa: E402

# ---- shared by several subcommands ------------------------------------------
SWEEP = Path(os.environ.get("SNR_SWEEP", REPO / "artifacts" / "snr_sweep.json"))
S = json.load(open(SWEEP))


# =============================================================================
# emergence
# =============================================================================
EMERGE_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_emergence.png"


DATASETS = os.environ.get("EMERGENCE_DS", "cities,counterfact_true_false").split(",")


LADDER = [
    ("EleutherAI/pythia-70m", "70m"),
    ("EleutherAI/pythia-410m", "410m"),
    ("EleutherAI/pythia-1.4b", "1.4b"),
    ("EleutherAI/pythia-2.8b", "2.8b"),
]


C_PLAIN, C_WHIT, C_NULL = "#c0392b", "#2c6fbb", "#7f8c8d"


def at_best_layer(model, dataset):
    node = S["models"][model]["datasets"][dataset]
    rec = next(r for r in node["layers"] if r["layer"] == node["best_layer"])
    return rec, node["d_model"], node["n"]


def collect(dataset):
    rows = []
    for model, label in LADDER:
        rec, d_model, n = at_best_layer(model, dataset)
        rows.append({
            "label": label,
            "d_model": d_model,
            "n": n,
            "layer": rec["layer"],
            "plain": rec["plain"]["auroc"],
            "plain_dp": rec["plain"]["d_prime"],
            "whit": rec["whitened"]["auroc"],
            "whit_dp": rec["whitened"]["d_prime"],
            "null": rec["null"]["auroc_p95"],
        })
    return rows


def run_emergence(argv=None):
    """Figure: emergence of the truth direction across the Pythia scale ladder --
    best-layer held-out AUROC of the mass-mean direction against the
    random-direction null, 70m -> 2.8b.

    Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

    Selection: the artifact's own `best_layer` (max held-out AUROC above that layer's
    own null). Both arms are drawn, because the claim is not "AUROC rises" but
    "nothing in this linear class clears the null until 410m" -- a statement about
    the whitened arm too.

    Supersedes an ad-hoc Jul-8 figure that had no generator and whose numbers no
    longer matched the Jul-9 sweep.
    """
    argparse.ArgumentParser(description=run_emergence.__doc__).parse_args(argv)
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
    panels = [(ds, collect(ds)) for ds in DATASETS]
    x = np.arange(len(LADDER))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (ds, rows) in zip(axes, panels):
        ax.fill_between(x, 0.5, [r["null"] for r in rows],
                        color=C_NULL, alpha=0.18, zorder=1,
                        label="random-direction null (to $p_{95}$)")
        ax.plot(x, [r["null"] for r in rows], color=C_NULL, lw=1.0, ls="--", zorder=2)

        ax.plot(x, [r["plain"] for r in rows], marker="o", ms=6, lw=2.2,
                color=C_PLAIN, zorder=4, label=r"mass-mean $\hat\theta$")
        ax.plot(x, [r["whit"] for r in rows], marker="s", ms=5, lw=1.4,
                color=C_WHIT, zorder=3, label=r"whitened $\hat\theta_{\mathrm{F}}$")

        ax.axhline(0.5, color="#bdc3c7", lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['label']}\n$d$={r['d_model']}" for r in rows])
        ax.set_xlabel("Pythia model size")
        ax.set_ylim(0.35, 1.02)
        ax.set_title(f"`{ds}`", fontsize=11, pad=10)

    axes[0].set_ylabel("held-out AUROC at best layer")
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower right")
    axes[0].annotate("inside the null:\nnothing recoverable",
                     xy=(0, panels[0][1][0]["plain"]), xytext=(0.18, 0.66),
                     fontsize=9.5, color=C_PLAIN,
                     arrowprops=dict(arrowstyle="->", color=C_PLAIN, lw=1.0))

    fig.tight_layout()
    EMERGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EMERGE_OUT, bbox_inches="tight")
    print(f"wrote {EMERGE_OUT}\n")

    hdr = (f"{'model':>6} {'L':>3} {'plain':>7} {'d_prime':>8} {'whit':>7} "
           f"{'null95':>7} {'margin':>8} {'clears?':>8}")
    for ds, rows in panels:
        print(f"=== {ds}")
        print(hdr)
        for r in rows:
            m = r["plain"] - r["null"]
            print(f"{r['label']:>6} {r['layer']:>3} {r['plain']:>7.3f} {r['plain_dp']:>8.3f} "
                  f"{r['whit']:>7.3f} {r['null']:>7.3f} {m:>+8.3f} {'yes' if m > 0 else 'NO':>8}")
        print("  whitened arm vs null:")
        for r in rows:
            m = r["whit"] - r["null"]
            print(f"  {r['label']:>6} whit {r['whit']:.3f} vs null {r['null']:.3f} -> {m:+.3f} "
                  f"{'clears' if m > 0 else 'INSIDE NULL'}")
        print()


# =============================================================================
# plausibility-depth
# =============================================================================
PLAUS_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_plausibility_depth.png"


C_LIKELY, C_TRUTH = "#8e44ad", "#c0392b"


BIG = "EleutherAI/pythia-2.8b"


SMALL = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b"]


STYLE = {"EleutherAI/pythia-410m": ":", "EleutherAI/pythia-1.4b": "-"}


def profile(model, dataset):
    node = S["models"][model]["datasets"][dataset]
    recs = sorted(node["layers"], key=lambda r: r["layer"])
    L = np.array([r["layer"] for r in recs], dtype=float)
    m = np.array([r["plain"]["auroc"] - r["null"]["auroc_p95"] for r in recs])
    return L / L.max(), m, L, node["best_layer"]


def run_plausibility_depth(argv=None):
    """Figure: depth profiles of the plausibility axis and the truth axis.

    Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

    Ordinate is the MARGIN m = held-out AUROC - the layer's own random-direction
    null p95, the same quantity the layer-selection rule maximises, because the null
    widens with depth and raw AUROC is not comparable across layers. Abscissa is
    fractional depth L / L_max so models of different depth are commensurable.

    Left: pythia-2.8b, where `likely` peaks early (L12 of 32) and `cities` late
    (L29), i.e. at opposite ends of the depth axis. Right: the same two curves at
    410m and 1.4b, where the ordering does NOT hold -- `cities` peaks at L11 and L7
    of 24, BEFORE `likely` at L15 and L13, and its late-depth structure appears only
    as a secondary rise over the final layers.

    Layer 0 is the embedding, excluded from selection, drawn hollow. On templated
    statements its last-token activation is nearly constant within a dataset, so
    `cities` is degenerate there; `likely` is not, and that is the point of the
    early-depth end of the left panel.
    """
    argparse.ArgumentParser(description=run_plausibility_depth.__doc__).parse_args(argv)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)

    # ---- left: pythia-2.8b -------------------------------------------------
    ax = axes[0]
    for ds, label, color in [("likely", "likely (plausibility)", C_LIKELY),
                             ("cities", "cities (truth)", C_TRUTH)]:
        x, m, L, best = profile(BIG, ds)
        sel = L != 0
        ax.plot(x[sel], m[sel], lw=2.0, color=color, marker="o", ms=3.4,
                label=label, zorder=3)
        ax.plot(x[~sel], m[~sel], lw=0, marker="o", ms=5, mfc="white",
                mec=color, zorder=3)
        j = int(np.where(L == best)[0][0])
        ax.plot([x[j]], [m[j]], marker="*", ms=15, lw=0, color="#d68910", zorder=5)
        ax.annotate(f"L{best}", xy=(x[j], m[j]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=color)

    ax.axhline(0.0, color="#bdc3c7", lw=0.8, zorder=1)
    ax.set_ylabel(r"margin $m = \mathrm{AUROC} - p_{95}$")
    ax.set_xlabel("fractional depth $L/L_{\\max}$")
    ax.set_title("pythia-2.8b: peaks at opposite ends", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.annotate("L0 = embedding\n(excluded)", xy=(0.0, 0.113), xytext=(0.055, 0.035),
                fontsize=8.5, color="#7f8c8d",
                arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=0.8))

    # ---- right: 410m and 1.4b ---------------------------------------------
    ax = axes[1]
    for model in SMALL:
        tag = model.split("-")[-1]
        for ds, color in [("likely", C_LIKELY), ("cities", C_TRUTH)]:
            x, m, L, best = profile(model, ds)
            sel = L != 0
            ax.plot(x[sel], m[sel], lw=1.8, ls=STYLE[model], color=color,
                    zorder=3)
            ax.plot(x[~sel], m[~sel], lw=0, marker="o", ms=5, mfc="white",
                    mec=color, zorder=3)
            j = int(np.where(L == best)[0][0])
            ax.plot([x[j]], [m[j]], marker="*", ms=13, lw=0, color="#d68910",
                    zorder=5)
            ax.annotate(f"L{best}", xy=(x[j], m[j]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=color)

    ax.axhline(0.0, color="#bdc3c7", lw=0.8, zorder=1)
    ax.set_xlabel("fractional depth $L/L_{\\max}$")
    ax.set_title("410m and 1.4b: the ordering reverses", fontsize=11, pad=10)
    handles = [Line2D([], [], color="#7f8c8d", lw=1.8, ls=STYLE[m_], label=t)
               for m_, t in zip(SMALL, ["410m", "1.4b"])]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right",
              title="model", title_fontsize=9.5)

    fig.tight_layout()
    PLAUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLAUS_OUT, bbox_inches="tight")
    print(f"wrote {PLAUS_OUT}\n")

    for model in [BIG] + SMALL:
        print(model)
        for ds in ["likely", "cities"]:
            x, m, L, best = profile(model, ds)
            j = int(np.where(L == best)[0][0])
            k = int(np.argmax(np.where(L == 0, -np.inf, m)))
            print(f"  {ds:<7} best_layer L{best:>2} (m={m[j]:+.3f}, "
                  f"depth {x[j]:.2f})   argmax m = L{int(L[k])} ({m[k]:+.3f})")
            print("     last three layers: " + ", ".join(
                f"L{int(l)} {v:+.3f}" for l, v in zip(L[-3:], m[-3:])))


# =============================================================================
# transfer
# =============================================================================
TRANSFER_OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_transfer.png"


MODEL = os.environ.get("TRANSFER_MODEL", "EleutherAI/pythia-2.8b")


FULL = os.environ.get("TRANSFER_FULL") == "1"


POLARITY = ["cities", "neg_cities", "larger_than", "smaller_than"]


SHORT = {
    "cities": "cities",
    "neg_cities": "neg_cities",
    "larger_than": "larger_than",
    "smaller_than": "smaller_than",
    "cities_cities_conj": "cc_conj",
    "cities_cities_disj": "cc_disj",
    "common_claim_true_false": "common_claim",
    "companies_true_false": "companies",
    "counterfact_true_false": "counterfact",
}


T = S["models"][MODEL]["transfer"]


def run_transfer(argv=None):
    """Figure: cross-dataset transfer of the mass-mean direction, pythia-2.8b.
    Held-out AUROC for a direction fit on each dataset (rows), evaluated on each
    (columns).

    Reads artifacts/snr_sweep.json only -- NO model, NO GPU.

    The off-diagonal is ORGANISED, not merely weak, which is why the colour map is
    diverging and centred on chance rather than sequential: the below-chance cells
    ARE the finding. Between a statement type and its logical negation the same
    vector reads truth BACKWARDS (cells near 0.07) -- a different thing from failing
    to transfer (cells near 0.5).

    TRANSFER_FULL=1 draws all nine main-tier datasets instead of the four-dataset
    polarity block. Worth running before claiming anything about "unrelated
    families": in the 9x9, cities -> common_claim is ~0.71, which is not chance.
    """
    argparse.ArgumentParser(description=run_transfer.__doc__).parse_args(argv)
    rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    })
    keys = [k for k in SHORT if k in T] if FULL else POLARITY
    M = np.array([[T[r][c] for c in keys] for r in keys])

    fig, ax = plt.subplots(figsize=(7.4, 6.0) if FULL else (5.9, 5.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(keys)))
    ax.set_yticks(np.arange(len(keys)))
    ax.set_xticklabels([SHORT[k] for k in keys], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([SHORT[k] for k in keys], fontsize=9)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("fit on")

    for i in range(len(keys)):
        for j in range(len(keys)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v - 0.5) > 0.30 else "#2c3e50",
                    fontweight="bold" if (i == j or v < 0.2) else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("held-out AUROC", fontsize=9.5)
    cb.ax.axhline(0.5, color="#2c3e50", lw=1.0)

    ax.set_title(f"Transfer is polarity- and family-structured ({MODEL.split('/')[-1]})",
                 fontsize=11, pad=10)

    fig.tight_layout()
    TRANSFER_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRANSFER_OUT, bbox_inches="tight")
    print(f"wrote {TRANSFER_OUT}\n")

    diag = [T[k][k] for k in keys]
    print(f"diagonal: {min(diag):.3f} - {max(diag):.3f}")
    print("\nnegation pairs (the anti-transfer finding):")
    for a, b in [("cities", "neg_cities"), ("larger_than", "smaller_than")]:
        if a in T and b in T:
            print(f"  {a:>12} -> {b:<14} {T[a][b]:.3f}")
            print(f"  {b:>12} -> {a:<14} {T[b][a]:.3f}")
    off = [T[r][c] for r in POLARITY for c in POLARITY
           if r != c and {r, c} not in [{"cities", "neg_cities"},
                                        {"larger_than", "smaller_than"}]]
    print(f"\n4x4 cross-family cells (excl. negation pairs): "
          f"{min(off):.3f} - {max(off):.3f}")
    print("\nfull-matrix cells the 4x4 hides (fit on cities) -- NOT chance:")
    for c, v in T["cities"].items():
        if c not in POLARITY:
            print(f"  cities -> {c:<26} {v:.3f}")


def run_all(argv=None):
    """Draw every figure in this group, each from matplotlib's default style."""
    argparse.ArgumentParser(description=run_all.__doc__).parse_args(argv)
    for run in (run_emergence, run_plausibility_depth, run_transfer,):
        matplotlib.rcdefaults()
        run([])


COMMANDS = {
    "emergence": run_emergence,
    "plausibility-depth": run_plausibility_depth,
    "transfer": run_transfer,
    "all": run_all,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
