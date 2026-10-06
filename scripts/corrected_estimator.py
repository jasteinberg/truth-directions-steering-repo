"""
A corrected linear estimator: Ledoit-Wolf shrinkage of the within-class covariance.

The post's section "A corrected linear estimator" (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
The Ledoit-Wolf intensity at every cached layer and the spectrum it implies; why
it is large (beta^2 against the Gaussian reference, v1 kurtosis); the held-out
d' of the Fisher direction along the shrinkage path; and rho chosen by
cross-validation on the train half.

Subcommands:

    intensity              Extract the Ledoit-Wolf shrinkage intensity rho actually used by ...
    decomposition          Decompose the Ledoit-Wolf shrinkage intensity and test whether ...
    rho-sweep              Does shrinkage explain why whitening falls short below layer 24?
    rho-cv                 Choose the shrinkage intensity by cross-validation INSIDE the ...

Run from the repo root:  python scripts/corrected_estimator.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib import steering
from utils.truthlib.estimators import (
    d_prime,
    fisher,
    massive_mask,
    split_indices,
    within_class_center,
)
from utils.truthlib.shrinkage import cv_layer, lw_decomposition, shrinkage_row

# ---- shared by several subcommands ------------------------------------------
CACHE = os.path.join(REPO, "artifacts", "act_cache")
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 0.9]


# =============================================================================
# intensity
# =============================================================================
INTENSITY_OUT = os.path.join(REPO, "artifacts", "shrinkage_intensity.json")


def run_intensity(argv=None):
    """Extract the Ledoit-Wolf shrinkage intensity rho actually used by the whitened
    (Fisher) direction, per layer, from the cached activations.

    `recoverability.py sweep` fits LedoitWolf(assume_centered=True) on the TRAIN half of the
    within-class-centered activations and never records lw.shrinkage_, so the post's
    methods appendix names the estimator without being able to quote the number.
    This reproduces that fit exactly -- same class-stratified 50/50 split at seed 0,
    same within-class centering, same float64 cast -- and reports rho along with the
    raw and shrunk spectra so the effective anisotropy can be compared against the
    raw lambda_1/lambda_2 the plane-rotation section uses.

    Reports per layer:
      rho          Ledoit-Wolf shrinkage intensity (0 = raw C, 1 = isotropic)
      l1/l2 raw    leading eigenvalue ratio of the raw within-class C (train half)
      l1/l2 shr    the same ratio after shrinkage, (1-rho)*l + rho*mu
      l1/tr raw    leading eigenvalue share of the raw C
      PR raw       participation ratio of the raw C
    """
    ap = argparse.ArgumentParser(description=run_intensity.__doc__)
    ap.add_argument("--out", default=INTENSITY_OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        z = np.load(p)
        y = z["y"]
        layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
        print(f"\n=== {key} ===", flush=True)
        print(f"{'L':>4} {'rho':>8} {'l1/l2 raw':>12} {'l1/l2 shr':>12} "
              f"{'l1/tr raw':>10} {'PR raw':>8}", flush=True)
        per_layer = {}
        for L in layers:
            r = shrinkage_row(z[f"L{L}"].astype(np.float64), y, seed=args.seed)
            per_layer[str(L)] = r
            print(f"{L:>4} {r['rho']:>8.4f} {r['lambda1_over_lambda2_raw']:>12.1f} "
                  f"{r['lambda1_over_lambda2_shrunk']:>12.1f} "
                  f"{r['lambda1_over_trace_raw']:>10.3f} {r['PR_raw']:>8.2f}",
                  flush=True)
        results[key] = per_layer

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed,
                              "split": "class-stratified 50/50, train half",
                              "estimator": "sklearn LedoitWolf(assume_centered=True)"},
                   "models": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


# =============================================================================
# decomposition
# =============================================================================
DECOMP_OUT = os.path.join(REPO, "artifacts", "shrinkage_decomposition.json")


def run_decomposition(argv=None):
    """Decompose the Ledoit-Wolf shrinkage intensity and test whether heavy tails, not
    concentration, drive it -- and compare train-half vs full-sample spectra.

    Two questions.

    (1) WHY IS rho LARGE FOR counterfact?  LW picks
            rho = min(beta2, delta2) / delta2,
            delta2 = (1/d) ||C - mu I||_F^2,
            beta2  = (1/(d N^2)) sum_i ||x_i x_i^T - C||_F^2
                    = (1/(d N^2)) [ sum_i ||x_i||^4 - N ||C||_F^2 ].
        For Gaussian x, E sum_i ||x_i||^4 = N[(tr C)^2 + 2 tr C^2], which gives a
        Gaussian reference rho_gauss. In the rank-1 limit rho_gauss -> 2/N. Reporting
        rho_obs / rho_gauss isolates the non-Gaussian part; excess kurtosis of the
        projection onto v1 says whether the rogue axis is where it lives.

    (2) WHICH SPLIT SHOULD THE SPECTRUM BE QUOTED ON?  The eigenvalues are descriptive
        statistics of Sigma, not a fitted predictor scored on itself, so the full
        sample is the better estimator -- PROVIDED any tuned quantity (rho) was chosen
        on train. But rho itself is N-dependent (beta2 ~ 1/N), so rho_train applied to
        a full-sample C over-shrinks. This reports rho at both N and the resulting
        lambda1/lambda2 under each choice, so the size of that effect is visible.
    """
    ap = argparse.ArgumentParser(description=run_decomposition.__doc__)
    ap.add_argument("--out", default=DECOMP_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma list; default all")
    args = ap.parse_args(argv)
    want = {int(s) for s in args.layers.split(",")} if args.layers else None

    results = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        key = os.path.basename(p)[:-4]
        z = np.load(p); y = z["y"]
        layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
        if want:
            layers = [L for L in layers if L in want]
        print(f"\n=== {key} ===", flush=True)
        print(f"{'L':>4} {'rho_tr':>8} {'rho_full':>9} {'rho/gauss':>10} "
              f"{'kurt_v1':>9} {'kurt_bulk':>10} {'l1/l2 tr':>10} {'l1/l2 full':>11} "
              f"{'l1/l2 f@rho_tr':>15}", flush=True)
        per_layer = {}
        for L in layers:
            r = lw_decomposition(z[f"L{L}"].astype(np.float64), y, seed=args.seed)
            per_layer[str(L)] = r
            t, f = r["train"], r["full"]
            print(f"{L:>4} {t['rho']:>8.4f} {f['rho']:>9.4f} "
                  f"{t['rho_over_gauss']:>10.1f} {t['excess_kurtosis_v1']:>9.2f} "
                  f"{t['excess_kurtosis_bulk50']:>10.2f} "
                  f"{t['lambda1_over_lambda2_raw']:>10.1f} "
                  f"{f['lambda1_over_lambda2_raw']:>11.1f} "
                  f"{r['full_shrunk_by_train_rho']['lambda1_over_lambda2']:>15.1f}",
                  flush=True)
        results[key] = per_layer

    with open(args.out, "w") as fh:
        json.dump({"config": {"seed": args.seed,
                              "split": "class-stratified 50/50",
                              "estimator": "sklearn LedoitWolf(assume_centered=True)"},
                   "models": results}, fh, indent=1)
    print(f"\nwrote {args.out}", flush=True)


# =============================================================================
# rho-sweep
# =============================================================================
RHOSWEEP_OUT = os.path.join(REPO, "artifacts", "shrinkage_sweep.json")


def run_rho_sweep(argv=None):
    """Does shrinkage explain why whitening falls short below layer 24?

    The post's new section observes that on counterfact at pythia-2.8b, removing the eleven
    massive-activation droppers raises held-out d'_F at L8-L20 (0.005-0.031 -> 0.129-0.385)
    while leaving L28 untouched (0.384 -> 0.385). The hypothesis in the text is that
    Ledoit-Wolf's rho ~ 0.137 is too large to let

        Sigma_hat(rho) = (1 - rho) * C_hat + rho * mu * I,     mu = tr C_hat / d

    invert an anisotropy of order 1e4, so theta_F = Sigma_hat(rho)^-1 delta_hat cannot rotate
    far enough off v1.

    PREDICTION IF TRUE: sweeping rho DOWN on the uncleaned data should recover some of the
    gain that deleting the eleven produces -- d'_F at L8-L20 should rise as rho falls, toward
    the cleaned values. PREDICTION IF FALSE: d'_F stays flat or degrades at every rho, and the
    shortfall is about something other than regularization strength.

    rho -> 0 is not available: C_hat is singular at N_train ~ 599 against d = 2560, so the
    grid stops at 1e-4 and the inverse is taken via solve on the shrunk matrix, which stays
    positive definite for any rho > 0.

    Everything is fit on the train half and scored on the full held-out half, matching the
    "clean_train" regime of `rogue_dimension.py outliers` so the numbers are directly comparable.
    """
    ap = argparse.ArgumentParser(description=run_rho_sweep.__doc__)
    ap.add_argument("--out", default=RHOSWEEP_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    args = ap.parse_args(argv)

    z = np.load(os.path.join(CACHE, f"{args.model}__{args.dataset}.npz"))
    y = z["y"]
    tr, te = split_indices(y, seed=args.seed)
    layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))

    results = {}
    hdr = "  ".join(f"{r:>7.4g}" for r in GRID)
    print(f"{'L':>4} {'rho_LW':>7} {'d_LW':>6} {'d_clean':>8} | {hdr}", flush=True)
    for L in layers:
        X = z[f"L{L}"].astype(np.float64)
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        rho_lw = float(LedoitWolf(assume_centered=True)
                       .fit(within_class_center(Xtr, ytr)).shrinkage_)

        def score(t):
            if (Xtr @ t)[ytr == 1].mean() < (Xtr @ t)[ytr == 0].mean():
                t = -t
            return d_prime(Xte @ t, yte)

        d_lw = score(fisher(Xtr, ytr, rho=rho_lw))

        # reference: the eleven removed from train only, at LW's own rho there
        keep = ~massive_mask(Xtr)[0]
        Xc2, yc2 = Xtr[keep], ytr[keep]
        rho_c = float(LedoitWolf(assume_centered=True)
                      .fit(within_class_center(Xc2, yc2)).shrinkage_)
        tc = fisher(Xc2, yc2, rho=rho_c)
        if (Xc2 @ tc)[yc2 == 1].mean() < (Xc2 @ tc)[yc2 == 0].mean():
            tc = -tc
        d_clean = d_prime(Xte @ tc, yte)

        row = {"rho_LW": rho_lw, "d_F_at_rho_LW": d_lw,
               "rho_LW_clean": rho_c, "d_F_clean_train": d_clean, "sweep": {}}
        vals = []
        for rho in GRID:
            v = score(fisher(Xtr, ytr, rho=rho))
            row["sweep"][str(rho)] = v
            vals.append(v)
        results[str(L)] = row
        print(f"{L:>4} {rho_lw:>7.4f} {d_lw:>6.3f} {d_clean:>8.3f} | "
              + "  ".join(f"{v:>7.3f}" for v in vals), flush=True)

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "model": args.model,
                              "dataset": args.dataset, "grid": GRID,
                              "note": "fit on train half, scored on FULL held-out half"},
                   "layers": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


# =============================================================================
# rho-cv
# =============================================================================
RHOCV_OUT = os.path.join(REPO, "artifacts", "shrinkage_cv.json")


def run_rho_cv(argv=None):
    """Choose the shrinkage intensity by cross-validation INSIDE the training half.

    `corrected_estimator.py rho-sweep` showed that Ledoit-Wolf's rho is far from optimal for the
    separation d'_F of the direction it produces -- but it found that by reading held-out d'
    off a grid, which is selection on the evaluation data and therefore not quotable. This
    does it properly:

        inner loop   K-fold CV within the TRAIN half only; for each rho, fit
                     theta_F on K-1 folds and score d' on the held-out fold;
                     rho* = argmax of the mean CV score.
        outer step   refit on the FULL train half at rho*, score ONCE on the
                     test half. The test half is touched exactly once per layer.

    Ledoit-Wolf minimizes E||Sigma_hat - Sigma||_F^2. That is not the objective anyone
    cares about here: the quantity that matters is d' of Sigma_hat^-1 delta_hat, which
    depends on the inverse and on one particular direction in it. CV optimizes the thing
    we actually report.

    Implementation note: Sigma(rho) = (1-rho) C + rho mu I shares eigenvectors with C, so
    one eigendecomposition per fold serves the whole rho grid --
        Sigma(rho)^-1 delta = V diag(1/((1-rho) lambda + rho mu)) V^T delta.
    Without this the sweep costs a 2560x2560 solve per (fold, rho).
    """
    ap = argparse.ArgumentParser(description=run_rho_cv.__doc__)
    ap.add_argument("--out", default=RHOCV_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    results = {}
    for model, n_blocks in (("pythia-2.8b", 32), ("pythia-1.4b", 24)):
        for ds in ("counterfact_true_false", "cities"):
            # refuses a missing cache or one without the planned layers
            Xs, y = steering.load_cached_acts(model, ds, n_blocks)
            tr, te = split_indices(y, seed=args.seed)
            layers = sorted(Xs)
            key = f"{model}__{ds}"
            print(f"\n=== {key} ===", flush=True)
            print(f"{'L':>4} {'rho*':>8} {'rho_LW':>8} {'d_F(cv)':>8} {'d_F(LW)':>8} "
                  f"{'null':>7} {'ratio':>6}", flush=True)
            per = {}
            for L in layers:
                r = cv_layer(Xs[L].astype(np.float64), y, tr, te,
                              args.folds, args.seed, GRID)
                per[str(L)] = r
                ratio = r["d_F_cv"] / r["d_F_LW"] if r["d_F_LW"] > 0 else float("inf")
                print(f"{L:>4} {r['rho_star']:>8.4g} {r['rho_LW']:>8.4f} "
                      f"{r['d_F_cv']:>8.3f} {r['d_F_LW']:>8.3f} "
                      f"{r['null_p95']:>7.3f} {ratio:>6.1f}", flush=True)
            results[key] = per

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "folds": args.folds, "grid": GRID,
                              "protocol": "rho chosen by K-fold CV inside the train "
                                          "half; test half scored once per layer"},
                   "results": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


COMMANDS = {
    "intensity": run_intensity,
    "decomposition": run_decomposition,
    "rho-sweep": run_rho_sweep,
    "rho-cv": run_rho_cv,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
