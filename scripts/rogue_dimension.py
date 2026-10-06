"""
The rogue dimension: within-class geometry, its origin, and the score gradient.

The post's section "The rogue dimension" with its subsections "The origin of the
rogue dimension" and "Susceptibility and the score gradient", and the appendices on
the full twelve-dataset benchmark, the eleven outlier statements and the OLMo
replication (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
Spectra, PR and cos(theta, v1) at the selected layer and at every layer; massive
coordinates and droppers at every layer; the same observables on the extra datasets
and on OLMo-2-1B; the outlier check; steering and decoding along v1 and theta_perp;
the per-pair score gradient and its fixed-seed bootstrap intervals.

Subcommands:

    observables            Order parameters for probe geometry that the interpretability ...
    all-layers             The geometry observables of `rogue_dimension.py observables` at EVERY ...
    massive                Does the massive activation survive to the final layers? ...
    extra-datasets         Rogue-dimension observables on the main-tier datasets the post ...
    olmo                   Does the rogue-dimension signature replicate outside the Pythia ...
    outliers               Is the rogue dimension eleven outlier statements? Four checks
    steer-arms             Is the large steering effect on `counterfact` a truth direction, ...
    gradient               Measures g = <sum_t grad_{x_t} ell>, the mean gradient of the ...
    gradient-ci            Regenerate the bootstrap intervals on the score-gradient ...

Run from the repo root:  python scripts/rogue_dimension.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib import acts, data, steering
from utils.truthlib import estimators as est
from utils.truthlib.acts import extract_all_layers, get_model
from utils.truthlib.data import cap_for, load_dataset
from utils.truthlib.estimators import (
    cos_bootstrap_interval,
    massive_mask,
    observables,
)
from utils.truthlib.geometry import (
    gram_observables,
    held_out_cleaning,
    massive_row,
    outlier_geometry,
    rogue_arms,
    rogue_spectrum,
    v1_outlier_mask,
)

# ---- shared by several subcommands ------------------------------------------
CACHE = os.path.join(REPO, "artifacts", "act_cache")
CAP = 1199


# =============================================================================
# observables
# =============================================================================
OBS_OUT = os.path.join(REPO, "artifacts", "geometry_observables.json")


def run_observables(argv=None):
    """Order parameters for probe geometry that the interpretability literature does not
    report. All are model-free: computed from cached last-token activations.

      PR      participation ratio (sum lam)^2 / sum lam^2 -- the effective number of
              directions carrying within-class variance. PR = 1 means rank-one.
      cos_v1  |cos(theta_hat, v_1)|, alignment of the probe with the dominant
              ("rogue" / massive-activation) direction.
      d_mm    mass-mean separation |delta.theta| / sqrt(pooled var along theta)
      d_maha  full Mahalanobis separation sqrt(delta^T Sigma^{-1} delta), i.e. the
              separation available to an optimal linear readout. d_maha / d_mm
              measures how much signal the naive estimator leaves on the table.
      var_th  Var(x . theta) -- the spontaneous fluctuation along the probe. Paired
              with the steering susceptibility chi it tests a fluctuation-response
              relation: in an equilibrium system response ~ fluctuation, and a
              transformer has no reason to obey that.
      overlap |cos(theta_L, theta_L')| across layers -- is the direction propagated
              through depth, or recomputed?
    """
    ap = argparse.ArgumentParser(description=run_observables.__doc__)
    ap.add_argument("--out", default=OBS_OUT)
    ap.add_argument("--no-shrink", dest="shrink", action="store_false")
    args = ap.parse_args(argv)

    results = {}
    # the four cached (model, dataset) pairs in the published order; each cache must hold its
    # model's planned layers (steering.load_cached_acts refuses a missing or truncated one)
    cells = [(m, ds, nb) for m, nb in (("pythia-1.4b", 24), ("pythia-2.8b", 32))
             for ds in ("cities", "counterfact_true_false")]
    for model, ds, n_blocks in cells:
        key = f"{model}__{ds}"
        Xs, y = steering.load_cached_acts(model, ds, n_blocks)
        layers = sorted(Xs)
        print(f"\n=== {model} / {ds} ===", flush=True)
        print(f"{'L':>4} {'PR':>8} {'cos_v1':>8} {'d_mm':>7} {'d_maha':>8} "
              f"{'hidden':>7} {'var_th':>11}", flush=True)
        per_layer, thetas = {}, {}
        for L in layers:
            o = observables(Xs[L].astype(np.float64), y, args.shrink)
            thetas[L] = np.array(o.pop("theta"))
            per_layer[str(L)] = o
            print(f"{L:>4} {o['PR']:>8.2f} {o['cos_theta_v1']:>8.3f} "
                  f"{o['d_mass_mean']:>7.3f} {o['d_mahalanobis']:>8.2f} "
                  f"{o['hidden_signal_ratio']:>7.1f} {o['var_along_theta']:>11.2f}",
                  flush=True)

        # layer-to-layer overlap of the probe direction
        ov = {}
        for L in layers:
            ov[str(L)] = {str(L2): float(abs(thetas[L] @ thetas[L2])) for L2 in layers}
        print("\n  layer overlap |cos(theta_L, theta_L')|", flush=True)
        print("      " + "".join(f"{L:>7}" for L in layers), flush=True)
        for L in layers:
            print(f"  L{L:<3}" + "".join(f"{ov[str(L)][str(L2)]:>7.2f}" for L2 in layers),
                  flush=True)

        results[key] = {"model": model, "dataset": ds,
                        "layers": per_layer, "layer_overlap": ov}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")


# =============================================================================
# all-layers
# =============================================================================
ALL_DEV = "mps"


ALL_OUT = os.path.join(REPO, "artifacts", "geometry_all_layers.json")


def run_all_layers(argv=None):
    """The geometry observables of `rogue_dimension.py observables` at EVERY layer, not the
    six-layer probed grid. Motivated by the tl;dr claim that on `counterfact` the
    class gap never dominates the spread "at any layer": the cached grid stops at
    28, and the one layer where the plain probe clears its null is 32.

    Conventions match truthlib.estimators.observables exactly:
      - full sample (not the held-out half), since these describe geometry
      - within-class centering, np.cov(ddof=1)
      - d_mm from the pooled projected variance, ddof=1

    Speed: with N < d the covariance is rank-deficient, so the nonzero spectrum of
    Xc^T Xc / (N-1) equals that of the N x N Gram matrix Xc Xc^T / (N-1). Working
    from the Gram matrix is exact and turns a 2560^3 eigh into a 1198^3 one.
    v1 is pulled back as Xc^T u1, renormalised.
    """
    ap = argparse.ArgumentParser(description=run_all_layers.__doc__)
    ap.add_argument("models", nargs="?", default="EleutherAI/pythia-2.8b", help="comma-separated")
    ap.add_argument("datasets", nargs="?", default="counterfact_true_false,cities", help="comma-separated")
    ap.add_argument("--out", default=ALL_OUT, help="results file (default: the published one; merged into if it exists)")
    args = ap.parse_args(argv)
    models, dsets = args.models.split(","), args.datasets.split(",")

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}

    for mname in models:
        tok, model = acts.get_model(mname, ALL_DEV, torch.float16)
        tag = mname.split("/")[-1]
        for ds in dsets:
            key = f"{tag}__{ds}"
            print(f"\n=== {key} ===", flush=True)
            stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
            print(f"  {len(stmts)} statements, one forward pass", flush=True)
            A = acts.extract_all_layers(stmts, tok, model, ALL_DEV, 16)
            print(f"  activations {A.shape}", flush=True)

            rows = {}
            for L in range(A.shape[0]):
                o = gram_observables(A[L], y)
                rows[str(L)] = o
                if o.get("degenerate"):
                    print(f"  L{L:>2}  degenerate (within-class covariance vanishes)",
                          flush=True)
                else:
                    print(f"  L{L:>2}  PR {o['PR']:8.2f}  l1/tr {o['lambda1_over_trace']:.3f}"
                          f"  l1/l2 {o['lambda1_over_lambda2']:10.1f}"
                          f"  cos {o['cos_theta_v1']:.3f}  d_mm {o['d_mass_mean']:.3f}",
                          flush=True)
            results[key] = {"model": mname, "dataset": ds, "n": int(len(y)),
                            "layers": rows}
            del A; gc.collect()
            json.dump(results, open(args.out, "w"), indent=1)
            print(f"  wrote {args.out}", flush=True)
        del model; gc.collect()


# =============================================================================
# massive
# =============================================================================
MASSIVE_DEV = "mps"


MASSIVE_OUT = os.path.join(REPO, "artifacts", "massive_all_layers.json")


def run_massive(argv=None):
    """Does the massive activation survive to the final layers? `rogue_dimension.py all-layers`
    found that `counterfact` escapes the rogue dimension at layers 31-32 on
    pythia-2.8b (cos 0.982 -> 0.653 -> 0.055). If the mechanism of *The origin of
    the rogue dimension* is right, the escape should coincide with the massive
    coordinates disappearing, or with every statement carrying them so that none
    drop.

    Criterion is truthlib.estimators.massive_mask verbatim, the same rule the
    eleven-outlier appendix uses. Never touches v1, the class means or the labels.
    """
    ap = argparse.ArgumentParser(description=run_massive.__doc__)
    ap.add_argument("models", nargs="?", default="EleutherAI/pythia-2.8b", help="comma-separated")
    ap.add_argument("datasets", nargs="?", default="counterfact_true_false,cities", help="comma-separated")
    ap.add_argument("--out", default=MASSIVE_OUT, help="results file (default: the published one; merged into if it exists)")
    args = ap.parse_args(argv)
    models, dsets = args.models.split(","), args.datasets.split(",")
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}

    for mname in models:
        tok, model = acts.get_model(mname, MASSIVE_DEV, torch.float16)
        tag = mname.split("/")[-1]
        for ds in dsets:
            key = f"{tag}__{ds}"
            print(f"\n=== {key} ===", flush=True)
            stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
            A = acts.extract_all_layers(stmts, tok, model, MASSIVE_DEV, 16)
            rows = {}
            for L in range(A.shape[0]):
                r = massive_row(A[L])
                rows[str(L)] = r
                print(f"  L{L:>2}  massive coords {r['n_massive_coords']:>2}"
                      f"  droppers {r['n_droppers']:>4}"
                      f"  max |med_j|/median {r['max_coord_ratio']:9.1f}", flush=True)
            results[key] = {"model": mname, "dataset": ds, "n": int(len(y)),
                            "layers": rows}
            del A; gc.collect()
            json.dump(results, open(args.out, "w"), indent=1)
            print(f"  wrote {args.out}", flush=True)
        del model; gc.collect()


# =============================================================================
# extra-datasets
# =============================================================================
EXTRA_MODEL = "EleutherAI/pythia-2.8b"


EXTRA_DATASETS = ["companies_true_false", "common_claim_true_false", "cities_cities_conj"]


LAYERS = [24, 28, 31]


EXTRA_DEV = "mps"                         # default for --device


EXTRA_OUT = REPO / "artifacts" / "geometry_extra_datasets.json"


def run_extra_datasets(argv=None):
    """Rogue-dimension observables on the main-tier datasets the post does not carry
    through the steering analysis.

    The twelve-dataset appendix infers that companies_true_false is "a second
    instance of the rogue-dimension pattern" from its plain/whitened gap alone.
    This measures the spectrum directly, at the three depths the post reasons
    about, on the datasets whose selected layer sits deep and whose plain d' is
    small. Same observables() as `rogue_dimension.py observables`, so the numbers are
    comparable to the cities / counterfact entries there.
    """
    ap = argparse.ArgumentParser(description=run_extra_datasets.__doc__)
    ap.add_argument("--device", default=EXTRA_DEV)
    args = ap.parse_args(argv)
    tok, model = acts.get_model(EXTRA_MODEL, args.device, torch.float16)
    out = {}
    for ds in EXTRA_DATASETS:
        st, y = data.load_dataset(ds, cap=1199, seed=0)
        A = acts.extract_all_layers(st, tok, model, args.device, 16)
        out[ds] = {}
        for L in LAYERS:
            o = est.observables(A[L].astype(np.float64), y, True); o.pop("theta")
            out[ds][str(L)] = o
            print(f"{ds:26s} L{L} PR={o['PR']:6.2f} lam1/tr={o['lambda1_over_trace']:.3f} "
                  f"lam1/lam2={o['lambda1_over_lambda2']:8.1f} cos(th,v1)={o['cos_theta_v1']:.3f} "
                  f"d_mm={o['d_mass_mean']:.2f}", flush=True)
        del A
    EXTRA_OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {EXTRA_OUT}")


# =============================================================================
# olmo
# =============================================================================
OLMO_ART = REPO / "artifacts"


OLMO_MODEL = "allenai/OLMo-2-0425-1B"     # default for --model


OLMO_DATASETS = ["counterfact_true_false", "cities"]


OLMO_OUT = OLMO_ART / "geometry_olmo.json"   # default for --out


DEVICE = "mps"


def run_olmo(argv=None):
    """Does the rogue-dimension signature replicate outside the Pythia family?

    Runs the geometry observables (PR, lambda_1/trace, lambda_1/lambda_2,
    |cos(theta_hat, v_1)|) on OLMo-2-1B for counterfact_true_false and cities, using
    the same loaders and the same estimator as the Pythia results, so the numbers are
    directly comparable. Activations only -- no steering, no GPU.
    """
    ap = argparse.ArgumentParser(description=run_olmo.__doc__)
    ap.add_argument("--model", default=OLMO_MODEL)
    ap.add_argument("--out", default=str(OLMO_OUT))
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    tok, model = get_model(args.model, DEVICE, torch.float32)
    res = {"model": args.model, "datasets": {}}
    for ds in OLMO_DATASETS:
        stmts, y = load_dataset(ds, cap=cap_for(ds))   # same cap as the pythia runs
        y = np.asarray(y)
        A = extract_all_layers(stmts, tok, model, DEVICE, batch_size=16)
        nL = A.shape[0]
        print(f"\n=== {args.model} / {ds}  ({len(y)} rows, {nL} layers) ===")
        print(f"{'L':>4} {'PR':>8} {'l1/tr':>8} {'l1/l2':>9} {'cos_v1':>8} {'d_mm':>7} {'d_maha':>8}")
        rows = []
        for L in range(nL):
            X = np.asarray(A[L], dtype=np.float64)
            o = observables(X, y)
            o.pop("theta", None)      # 2048 floats per layer; not needed
            o["layer"] = L
            rows.append(o)
            print(f"{L:>4} {o['PR']:>8.2f} {o['lambda1_over_trace']:>8.3f} "
                  f"{o['lambda1_over_lambda2']:>9.1f} {o['cos_theta_v1']:>8.3f} "
                  f"{o['d_mass_mean']:>7.2f} {o['d_mahalanobis']:>8.2f}")
        res["datasets"][ds] = rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


# =============================================================================
# outliers
# =============================================================================
OUTLIER_OUT = os.path.join(REPO, "artifacts", "outlier_check.json")


def statements_for(dataset, cap, seed=0):
    """Regenerate the loader's statement list so flagged rows can be read."""
    return data.load_dataset(dataset, cap=cap, seed=seed)


def run_outliers(argv=None):
    """Is the rogue dimension eleven outlier statements? Four checks.

    The finding that prompted this was selected circularly: points were thresholded on
    their projection onto v1, then v1 was recomputed without them. Removing the top of
    a distribution shortens it, so that alone shows nothing. This re-derives the outlier
    set from a criterion that never looks at v1, then measures held-out.

      (1) SELECTION INDEPENDENT OF v1. Sun et al.'s massive-activation rule, applied
          coordinate-wise to the raw activations: |x_ij| > 100 and > 1000x the median
          |x| of that layer. Reports the overlap with the v1-selected set. If they
          coincide, the v1 result is real; if not, it was an artifact of selection.

      (2) WHICH STATEMENTS. Regenerates the statement list with the same loader, seed
          and cap, verifies the labels match the cache, and prints the flagged ones.

      (3) HELD-OUT. Refits theta_mm and theta_F on the train half and scores d' on the
          test half, for: all points; both halves cleaned; and train cleaned with the
          full test (the practical question -- does dropping them help a probe that is
          still deployed on contaminated data?). d'_M has no held-out analogue (it is a
          max over directions), so it is reported in-sample and labelled as such.

      (4) GENERALIZATION. Same for pythia-1.4b, and for cities as the control that
          should show nothing.
    """
    ap = argparse.ArgumentParser(description=run_outliers.__doc__)
    ap.add_argument("--out", default=OUTLIER_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=1199)
    args = ap.parse_args(argv)

    results = {}
    for model in ("pythia-2.8b", "pythia-1.4b"):
        for ds in ("counterfact_true_false", "cities"):
            path = os.path.join(CACHE, f"{model}__{ds}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path); y = z["y"]
            layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
            key = f"{model}__{ds}"
            print(f"\n{'='*74}\n=== {key} ===", flush=True)

            stmts, y_re = statements_for(ds, args.cap, args.seed)   # a loader failure raises
            match = len(y_re) == len(y) and bool((y_re == y).all())
            print(f"statement list regenerated, labels match cache: {match}", flush=True)

            per_layer = {}
            for L in layers:
                X = z[f"L{L}"].astype(np.float64)
                mass, med, hit = massive_mask(X)
                circ = v1_outlier_mask(X, y)
                inter = int((mass & circ).sum())
                per_layer[str(L)] = {
                    "median_abs_activation": med,
                    "n_massive": int(mass.sum()), "n_v1_selected": int(circ.sum()),
                    "n_overlap": inter,
                    "geometry_all": outlier_geometry(X, y),
                    "geometry_clean": outlier_geometry(X[~mass], y[~mass])
                                      if 0 < mass.sum() < len(y) - 20 else None,
                    "held_out": held_out_cleaning(X, y, mass, seed=args.seed),
                }
                g = per_layer[str(L)]
                gc = g["geometry_clean"] or {}
                ho = g["held_out"]
                print(f"  L{L:<3} massive={g['n_massive']:<4} v1sel={g['n_v1_selected']:<4} "
                      f"overlap={inter:<4} | l1/l2 {g['geometry_all']['lambda1_over_lambda2']:>9.1f}"
                      f" -> {gc.get('lambda1_over_lambda2', float('nan')):>7.1f}"
                      f" | cos {g['geometry_all']['cos_theta_v1']:.3f}"
                      f" -> {gc.get('cos_theta_v1', float('nan')):.3f}", flush=True)
                for tag in ("all", "clean_both", "clean_train"):
                    if tag in ho:
                        r = ho[tag]
                        print(f"        {tag:<12} n={r['n_test']:<5} d_mm={r['d_mm']:.3f} "
                              f"d_F={r['d_F']:.3f}  null_p95={r['null_p95']:.3f}", flush=True)

                if stmts is not None and mass.sum() and L == layers[-1]:
                    idx = np.where(mass)[0]
                    print(f"    flagged statements ({len(idx)}):", flush=True)
                    for i in idx[:15]:
                        print(f"      y={y[i]}  {stmts[i][:96]}", flush=True)
                    per_layer[str(L)]["flagged"] = [
                        {"i": int(i), "y": int(y[i]), "statement": stmts[i]} for i in idx]
            results[key] = per_layer

    with open(args.out, "w") as f:
        json.dump({"config": {"seed": args.seed, "cap": args.cap,
                              "selection": "Sun et al. |x|>100 and >100x median |x| (relaxed from their 1000x; see truthlib.estimators.massive_mask)"},
                   "results": results}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


# =============================================================================
# steer-arms
# =============================================================================
ROGUE_ART, ROGUE_DEV = os.path.join(REPO, "artifacts"), "mps"


def run_steer_arms(argv=None):
    """Is the large steering effect on `counterfact` a truth direction, or the model's
    massive-activation ("rogue") dimension?

    Diagnosis. At pythia-2.8b layer 20 the within-class covariance of counterfact
    activations has lambda_1 / tr(Sigma) = 0.997 -- ONE eigenvalue holds 99.7% of the
    variance -- and |cos(theta_hat, v_1)| = 1.000. The mean shift is weak
    (||delta|| = 9.5 vs sqrt(tr Sigma) = 107), so the difference-in-means estimator has
    no truth signal to lock onto and collapses onto the highest-variance direction.
    Steering then injects alpha * sqrt(lambda_1) along the model's most influential
    axis. The output moves; truth has nothing to do with it.

    Arms, each a unit vector u steered as alpha * std(X @ u) * u ("alpha sigmas along u"):
      theta       mass-mean direction (what the study used)
      v1          top eigenvector of the within-class covariance -- NO LABELS USED
      theta_perp  mass-mean refit after projecting v_1 out of the activations
      random      matched, meaningless

    Predictions registered BEFORE running:
      counterfact: A(v1) ~= A(theta) ~= +1.0..1.1  (they are the same vector)
                   A(theta_perp) ~= 0              => effect was entirely the rogue dim
      cities:      A(v1) < A(theta), A(theta_perp) retains most of A(theta)
                   (|cos(theta,v1)| = 0.331 there; only 6.9% of variance along theta)

    Falsifier: A(v1) ~= 0 on counterfact while A(theta) ~= +1.09, despite cos = 1.000.
    That would be internally contradictory, and would rescue the original claim.
    """
    ap = argparse.ArgumentParser(description=run_steer_arms.__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--datasets", default="counterfact_true_false,cities")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROGUE_ART, "rogue_dimension.json"))
    ap.add_argument("--layers", default="", help="comma list, a subset of the six swept layers (default: all)")
    ap.add_argument("--arms", default="theta,v1,theta_perp,random", help="comma list of arms to steer (default: all four)")
    args = ap.parse_args(argv)
    names = [a for a in args.arms.split(",") if a]
    if set(names) - {"theta", "v1", "theta_perp", "random"}:
        raise SystemExit(f"unknown arm(s) {sorted(set(names) - {'theta', 'v1', 'theta_perp', 'random'})}")
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    want = {int(x) for x in args.layers.split(",") if x.strip()}

    tok, model = acts.get_model(args.model, ROGUE_DEV, torch.float16)
    dtype = next(model.parameters()).dtype
    nL = model.config.num_hidden_layers
    layers = steering.planned_layers(nL)
    if want - set(layers):
        raise SystemExit(f"--layers {sorted(want - set(layers))} not among {layers}")
    results = {}

    for ds in [d for d in args.datasets.split(",") if d]:
        Xs, y = steering.get_acts(model, tok, args.model, ds, layers, args.cap)
        pool = data.load_pairs(ds, args.pairs, 0)
        results[ds] = {}
        print(f"\n{'='*78}\n{args.model} / {ds}\n{'='*78}", flush=True)

        for L in [L for L in layers if not want or L in want]:
            X = Xs[L].astype(np.float64)
            sp, _, _ = rogue_spectrum(X, y)
            print(f"\n--- layer {L} ---", flush=True)
            print(f"  lambda1/tr={sp['lambda1_over_trace']:.4f}  "
                  f"lambda1={sp['lambda1']:.1f} lambda2={sp['lambda2']:.1f}  "
                  f"|cos(theta,v1)|={sp['cos_theta_v1']:.3f}  "
                  f"||delta||/sqrt(tr)={sp['delta_over_sqrt_trace']:.4f}", flush=True)

            per_arm = {k: {str(a): [] for a in alphas} for k in names}
            aurocs = {k: [] for k in per_arm}

            for sd in seeds:
                arms, aur, te = rogue_arms(X, y, seed=sd)
                pairs = steering.pairs_for_seed(pool, sd, args.pairs_per_seed)
                for k in aurocs: aurocs[k].append(aur[k])
                with steering.Steerer(model, L) as st:
                    st.set(None, 0, ROGUE_DEV, dtype)
                    base = steering.score_pairs(model, tok, pairs, args.bs)
                    # norm-matched: every arm displaced by alpha*||delta||.
                    # Scaling each arm by its own std(X@u) gave v1 a push of
                    # sqrt(lambda_1) ~ 104 and theta_perp a tiny one -- the arms
                    # were not comparable.
                    sc = est.class_gap(X, y)
                    for name, (u, Xa) in arms.items():
                        if name not in per_arm:
                            continue
                        for a in alphas:
                            A, _ = steering.antisym(model, tok, st, u, a, sc,
                                              pairs, base, args.bs, dtype)
                            per_arm[name][str(a)].append(A)
                    st.set(None, 0, ROGUE_DEV, dtype)

            a_top = str(alphas[-1])
            print(f"  {'arm':<12} {'AUROC':>7} {'A(a=%s)'%a_top:>18}", flush=True)
            for name in names:
                v = np.array(per_arm[name][a_top])
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                print(f"  {name:<12} {np.mean(aurocs[name]):>7.3f} "
                      f"{v.mean():>+11.3f}+-{se:.3f}", flush=True)

            results[ds][str(L)] = {"spectrum": sp, "arms": per_arm,
                                   "auroc": {k: float(np.mean(v)) for k, v in aurocs.items()}}
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        del Xs; gc.collect()

    del model, tok; gc.collect(); torch.mps.empty_cache()
    print(f"\nwrote {args.out}\ndone.")


# =============================================================================
# gradient
# =============================================================================
GRAD_ART = os.path.join(REPO, "artifacts")


GRAD_DEV = "mps"


def run_gradient(argv=None):
    """Measures g = <sum_t grad_{x_t} ell>, the mean gradient of the behavioural score
    with respect to the layer-L residual stream, summed over token positions.

    Why the position sum: the steering hook (truthlib.steering.Steerer) adds the
    same vector at EVERY position, so the linear response of the score to a push
    along w is  d(ell)/dh = c * (w . sum_t grad_{x_t} ell).  Differentiating only
    the final token would not be the quantity the steering arms measure.

    ell is taken from truthlib.steering.score_pairs by construction: same
    contrastive pairs, same tokenisation, same summation over completion tokens,
    same pool/seed selection. The score is a DIFFERENCE of two forward passes over
    different sequences, so the gradient is likewise the difference of the two
    per-sequence gradients.

    Outputs cos(g, .) against the mass-mean, whitened, rogue and orthogonal-gap
    directions, plus the predicted small-h susceptibility chi_lin(w) = scale*(w.g)
    for comparison against the measured chi (a through-origin fit over
    alpha in {0.5,1,2,4}, hence NOT a pure h->0 derivative).
    """
    ap = argparse.ArgumentParser(description=run_gradient.__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    ap.add_argument("--layers", default="28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=400)        # pool, as in steer_confirm2
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(GRAD_ART, "score_gradient.json"))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]

    tok, model = acts.get_model(args.model, GRAD_DEV, torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stmts, y = data.load_dataset(args.dataset, cap=args.cap, seed=args.seed)
    print(f"extracting activations ({len(stmts)} statements)...", flush=True)
    A = acts.extract_all_layers(stmts, tok, model, GRAD_DEV, 16)

    pool = data.load_pairs(args.dataset, args.pairs, 0)
    pairs = steering.pairs_for_seed(pool, args.seed, args.pairs_per_seed)
    print(f"{len(pairs)} contrastive pairs\n", flush=True)

    out = {"model": args.model, "dataset": args.dataset, "seed": args.seed,
           "n_pairs": len(pairs), "layers": {}}

    for L in layers:
        X = A[L].astype(np.float64)
        tr, te = est.split_indices(y, seed=args.seed)
        th, thw, _, _ = steering.fit_dirs(X, y, args.seed)
        v1, e2, lam = est.rogue_and_gap(X, y, tr)
        scale = est.class_gap(X[tr], y[tr])
        block = model.gpt_neox.layers[L]

        G = steering.pair_gradients(model, tok, block, pairs, args.bs, log=f"L{L}")   # (n_pairs, d)
        g = G.mean(0)
        gn = float(np.linalg.norm(g))

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        rec = {
            "alpha_unit": scale,
            "lam1_over_tr": float(lam[0] / lam.sum()),
            "lam1_over_lam2": float(lam[0] / lam[1]),
            "g_norm": gn,
            "cos_g_v1": cos(g, v1),
            "cos_g_theta": cos(g, th),
            "cos_g_theta_whitened": cos(g, thw),
            "cos_g_e2": cos(g, e2),
            "cos_theta_v1": cos(th, v1),
            "plane_fraction": float(np.sqrt((g @ v1) ** 2 + (g @ e2) ** 2) / gn),
            "chi_lin": {
                "plain": scale * float(th @ g),
                "whitened": scale * float(thw @ g),
                "v1": scale * float(v1 @ g),
                "e2": scale * float(e2 @ g),
                "g_ceiling": scale * gn,
            },
        }
        out["layers"][str(L)] = rec
        print(json.dumps(rec, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        # Save the vectors themselves: the summary above answers the part-1
        # question, but g projected on the eigenbasis, the steering ceiling,
        # and any audit of a published direction all need g, v1, e2 and the
        # spectrum, not their pairwise cosines.
        vec_path = args.out.replace(".json", "_vectors.npz")
        prev = dict(np.load(vec_path)) if os.path.exists(vec_path) else {}
        prev.update({f"L{L}_g": g, f"L{L}_g_per_pair": G.astype(np.float32),
                     f"L{L}_v1": v1, f"L{L}_e2": e2, f"L{L}_theta": th,
                     f"L{L}_theta_whitened": thw, f"L{L}_lam": lam,
                     f"L{L}_scale": np.array([scale])})
        np.savez_compressed(vec_path, **prev)
        print(f"  vectors -> {vec_path}", flush=True)
        gc.collect()

    print(f"\nwrote {args.out}\ndone.")


# =============================================================================
# gradient-ci
# =============================================================================
GRADCI_ART = REPO / "artifacts"


SEED = 0


B = 10000


NAMES = ["cities", "counterfact"]


KEYS = {"cos_g_v1": "v1", "cos_g_theta": "theta",
        "cos_g_theta_whitened": "theta_whitened", "cos_g_e2": "e2"}


def run_gradient_ci(argv=None):
    """Regenerate the bootstrap intervals on the score-gradient overlaps, with a FIXED
    seed and a resample count large enough that the Monte-Carlo error of the
    bootstrap sits below the precision quoted in the post.

    Why this script exists: the point estimates cos(g, w) are deterministic
    functions of the stored per-pair gradients, but the intervals were previously
    drawn from an unseeded generator, so every regeneration of the JSON moved the
    endpoints by 1-2 in the third decimal -- the precision the post quotes at. The
    seed and B below are now part of the recorded method.

    Reads artifacts/score_gradient_{name}_vectors.npz, rewrites the interval
    triples [point, lo, hi] in artifacts/score_gradient_{name}.json in place,
    leaving every other field untouched.
    """
    ap = argparse.ArgumentParser(description=run_gradient_ci.__doc__)
    ap.add_argument("--suffix", default="", help="read and rewrite score_gradient_{name}{suffix}.json (e.g. _v2)")
    args = ap.parse_args(argv)
    for name in NAMES:
        z = np.load(GRADCI_ART / f"score_gradient_{name}{args.suffix}_vectors.npz")
        path = GRADCI_ART / f"score_gradient_{name}{args.suffix}.json"
        doc = json.load(open(path))
        layers = sorted(doc["layers"], key=int)
        for L in layers:
            G = z[f"L{L}_g_per_pair"].astype(np.float64)
            rng = np.random.default_rng(SEED + int(L))
            for jk, vk in KEYS.items():
                w = z[f"L{L}_{vk}"].astype(np.float64)
                old = doc["layers"][L][jk]
                old = old if isinstance(old, list) else [old, float("nan"), float("nan")]   # a fresh run stores the point only
                new = cos_bootstrap_interval(G, w, rng, B)
                doc["layers"][L][jk] = new
                if jk == "cos_g_v1" or jk == "cos_g_theta_whitened":
                    print(f"  {name} L{L} {jk:<22} "
                          f"[{old[0]:+.4f} {old[1]:+.4f} {old[2]:+.4f}] -> "
                          f"[{new[0]:+.4f} {new[1]:+.4f} {new[2]:+.4f}]", flush=True)
        doc["note"] = (f"regenerated from *_vectors.npz; 95% bootstrap CIs over "
                       f"pairs, B={B}, seed={SEED}+layer")
        json.dump(doc, open(path, "w"), indent=2)
        print(f"-> wrote {path}", flush=True)


COMMANDS = {
    "observables": run_observables,
    "all-layers": run_all_layers,
    "massive": run_massive,
    "extra-datasets": run_extra_datasets,
    "olmo": run_olmo,
    "outliers": run_outliers,
    "steer-arms": run_steer_arms,
    "gradient": run_gradient,
    "gradient-ci": run_gradient_ci,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
