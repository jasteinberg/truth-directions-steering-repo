"""
A causal test of the mass-mean direction: steering sweeps and chi.

The post's sections "A causal test of the mass-mean direction" and "Steering
without a rogue dimension" (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
Steering cells for the plain and whitened directions at every (model, dataset,
layer, seed) in artifacts/steer_ckpt/, the random-direction steering nulls, and the
per-layer susceptibility chi with its rank test against the null.

Subcommands:

    sweep                  Does causal steering efficacy anticorrelate with decodability ...
    extend-null            The random-direction null for steering is currently estimated ...
    chi                    Estimator-vs-function-class analysis for counterfact steering, ...

Run from the repo root:  python scripts/causal_steering.py <subcommand> [-h]

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
from utils.truthlib.steering import (
    SEED_PLAN,
    cell_path,
    chi_summary,
    get_acts,
    null_path,
    run_cell,
    run_null,
)

# ---- shared by several subcommands ------------------------------------------
DEV = "mps"


# =============================================================================
# sweep
# =============================================================================
ART = os.path.join(REPO, "artifacts")


CKPT = os.path.join(ART, "steer_ckpt")  # class-gap alpha units


ACTS = os.path.join(ART, "act_cache")


def run_sweep(argv=None):
    """Does causal steering efficacy anticorrelate with decodability across depth, and
    does whitening trade intervention efficacy for readout quality?

    DESIGN

    Decomposition. A steering effect must be split into
        antisym  A = [shift(+a) - shift(-a)] / 2    <- real steering
        sym      S = [shift(+a) + shift(-a)] / 2    <- perturbation damage
    Only A is evidence of a causal direction: a generic out-of-distribution
    perturbation degrades the forward pass for BOTH signs, contributing to S while
    leaving A near zero. The pilot showed |z| up to 38 at mid layers with A ~ 0.

    Seed allocation. Layers are fractions of depth, so models of different depth are
    comparable. The early/mid window is where a causal variable is PREDICTED to live
    (it must be upstream of the computation consuming it); the deepest layer is the
    control, where decoding is near-perfect and the prediction is A ~ 0. Both ends of
    the claim get 10 seeds, the transition region 5. The allocation follows the
    prediction, not the pilot's observed peak -- otherwise the tightest error bars
    would sit exactly where noise happened to be favourable.

    Two noise sources. The split seed varies which statements fit theta (estimation
    noise in the direction); the pair sample varies the readout. Both are resampled
    per seed, so error bars cover resamples of the probing set AND the evaluation set.

    Random controls involve no fitting, so they do not depend on the split seed: they
    are drawn once per (model, dataset, layer, alpha) with n_rand draws.

    RESUMABILITY

    Every (model, dataset, layer, seed) cell is written to its own file under
    artifacts/steer_ckpt/. Seed semantics are stable -- seed k always means the same
    split and the same pair subsample -- so seeds can be appended later with
        --seeds 10,11,...,19
    and nothing already computed is recomputed. Activations for the probed layers are
    cached to artifacts/act_cache/, so a seed extension costs only its scoring passes.
    The notebook loads whatever cells exist via load_cells().
    """
    ap = argparse.ArgumentParser(description=run_sweep.__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b,EleutherAI/pythia-1.4b")
    ap.add_argument("--datasets", default="cities,counterfact_true_false")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="", help="explicit seed list, e.g. 10,11,12; "
                    "default uses SEED_PLAN per layer")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_rand", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--layers", default="", help="comma list, a subset of the planned layers (default: all planned)")
    ap.add_argument("--ckpt_dir", default=CKPT, help="where cells and nulls are written (default: the published steer_ckpt/)")
    args = ap.parse_args(argv)
    alphas = [float(x) for x in args.alphas.split(",")]
    explicit = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    want = {int(x) for x in args.layers.split(",") if x.strip()}
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for mname in [m for m in args.models.split(",") if m]:
        tok, model = acts.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        nL = model.config.num_hidden_layers
        plan = {max(1, int(round(f * nL))): n for f, n in SEED_PLAN.items()}
        print(f"\n=== {mname} ({nL} layers) plan={plan} ===", flush=True)

        for ds in [d for d in args.datasets.split(",") if d]:
            print(f"  [{ds}]", flush=True)
            layers = sorted(plan)
            if want - set(layers):
                raise SystemExit(f"--layers {sorted(want - set(layers))} not in the plan {layers}")
            Xs, y = get_acts(model, tok, mname, ds, layers, args.cap)   # all planned layers: the cache key
            pool = data.load_pairs(ds, args.pairs, 0)

            for L in [L for L in layers if not want or L in want]:
                seeds = explicit if explicit is not None else list(range(plan[L]))
                for sd in seeds:
                    cp = os.path.join(args.ckpt_dir, os.path.basename(cell_path(mname, ds, L, sd)))
                    if os.path.exists(cp) and not args.force:
                        continue
                    rec, _ = run_cell(model, tok, Xs[L], y, pool, L, alphas,
                                      sd, args.bs, dtype, args.pairs_per_seed)
                    with open(cp, "w") as f:
                        json.dump(rec, f, indent=2)
                    a8 = rec["alphas"][str(alphas[-1])]
                    print(f"    L{L:<3} s{sd:<3} AUROC={rec['probe_auroc']:.3f} "
                          f"plain A={a8['plain']['antisym']:+.3f} "
                          f"whit A={a8['whitened']['antisym']:+.3f}", flush=True)

                np_ = os.path.join(args.ckpt_dir, os.path.basename(null_path(mname, ds, L)))
                if not os.path.exists(np_) or args.force:
                    nl = run_null(model, tok, Xs[L], y, pool, L, alphas,
                                  args.bs, dtype, args.pairs_per_seed, args.n_rand)
                    with open(np_, "w") as f:
                        json.dump(nl, f, indent=2)
                    n8 = nl["alphas"][str(alphas[-1])]
                    print(f"    L{L:<3} NULL   rand A={n8['antisym_mean']:+.3f}"
                          f"+-{n8['antisym_std']:.3f}", flush=True)
            del Xs; gc.collect()

        del model, tok; gc.collect(); torch.mps.empty_cache()
    print(f"\ndone. cells in {args.ckpt_dir}")


# =============================================================================
# extend-null
# =============================================================================
def run_extend_null(argv=None):
    """The random-direction null for steering is currently estimated from 6 draws, so
    its spread (+-0.2 to +-0.9) is far too uncertain to support significance claims.
    This extends it to n_rand draws per (model, dataset, layer, alpha) and reports a
    p95, matching how the probe null is reported elsewhere in the study.

    Draws are NOT seeds. A seed varies the train/test split, hence the fitted theta;
    averaging over seeds gives the standard error on the truth direction's effect. A
    draw varies the random control vector -- no fitting is involved, so draws do not
    depend on the split. What we want from the null is the *distribution* of the
    antisymmetric effect produced by meaningless directions, and specifically its
    p95: the effect size a random direction exceeds only 5% of the time.

    Existing draws are reused; only the shortfall is computed. Results merge into the
    same NULL files, so this is safe to re-run and to interrupt.
    """
    ap = argparse.ArgumentParser(description=run_extend_null.__doc__)
    ap.add_argument("--models", default="EleutherAI/pythia-2.8b,EleutherAI/pythia-1.4b")
    ap.add_argument("--datasets", default="cities,counterfact_true_false")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--n_rand", type=int, default=30, help="TOTAL draws wanted")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--layers", default="", help="comma list; default all in SEED_PLAN")
    ap.add_argument("--ckpt_dir", default=CKPT, help="where the nulls are read and written (default: the published steer_ckpt/)")
    args = ap.parse_args(argv)
    want = {int(s) for s in args.layers.split(",") if s}
    alphas = [float(a) for a in args.alphas.split(",")]

    for mname in [m for m in args.models.split(",") if m]:
        tok, model = acts.get_model(mname, DEV, torch.float16)
        dtype = next(model.parameters()).dtype
        nL = model.config.num_hidden_layers
        planned = steering.planned_layers(nL)
        if want - set(planned):
            raise SystemExit(f"--layers {sorted(want - set(planned))} not in the plan {planned}")
        layers = [L for L in planned if not want or L in want]
        print(f"\n=== {mname} ({nL} layers) -> {layers} ===", flush=True)

        for ds in [d for d in args.datasets.split(",") if d]:
            # all planned layers: the cache key (a subset would re-extract and overwrite the cache)
            Xs, y = steering.get_acts(model, tok, mname, ds, planned, args.cap)
            pool = data.load_pairs(ds, args.pairs, 0)
            pairs = steering.pairs_for_seed(pool, 0, args.pairs_per_seed)

            for L in layers:
                p = os.path.join(args.ckpt_dir, os.path.basename(steering.null_path(mname, ds, L)))
                cur = json.load(open(p)) if os.path.exists(p) else {"layer": L, "alphas": {}}
                X = Xs[L].astype(np.float64)
                # Scale in class-gap units, matching run_null in steer_confirm2
                # ("same unit as the theta arms"). Note this is not std(X @ th),
                # which differs by sigma_along_theta and is layer-dependent.
                scale = est.class_gap(X, y)

                with steering.Steerer(model, L) as st:
                    st.set(None, 0, DEV, dtype)
                    base = steering.score_pairs(model, tok, pairs, args.bs)
                    for a in alphas:
                        k = str(a)
                        have = cur["alphas"].get(k, {}).get("draws", [])
                        need = args.n_rand - len(have)
                        if need <= 0:
                            print(f"    L{L} a={a}: already {len(have)} draws", flush=True)
                            continue
                        # offset the rng so new draws are independent of the old
                        rng = np.random.default_rng(9000 + L * 17 + 101 * len(have))
                        vals = list(have)
                        for _ in range(need):
                            r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
                            av, _ = steering.antisym(model, tok, st, r, a, scale,
                                               pairs, base, args.bs, dtype)
                            vals.append(av)
                            # Checkpoint every 10 draws so a long extension survives
                            # interruption. Mid-alpha resume reseeds the rng from the
                            # new len(have) -- draws stay independent; no consumer
                            # needs draw alignment across alphas (signflip reads the
                            # per-alpha p95, chi_whitening_analysis the alpha=1 draws).
                            if len(vals) % 10 == 0:
                                v = np.array(vals)
                                cur["alphas"][k] = {
                                    "antisym_mean": float(v.mean()),
                                    "antisym_std": float(v.std(ddof=1)),
                                    "antisym_p95": float(np.percentile(np.abs(v), 95)),
                                    "n_draws": len(v), "draws": [float(x) for x in v]}
                                with open(p, "w") as f:
                                    json.dump(cur, f, indent=2)
                                print(f"    L{L} a={a}: {len(vals)}/{args.n_rand} draws",
                                      flush=True)
                        v = np.array(vals)
                        cur["alphas"][k] = {
                            "antisym_mean": float(v.mean()),
                            "antisym_std": float(v.std(ddof=1)),
                            "antisym_p95": float(np.percentile(np.abs(v), 95)),
                            "n_draws": len(v), "draws": [float(x) for x in v]}
                        print(f"    L{L} a={a}: {len(v)} draws  mean={v.mean():+.3f} "
                              f"sd={v.std(ddof=1):.3f}  |A|p95={np.percentile(np.abs(v),95):.3f}",
                              flush=True)
                    st.set(None, 0, DEV, dtype)
                cur["n_rand"] = args.n_rand
                with open(p, "w") as f:
                    json.dump(cur, f, indent=2)
            del Xs; gc.collect()
        del model, tok; gc.collect(); torch.mps.empty_cache()
    print("\ndone.")


# =============================================================================
# chi
# =============================================================================
ALPHAS = [0.5, 1.0, 2.0, 4.0]


MODEL = "pythia-2.8b"


DATASETS = ["cities", "counterfact_true_false"]


LAYERS = {"cities": [8, 12, 16, 20, 24, 28],
          "counterfact_true_false": [8, 12, 16, 20, 24, 28]}


def run_chi(argv=None):
    """Estimator-vs-function-class analysis for counterfact steering, via
    chi = dA/dalpha.
    Reads existing artifacts/steer_ckpt/*.json -- NO new activations/compute.

    Question: is weak/wrong-signed counterfact steering explained by
    (a) rogue-dimension collapse of the mass-mean estimator (our diagnosis), or
    (b) the linear function class itself being too weak to move the behaviour?

    Note on (b): this is a natural hypothesis to rule out, not a position taken from
    the literature. Braun et al. (2025), "Understanding (Un)Reliability of Steering
    Vectors in Language Models" (arXiv:2505.22637), argue something adjacent but
    different -- that steerability tracks the directional coherence of a behaviour's
    activation differences and their separability along the difference-of-means line,
    i.e. steering fails when the behaviour is not a coherent linear direction. The
    result here supplies a mechanism for one such failure rather than contradicting it.

    Test: if whitening (Mahalanobis-correcting for the rogue dimension) alone
    recovers correctly-signed, non-null causal steering with a single rank-1
    direction, that supports (a) -- no need to invoke subspace/nonlinear
    expressivity limits. If whitened steering stays weak/wrong-signed too,
    that's evidence for (b) or a mix.
    """
    argparse.ArgumentParser(description=run_chi.__doc__).parse_args(argv)
    for dataset in DATASETS:
        print(f"\n=== {MODEL} / {dataset} ===")
        hdr = f"{'L':>3} {'n':>2} | {'chi_plain':>10} {'chi_whit':>10} | " \
              f"{'plain+%':>7} {'whit+%':>7} | {'plain_z':>7} {'whit_z':>7} | " \
              f"{'plain_p':>7} {'whit_p':>7}"
        print(hdr)
        print("-" * len(hdr))
        for L in LAYERS[dataset]:
            r = chi_summary(MODEL, dataset, L, ALPHAS)
            if r is None:
                print(f"{L:>3}  -- missing --")
                continue
            print(f"{r['layer']:>3} {r['n']:>2} | "
                  f"{r['chi_plain']:+10.4f} {r['chi_whit']:+10.4f} | "
                  f"{r['plain_pos_frac']*100:6.0f}% {r['whit_pos_frac']*100:6.0f}% | "
                  f"{r['plain_z']:+7.2f} {r['whit_z']:+7.2f} | "
                  f"{r['plain_p']:7.3f} {r['whit_p']:7.3f}")


COMMANDS = {
    "sweep": run_sweep,
    "extend-null": run_extend_null,
    "chi": run_chi,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
