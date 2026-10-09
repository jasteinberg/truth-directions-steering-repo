"""
Dimensional slack and sample size: the shuffled-label law.

The post's section "Dimensional slack and sample size" (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
A mass-mean direction fit to shuffled labels separates in-sample while N is below
Cover's capacity 2d; its excess AUROC follows a power law in N/2d whose amplitude
collapses with the participation ratio. Per-dataset curves and fits, the Gaussian
check of the d'-PR relation, the pool-size control, and the planted in-sample vs
held-out attenuation control.

Subcommands:

    by-dataset             Does the shuffled-label amplitude depend on the dataset?
    by-dataset-refit       Recompute the fits and collapse from the stored curves
    gaussian-check         Why does sp_en_trans miss the sqrt(PR/N) collapse?
    pool-control           Is the sp_en_trans anomaly a property of the dataset, or of ...
    insample-attenuation   In-sample versus held-out d' on a synthetic control with a ...
    no-droppers            The counterfact shuffled-label sweep without the dropper statements
    kurtosis-droppers      Is counterfact's excess kurtosis at layer 31 the droppers?

Run from the repo root:  python scripts/dimensional_slack.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib import acts, data
from utils.truthlib import estimators as est
from utils.truthlib.nulls import (
    collapse,
    cover_spectrum,
    excess_curve,
    geometric_grid,
    planted_attenuation,
    pool_stats,
    powerlaw_fit,
    shuffled_draw,
)

# ---- shared by several subcommands ------------------------------------------
MODEL = "EleutherAI/pythia-2.8b"          # default for --model
DATASETS = ["counterfact_true_false", "cities", "larger_than", "sp_en_trans"]
CAP = 3000          # applies only to counterfact (N=31,964); the rest are smaller
DEV = "mps"                               # default for --device
SWEEP = json.load(open(REPO / "artifacts" / "snr_sweep.json"))
NS = [100, 200, 354]


# =============================================================================
# by-dataset
# =============================================================================
N_MIN = 100


N_PTS = 7           # geometric points from N_MIN to each dataset's own N_total


BYDS_N_REP = 16


BYDS_OUT = REPO / "artifacts" / "cover_by_dataset.json"


def run_by_dataset(argv=None):
    """Does the shuffled-label amplitude depend on the dataset?

    The blog fits AUROC_shuffled - 1/2 ~ a (N/2d)^b on `counterfact` only, pooling
    four models, and gets a = 0.045, b = -0.49. The exponent is predicted by the
    estimator: a difference of two sample means of noise is O(sqrt(d/N)), so the
    in-sample excess must fall as N^{-1/2} at fixed d whatever the covariance. The
    amplitude has no such derivation -- it should absorb the SHAPE of the
    within-class spectrum, i.e. how many directions the noise effectively occupies.

    Effective-dimension hypothesis: if the spurious separation scales as
    sqrt(d_eff/N) with d_eff the participation ratio (sum lam)^2 / sum lam^2 of the
    within-class covariance, then a dataset with a strongly concentrated spectrum
    (small PR) should sit LOWER, not higher.

    This script measures a and b per dataset at a fixed model, averaging over
    several label permutations per N, and records the spectrum diagnostics
    alongside so the two can be compared.

    The control is scored in-sample by design, so no half has to be held back and
    the grid can run to the full set. Each dataset therefore gets its own geometric
    grid from N=100 to its own N_total rather than one shared grid truncated at the
    smallest dataset -- `sp_en_trans` (N=354) got two points under the shared grid,
    which is a zero-residual fit and an uninformative exponent.

    The spectrum is also recomputed at each N, not only on the full set, because
    the collapse excess ~ C sqrt(PR/N) treats PR as a measured quantity: if the
    sample PR drifts with the number of samples it was estimated from, C inherits
    that drift and the small datasets are penalised for being small.
    """
    ap = argparse.ArgumentParser(description=run_by_dataset.__doc__)
    ap.add_argument("--sweep", default=str(REPO / "artifacts" / "snr_sweep.json"), help="readout sweep whose best layers are used")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default=str(BYDS_OUT))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args(argv)
    sweep = json.load(open(args.sweep))
    datasets = args.datasets.split(",")
    out_path = Path(args.out)
    tok, model = acts.get_model(args.model, args.device, torch.float16 if args.device == "mps" else torch.float32)
    out = {"config": {"model": args.model, "datasets": datasets, "sweep": os.path.basename(args.sweep),
                      "n_min": N_MIN, "n_pts": N_PTS, "cap": CAP,
                      "n_rep": BYDS_N_REP,
                      "note": "in-sample shuffled-label excess; per-dataset "
                              "geometric N grid to the full set"},
           "results": {}}
    for ds in datasets:
        best = sweep["models"][args.model]["datasets"][ds]["best_layer"]
        stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
        print(f"[{ds}] N={len(y)} best_layer={best}", flush=True)
        A = acts.extract_all_layers(stmts, tok, model, args.device, 16)
        X = A[best].astype(np.float64)
        d_model = X.shape[1]
        ns = geometric_grid(len(y), N_MIN, N_PTS)
        print(f"   grid {ns}", flush=True)
        rows = excess_curve(X, y, d_model, ns, BYDS_N_REP, seed=0)
        fit = powerlaw_fit(rows)
        out["results"][ds] = {"best_layer": int(best), "N_total": int(len(y)),
                              "ns": ns,
                              "spectrum": cover_spectrum(X, y), "curve": rows,
                              "fit": fit,
                              "collapse": collapse(rows, cover_spectrum(X, y)["participation_ratio"])}
        print(f"   fit a={fit['amplitude']:.4f} b={fit['exponent']:+.3f} "
              f"R2={fit['r2']:.3f}  PR={out['results'][ds]['spectrum']['participation_ratio']:.1f}",
              flush=True)
        if "exponent_wls" in fit:
            print(f"   wls a={fit['amplitude_wls']:.4f} b={fit['exponent_wls']:+.3f}",
                  flush=True)
        for r in rows:
            print(f"     N={r['N']:>5} N/2d={r['N_over_2d']:.3f} "
                  f"excess={r['excess_mean']:.4f}±{r['excess_sd']:.4f} "
                  f"true={r['true_auroc_mean']:.3f} PR@N={r['pr_at_N']:.1f}",
                  flush=True)
        del A, X
        gc.collect()
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  -> wrote {out_path}", flush=True)


def refit():
    """Recompute fits and collapse from the stored curves, no model needed."""
    out = json.loads(BYDS_OUT.read_text())
    for ds, v in out["results"].items():
        if "curve" not in v:
            continue
        v["fit"] = powerlaw_fit(v["curve"])
        v["collapse"] = collapse(v["curve"], v["spectrum"]["participation_ratio"])
        f = v["fit"]
        print(f"{ds:24s} b={f['exponent']:+.3f}±{f['exponent_se']:.3f} "
              f"a={f['amplitude']:.4f} R2={f['r2']:.3f}", flush=True)
        for w, wv in f["windows"].items():
            print(f"{'':24s}   {w:<8s} b={wv['exponent']:+.3f}±{wv['exponent_se']:.3f} "
                  f"({wv['n_points']} pts)", flush=True)
    BYDS_OUT.write_text(json.dumps(out, indent=2))
    print(f"-> rewrote {BYDS_OUT}")


def run_by_dataset_refit(argv=None):
    """Recompute the fits and collapse from the stored curves; no model needed."""
    argparse.ArgumentParser(description=run_by_dataset_refit.__doc__).parse_args(argv)
    refit()


# =============================================================================
# gaussian-check
# =============================================================================
GAUSS_N_REP = 32


GAUSS_OUT = REPO / "artifacts" / "cover_gaussian_check.json"


def pr_of(X, y):
    lam = est.eigvals_desc(est.within_class_cov(X, y, ddof=1, class_sorted=True))
    return est.participation_ratio(lam), lam


def run_gaussian_check(argv=None):
    """Why does sp_en_trans miss the sqrt(PR/N) collapse?

    The collapse constant is not free. Under shuffled labels there is no signal, so
    x ~ N(0, Sigma) within class and the mass-mean direction is

        theta = mu_+ - mu_- = (2/N) sum_i eps_i x_i,   eps_i = +-1,

    so theta ~ N(0, (4/N) Sigma). The in-sample separation it produces is

        E|theta|^2       = (4/N) tr Sigma
        E theta' Sigma theta = (4/N) tr Sigma^2

        d'_in = |theta|^2 / sqrt(theta' Sigma theta) = 2 sqrt(PR / N),

    with PR = (tr Sigma)^2 / tr Sigma^2. If the projections are Gaussian then
    AUROC = Phi(d'/sqrt2), and for small d'

        AUROC - 1/2 ~ d' / (2 sqrt(pi)) = sqrt(PR/N) / sqrt(pi),

    i.e. C = 1/sqrt(pi) = 0.5642, a parameter-free prediction.

    That splits the collapse into two independent steps:

      (A) geometry     d'_in  =?  2 sqrt(PR/N)      -- second moments only
      (B) Gaussianity  AUROC  =?  Phi(d'_in/sqrt2)  -- shape of the projection

    C can be inflated by a failure of either. This script measures both separately
    per dataset, plus the excess kurtosis of the within-class projection, so the
    sp_en_trans anomaly can be localised to one step.
    """
    ap = argparse.ArgumentParser(description=run_gaussian_check.__doc__)
    ap.add_argument("--sweep", default=str(REPO / "artifacts" / "snr_sweep.json"), help="readout sweep whose best layers are used")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default=str(GAUSS_OUT))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args(argv)
    sweep = json.load(open(args.sweep))
    out_path = Path(args.out)
    tok, model = acts.get_model(args.model, args.device, torch.float16 if args.device == "mps" else torch.float32)
    out = {"config": {"model": args.model, "ns": NS, "n_rep": GAUSS_N_REP,
                      "C_predicted": 1.0 / math.sqrt(math.pi), "sweep": os.path.basename(args.sweep)},
           "results": {}}
    for ds in args.datasets.split(","):
        best = sweep["models"][args.model]["datasets"][ds]["best_layer"]
        stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
        A = acts.extract_all_layers(stmts, tok, model, args.device, 16)
        X = A[best].astype(np.float64)
        PR, lam = pr_of(X, y)
        print(f"[{ds}] N_total={len(y)} L={best} PR={PR:.1f}", flush=True)
        rows = []
        rng = np.random.default_rng(0)
        for N in NS:
            if N > len(y):
                continue
            dp, au, ku = [], [], []
            for _ in range(GAUSS_N_REP):
                r = shuffled_draw(X, y, rng, N)
                if r:
                    dp.append(r[0]); au.append(r[1]); ku.append(r[2])
            dp_m, au_m, ku_m = float(np.mean(dp)), float(np.mean(au)), float(np.mean(ku))
            dp_pred = 2.0 * math.sqrt(PR / N)
            au_from_dp = float(norm.cdf(dp_m / math.sqrt(2)))
            rows.append({
                "N": N, "dprime_meas": dp_m, "dprime_pred": dp_pred,
                "step_A_ratio": dp_m / dp_pred,
                "auroc_meas": au_m, "auroc_from_dprime": au_from_dp,
                "step_B_ratio": (au_m - 0.5) / (au_from_dp - 0.5),
                "kurtosis": ku_m,
                "C_meas": (au_m - 0.5) * math.sqrt(N / PR),
            })
            r = rows[-1]
            print(f"   N={N:5d}  d'meas={dp_m:.3f} d'pred={dp_pred:.3f} (A={r['step_A_ratio']:.3f})"
                  f"  AUROC={au_m:.4f} from-d'={au_from_dp:.4f} (B={r['step_B_ratio']:.3f})"
                  f"  kurt={ku_m:+.2f}  C={r['C_meas']:.3f}", flush=True)
        # spectral shape beyond PR: normalised eigenvalue moments
        p = lam / lam.sum()
        out["results"][ds] = {
            "best_layer": int(best), "N_total": int(len(y)), "PR": PR,
            "participation_ratios": {
                "PR2": float(1.0 / (p ** 2).sum()),
                "PR3": float(1.0 / (p ** 3).sum() ** 0.5),
                "shannon_exp": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
                "lambda1_over_trace": float(p[0]),
                "top10_share": float(p[:10].sum()),
                "top50_share": float(p[:50].sum()),
            },
            "curve": rows,
        }
        del A, X
        gc.collect()
        out_path.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {out_path}")


# =============================================================================
# pool-control
# =============================================================================
POOLS = [354, 800, 1980, 3000]


POOL_N_REP = 24


N_POOL_REP = 4


POOL_OUT = REPO / "artifacts" / "cover_pool_control.json"


def run_pool_control(argv=None):
    """Is the sp_en_trans anomaly a property of the dataset, or of measuring PR from
    only 354 samples?

    `dimensional_slack.py gaussian-check` localised the miss to step (A): sp_en_trans's in-sample
    d' runs ~50% above 2 sqrt(PR/N) where the other three sit within 3-17%. Step
    (B), AUROC = Phi(d'/sqrt2), holds to 2% everywhere, so the projection shape is
    not the culprit.

    PR here is a SAMPLE participation ratio. Sigma-hat from P samples in d=2560 has
    rank <= P-2, and PR@N was already seen to rise with N and not saturate until
    N ~ 1000 on counterfact. sp_en_trans has N_total = 354, so its PR can only ever
    be read at the small-P end of that curve. If that is the whole story, then
    restricting a LARGE dataset to a pool of 354 and re-measuring PR from that pool
    should reproduce the same inflated A.

    That is the control: vary the pool size P at fixed dataset and watch A. If A(P)
    falls toward 1 as P grows, the anomaly belongs to the estimate of PR, not to
    sp_en_trans. If counterfact holds A ~ 1 even at P = 354, sp_en_trans is
    genuinely different.

    Also decomposes A into its numerator and denominator against the second-moment
    prediction:

        num = |theta|^2  / ((4/N) tr Sigma)
        den = theta' Sigma theta / ((4/N) tr Sigma^2)
        A   = d'_meas / (2 sqrt(PR/N)) = num / sqrt(den)
    """
    ap = argparse.ArgumentParser(description=run_pool_control.__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args(argv)
    tok, model = acts.get_model(args.model, args.device, torch.float16 if args.device == "mps" else torch.float32)
    out = {"config": {"model": args.model, "pools": POOLS, "ns": NS,
                      "n_rep": POOL_N_REP, "n_pool_rep": N_POOL_REP}, "results": {}}
    for ds in DATASETS:
        best = SWEEP["models"][args.model]["datasets"][ds]["best_layer"]
        stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
        A_all = acts.extract_all_layers(stmts, tok, model, args.device, 16)
        X = A_all[best].astype(np.float64)
        print(f"[{ds}] N_total={len(y)} L={best}", flush=True)
        rows = []
        rng = np.random.default_rng(0)
        for P in POOLS:
            if P > len(y):
                continue
            for _ in range(N_POOL_REP if P < len(y) else 1):
                pid = rng.choice(len(y), size=P, replace=False)
                Xp, yp = X[pid], y[pid]
                Sig, tr, tr2, PR = pool_stats(Xp, yp)
                for N in NS:
                    if N > P:
                        continue
                    a_s, num_s, den_s = [], [], []
                    for _ in range(POOL_N_REP):
                        idx = rng.choice(P, size=N, replace=False)
                        Xs = Xp[idx]
                        ysh = rng.permutation(yp[idx])
                        if len(np.unique(ysh)) < 2:
                            continue
                        th = est.mass_mean(Xs, ysh)
                        th = th * np.linalg.norm(th) / max(np.linalg.norm(th), 1e-12)
                        # mass_mean_direction returns a unit vector; rebuild the raw gap
                        mu1 = Xs[ysh == 1].mean(0)
                        mu0 = Xs[ysh == 0].mean(0)
                        raw = mu1 - mu0
                        n2 = float(raw @ raw)
                        q = float(raw @ (Sig @ raw))
                        z = Xs @ raw
                        z1, z0 = z[ysh == 1], z[ysh == 0]
                        sp = math.sqrt(0.5 * (z1.var(ddof=1) + z0.var(ddof=1)))
                        if sp <= 0:
                            continue
                        a_s.append(float((z1.mean() - z0.mean()) / sp)
                                   / (2.0 * math.sqrt(PR / N)))
                        num_s.append(n2 / ((4.0 / N) * tr))
                        den_s.append(q / ((4.0 / N) * tr2))
                    rows.append({"P": P, "PR_pool": PR, "N": N,
                                 "A": float(np.mean(a_s)),
                                 "num": float(np.mean(num_s)),
                                 "den": float(np.mean(den_s))})
        # average over pool repeats
        agg = {}
        for r in rows:
            agg.setdefault((r["P"], r["N"]), []).append(r)
        curve = []
        for (P, N), rs in sorted(agg.items()):
            curve.append({"P": P, "N": N,
                          "PR_pool": float(np.mean([r["PR_pool"] for r in rs])),
                          "A": float(np.mean([r["A"] for r in rs])),
                          "num": float(np.mean([r["num"] for r in rs])),
                          "den": float(np.mean([r["den"] for r in rs]))})
            c = curve[-1]
            print(f"   P={P:5d} N={N:4d}  PR_pool={c['PR_pool']:6.1f}  A={c['A']:.3f}"
                  f"   num={c['num']:.3f} den={c['den']:.3f}", flush=True)
        out["results"][ds] = {"best_layer": int(best), "N_total": int(len(y)),
                              "curve": curve}
        del A_all, X
        gc.collect()
        POOL_OUT.write_text(json.dumps(out, indent=2))
    print(f"-> wrote {POOL_OUT}")


# =============================================================================
# insample-attenuation
# =============================================================================
PLANTED_OUT = REPO / "artifacts" / "check_insample_attenuation.json"


PLANTED_N_REP = 20


def run_insample_attenuation(argv=None):
    """In-sample versus held-out d' on a synthetic control with a planted separation.

    The post quotes a synthetic control in *Setup*: "with a planted separation of
    d' = 1, the in-sample estimate returns 1.31 and the held-out estimate 0.58".
    No script produced those numbers and they are not reproducible at the post's
    own d/N. This script does the control at the post's regime and at a ladder of
    d/N so the quoted sentence can carry numbers that exist.

    Model: x ~ N(+-delta/2, I_d), delta = u (unit), so the population d' along u
    is exactly 1. Mass-mean direction fit on a class-balanced training half, scored
    in-sample on that half and held-out on the other. Prediction from the
    shuffled-label derivation in the post: in-sample d'^2 ~ 1 + 4d/N_train (the
    noise part of theta-hat is aligned with the sample fluctuations that defined
    it), held-out d' ~ cos(theta-hat, u) ~ 1/sqrt(1 + 4d/N_train).
    """
    argparse.ArgumentParser(description=run_insample_attenuation.__doc__).parse_args(argv)
    rng = np.random.default_rng(0)
    rows = []
    for d, N in [(2560, 1198), (2048, 1198), (1024, 1198), (512, 1198), (2560, 354), (128, 1198)]:
        r = np.array([planted_attenuation(d, N, rng) for _ in range(PLANTED_N_REP)])
        ntr = N // 2
        pred_in = float(np.sqrt(1 + 4 * d / ntr))
        pred_out = float(1 / np.sqrt(1 + 4 * d / ntr))
        rows.append({"d": d, "N": N, "N_train": ntr, "d_over_Ntrain": d / ntr,
                     "insample_mean": float(r[:, 0].mean()), "insample_sd": float(r[:, 0].std(ddof=1)),
                     "heldout_mean": float(r[:, 1].mean()), "heldout_sd": float(r[:, 1].std(ddof=1)),
                     "cos_mean": float(r[:, 2].mean()),
                     "pred_insample": pred_in, "pred_heldout": pred_out})
        print(f"d={d:5d} N={N:5d} d/Ntr={d/ntr:5.2f}  in-sample {r[:,0].mean():.2f}±{r[:,0].std():.2f} (pred {pred_in:.2f})"
              f"  held-out {r[:,1].mean():.2f}±{r[:,1].std():.2f} (pred {pred_out:.2f})  cos={r[:,2].mean():.2f}")
    PLANTED_OUT.write_text(json.dumps({"planted_dprime": 1.0, "n_rep": PLANTED_N_REP, "rows": rows}, indent=2))
    print(f"-> wrote {PLANTED_OUT}")


# =============================================================================
# no-droppers
# =============================================================================
DROPPER_FLAG_LAYER = 28     # the dropper mask: massive_mask at this layer (label-free)


NODROP_OUT = REPO / "artifacts" / "cover_no_droppers.json"


def run_no_droppers(argv=None):
    """The per-dataset shuffled-label sweep for counterfact with the dropper statements excluded.

    Same protocol as `dimensional_slack.py by-dataset` (cap, geometric N grid, repetitions,
    seed, best layer from snr_sweep.json), on pythia-2.8b counterfact at its best layer, with
    the statements that lack the massive activation (massive_mask at layer 28, label-free)
    removed first. Feeds the appendix panel of the effective-dimension collapse.
    Writes artifacts/cover_no_droppers.json.
    """
    argparse.ArgumentParser(description=run_no_droppers.__doc__).parse_args(argv)
    ds = "counterfact_true_false"
    sweep = json.load(open(REPO / "artifacts" / "snr_sweep.json"))
    best = sweep["models"][MODEL]["datasets"][ds]["best_layer"]
    tok, model = acts.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    stmts, y = data.load_dataset(ds, cap=CAP, seed=0)
    y = np.asarray(y)
    A = acts.extract_all_layers(stmts, tok, model, DEV, 16)
    drop, _, _ = est.massive_mask(A[DROPPER_FLAG_LAYER].astype(np.float64))
    X, yk = A[best].astype(np.float64)[~drop], y[~drop]
    ns = geometric_grid(len(yk), N_MIN, N_PTS)
    rows = excess_curve(X, yk, X.shape[1], ns, BYDS_N_REP, seed=0)
    spec_ = cover_spectrum(X, yk)
    fit = powerlaw_fit(rows)
    col = collapse(rows, spec_["participation_ratio"])
    print(f"L{best}: excluded {int(drop.sum())} of {len(y)}; PR={spec_['participation_ratio']:.1f} "
          f"a={fit['amplitude']:.4f} b={fit['exponent']:+.3f}+-{fit['exponent_se']:.3f} "
          f"C={min(c['C'] for c in col):.2f}-{max(c['C'] for c in col):.2f}", flush=True)
    out = {"model": MODEL, "dataset": ds, "best_layer": int(best), "flag_layer": DROPPER_FLAG_LAYER,
           "n_total": int(len(y)), "n_excluded": int(drop.sum()), "N_total": int(len(yk)), "ns": ns,
           "spectrum": spec_, "curve": rows, "fit": fit, "collapse": col}
    NODROP_OUT.write_text(json.dumps(out, indent=2))


# =============================================================================
# kurtosis-droppers
# =============================================================================
KURT_LAYERS = [30, 31, 32]


KURT_OUT = REPO / "artifacts" / "kurtosis_droppers.json"


def _gaussian_steps(X, y, label):
    """gaussian-check's two steps and the excess kurtosis, on one activation set."""
    PR, lam = pr_of(X, y)
    rng = np.random.default_rng(0)
    rows = []
    for N in NS:
        if N > len(y):
            continue
        dp, au, ku = [], [], []
        for _ in range(GAUSS_N_REP):
            r = shuffled_draw(X, y, rng, N)
            if r:
                dp.append(r[0]); au.append(r[1]); ku.append(r[2])
        dp_m, au_m, ku_m = map(lambda v: float(np.mean(v)), (dp, au, ku))
        au_from = float(norm.cdf(dp_m / math.sqrt(2)))
        rows.append({"N": N, "step_A": dp_m / (2 * math.sqrt(PR / N)),
                     "step_B": (au_m - 0.5) / (au_from - 0.5), "kurtosis": ku_m,
                     "C": (au_m - 0.5) * math.sqrt(N / PR)})
    print(f"  {label:12s} n={len(y):5d} PR={PR:6.1f} lam1/tr={lam[0]/lam.sum():.3f} | " + "  ".join(
        f"N={r['N']}: A={r['step_A']:.2f} B={r['step_B']:.2f} k={r['kurtosis']:+.1f} C={r['C']:.2f}" for r in rows), flush=True)
    return {"n": int(len(y)), "PR": float(PR), "lambda1_over_trace": float(lam[0] / lam.sum()), "curve": rows}


def run_kurtosis_droppers(argv=None):
    """Is the excess kurtosis of counterfact's shuffled-label projections at layer 31 the droppers?

    Hypothesis: the statements that lack the massive activation (flagged at layer 28 by the
    coordinate rule, truthlib.estimators.massive_mask, which never touches v1, the class means
    or the labels) still sit off the bulk at layer 31, after the massive coordinates themselves
    have gone. A fraction p of outliers gives excess kurtosis up to 1/p - 6, about 100 at
    p ~ 0.01.

    Reruns the Gaussian check (same N grid, repetitions, seed and cap) on pythia-2.8b
    counterfact at layers 30, 31 and 32 (32 = the last block's pre-norm output), on all
    statements and with the droppers excluded, plus where the droppers sit relative to the
    bulk. Writes artifacts/kurtosis_droppers.json.
    """
    argparse.ArgumentParser(description=run_kurtosis_droppers.__doc__).parse_args(argv)
    tok, model = acts.get_model(MODEL, DEV, torch.float16 if DEV == "mps" else torch.float32)
    stmts, y = data.load_dataset("counterfact_true_false", cap=CAP, seed=0)
    y = np.asarray(y)
    A = acts.extract_all_layers(stmts, tok, model, DEV, 16)
    drop, _, massive = est.massive_mask(A[DROPPER_FLAG_LAYER].astype(np.float64))
    print(f"N={len(y)}  droppers flagged at L{DROPPER_FLAG_LAYER}: {int(drop.sum())} (p = {drop.mean():.4f}, "
          f"1/p - 6 = {1/drop.mean() - 6:.0f}); massive coords there: {int(massive.sum())}", flush=True)
    out = {"model": MODEL, "dataset": "counterfact_true_false", "n_total": int(len(y)),
           "flag_layer": DROPPER_FLAG_LAYER, "n_droppers": int(drop.sum()), "layers": {}}
    for L in KURT_LAYERS:
        X = A[L].astype(np.float64)
        Xc = est.within_class_center(X, y)
        lam, V = est.eigh_desc(est.within_class_cov(X, y, ddof=1))
        z = Xc @ V[:, 0]
        s = np.std(z[~drop])
        zd = np.abs(z[drop] - np.median(z[~drop])) / s
        print(f"\nL{L}: droppers along v1 of this layer: median |z| = {np.median(zd):.1f} bulk sd "
              f"(bulk: {np.median(np.abs(z[~drop] - np.median(z[~drop])) / s):.2f}); "
              f"share of lambda1 from droppers = {(z[drop]**2).sum() / (z**2).sum():.2f}", flush=True)
        out["layers"][str(L)] = {
            "dropper_median_abs_z_on_v1": float(np.median(zd)),
            "dropper_share_of_lambda1": float((z[drop] ** 2).sum() / (z ** 2).sum()),
            "all": _gaussian_steps(X, y, "all"),
            "no_droppers": _gaussian_steps(X[~drop], y[~drop], "no droppers"),
        }
    KURT_OUT.write_text(json.dumps(out, indent=1)); print(f"\nwrote {KURT_OUT}")


COMMANDS = {
    "by-dataset": run_by_dataset,
    "by-dataset-refit": run_by_dataset_refit,
    "gaussian-check": run_gaussian_check,
    "pool-control": run_pool_control,
    "insample-attenuation": run_insample_attenuation,
    "no-droppers": run_no_droppers,
    "kurtosis-droppers": run_kurtosis_droppers,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
