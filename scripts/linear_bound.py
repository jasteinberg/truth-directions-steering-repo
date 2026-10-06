"""
The linear bound: steering along g_hat, and the noise floor of its estimate.

The section "The linear bound" of part 2, "Steering Vectors and the Limits of Linear Response", with its appendix "The linear-response
relation in full".
The g-arm runner (steer along g_hat, the plain direction, controls, shuffles and a
random null, in half or complement split; also the runner behind the control and
depth sections), the noise-floor curve over the number of fit pairs and its
small-alpha reading, and the finite-difference check of the gradient convention.

Subcommands:

    g-arm                  Steer along g-hat itself and compare against the linear bound ...
    noise-floor            The linear bound against the size of the gradient sample
    noise-floor-small      Same as `linear_bound.py noise-floor` but the measured ...
    fd-check               Finite-difference check of the truth-score gradient (the ...

Run from the repo root:  python scripts/linear_bound.py <subcommand> [-h]

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
import torch
from matplotlib import rcParams

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib import acts, data, steering
from utils.truthlib.estimators import chi_origin
from utils.truthlib.steering import DEV, noise_floor_ratio, pairs_for_seed, response_slopes

# ---- shared by several subcommands ------------------------------------------
ART = REPO / "artifacts"
NS = [25, 50, 125, 250]


# =============================================================================
# g-arm
# =============================================================================
ALPHAS = [0.5, 1.0, 2.0, 4.0]


def run_g_arm(argv=None):
    """Steer along g-hat itself and compare against the linear bound c*||g||.

    chi(w) = c (w . g) is a linear-response statement, so c*||g|| bounds chi only
    in the small-h window. This script measures what happens when you steer along
    g-hat at the alphas the rest of the sweep uses. Two split modes, both held out
    by construction (g is never estimated on a pair it is scored on):

      half        g on the first 125 of the seed's 250 pairs, scored on the other
                  125. The original ten-seed protocol (rl-align 1c51012).
      complement  g on all 250 of the seed's pairs, scored on the 150 pairs of the
                  400-pair pool the seed did not draw. Doubles n for g; the
                  prediction under linear response is that chi(g_hat)/(c||g_hat||)
                  rises from cos^2(g_125, g) to cos^2(g_250, g), i.e. the slack in
                  the bound is estimation noise in g_hat, not curvature.

    Per-pair gradients come from the *_vectors.npz written by `rogue_dimension.py gradient` (part 1's section file). Arms: g, plain
    mass-mean, and random directions for the null. Same alpha grid, class-gap unit
    and antisymmetrisation as steer_confirm2, so the numbers drop into its tables.
    The file also records ||g_fit|| and tr(Sigma_G)/n so the noise floor of the
    bound can be read off without the npz.

    Shuffled-label control (--n_shuffle): see the flag help; the analogue of part 1's
    shuffled-label probe fits.

    Controls (--controls, additive; the default arms are unchanged so the existing
    artifacts reproduce). For each named target direction t in {theta, theta_F, e2}:
      t         the direction itself (skip via --arms bookkeeping if already measured)
      t_perp    t with its g_hat component projected out, renormalised: the identity
                predicts chi = 0 on the null scale, for a direction that is >99% of t
      match_t_i n_match random directions with the SAME cosine to g_hat as t,
                w = rho g_hat + sqrt(1-rho^2) u, u a random unit vector orthogonal
                to g_hat: the identity predicts chi(w) = chi(t) exactly, with none
                of t's structure
    Together they test that the projection onto g is necessary and sufficient.
    Predictions c (t . g_hat) ||g_hat|| are written into the file.
    """
    ap = argparse.ArgumentParser(description=run_g_arm.__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["half", "complement"], default="half")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pairs_per_seed", type=int, default=250)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_rand", type=int, default=10)
    ap.add_argument("--n_fit", type=int, default=None,
                    help="complement mode only: fit g on the first n_fit of the seed's pairs "
                         "(default all); the eval set is unchanged, so the null and plain arm "
                         "measured at n_fit=all still apply")
    ap.add_argument("--arms", default="g,plain,rand",
                    help="comma list from g, plain, rand (skip arms already measured on this eval set)")
    ap.add_argument("--controls", default="",
                    help="comma list from theta, theta_F, e2: adds t, t_perp and match_t_i arms")
    ap.add_argument("--n_match", type=int, default=5)
    ap.add_argument("--alphas", default=None, help="comma list; default 0.5,1,2,4 (the sweep grid)")
    ap.add_argument("--n_shuffle", type=int, default=0,
                    help="shuffled-label control: n_shuffle arms, each the unit vector of the fit-half "
                         "gradient mean with a random sign flip per pair (a flipped pair label negates "
                         "ell_i and G_i). Predicted chi = c||g_hat|| cos(gshuf, g_hat), i.e. only the "
                         "chance overlap; the arm tests that g_hat is not a label-free direction")
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else ALPHAS

    short = args.dataset.replace("_true_false", "")
    # The pooled npz is seed 0's. Every other seed has its own per-pair gradients
    # and its own class-gap unit, so the file must be named explicitly: running a
    # nonzero seed against the pooled file fits g on seed 0's pairs and scores it
    # on pairs that overlap them (in-sample).
    if args.vectors is None and args.seed != 0:
        raise SystemExit("--vectors is required for seed != 0 (the default npz is seed 0's)")
    vec_path = args.vectors or str(ART / f"score_gradient_{short}_vectors.npz")
    suffix = "" if args.mode == "half" else "_comp"
    if args.n_fit is not None:
        suffix += f"_n{args.n_fit}"
    out_path = args.out or str(ART / f"steer_g_arm_{short}_L{args.layer}{suffix}_s{args.seed}.json")

    z = np.load(vec_path)
    G = z[f"L{args.layer}_g_per_pair"].astype(np.float64)
    theta = z[f"L{args.layer}_theta"]
    scale = float(z[f"L{args.layer}_scale"][0])

    pool = data.load_pairs(args.dataset, args.pairs, 0)
    pairs = pairs_for_seed(pool, args.seed, args.pairs_per_seed)
    assert G.shape[0] == len(pairs), "per-pair gradients do not match the seed's pairs"
    steering.check_gradient_file(vec_path, args.dataset, args.seed, args.pairs_per_seed)
    assert len({repr(p) for p in pool}) == len(pool), "duplicate pairs in the pool: fit and eval could overlap"

    if args.mode == "half":
        half = len(pairs) // 2
        fit_G = G[:half]
        eval_pairs = pairs[half:]
    else:
        fit_G = G if args.n_fit is None else G[:args.n_fit]
        rng = np.random.default_rng(777 + args.seed)  # same draw as pairs_for_seed
        drawn = rng.choice(len(pool), size=min(args.pairs_per_seed, len(pool)), replace=False)
        assert [pool[i] for i in drawn] == pairs, "complement draw does not reproduce pairs_for_seed"
        chosen = set(drawn.tolist())
        eval_pairs = [p for i, p in enumerate(pool) if i not in chosen]
        assert not {repr(p) for p in pairs} & {repr(p) for p in eval_pairs}, "fit and eval pairs overlap"

    g_fit = fit_G.mean(0)
    g_norm = float(np.linalg.norm(g_fit))
    g_hat = g_fit / g_norm
    ceiling = scale * g_norm
    tr_sigma_over_n = float(fit_G.var(0, ddof=1).sum() / fit_G.shape[0])

    rng = np.random.default_rng(args.seed)
    d = len(g_hat)
    want = set(args.arms.split(","))
    arms = {}
    if "g" in want: arms["g"] = g_hat
    if "plain" in want: arms["plain"] = theta
    if "rand" in want:
        for i in range(args.n_rand):
            r = rng.standard_normal(d)
            arms[f"rand{i}"] = r / np.linalg.norm(r)
    control_pred = {}
    targets = {"theta": theta, "theta_F": z[f"L{args.layer}_theta_whitened"], "e2": z[f"L{args.layer}_e2"]}
    # LayerNorm-mechanism arms (13 Sept): delta_hat from the act cache (all 1198 statements,
    # so the DIRECTION sees eval statements; the window statement does not depend on that,
    # the chi prediction uses g_eval and is mildly optimistic), the massive coordinate e_m
    # (argmax |mean residual|), and delta_hat with e_m projected out. Hypothesis: the
    # counterfact window is set by the push along e_m (LN nonlinearity), so delta_perp_m
    # should stay linear to alpha = 4 and e_m alone should close early with tiny chi.
    want_ln = {"delta", "delta_perp_massive", "e_massive"} & set(args.controls.split(","))
    if want_ln:
        tag = "counterfact_true_false" if "counterfact" in args.dataset else "cities"
        A = np.load(str(REPO / "artifacts" / "act_cache" / f"pythia-2.8b__{tag}.npz"))
        X = A[f"L{args.layer}"].astype(np.float64); yy = A["y"]
        delta = X[yy == 1].mean(0) - X[yy == 0].mean(0); m = int(np.argmax(np.abs(X.mean(0))))
        em = np.zeros(d); em[m] = 1.0
        dpm = delta.copy(); dpm[m] = 0.0
        targets.update({"delta": delta, "delta_perp_massive": dpm, "e_massive": em})
        print(f"LN arms: massive coordinate {m}, delta weight on it {delta[m]**2 / (delta @ delta):.3f}, "
              f"|delta| {np.linalg.norm(delta):.3f} vs c {scale:.3f}", flush=True)
    crng = np.random.default_rng(1000 + args.seed)
    for t in [x for x in args.controls.split(",") if x]:
        v = np.asarray(targets[t], np.float64); v = v / np.linalg.norm(v)
        rho = float(v @ g_hat)
        control_pred[t] = {"cos_g_hat": rho, "pred_chi": ceiling * rho, "pred_chi_perp": 0.0}
        arms[t] = v
        vp = v - rho * g_hat; arms[f"{t}_perp"] = vp / np.linalg.norm(vp)
        for i in range(args.n_match):
            u = crng.standard_normal(d); u -= (u @ g_hat) * g_hat; u /= np.linalg.norm(u)
            arms[f"match_{t}_{i}"] = rho * g_hat + np.sqrt(1 - rho ** 2) * u

    srng = np.random.default_rng(2000 + args.seed)
    for i in range(args.n_shuffle):
        sgn = srng.choice([-1.0, 1.0], size=fit_G.shape[0])
        gs = (sgn[:, None] * fit_G).mean(0); gs_norm = float(np.linalg.norm(gs)); gs /= gs_norm
        rho = float(gs @ g_hat)
        control_pred[f"gshuf{i}"] = {"cos_g_hat": rho, "pred_chi": ceiling * rho, "shuf_norm": gs_norm,
                                     "shuf_norm_over_g_norm": gs_norm / g_norm}
        arms[f"gshuf{i}"] = gs

    tok, model = acts.get_model(args.model, DEV, torch.float16)
    base = steering.score_pairs(model, tok, eval_pairs, args.bs)
    out = {"model": args.model, "dataset": args.dataset, "layer": args.layer,
           "seed": args.seed, "mode": args.mode,
           "n_fit_pairs": int(fit_G.shape[0]), "n_eval_pairs": int(len(eval_pairs)),
           "alpha_unit": scale, "g_fit_norm": g_norm, "c_times_g_norm": ceiling,
           "tr_sigma_g_over_n": tr_sigma_over_n,
           "baseline_score": float(base.mean()), "null_scale": ceiling / np.sqrt(d),
           "controls": control_pred, "arms": {}}
    print(f"mode={args.mode} n_fit={fit_G.shape[0]} n_eval={len(eval_pairs)}  "
          f"c||g|| = {ceiling:.4f}  ||g_fit||^2 = {g_norm**2:.5f}  "
          f"trS/n = {tr_sigma_over_n:.5f}  baseline = {base.mean():+.4f}", flush=True)

    for name, vec in arms.items():
        rec = {"antisym": {}}
        with steering.Steerer(model, args.layer) as st:
            for a in alphas:
                st.set(vec, a * scale, DEV, torch.float16)
                sp = steering.score_pairs(model, tok, eval_pairs, args.bs)
                st.set(vec, -a * scale, DEV, torch.float16)
                sm = steering.score_pairs(model, tok, eval_pairs, args.bs)
                rec["antisym"][str(a)] = float((sp.mean() - sm.mean()) / 2)
            st.set(None, 0, DEV, torch.float16)
        rec["chi"] = chi_origin(alphas, [rec["antisym"][str(a)] for a in alphas])
        rec["chi_over_ceiling"] = rec["chi"] / ceiling if ceiling else float("nan")
        out["arms"][name] = rec
        rec["chi_small"] = rec["antisym"][str(alphas[0])] / alphas[0]
        if not name.startswith("rand"):
            print(f"  {name:14s} chi={rec['chi']:+.4f} chi_small={rec['chi_small']:+.4f}  "
                  f"chi/ceiling={rec['chi_over_ceiling']:+.4f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)

    rc = [out["arms"][k]["chi"] for k in out["arms"] if k.startswith("rand")]
    if rc:
        out["rand_chi_mean"] = float(np.mean(rc))
        out["rand_chi_p95"] = float(np.percentile(np.abs(rc), 95))
        print(f"\nrandom |chi| p95 = {out['rand_chi_p95']:.4f}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")


# =============================================================================
# noise-floor
# =============================================================================
matplotlib.use("Agg")


OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures")) / "truth_noise_floor.png"


SEEDS = range(10)


def run_noise_floor(argv=None):
    """The linear bound against the size of the gradient sample. For each g-arm seed
    at counterfact L28 the direction g_hat was fit on n in {25, 50, 125, 250} of
    the seed's pairs and steered on the same 150 held-out pairs. Under linear
    response the ratio chi(g_hat)/(c ||g_hat||) equals cos^2(g_hat, g) =
    ||g||^2 / (||g||^2 + tr Sigma_G / n), with ||g||^2 and tr Sigma_G read from
    the per-pair gradients, so the curve has no free parameter.

    Reads artifacts/steer_g_arm_counterfact_L28_comp{_n{n}}_s{s}.json
    Writes artifacts/noise_floor_curve.json and $FIGURE_DIR/truth_noise_floor.png
    """
    ap = argparse.ArgumentParser(description=run_noise_floor.__doc__)
    ap.add_argument("--slope", action="store_true",
                    help="plot chi from the alpha=0.5 slope instead of the through-origin grid fit (the grid fit "
                         "under-reports chi where alpha=4 leaves the linear window; see `linear_bound.py noise-floor-small`). "
                         "The JSON summary keeps the grid-fit numbers. The post's truth_noise_floor.png uses --slope.")
    SLOPE = ap.parse_args(argv).slope
    KEY = "ratio_slope" if SLOPE else "ratio"
    rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.6, "grid.alpha": 0.55, "font.size": 11,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
    rows = {}
    for s in SEEDS:
        full = json.load(open(ART / f"steer_g_arm_counterfact_L28_comp_s{s}.json"))
        g2, trn = full["g_fit_norm"] ** 2, full["tr_sigma_g_over_n"]
        g_true2, trS = g2 - trn, trn * full["n_fit_pairs"]
        rows[s] = {"g_true2": g_true2, "trS": trS, "cells": {}}
        for n in NS:
            p = ART / (f"steer_g_arm_counterfact_L28_comp_s{s}.json" if n == 250 else f"steer_g_arm_counterfact_L28_comp_n{n}_s{s}.json")
            if not p.exists():
                continue
            d = json.load(open(p))
            ratio = d["arms"]["g"]["chi"] / d["c_times_g_norm"]
            pred = noise_floor_ratio(full, n)
            rows[s]["cells"][n] = {"ratio": ratio, "pred": pred, "ratio_true": d["arms"]["g"]["chi"] / (d["alpha_unit"] * np.sqrt(g_true2)),
                                   "ratio_slope": d["arms"]["g"]["antisym"]["0.5"] / 0.5 / d["c_times_g_norm"]}

    summary = {}
    print("n    measured mean (SE)   predicted mean   measured/predicted (SE)   k")
    for n in NS:
        m = np.array([r["cells"][n]["ratio"] for r in rows.values() if n in r["cells"]])
        p = np.array([r["cells"][n]["pred"] for r in rows.values() if n in r["cells"]])
        if len(m) == 0:
            continue
        q = m / p
        summary[str(n)] = {"k": int(len(m)), "measured_mean": float(m.mean()), "measured_se": float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else None,
                           "predicted_mean": float(p.mean()), "ratio_to_pred_mean": float(q.mean()),
                           "ratio_to_pred_se": float(q.std(ddof=1) / np.sqrt(len(q))) if len(q) > 1 else None}
        print(f"{n:<4d} {m.mean():.3f} ({summary[str(n)]['measured_se'] or 0:.3f})       {p.mean():.3f}            {q.mean():.3f} ({summary[str(n)]['ratio_to_pred_se'] or 0:.3f})        {len(m)}")
    json.dump({"seeds": {str(s): {"g_true2": r["g_true2"], "trS": r["trS"], "cells": {str(n): c for n, c in r["cells"].items()}} for s, r in rows.items()},
               "summary": summary}, open(ART / "noise_floor_curve.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    nn = np.logspace(np.log10(15), np.log10(400), 100)
    for s, r in rows.items():
        ax.plot(nn, r["g_true2"] / (r["g_true2"] + r["trS"] / nn), color="#c0392b", lw=0.8, alpha=0.35)
        ns = sorted(r["cells"]); ax.plot(ns, [r["cells"][n][KEY] for n in ns], "o-", color="#2c3e50", ms=3.5, lw=0.8, alpha=0.55)
    g2m = np.mean([r["g_true2"] for r in rows.values()]); trm = np.mean([r["trS"] for r in rows.values()])
    ax.plot(nn, g2m / (g2m + trm / nn), color="#c0392b", lw=2, label=r"prediction $\|g\|^2/(\|g\|^2 + \mathrm{tr}\,\Sigma_G/n)$")
    ax.plot([], [], "o-", color="#2c3e50", label=r"measured $\chi(\hat g)/(c\|\hat g\|)$, ten seeds" + (r" ($\alpha=0.5$ slope)" if SLOPE else ""))
    ax.set_xscale("log"); ax.set_xlabel(r"$n$ (pairs used to estimate $g$)"); ax.set_ylabel("fraction of the linear bound achieved")
    ax.set_ylim(0, 1); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title(r"`counterfact`, layer 28: the bound's slack is the estimate of $g$")
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


# =============================================================================
# noise-floor-small
# =============================================================================
def cell(path):
    d = json.load(open(path))
    return d, response_slopes(d["arms"]["g"])


def run_noise_floor_small(argv=None):
    """Same as `linear_bound.py noise-floor` but the measured susceptibility is the small-alpha
    slope A(0.5)/0.5 (and A(1)/1 for comparison) rather than the through-origin fit
    over alpha in {0.5,1,2,4}. Where the linear window closes below alpha=4 the
    through-origin fit under-reports chi; the small-alpha slope is the quantity the
    identity actually bounds. Adds the per-alpha linearity check A(alpha)/alpha.

    Writes artifacts/noise_floor_curve_smallalpha.json
    """
    argparse.ArgumentParser(description=run_noise_floor_small.__doc__).parse_args(argv)
    rows = {}
    for s in range(10):
        full, sl = cell(ART / f"steer_g_arm_counterfact_L28_comp_s{s}.json")
        rows[s] = {"lin_check_n250": sl, "cells": {}}
        for n in NS:
            p = ART / (f"steer_g_arm_counterfact_L28_comp_s{s}.json" if n == 250 else f"steer_g_arm_counterfact_L28_comp_n{n}_s{s}.json")
            d, sl = cell(p)
            pred = noise_floor_ratio(full, n)
            rows[s]["cells"][n] = {"pred": pred, "fit": d["arms"]["g"]["chi"] / d["c_times_g_norm"],
                                   "a05": sl[0.5] / d["c_times_g_norm"], "a1": sl[1.0] / d["c_times_g_norm"],
                                   "a2": sl[2.0] / d["c_times_g_norm"], "a4": sl[4.0] / d["c_times_g_norm"]}
    print("linearity at n=250, A(alpha)/alpha relative to A(0.5)/0.5, ten seeds (mean):")
    for a in (1.0, 2.0, 4.0):
        print(f"  alpha {a}: {np.mean([r['lin_check_n250'][a] / r['lin_check_n250'][0.5] for r in rows.values()]):.3f}")
    print("\nn     pred    fit/pred (SE)     a0.5/pred (SE)     a1/pred (SE)     a2/pred     a4/pred")
    summary = {}
    for n in NS:
        P = np.array([rows[s]["cells"][n]["pred"] for s in rows])
        line = f"{n:<5d} {P.mean():.3f}"
        summary[str(n)] = {"pred": float(P.mean())}
        for key in ("fit", "a05", "a1", "a2", "a4"):
            Q = np.array([rows[s]["cells"][n][key] for s in rows]) / P
            summary[str(n)][key] = {"ratio_to_pred": float(Q.mean()), "se": float(Q.std(ddof=1) / np.sqrt(len(Q)))}
            line += f"   {Q.mean():.3f} ({Q.std(ddof=1)/np.sqrt(len(Q)):.3f})"
        print(line)
    json.dump({"seeds": {str(s): {"lin_check_n250": {str(k): v for k, v in r["lin_check_n250"].items()},
                                  "cells": {str(n): c for n, c in r["cells"].items()}} for s, r in rows.items()},
               "summary": summary}, open(ART / "noise_floor_curve_smallalpha.json", "w"), indent=1)
    print("wrote", ART / "noise_floor_curve_smallalpha.json")


# =============================================================================
# fd-check
# =============================================================================
def run_fd_check(argv=None):
    """Finite-difference check of the truth-score gradient (the counterpart of
    `refusal_corner.py fd-check` for Pythia). For a few pairs and directions w:

      fd_i   = (ell_i(+h w) - ell_i(-h w)) / 2h, with h w added to the residual at
               every position of block L (truthlib.steering.Steerer, as in steering)
      pred_i = G_i . w, G_i the per-pair gradient exactly as `rogue_dimension.py
               gradient` builds it (grad_for_texts on the true and false completions,
               difference, summed over positions)

    Agreement at small h validates the position-sum convention and the sign. The
    fp16 loss scaling is checked against the same gradients computed in float32:
    the relative error and the fraction of entries that underflow to exactly zero.
    """
    ap = argparse.ArgumentParser(description=run_fd_check.__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dataset", default="counterfact_true_false")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n", type=int, default=8, help="pairs")
    ap.add_argument("--h", type=float, default=0.05, help="push in residual units")
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args(argv)

    pairs = steering.pairs_for_seed(data.load_pairs(args.dataset, 400, 0), 0, args.n)

    def load(dtype):
        tok, model = acts.get_model(args.model, steering.DEV, dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        return tok, model

    # fp16, with LOSS_SCALE: the gradients every experiment used
    tok, model = load(torch.float16)
    G16 = steering.pair_gradients(model, tok, model.gpt_neox.layers[args.layer], pairs, args.bs)
    del model
    # fp32: the reference, and the model the finite difference runs on (an fp16
    # residual cannot resolve a push of h; in fp32 the difference is clean)
    tok, model = load(torch.float32)
    G32 = steering.pair_gradients(model, tok, model.gpt_neox.layers[args.layer], pairs, args.bs)
    d = G32.shape[1]
    rng = np.random.default_rng(0)
    dirs = {"g_hat": G32.mean(0) / np.linalg.norm(G32.mean(0)), "rand": rng.standard_normal(d)}
    dirs["rand"] /= np.linalg.norm(dirs["rand"])

    print(f"{args.model} {args.dataset} L{args.layer}: {len(pairs)} pairs, h = {args.h}, LOSS_SCALE = {steering.LOSS_SCALE}")
    with steering.Steerer(model, args.layer) as st:
        for name, w in dirs.items():
            st.set(w, +args.h, steering.DEV, torch.float32)
            sp = steering.score_pairs(model, tok, pairs, args.bs)
            st.set(w, -args.h, steering.DEV, torch.float32)
            sm = steering.score_pairs(model, tok, pairs, args.bs)
            st.set(None, 0, steering.DEV, torch.float32)
            fd = (sp - sm) / (2 * args.h)
            for label, G in (("fp32", G32), ("fp16", G16)):
                pred = G @ w
                rel = np.abs(fd - pred) / (np.abs(pred) + 1e-6)
                print(f"  {name:6s} fd vs G{label}.w: median rel err {np.median(rel):.4f}, max {rel.max():.4f}; "
                      f"means fd {fd.mean():+.4f} G.w {pred.mean():+.4f}")
    rel = np.linalg.norm(G16 - G32, axis=1) / np.linalg.norm(G32, axis=1)
    cosg = float(G16.mean(0) @ G32.mean(0) / np.linalg.norm(G16.mean(0)) / np.linalg.norm(G32.mean(0)))
    print(f"  fp16 vs fp32 per-pair gradient: relative error median {np.median(rel):.4f}, max {rel.max():.4f}; "
          f"cos(g_hat16, g_hat32) {cosg:.6f}; exact zeros fp16 {np.mean(G16 == 0):.5f} fp32 {np.mean(G32 == 0):.5f}")


COMMANDS = {
    "g-arm": run_g_arm,
    "noise-floor": run_noise_floor,
    "noise-floor-small": run_noise_floor_small,
    "fd-check": run_fd_check,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
