"""
Depth and dataset profile: the bound and the linear window across layers.

The section "Depth and dataset profile" of part 2, "Steering Vectors and the Limits of Linear Response".
The g-arm against its noise-floor prediction at every swept layer, the noise
floor by depth, the seven-point window curves (and cities seeds, refusal
P(refusal)), and the LayerNorm and relative-push geometry of the window.

Subcommands:

    arms                   Per-layer complement arm (seed 0, both datasets) against the ...
    noise-floor            Noise floor of the gradient estimate by depth
    windows                Reads the overnight 13/14 Sept artifacts (rl-align 2b8e72a) and ...
    layernorm-window       Why the linear window closes where it does (draft E14, E16, P5)

Run from the repo root:  python scripts/depth_profile.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib.steering import noise_floor_ratio, response_slopes

# ---- shared by several subcommands ------------------------------------------
ART = REPO / "artifacts"


# =============================================================================
# arms
# =============================================================================
ARMS_R = []


def run_arms(argv=None):
    """Per-layer complement arm (seed 0, both datasets) against the noise-floor
    prediction, using the small-alpha slope A(0.5)/0.5 and the through-origin fit,
    plus the linear-window check A(alpha)/alpha relative to alpha = 0.5. Also the
    cities L28 seeds 0-4.

    Writes artifacts/depth_arms_analysis.json
    """
    argparse.ArgumentParser(description=run_arms.__doc__).parse_args(argv)
    depth = json.load(open(ART / "noise_floor_depth.json"))["datasets"]
    out = {"layers": {}, "cities_seeds": {}}
    print("ds          L   pred   a0.5/pred  fit/pred | A/a rel to 0.5 @1,2,4 | g/p95  plain   p95")
    for short in ["counterfact", "cities"]:
        out["layers"][short] = {}
        for L in [8, 12, 16, 20, 24, 28]:
            d = json.load(open(ART / f"steer_g_arm_{short}_L{L}_comp_s0.json"))
            a = d["arms"]["g"]["antisym"]; cg = d["c_times_g_norm"]
            sl = response_slopes(d["arms"]["g"])
            pred = depth[short][str(L)]["cos2_pred_n250"]
            rel = [sl[k] / sl[0.5] for k in (1.0, 2.0, 4.0)]
            rec = {"pred": pred, "a05_over_pred": sl[0.5] / cg / pred, "fit_over_pred": d["arms"]["g"]["chi"] / cg / pred,
                   "lin_rel": rel, "g_over_p95": d["arms"]["g"]["chi"] / d["rand_chi_p95"],
                   "plain_chi": d["arms"]["plain"]["chi"], "p95": d["rand_chi_p95"], "c_g_hat": cg}
            out["layers"][short][str(L)] = rec
            print(f"{short:11s} {L:<3d} {pred:.3f}  {rec['a05_over_pred']:.3f}      {rec['fit_over_pred']:.3f}    | "
                  f"{rel[0]:.2f} {rel[1]:.2f} {rel[2]:.2f}      | {rec['g_over_p95']:5.1f}  {rec['plain_chi']:+.3f}  {rec['p95']:.3f}")
    print("\ncities L28: seed  pred   a0.5/pred  fit/pred  noise  plain   p95")
    for s in range(5):
        d = json.load(open(ART / f"steer_g_arm_cities_L28_comp_s{s}.json"))
        g2, trn = d["g_fit_norm"] ** 2, d["tr_sigma_g_over_n"]; pred = noise_floor_ratio(d)
        a = d["arms"]["g"]["antisym"]; cg = d["c_times_g_norm"]
        rec = {"pred": pred, "a05_over_pred": a["0.5"] / 0.5 / cg / pred, "fit_over_pred": d["arms"]["g"]["chi"] / cg / pred,
               "noise_share": trn / g2, "plain_chi": d["arms"]["plain"]["chi"], "p95": d["rand_chi_p95"]}
        out["cities_seeds"][str(s)] = rec; ARMS_R.append(rec)
        print(f"            {s}     {pred:.3f}  {rec['a05_over_pred']:.3f}      {rec['fit_over_pred']:.3f}     {rec['noise_share']:.3f}  {rec['plain_chi']:+.3f}  {rec['p95']:.3f}")
    q = np.array([r["a05_over_pred"] for r in ARMS_R]); f = np.array([r["fit_over_pred"] for r in ARMS_R])
    print(f"five seeds: a0.5/pred {q.mean():.3f} (SE {q.std(ddof=1)/np.sqrt(5):.3f}), fit/pred {f.mean():.3f} (SE {f.std(ddof=1)/np.sqrt(5):.3f})")
    json.dump(out, open(ART / "depth_arms_analysis.json", "w"), indent=1)


# =============================================================================
# noise-floor
# =============================================================================
NFD_LAYERS = [8, 12, 16, 20, 24, 28]


def run_noise_floor(argv=None):
    """Noise floor of the gradient estimate by depth. From the per-pair gradients in
    score_gradient_{dataset}_vectors.npz (seed 0, n = 250 pairs; written by
    `rogue_dimension.py gradient`), at each swept
    layer: ||g_hat||^2, tr(Sigma_G)/n, the noise share, the noise-corrected
    ||g||, and the linear-response prediction for the complement-arm ratio,
    cos^2 = ||g||^2 / (||g||^2 + tr Sigma_G / n), at n = 250 and at n = 125.

    Writes artifacts/noise_floor_depth.json. No GPU.
    """
    argparse.ArgumentParser(description=run_noise_floor.__doc__).parse_args(argv)
    out = {"note": "seed-0 npz, n=250 pairs; cos2 = predicted chi(g_hat)/(c||g_hat||) under linear response", "datasets": {}}
    for short in ["counterfact", "cities"]:
        z = np.load(ART / f"score_gradient_{short}_vectors.npz")
        out["datasets"][short] = {}
        print(f"{short}: L   ||g_hat||^2   trS/n    share   ||g||    c    c||g_hat||  c||g||   cos2(250)  cos2(125)")
        for L in NFD_LAYERS:
            G = z[f"L{L}_g_per_pair"].astype(np.float64); n = G.shape[0]
            g = G.mean(0); g2 = float(g @ g); trS = float(G.var(0, ddof=1).sum())
            c = float(z[f"L{L}_scale"][0])
            gt2 = g2 - trS / n
            rec = {"n": n, "g_hat2": g2, "trS_over_n": trS / n, "noise_share": trS / n / g2,
                   "g_true": float(np.sqrt(max(gt2, 0))), "c": c, "c_g_hat": c * np.sqrt(g2),
                   "c_g_true": c * float(np.sqrt(max(gt2, 0))),
                   "cos2_pred_n250": gt2 / (gt2 + trS / 250), "cos2_pred_n125": gt2 / (gt2 + trS / 125)}
            out["datasets"][short][str(L)] = rec
            print(f"          {L:<3d} {g2:.5f}     {trS/n:.5f}  {rec['noise_share']:.3f}   {rec['g_true']:.4f}  {c:.2f}  {rec['c_g_hat']:.3f}      {rec['c_g_true']:.3f}    {rec['cos2_pred_n250']:.3f}      {rec['cos2_pred_n125']:.3f}")
    json.dump(out, open(ART / "noise_floor_depth.json", "w"), indent=1)
    print("wrote", ART / "noise_floor_depth.json")


# =============================================================================
# windows
# =============================================================================
ALPHAS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


WIN_LAYERS = {"counterfact": [8, 12, 16, 20, 24, 28], "cities": [8, 12, 16, 20, 24, 28]}


SEEDS = {"counterfact": [0, 4, 8], "cities": [0]}   # 25 Sept: seeds 4, 8 and cities L12-L24 added


def load(p):
    return json.load(open(ART / p))


def curve(d, arm="g"):
    sl = response_slopes(d["arms"][arm])
    assert list(sl) == ALPHAS, "not a fine-grid run"
    return list(sl.values())


WIN_R = {"alphas": ALPHAS, "window": {}, "cities_seeds": {}, "refusal_P": {}}


def fmt(v):
    return ", ".join(f"{x:.2f}" for x in v)


def run_windows(argv=None):
    """Reads the overnight 13/14 Sept artifacts (rl-align 2b8e72a) and writes
    artifacts/overnight_analysis.json plus the markdown tables the draft quotes.

    A. Seven-point window curves along g_hat, alpha in {1/16 .. 4}, seed 0, complement
       split: A(alpha)/alpha relative to its alpha = 1/16 value, per layer, both datasets
       (steer_g_arm_{ds}_L{L}_comp_fine_s0.json). Also the bias of the alpha = 0.5 slope
       used everywhere else: A(0.5)/0.5 relative to A(1/16)/(1/16).
    B. cities L28 complement seeds 5 and 6: g_hat slope over the noise-floor formula, the
       plain (theta_hat) arm against its null (steer_g_arm_cities_L28_comp_s{5,6}.json).
    C. Refusal P(refusal) along r_hat and g_hat at L12/L15 seed 0
       (refusal_steer_arm_L{12,15}_s0_P.json): the three regimes.

    No GPU.
    """
    argparse.ArgumentParser(description=run_windows.__doc__).parse_args(argv)
    # ---- A. window curves
    for ds, Ls in WIN_LAYERS.items():
        WIN_R["window"][ds] = {}
        for s in SEEDS[ds]:
          for L in Ls:
            p = ART / f"steer_g_arm_{ds}_L{L}_comp_fine_s{s}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            cv = curve(d)
            ref = cv[0] if ds == "counterfact" else cv[2]  # cities: alpha=1/16 is below fp16 resolution
            WIN_R["window"][ds][f"{L}" if s == 0 else f"{L}_s{s}"] = {
                "A_over_alpha": cv, "rel": [x / ref for x in cv], "ref_alpha": ALPHAS[0] if ds == "counterfact" else ALPHAS[2],
                "c": d["alpha_unit"], "c_times_g_norm": d["c_times_g_norm"],
                "slope_1_16_over_bound": cv[0] / d["c_times_g_norm"], "slope_1_4_over_bound": cv[2] / d["c_times_g_norm"],
                "slope_0_5_over_bound": cv[3] / d["c_times_g_norm"], "formula_pred": noise_floor_ratio(d),
                "bias_0_5_vs_ref": cv[3] / ref, "null_scale": d["null_scale"], "push_1_16_units": d["alpha_unit"] / 16,
            }

    # ---- B. cities seeds 5, 6
    for s in (5, 6):
        d = load(f"steer_g_arm_cities_L28_comp_s{s}.json")
        a = d["arms"]["g"]["antisym"]
        sl = a["0.5"] / 0.5
        pl = d["arms"]["plain"]["antisym"]["0.5"] / 0.5
        WIN_R["cities_seeds"][str(s)] = {
            "slope": sl, "ratio": sl / d["c_times_g_norm"], "formula_pred": noise_floor_ratio(d),
            "slope_over_formula": sl / d["c_times_g_norm"] / noise_floor_ratio(d), "c": d["alpha_unit"],
            "plain_chi": pl, "null_p95": d["rand_chi_p95"], "lin_rel": [a[str(x)] / x / sl for x in (1.0, 2.0, 4.0)],
        }

    # ---- C. refusal probabilities
    for L in (12, 15):
        d = load(f"refusal_steer_arm_L{L}_s0_P.json")
        out = {"c": d["alpha_unit"], "baseline_P_mean": d["baseline_P_mean"]}
        for arm in ("r", "g"):
            x = d["arms"][arm]
            out[arm] = {k: [x[k][str(a)] for a in ALPHAS] for k in ("A", "P_plus", "P_minus", "P_plus_harmless", "P_minus_harmful")}
            out[arm]["A_over_alpha"] = x["A_over_alpha"]
            out[arm]["rel"] = [v / x["A_over_alpha"][0] for v in x["A_over_alpha"]]
        WIN_R["refusal_P"][str(L)] = out

    json.dump(WIN_R, open(ART / "overnight_analysis.json", "w"), indent=1)


    print("### A. window along g_hat, A(alpha)/alpha relative to the reference alpha (cf: 1/16; cities: 1/4)")
    print("| layer | rel. A(alpha)/alpha over alpha = 1/16 ... 4 | A(0.5)/0.5 rel. ref | slope(ref)/bound | formula | ratio |")
    print("|---|---|---|---|---|---|")
    for ds in WIN_LAYERS:
        for L, w in WIN_R["window"][ds].items():
            key = "slope_1_16_over_bound" if ds == "counterfact" else "slope_1_4_over_bound"
            print(f"| `{ds}` L{L} | {fmt(w['rel'])} | {w['bias_0_5_vs_ref']:.2f} | {w[key]:.3f} | {w['formula_pred']:.3f} | {w[key]/w['formula_pred']:.2f} |")
    print("\n(cities L8: c = %.2f, push at alpha = 1/16 is %.3f residual units; null scale %.3f; A/alpha raw: %s)" % (
        WIN_R["window"]["cities"]["8"]["c"], WIN_R["window"]["cities"]["8"]["push_1_16_units"], WIN_R["window"]["cities"]["8"]["null_scale"],
        fmt(WIN_R["window"]["cities"]["8"]["A_over_alpha"])))

    print("\n### B. cities L28 complement, seeds 5 and 6")
    print("| seed | c | slope | ratio | formula | slope/formula | plain chi | null p95 | A(1,2,4)/alpha rel. 0.5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s, v in WIN_R["cities_seeds"].items():
        print(f"| {s} | {v['c']:.2f} | {v['slope']:+.4f} | {v['ratio']:.3f} | {v['formula_pred']:.3f} | {v['slope_over_formula']:.2f} | {v['plain_chi']:+.4f} | {v['null_p95']:.4f} | {fmt(v['lin_rel'])} |")

    print("\n### C. refusal P(refusal) along r_hat and g_hat, seed 0")
    for L, v in WIN_R["refusal_P"].items():
        print(f"\nL{L}, c = {v['c']:.1f}, baseline mean P = {v['baseline_P_mean']:.2f}")
        print("| arm | alpha | A/alpha (log-odds) | rel. | P+ harmless | P- harmful |")
        print("|---|---|---|---|---|---|")
        for arm in ("r", "g"):
            x = v[arm]
            for i, a in enumerate(ALPHAS):
                print(f"| {arm} | {a:g} | {x['A_over_alpha'][i]:.2f} | {x['rel'][i]:.2f} | {x['P_plus_harmless'][i]:.2f} | {x['P_minus_harmful'][i]:.3f} |")
    print("\nwrote", ART / "overnight_analysis.json")


# =============================================================================
# layernorm-window
# =============================================================================
LNW_LAYERS = (8, 12, 16, 20, 24, 28)


REFUSAL_LAYERS = (9, 12, 15, 18)


CACHE_TAG = {"counterfact": "counterfact_true_false", "cities": "cities"}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def mean_abs_cos(X: np.ndarray, u: np.ndarray) -> float:
    """mean_i |cos(x_i, u)|, u a unit vector (row by row, as on 13 Sept)."""
    return float(np.mean([abs(float(x @ u) / float(np.linalg.norm(x))) for x in X]))


def pythia_cell(ds: str, L: int) -> tuple[dict, dict]:
    A = np.load(ART / "act_cache" / f"pythia-2.8b__{CACHE_TAG[ds]}.npz")
    X = A[f"L{L}"].astype(np.float64)
    y = A["y"]
    g = np.load(ART / f"score_gradient_{ds}_vectors.npz")[f"L{L}_g"].astype(np.float64)
    run = json.load(open(ART / f"steer_g_arm_{ds}_L{L}_comp_s0.json"))
    c = run["alpha_unit"]
    av = {a: run["arms"]["g"]["antisym"][str(a)] / a for a in (0.5, 1.0, 2.0, 4.0)}

    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    xbar = X.mean(0)
    xn = np.linalg.norm(X, axis=1)
    m = int(np.argmax(np.abs(xbar)))                 # the massive coordinate
    Xm = X.copy(); Xm[:, m] = 0.0
    gu = unit(g)
    push4_over_x = 4 * c / float(xn.mean())
    mac_g = mean_abs_cos(X, gu)

    geom = {"ds": ds, "L": L, "c": c,
            "delta_norm": float(np.linalg.norm(delta)),
            "mean_x_norm": float(xn.mean()),
            "push4_over_x": push4_over_x,
            "cos_d_xbar": float(unit(delta) @ xbar / np.linalg.norm(xbar)),
            "mean_abs_cos_d_x": mean_abs_cos(X, unit(delta)),
            "cos_g_xbar": float(gu @ xbar / np.linalg.norm(xbar)),
            "mean_abs_cos_g_x": mac_g,
            "massive_idx": m,
            "delta_frac_massive": float((delta[m] / np.linalg.norm(delta)) ** 2),
            "xbar_frac_massive": float(xbar[m] ** 2 / (xbar @ xbar)),
            "window_A4rel": round(av[4.0] / av[0.5], 2)}
    window = {"ds": ds, "L": L, "c": c,
              "push4_over_xnorm": push4_over_x,
              "push4_over_xnorm_nomassive": 4 * c / float(np.linalg.norm(Xm, axis=1).mean()),
              "push4_over_sd_xg": 4 * c / float((X @ gu).std()),
              "mean_abs_cos_g_x": mac_g,
              "w2": av[2.0] / av[0.5], "w4": av[4.0] / av[0.5]}
    return geom, window


def refusal_cell(z, L: int) -> dict:
    X = z[f"L{L}_X_last"].astype(np.float64)
    gu = unit(z[f"L{L}_g"].astype(np.float64))
    c = float(z[f"L{L}_scale"][0])
    run = json.load(open(ART / f"refusal_steer_arm_L{L}_s0.json"))
    al = run["alphas"]
    A = run["arms"]["g"]["A"]
    av = {a: (A[str(a)] if isinstance(A, dict) else A[al.index(a)]) / a for a in al}
    push4 = 4 * c / float(np.linalg.norm(X, axis=1).mean())
    return {"ds": "refusal", "L": L, "c": c,
            "push4_over_xnorm": push4,
            "push4_over_xnorm_nomassive": push4,      # Qwen has no dominant coordinate
            "push4_over_sd_xg": 4 * c / float((X @ gu).std()),
            "mean_abs_cos_g_x": mean_abs_cos(X, gu),
            "w2": av[0.5] / av[0.0625], "w4": av[1.0] / av[0.0625]}


def run_layernorm_window(argv=None):
    """Why the linear window closes where it does (draft E14, E16, P5).

      layernorm_geometry.json   Pythia-2.8b, counterfact and cities, L8-L28: the push
          4c against the residual norm, cos(delta_hat, x) and cos(g_hat, x) (mean and
          per statement), and the weight of the massive coordinate in delta_hat and in
          the mean residual, next to the measured window A(4)/4 relative to alpha = 0.5.
          On counterfact delta_hat IS the massive coordinate at L8-L20, which prompted
          the LayerNorm hypothesis (notes/layernorm_arms_prediction.md).
      window_vs_push.json       the same cells plus refusal (Qwen1.5-1.8B-Chat,
          L9-L18): relative push 4c/||x|| with and without the massive coordinate,
          4c against the spread of x . g_hat, and the window ratios w2, w4.

    Window ratios. On Pythia, w2 and w4 are A(alpha)/alpha at alpha = 2 and 4 relative
    to alpha = 0.5 (pushes 4x and 8x the reference). On refusal they are at alpha =
    0.5 and 1 relative to alpha = 1/16 (8x and 16x), as computed on 13 Sept: the two
    models' columns are not the same multiple of the reference push.
    [Flagged for Julia, 1 Oct: E16 compares windows across models.]

    Inputs: the act cache and score_gradient_{ds}_vectors.npz (L*_g, from
    `rogue_dimension.py gradient` (part 1's section file)) for Pythia, the
    seed-0 complement runs steer_g_arm_{ds}_L*_comp_s0.json; for refusal
    refusal_gradient_vectors.npz (L*_X_last, L*_g, L*_scale) and
    refusal_steer_arm_L*_s0.json. No GPU.

    Regenerates both committed files byte for byte.
    """
    argparse.ArgumentParser(description=run_layernorm_window.__doc__).parse_args(argv)
    geometry, windows = [], []
    for ds in ("counterfact", "cities"):
        for L in LNW_LAYERS:
            geom, win = pythia_cell(ds, L)
            geometry.append(geom); windows.append(win)
            print(f"{ds:12s} L{L:<2d} 4c/|x| {geom['push4_over_x']:.3f}  |cos(d,x)| {geom['mean_abs_cos_d_x']:.3f}  "
                  f"massive {geom['massive_idx']} (delta {geom['delta_frac_massive']:.2f})  A(4)/4 rel {geom['window_A4rel']:.2f}")
    z = np.load(ART / "refusal_gradient_vectors.npz")
    for L in REFUSAL_LAYERS:
        win = refusal_cell(z, L)
        windows.append(win)
        print(f"{'refusal':12s} L{L:<2d} 4c/|x| {win['push4_over_xnorm']:.3f}  w2 {win['w2']:.2f}  w4 {win['w4']:.2f}")
    for name, obj in (("layernorm_geometry", geometry), ("window_vs_push", windows)):
        with open(ART / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=1)
        print(f"wrote artifacts/{name}.json")


COMMANDS = {
    "arms": run_arms,
    "noise-floor": run_noise_floor,
    "windows": run_windows,
    "layernorm-window": run_layernorm_window,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
