"""
Steering controls and the exact response.

The section "Steering controls and the exact response" of part 2, "Steering Vectors and the Limits of Linear Response".
The exact linear response chi(w) = c (w . g_eval) over every steered arm, the
projected-out / matched / shuffled controls and the refusal summary
(controls_analysis.json); and where the projected-out residual e2_perp lives.

Subcommands:

    controls               Assembles the steering-control tables for part 2 from the ...

Run from the repo root:  python scripts/steering_controls.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib.steering import (
    complement_arms,
    noise_floor_ratio,
    pool_gradients,
    response_slopes,
    small_alpha_slope,
)

# ---- shared by several subcommands ------------------------------------------
ART = REPO / "artifacts"


# =============================================================================
# controls
# =============================================================================
CTRL_POOL = 400


def load(p):
    return json.load(open(p))


def run_controls(argv=None):
    """Assembles the steering-control tables for part 2 from the artifacts, no GPU.

    1. Exact linear response at L28 (both datasets): for every arm in every complement
       run (g_hat, plain, rand*, controls, matches, shuffles), reconstruct the unit
       direction w from the recorded seeds and compare the measured alpha=0.5 slope
       with c (w . g_eval), g_eval = mean per-pair gradient over the pairs the seed
       did not draw (pool gradients assembled from the per-seed npz files).
    2. Control tables per target (perp / match), per seed, with the a-priori prediction
       c||g_hat|| cos(t, g_hat), the exact prediction c (t . g_eval), the null p95.
    3. Shuffle table.
    4. Refusal summary over layers and prompt seeds.
    5. counterfact depth cells over seeds.

    Writes artifacts/controls_analysis.json and prints markdown tables.
    """
    argparse.ArgumentParser(description=run_controls.__doc__).parse_args(argv)
    R = {"exact_linear_response": {}, "controls": {}, "shuffle": {}, "refusal": {}, "depth": {}}
    for ds in ("counterfact", "cities"):
        seeds = range(10) if ds == "counterfact" else range(5)
        Gp, chosen, cov = pool_gradients(ds, seeds)
        R["exact_linear_response"][ds] = {"pool_coverage": cov, "arms": []}
        ctrl_rows, shuf_rows = [], []
        for s in seeds:
            if s not in chosen:
                continue
            ev = [i for i in range(CTRL_POOL) if i not in chosen[s] and not np.isnan(Gp[i, 0])]
            g_eval = Gp[ev].mean(0)
            comp = ART / f"steer_g_arm_{ds}_L28_comp_s{s}.json"
            if not comp.exists():
                continue
            base = load(comp)
            c = base["alpha_unit"]; bound = base["c_times_g_norm"]
            p95 = [v for k, v in base.items() if "p95" in k][0]
            for kind in ("comp", "comp_ctrl", "comp_shuf"):
                p = ART / f"steer_g_arm_{ds}_L28_{kind}_s{s}.json"
                if not p.exists():
                    continue
                d = load(p); W = complement_arms(ds, s, d)
                for name, rec in d["arms"].items():
                    if name not in W:
                        continue
                    R["exact_linear_response"][ds]["arms"].append(
                        {"seed": s, "arm": name, "measured": small_alpha_slope(rec), "exact": c * float(W[name] @ g_eval)})
                if kind == "comp_ctrl":
                    for t, cp in d["controls"].items():
                        m = [small_alpha_slope(d["arms"][k]) for k in d["arms"] if k.startswith(f"match_{t}_")]
                        if f"{t}_perp" not in d["arms"] or len(m) < 2:
                            continue  # run still in progress
                        ctrl_rows.append({"seed": s, "target": t, "cos_g_hat": cp["cos_g_hat"], "pred_chi": cp["pred_chi"],
                                          "exact_chi": c * float(W[t] @ g_eval), "measured_t": small_alpha_slope(d["arms"][t]),
                                          "measured_perp": small_alpha_slope(d["arms"][f"{t}_perp"]),
                                          "exact_perp": c * float(W[f"{t}_perp"] @ g_eval),
                                          "match_mean": float(np.mean(m)), "match_sd": float(np.std(m, ddof=1)) if len(m) > 1 else None,
                                          "null_p95": p95, "c_g_norm": bound})
                if kind == "comp_shuf":
                    for i, (k, cp) in enumerate(d["controls"].items()):
                        if k not in d["arms"]:
                            continue
                        shuf_rows.append({"seed": s, "arm": k, "measured": small_alpha_slope(d["arms"][k]), "pred_in_sample": cp["pred_chi"],
                                          "exact": c * float(W[k] @ g_eval), "norm_over_g": cp["shuf_norm_over_g_norm"], "null_p95": p95})
        A = R["exact_linear_response"][ds]["arms"]
        if A:
            m = np.array([a["measured"] for a in A]); e = np.array([a["exact"] for a in A])
            R["exact_linear_response"][ds].update({"n_arms": len(A), "slope_through_origin": float((m * e).sum() / (e * e).sum()),
                                                   "corr": float(np.corrcoef(m, e)[0, 1]),
                                                   "rms_resid": float(np.sqrt(np.mean((m - e) ** 2))), "rms_measured": float(np.sqrt(np.mean(m ** 2)))})
        R["controls"][ds] = ctrl_rows; R["shuffle"][ds] = shuf_rows
        if shuf_rows:
            m = np.array([r["measured"] for r in shuf_rows]); e = np.array([r["exact"] for r in shuf_rows]); b = np.array([r["pred_in_sample"] for r in shuf_rows])
            R["shuffle"][ds + "_summary"] = {"corr_exact": float(np.corrcoef(m, e)[0, 1]), "slope_exact": float((m * e).sum() / (e * e).sum()),
                                             "corr_in_sample": float(np.corrcoef(m, b)[0, 1]), "slope_in_sample": float((m * b).sum() / (b * b).sum())}
        if ctrl_rows:
            for t in sorted({r["target"] for r in ctrl_rows}):
                rr = [r for r in ctrl_rows if r["target"] == t]
                m = np.array([r["measured_t"] for r in rr]); p = np.array([r["pred_chi"] for r in rr]); e = np.array([r["exact_chi"] for r in rr])
                mm = np.array([r["match_mean"] for r in rr]); pp = np.array([r["measured_perp"] for r in rr]); n95 = np.array([r["null_p95"] for r in rr])
                R["controls"][f"{ds}_{t}_summary"] = {
                    "n_seeds": len(rr), "slope_t_vs_pred": float((m * p).sum() / (p * p).sum()), "corr_t_vs_pred": float(np.corrcoef(m, p)[0, 1]) if len(rr) > 2 else None,
                    "slope_t_vs_exact": float((m * e).sum() / (e * e).sum()), "slope_match_vs_pred": float((mm * p).sum() / (p * p).sum()),
                    "perp_inside_null": int((np.abs(pp) <= n95).sum()), "perp_abs_mean": float(np.abs(pp).mean()), "null_p95_mean": float(n95.mean())}

    # refusal
    for p in sorted(glob.glob(str(ART / "refusal_steer_arm_L*_s*.json"))):
        if p.endswith("_P.json"):
            continue  # P(refusal)-recording reruns of s0's r and g arms: a separate record (overnight_analysis), not extra arms
        d = load(p); key = os.path.basename(p)[len("refusal_steer_arm_"):-5]
        a = d["arms"]; m = [a[k]["chi_small"] for k in a if k.startswith("match")]
        R["refusal"][key] = {"layer": d["layer"], "seed": d["seed"], "pred_chi_r": d.get("pred_chi_r"), "pred_ratio_r": d.get("pred_ratio_r"), "pred_ratio_g": d.get("pred_ratio_g"),
                             "c_g_norm": d["c_times_g_norm"], "cos_r_g": d.get("cos_r_g"),
                             "r": a["r"]["chi_small"] if "r" in a else None, "r_ratio": a["r"]["ratio_small"] if "r" in a else None,
                             "g_ratio": a["g"]["ratio_small"] if "g" in a else None, "rperp": a["rperp"]["chi_small"] if "rperp" in a else None,
                             "match_mean": float(np.mean(m)) if m else None, "match_sd": float(np.std(m, ddof=1)) if len(m) > 1 else None,
                             "null_p95": d.get("rand_chi_small_p95"), "window_r": a["r"]["A_over_alpha"] if "r" in a else None,
                             "window_g": a["g"]["A_over_alpha"] if "g" in a else None}
    # refusal: exact linear response with the eval prompts' own mean gradient
    R["refusal_exact"] = {}
    for key, r in R["refusal"].items():
        L, seed = r["layer"], r["seed"]
        cands = [("refusal_gradient_vectors.npz", "refusal_prompts.json")] if seed == 0 else \
                [(f"refusal_gradient_s{seed}_vectors.npz", f"refusal_prompts_s{seed}.json"), (f"refusal_gradient_s{seed}b_vectors.npz", f"refusal_prompts_s{seed}b.json")]
        z = P = None
        for v_, p_ in cands:  # a seed's layers may be split across two gradient files
            if (ART / v_).exists() and (ART / p_).exists() and f"L{L}_g_per_prompt" in np.load(ART / v_).files:
                z = np.load(ART / v_); P = load(ART / p_); break
        if z is None:
            continue
        Gall = z[f"L{L}_g_per_prompt"].astype(np.float64); tr = z["train_idx"]; ev = P["eval_idx"]
        # the fit half comes from the gradient file and the eval half from the prompts file:
        # they must be one split (the 11 Sept contamination was a correct split read from the wrong file)
        assert list(P["train_idx"]) == tr.tolist() and list(P["y"]) == z["y"].tolist(), f"{key}: gradient and prompts files are not a pair"
        assert not set(tr.tolist()) & set(ev), f"{key}: eval prompts overlap the fit half"
        g_eval = Gall[ev].mean(0); Gf = Gall[tr]; g_fit = Gf.mean(0); g_hat = g_fit / np.linalg.norm(g_fit)
        r_hat = z[f"L{L}_r_hat"].astype(np.float64); c = float(z[f"L{L}_scale"][0])
        cos_rg = float(r_hat @ g_hat); vp = r_hat - cos_rg * g_hat; rperp = vp / np.linalg.norm(vp)
        d = load(ART / f"refusal_steer_arm_{key}.json"); W = {}
        if "r" in d["arms"]: W["r"] = r_hat
        if "g" in d["arms"]: W["g"] = g_hat
        if "rperp" in d["arms"]: W["rperp"] = rperp
        crng = np.random.default_rng(1000 + seed)
        for i in range(sum(k.startswith("match") for k in d["arms"])):
            u = crng.standard_normal(len(g_hat)); u -= (u @ g_hat) * g_hat; u /= np.linalg.norm(u)
            W[f"match{i}"] = cos_rg * g_hat + np.sqrt(1 - cos_rg ** 2) * u
        rng = np.random.default_rng(seed)
        for i in range(sum(k.startswith("rand") for k in d["arms"])):
            v = rng.standard_normal(len(g_hat)); W[f"rand{i}"] = v / np.linalg.norm(v)
        rows = [{"arm": k, "measured": d["arms"][k]["chi_small"], "exact": c * float(W[k] @ g_eval)} for k in d["arms"] if k in W]
        m = np.array([x["measured"] for x in rows]); e = np.array([x["exact"] for x in rows])
        R["refusal_exact"][key] = {"n_arms": len(rows), "arms": rows, "slope": float((m * e).sum() / (e * e).sum()), "corr": float(np.corrcoef(m, e)[0, 1]),
                                   "rms_resid": float(np.sqrt(np.mean((m - e) ** 2))), "rms_measured": float(np.sqrt(np.mean(m ** 2))),
                                   "named": {x["arm"]: (x["measured"], x["exact"]) for x in rows if not x["arm"].startswith("rand")},
                                   "noise_share_fit": float((Gf.var(0, ddof=1).sum() / Gf.shape[0]) / (g_fit @ g_fit)),
                                   "cos_gfit_geval": float(g_hat @ g_eval / np.linalg.norm(g_eval))}
    # depth
    for L in (8, 12, 16, 20, 24, 28):
        for s in (0, 4, 8):
            p = ART / f"steer_g_arm_counterfact_L{L}_comp_s{s}.json"
            if not p.exists():
                continue
            d = load(p); g = d["arms"]["g"]; sl = small_alpha_slope(g) / d["c_times_g_norm"]
            pred = noise_floor_ratio(d)
            av = list(response_slopes(g).values())
            R["depth"][f"L{L}_s{s}"] = {"ratio_slope": sl, "pred": pred, "ratio_over_pred": sl / pred, "c": d["alpha_unit"], "g_norm": d["g_fit_norm"],
                                        "window_rel": [x / av[0] for x in av[1:]]}
    with open(ART / "controls_analysis.json", "w") as f:
        json.dump(R, f, indent=1)

    # ---- markdown ----
    for ds in ("counterfact", "cities"):
        E = R["exact_linear_response"][ds]
        if E.get("n_arms"):
            print(f"\n### exact linear response, {ds} L28: {E['n_arms']} arms, pool coverage {E['pool_coverage']}/400, slope {E['slope_through_origin']:.3f}, corr {E['corr']:.4f}, rms resid {E['rms_resid']:.4f} vs rms measured {E['rms_measured']:.4f}")
        if R["controls"][ds]:
            print(f"\n### controls, {ds} L28 (alpha=0.5 slope)\n| seed | target | cos(t,g) | pred | exact | measured t | t_perp | exact perp | match mean (sd) | null p95 |\n|---|---|---|---|---|---|---|---|---|---|")
            for r in R["controls"][ds]:
                print(f"| {r['seed']} | {r['target']} | {r['cos_g_hat']:+.3f} | {r['pred_chi']:+.4f} | {r['exact_chi']:+.4f} | {r['measured_t']:+.4f} | {r['measured_perp']:+.4f} | {r['exact_perp']:+.4f} | {r['match_mean']:+.4f} ({r['match_sd']:.4f}) | {r['null_p95']:.3f} |")
            for k, v in R["controls"].items():
                if k.startswith(ds) and k.endswith("summary"): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
        if R["shuffle"][ds]:
            S = R["shuffle"][ds + "_summary"]
            print(f"\n### shuffle, {ds}: measured vs exact slope {S['slope_exact']:.3f} corr {S['corr_exact']:.3f}; vs in-sample slope {S['slope_in_sample']:.3f} corr {S['corr_in_sample']:.3f}")
    print("\n### refusal\n| run | c||g|| | cos(r,g) | pred chi(r) | r | r ratio (pred) | g ratio (pred) | r_perp | match mean (sd) | null p95 |\n|---|---|---|---|---|---|---|---|---|---|")
    for k, r in R["refusal"].items():
        f = lambda x, n=3: "-" if x is None else f"{x:+.{n}f}"
        msd = "-" if r["match_sd"] is None else f"{r['match_sd']:.2f}"
        n95 = "-" if r["null_p95"] is None else f"{r['null_p95']:.2f}"
        print(f"| {k} | {r['c_g_norm']:.1f} | {f(r['cos_r_g'])} | {f(r['pred_chi_r'],2)} | {f(r['r'],2)} | {f(r['r_ratio'])} ({f(r['pred_ratio_r'])}) | {f(r['g_ratio'])} ({f(r['pred_ratio_g'])}) | {f(r['rperp'],2)} | {f(r['match_mean'],2)} ({msd}) | {n95} |")
    print("\n### refusal, exact linear response (c w.g_eval): slope, corr, named arms measured/exact")
    for k, v in R["refusal_exact"].items():
        nm = "  ".join(f"{a} {m:+.2f}/{e:+.2f}" for a, (m, e) in v["named"].items() if not a.startswith("match"))
        mt = [v["named"][a] for a in v["named"] if a.startswith("match")]
        ms = f"  match {np.mean([x[0] for x in mt]):+.2f}/{np.mean([x[1] for x in mt]):+.2f}" if mt else ""
        print(f"- {k}: {v['n_arms']} arms slope {v['slope']:.3f} corr {v['corr']:.4f} resid {v['rms_resid']:.3f}/{v['rms_measured']:.2f}; noise share {v['noise_share_fit']:.3f} cos(g_fit,g_eval) {v['cos_gfit_geval']:.3f} | {nm}{ms}")
    print("\n### counterfact depth, slope/pred by seed\n| layer | s0 | s4 | s8 | c (s0,s4,s8) | window A(4)/4 rel (s0,s4,s8) |\n|---|---|---|---|---|---|")
    for L in (8, 12, 16, 20, 24, 28):
        cells = [R["depth"].get(f"L{L}_s{s}") for s in (0, 4, 8)]
        print(f"| {L} | " + " | ".join("-" if c is None else f"{c['ratio_over_pred']:.3f}" for c in cells) + " | " + ", ".join("-" if c is None else f"{c['c']:.1f}" for c in cells)
              + " | " + ", ".join("-" if c is None else f"{c['window_rel'][2]:.2f}" for c in cells) + " |")


# =============================================================================
# e2-perp  (was `steering_controls.py e2-perp`)
# =============================================================================
SEEDS = (0, 4)                 # the counterfact seeds whose control run steered e2


E2P_POOL, N_SUB, N_BOOT = 400, 150, 2000


def g_eval(Gp: np.ndarray, chosen: dict, s: int) -> tuple[np.ndarray, list[int]]:
    """Mean per-pair gradient over the pool pairs draw s did not fit on."""
    ev = [i for i in range(E2P_POOL) if i not in chosen[s] and not np.isnan(Gp[i, 0])]
    return Gp[ev].mean(0), ev


def decompose(w: np.ndarray, G: np.ndarray, c: float, m: int) -> dict:
    """Per-pair split of chi_exact = sum_i c (w . G_i) / n over the rows of G."""
    n = len(G)
    proj = G @ w
    contrib = c * proj / n
    tot = contrib.sum()
    norms = np.linalg.norm(G, axis=1)
    cos = proj / norms

    def top_share(frac: float) -> float:
        k = int(np.ceil(frac * n))
        return float(contrib[np.argsort(-np.abs(contrib))[:k]].sum() / tot)

    return {"chi_exact": float(tot),
            "top5_share": top_share(0.05),
            "top20_share": top_share(0.20),
            "normtail_share": float(contrib[norms >= np.quantile(norms, 0.95)].sum() / tot),
            "mean_cos": float(cos.mean()),
            "sd_cos": float(cos.std()),
            "same_sign_frac": float(np.mean(np.sign(contrib) == np.sign(tot))),
            "massive_share": float((G[:, m] * w[m]).sum() / proj.sum()),
            "w_massive_weight": float(w[m] ** 2)}


def run_e2_perp(argv=None):
    """Where the projected-out residual lives: e2_perp on counterfact L28, complement
    protocol (notes/exact_linear_response_13sept.md, addendum; draft P6, E17).

    e2_perp is e2 with its g_hat component removed (steer_g_arm --controls e2). Two
    questions, at the two seeds whose control runs carry e2 (0 and 4):

      e2_perp_across_complements.json
          Score the same e2_perp against every seed's complement, c (w . g_eval(k)),
          against the whole 400-pair pool, and against 2000 random 150-pair subsets
          of the pool (rng 0, without replacement): is the own-complement number a
          real overlap or eval-side sampling noise?
      e2_perp_decomposition.json
          Split chi_exact = sum_i c (w . G_i) / n over the seed's eval pairs, for e2,
          e2_perp and g_hat: signed share of the top 5% and 20% pairs by |contribution|,
          share of the pairs in the top 5% by ||G_i|| (the heavy tail), cos(w, G_i), and
          the share carried by the massive coordinate.

    Reads the per-seed *_vectors.npz written by `rogue_dimension.py gradient`, the
    steer_g_arm complement runs and the act cache. No GPU.

    Regenerates both committed files byte for byte.
    """
    argparse.ArgumentParser(description=run_e2_perp.__doc__).parse_args(argv)
    Gp, chosen, cov = pool_gradients("counterfact", range(10))
    assert cov == E2P_POOL, f"pool coverage {cov}/{E2P_POOL}: a seed's npz is missing"
    X = np.load(ART / "act_cache" / "pythia-2.8b__counterfact_true_false.npz")["L28"].astype(np.float64)
    m = int(np.argmax(np.abs(X.mean(0))))          # the massive coordinate

    across, decomp = {}, {}
    for s in SEEDS:
        comp = json.load(open(ART / f"steer_g_arm_counterfact_L28_comp_s{s}.json"))
        ctrl = json.load(open(ART / f"steer_g_arm_counterfact_L28_comp_ctrl_s{s}.json"))
        c = comp["alpha_unit"]
        W = complement_arms("counterfact", s, ctrl)
        dirs = {"e2": W["e2"], "e2_perp": W["e2_perp"], "g_hat": complement_arms("counterfact", s, comp)["g"]}

        w = dirs["e2_perp"]
        ge, ev = g_eval(Gp, chosen, s)
        rng = np.random.default_rng(0)
        boot = np.array([c * float(w @ Gp[rng.choice(E2P_POOL, N_SUB, replace=False)].mean(0)) for _ in range(N_BOOT)])
        across[s] = {"own": c * float(w @ ge),
                     "others": {k: c * float(w @ g_eval(Gp, chosen, k)[0]) for k in range(10)},
                     "pool": c * float(w @ Gp.mean(0)),
                     "boot_mean": float(boot.mean()), "boot_sd": float(boot.std())}
        for name, v in dirs.items():
            decomp[f"s{s}_{name}"] = decompose(v, Gp[ev], c, m)
        a = across[s]
        print(f"s{s}: e2_perp own {a['own']:+.4f}  pool {a['pool']:+.4f}  boot {a['boot_mean']:+.4f} +- {a['boot_sd']:.4f}  "
              f"normtail share {decomp[f's{s}_e2_perp']['normtail_share']:+.2f}")

    for name, obj in (("e2_perp_across_complements", across), ("e2_perp_decomposition", decomp)):
        with open(ART / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=1)
        print(f"wrote artifacts/{name}.json")


COMMANDS = {
    "controls": run_controls,
    "e2-perp": run_e2_perp,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
