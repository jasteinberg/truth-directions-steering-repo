"""
Refusal as the opposite corner: the identity on a safety-tuned model.

The section "Refusal as the opposite corner" of part 2, "Steering Vectors and the Limits of Linear Response".
Qwen1.5-1.8B-Chat on AdvBench / Alpaca: the refusal gradient row, steering along
r_hat, g_hat and the controls, generations under steering, and the
finite-difference check of the refusal gradient.

Subcommands:

    gradient               The gradient identity on the refusal direction of a safety-tuned ...
    steer                  Steering arm for the refusal audit row
    generations            Greedy generations under steering, to read the three regimes by ...
    fd-check               Finite-difference check of `refusal_corner.py gradient`'s ...

Run from the repo root:  python scripts/refusal_corner.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.truthlib import estimators as est
from utils.truthlib import refusal
from utils.truthlib.estimators import chi_origin
from utils.truthlib.refusal import (
    DEV,
    MODEL,
    QwenSteerer,
    batch_forward,
    boot_cos,
    chat_wrap,
    load_prompts,
    refusal_ids,
    score,
)

# ---- shared by several subcommands ------------------------------------------
ART = REPO / "artifacts"


# =============================================================================
# gradient
# =============================================================================
def run_gradient(argv=None):
    """The gradient identity on the refusal direction of a safety-tuned model.

    Model: Qwen/Qwen1.5-1.8B-Chat (24 layers), as in the earlier refusal scripts.
    Prompts: harmful = AdvBench, harmless = Alpaca, balanced, chat-formatted with
    the generation prompt appended, so the score is read at the position where the
    model would emit its first response token.

    Behavioural score: ell = log P(refusal) - log(1 - P(refusal)), the log-odds that
    the first generated token is one of a fixed refusal-token set (the same set the
    causal-reproduction script uses for its probability score). The log-odds is the
    analogue of the truth score log p(correct) - log p(incorrect) and does not
    saturate, so a bounded probability cannot masquerade as curvature.

    Per layer L in the swept set, at the residual stream entering block L (layer L,
    docs/model_conventions.md):
      - last-token activations of every prompt -> r_hat = mean(harmful) -
        mean(harmless) (the refusal direction of Arditi et al.), the within-class
        spectrum (lambda_1/tr, PR, v_1), cos(r_hat, v_1), and the class-gap unit
        c = ||mu_harmful - mu_harmless|| on the fit half;
      - per-prompt gradients G_i = sum_t d ell_i / d h_t (summed over positions,
        the convention of `rogue_dimension.py gradient`, since steering adds at all
        positions), for all prompts; g_all, g_harmful, g_harmless;
      - the audit row: cos(r_hat, g) with a bootstrap interval against 1/sqrt(d),
        cos(g, v_1), ||g||, tr(Sigma_G)/n, noise share, c||g||, and the linear-
        response prediction for the complement-arm ratio cos^2(g_hat, g).

    Fit/eval split: class-stratified halves under the seed, as in truthlib; r_hat,
    c, the spectrum and g are all fit-half quantities, the eval half is reserved
    for `refusal_corner.py steer`. Prompt lists and split indices are saved so the arm
    scores exactly the held-out prompts.

    Writes artifacts/refusal_gradient.json and artifacts/refusal_gradient_vectors.npz.
    """
    ap = argparse.ArgumentParser(description=run_gradient.__doc__)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="9,12,15,18")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ART / "refusal_gradient.json"))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rids = refusal_ids(tok)

    harmful, harmless = load_prompts(args.n, args.seed)
    prompts = harmful + harmless
    y = np.array([1] * len(harmful) + [0] * len(harmless))
    texts = [chat_wrap(tok, p) for p in prompts]
    tr, te = est.split_indices(y, seed=args.seed)
    d = model.config.hidden_size
    out = {"model": MODEL, "n_prompts": len(prompts), "seed": args.seed, "refusal_token_ids": rids,
           "score": "log-odds of a refusal first token", "layers": {}}
    vec = {"y": y, "train_idx": tr, "eval_idx": te}
    # prompts file named after --out so a second seed does not overwrite seed 0's split
    with open(args.out.replace("refusal_gradient", "refusal_prompts"), "w") as f:
        json.dump({"prompts": prompts, "y": y.tolist(), "train_idx": tr.tolist(), "eval_idx": te.tolist()}, f)

    for L in layers:
        block = model.model.layers[L]
        ell, X, G = [], [], []
        for i in range(0, len(texts), args.bs):
            e, a, g = batch_forward(model, tok, block, texts[i:i + args.bs], rids, True)
            ell.append(e); X.append(a); G.append(g)
            gc.collect(); torch.mps.empty_cache()
        ell, X, G = np.concatenate(ell), np.vstack(X), np.vstack(G)

        Xtr, ytr, Gtr = X[tr], y[tr], G[tr]
        delta = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        c = float(np.linalg.norm(delta)); r_hat = delta / c
        C = est.within_class_cov(Xtr, ytr)
        w, V = np.linalg.eigh(C); w, V = w[::-1], V[:, ::-1]; v1 = V[:, 0]
        g_all = Gtr.mean(0); g_norm = float(np.linalg.norm(g_all)); g_hat = g_all / g_norm
        trS = float(Gtr.var(0, ddof=1).sum()); n = len(Gtr)
        g_true2 = g_norm ** 2 - trS / n
        rec = {
            "n_fit": int(n), "d": int(d),
            "ell_harmful_mean": float(ell[y == 1].mean()), "ell_harmless_mean": float(ell[y == 0].mean()),
            "p_refusal_harmful": float((1 / (1 + np.exp(-ell[y == 1]))).mean()),
            "p_refusal_harmless": float((1 / (1 + np.exp(-ell[y == 0]))).mean()),
            "c": c, "lam1_over_tr": float(w[0] / w.sum()), "lam1_over_lam2": float(w[0] / w[1]),
            "PR": est.participation_ratio(w), "cos_r_v1": float(abs(r_hat @ v1)),
            "cos_r_g": [float(r_hat @ g_hat)] + boot_cos(Gtr, r_hat, args.boot, args.seed + L),
            "cos_g_v1": [float(abs(g_hat @ v1))] + [abs(x) for x in boot_cos(Gtr, v1, args.boot, args.seed + L + 100)],
            "cos_r_g_harmful": float(r_hat @ Gtr[ytr == 1].mean(0) / np.linalg.norm(Gtr[ytr == 1].mean(0))),
            "cos_r_g_harmless": float(r_hat @ Gtr[ytr == 0].mean(0) / np.linalg.norm(Gtr[ytr == 0].mean(0))),
            "random_scale": float(1 / np.sqrt(d)),
            "g_norm": g_norm, "tr_sigma_g_over_n": trS / n, "noise_share": trS / n / g_norm ** 2,
            "g_true": float(np.sqrt(max(g_true2, 0))), "c_times_g_norm": c * g_norm,
            "cos2_pred_complement": float(g_true2 / (g_true2 + trS / n)) if g_true2 > 0 else 0.0,
            "predicted_chi_r": float(c * (r_hat @ g_all)),
        }
        out["layers"][str(L)] = rec
        vec.update({f"L{L}_r_hat": r_hat, f"L{L}_v1": v1, f"L{L}_g": g_all, f"L{L}_g_per_prompt": G,
                    f"L{L}_lam": w, f"L{L}_scale": np.array([c]), f"L{L}_ell": ell, f"L{L}_X_last": X})
        print(f"L{L}: P(ref) harmful {rec['p_refusal_harmful']:.3f} harmless {rec['p_refusal_harmless']:.3f} | "
              f"lam1/tr {rec['lam1_over_tr']:.3f} PR {rec['PR']:.1f} cos(r,v1) {rec['cos_r_v1']:.3f} | "
              f"cos(r,g) {rec['cos_r_g'][0]:+.3f} [{rec['cos_r_g'][1]:+.3f},{rec['cos_r_g'][2]:+.3f}] "
              f"(rand {rec['random_scale']:.3f}) cos(g,v1) {rec['cos_g_v1'][0]:.3f} | "
              f"c {c:.2f} |g| {g_norm:.4f} noise {rec['noise_share']:.2f} c|g| {rec['c_times_g_norm']:.3f} "
              f"pred chi(r) {rec['predicted_chi_r']:+.3f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        np.savez_compressed(args.out.replace(".json", "_vectors.npz"), **vec)
    print("wrote", args.out)


# =============================================================================
# steer
# =============================================================================
ALPHAS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def run_steer(argv=None):
    """Steering arm for the refusal audit row: steer Qwen1.5-1.8B-Chat along the
    refusal direction r_hat, along g_hat, and along random unit directions, and
    compare the measured susceptibilities against the linear bound c||g_hat||.

    Everything steered is scored on the 128 eval prompts that `refusal_corner.py gradient`
    held out (artifacts/refusal_prompts.json, eval_idx); r_hat, g_hat and c are the
    fit-half quantities in artifacts/refusal_gradient_vectors.npz. Protocol as in
    `linear_bound.py g-arm`: push alpha * c * w at every position of the residual
    stream entering block L, antisymmetrise A(alpha) = (mean ell(+) - mean ell(-)) / 2, chi by the
    through-origin fit over the alpha grid AND by the smallest-alpha slope, since
    c is the full class gap here and the linear window closes early (the smoke
    test showed A/alpha along g_hat halving between alpha = 0.25 and 0.5, hence
    the two extra steps below 0.25).
    The symmetric part S(alpha) is kept too (degradation lands there).

    Control: r_perp, the refusal direction with its g_hat component projected out and
    renormalised. It is 99% of r_hat by cosine and gets the same push, and the identity
    predicts chi(r_perp) = 0 to within the null scale c||g||/sqrt(d). This is the test
    that the effect of r_hat runs entirely through its overlap with g, which the
    random-direction null (a bound, not a prediction) cannot make.

    Predictions recorded before the run: notes/refusal_steer_arm_prediction.md.
    """
    ap = argparse.ArgumentParser(description=run_steer.__doc__)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n_rand", type=int, default=30)
    ap.add_argument("--n_match", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="r,g,rperp,match,rand",
                    help="r, g, rperp (r_hat with its g_hat component removed: predicted chi = 0), "
                         "match (n_match random directions with the same cosine to g_hat as r_hat: "
                         "predicted chi = chi(r_hat)), rand")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--n_eval", type=int, default=None, help="smoke test: score only the first n eval prompts")
    ap.add_argument("--vectors", default=None, help="refusal gradient npz; required for seed != 0 (default: seed 0's)")
    ap.add_argument("--prompts", default=None, help="prompts/split json; required for seed != 0 (default: seed 0's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    L = args.layer
    alphas = [float(a) for a in args.alphas.split(",")]
    out_path = args.out or str(ART / f"refusal_steer_arm_L{L}_s{args.seed}.json")
    if args.seed != 0 and (args.vectors is None or args.prompts is None):
        raise SystemExit("--vectors and --prompts are required for seed != 0 (the defaults are seed 0's files)")
    args.vectors = args.vectors or str(ART / "refusal_gradient_vectors.npz")
    args.prompts = args.prompts or str(ART / "refusal_prompts.json")

    z = np.load(args.vectors)
    r_hat = z[f"L{L}_r_hat"].astype(np.float64)
    g_fit = z[f"L{L}_g"].astype(np.float64)
    G = z[f"L{L}_g_per_prompt"].astype(np.float64)[z["train_idx"]]
    c = float(z[f"L{L}_scale"][0])
    g_norm = float(np.linalg.norm(g_fit)); g_hat = g_fit / g_norm
    bound = c * g_norm
    trS_n = float(G.var(0, ddof=1).sum() / G.shape[0])
    g_true2 = g_norm ** 2 - trS_n
    pred_g = g_true2 / (g_true2 + trS_n)
    cos_rg = float(r_hat @ g_hat)
    pred_r = cos_rg * float(np.sqrt(max(pred_g, 0)))

    P = json.load(open(args.prompts))
    assert list(P["train_idx"]) == z["train_idx"].tolist() and list(P["y"]) == z["y"].tolist(), \
        f"{args.vectors} and {args.prompts} are not one split"
    assert not set(P["eval_idx"]) & set(z["train_idx"].tolist()), "eval prompts overlap the fit half"
    ev = P["eval_idx"] if args.n_eval is None else P["eval_idx"][:args.n_eval]
    y = np.array(P["y"])[ev]
    assert not set(ev) & set(z["train_idx"].tolist()), "eval prompts overlap the fit half"

    tok = AutoTokenizer.from_pretrained(refusal.MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(refusal.MODEL, torch_dtype=torch.float16).to(refusal.DEV).eval()
    rids = refusal.refusal_ids(tok)
    assert rids == json.load(open(ART / "refusal_gradient.json"))["refusal_token_ids"], "refusal-token set changed"
    texts = [refusal.chat_wrap(tok, P["prompts"][i]) for i in ev]

    base = score(model, tok, texts, rids, args.bs)
    rng = np.random.default_rng(args.seed)
    want = set(args.arms.split(","))
    arms = {}
    if "r" in want: arms["r"] = r_hat
    if "g" in want: arms["g"] = g_hat
    if "rperp" in want:
        v = r_hat - (r_hat @ g_hat) * g_hat; arms["rperp"] = v / np.linalg.norm(v)
    if "match" in want:
        # random directions with the same cosine to g_hat as r_hat: predicted chi = chi(r_hat)
        crng = np.random.default_rng(1000 + args.seed)
        for i in range(args.n_match):
            u = crng.standard_normal(len(g_hat)); u -= (u @ g_hat) * g_hat; u /= np.linalg.norm(u)
            arms[f"match{i}"] = cos_rg * g_hat + np.sqrt(1 - cos_rg ** 2) * u
    if "rand" in want:
        for i in range(args.n_rand):
            v = rng.standard_normal(len(g_hat)); arms[f"rand{i}"] = v / np.linalg.norm(v)

    out = {"baseline_P_mean": float((1 / (1 + np.exp(-base))).mean()), "model": refusal.MODEL, "layer": L, "seed": args.seed, "alphas": alphas,
           "n_fit": int(G.shape[0]), "n_eval": int(len(ev)),
           "alpha_unit": c, "g_fit_norm": g_norm, "c_times_g_norm": bound,
           "tr_sigma_g_over_n": trS_n, "cos_r_g": cos_rg,
           "pred_ratio_g": pred_g, "pred_ratio_r": pred_r, "pred_chi_r": c * float(r_hat @ g_fit),
           "pred_chi_rperp": 0.0, "null_scale": bound / np.sqrt(len(g_hat)),
           "baseline_ell_mean": float(base.mean()),
           "baseline_ell_harmful": float(base[y == 1].mean()), "baseline_ell_harmless": float(base[y == 0].mean()),
           "arms": {}}
    print(f"L{L}: n_eval={len(ev)} c={c:.3f} ||g||={g_norm:.4f} c||g||={bound:.3f} trS/n={trS_n:.4f} "
          f"pred ratio g {pred_g:.3f} r {pred_r:.3f} pred chi(r) {out['pred_chi_r']:+.3f} "
          f"baseline {base.mean():+.3f} (harmful {out['baseline_ell_harmful']:+.3f}, harmless {out['baseline_ell_harmless']:+.3f})",
          flush=True)

    for name, w in arms.items():
        rec = {"A": {}, "S": {}, "A_harmful": {}, "A_harmless": {}}
        with QwenSteerer(model, L) as st:
            for a in alphas:
                st.set(w, a * c, refusal.DEV, torch.float16); sp = score(model, tok, texts, rids, args.bs)
                st.set(w, -a * c, refusal.DEV, torch.float16); sm = score(model, tok, texts, rids, args.bs)
                A = (sp - sm) / 2; S = (sp + sm) / 2 - base
                sig = lambda x: 1 / (1 + np.exp(-x))
                rec.setdefault("P_plus", {})[str(a)] = float(sig(sp).mean()); rec.setdefault("P_minus", {})[str(a)] = float(sig(sm).mean())
                rec.setdefault("P_plus_harmless", {})[str(a)] = float(sig(sp[y == 0]).mean()); rec.setdefault("P_minus_harmful", {})[str(a)] = float(sig(sm[y == 1]).mean())
                rec["A"][str(a)] = float(A.mean()); rec["S"][str(a)] = float(S.mean())
                rec["A_harmful"][str(a)] = float(A[y == 1].mean()); rec["A_harmless"][str(a)] = float(A[y == 0].mean())
            st.set(None, 0, refusal.DEV, torch.float16)
        Av = [rec["A"][str(a)] for a in alphas]
        rec["chi_fit"] = chi_origin(alphas, Av)
        rec["chi_small"] = Av[0] / alphas[0]
        rec["A_over_alpha"] = [v / a for v, a in zip(Av, alphas)]
        rec["ratio_fit"] = rec["chi_fit"] / bound
        rec["ratio_small"] = rec["chi_small"] / bound
        out["arms"][name] = rec
        if not name.startswith("rand"):
            print(f"  {name:5s} A/alpha={[round(v, 3) for v in rec['A_over_alpha']]}  "
                  f"chi_small={rec['chi_small']:+.3f} ratio_small={rec['ratio_small']:+.3f}  "
                  f"chi_fit={rec['chi_fit']:+.3f} ratio_fit={rec['ratio_fit']:+.3f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)

    mc = [out["arms"][k]["chi_small"] for k in out["arms"] if k.startswith("match")]
    if mc:
        out["match_chi_small_mean"] = float(np.mean(mc)); out["match_chi_small_sd"] = float(np.std(mc, ddof=1)) if len(mc) > 1 else None
        print(f"match: chi_small {np.mean(mc):+.3f} sd {out['match_chi_small_sd'] or 0:.3f} (pred = chi(r_hat) {out['pred_chi_r']:+.3f})", flush=True)
    rc = [(k, out["arms"][k]) for k in out["arms"] if k.startswith("rand")]
    if rc:
        for key in ("chi_small", "chi_fit"):
            v = np.array([r[key] for _, r in rc])
            out[f"rand_{key}_mean"] = float(v.mean()); out[f"rand_{key}_p95"] = float(np.percentile(np.abs(v), 95))
            out[f"rand_{key}_sd"] = float(v.std(ddof=1))
        print(f"random: |chi_small| p95 {out['rand_chi_small_p95']:.4f} sd {out['rand_chi_small_sd']:.4f}; "
              f"|chi_fit| p95 {out['rand_chi_fit_p95']:.4f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_path)


# =============================================================================
# generations
# =============================================================================
@torch.no_grad()
def generate(model, tok, texts, max_new):
    enc = tok(texts, return_tensors="pt", padding=True).to(refusal.DEV)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def run_generations(argv=None):
    """Greedy generations under steering, to read the three regimes by eye rather
    than infer them from the first-token distribution. For each arm (r_hat, g_hat)
    and alpha in --alphas, push +alpha c w (toward refusal) on harmless eval
    prompts and -alpha c w (away) on harmful eval prompts, at every position of
    block L, and generate --max_new tokens. alpha = 0 is the unsteered baseline.

    Writes artifacts/refusal_generations_L{L}_s{seed}.json:
      {arm: {alpha: [{prompt, y, sign, text, first_token_refusal}]}}.
    Same vectors, prompts and hook as `refusal_corner.py steer`.
    """
    ap = argparse.ArgumentParser(description=run_generations.__doc__)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_each", type=int, default=8, help="harmless and harmful eval prompts per cell")
    ap.add_argument("--alphas", default="0,0.25,1,2,4")
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--vectors", default=None, help="refusal gradient npz; required for seed != 0 (default: seed 0's)")
    ap.add_argument("--prompts", default=None, help="prompts/split json; required for seed != 0 (default: seed 0's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    L = args.layer
    alphas = [float(a) for a in args.alphas.split(",")]
    out_path = args.out or str(ART / f"refusal_generations_L{L}_s{args.seed}.json")
    if args.seed != 0 and (args.vectors is None or args.prompts is None):
        raise SystemExit("--vectors and --prompts are required for seed != 0 (the defaults are seed 0's files)")
    args.vectors = args.vectors or str(ART / "refusal_gradient_vectors.npz")
    args.prompts = args.prompts or str(ART / "refusal_prompts.json")

    z = np.load(args.vectors)
    r_hat = z[f"L{L}_r_hat"].astype(np.float64)
    g_fit = z[f"L{L}_g"].astype(np.float64); g_hat = g_fit / np.linalg.norm(g_fit)
    c = float(z[f"L{L}_scale"][0])
    P = json.load(open(args.prompts))
    assert list(P["train_idx"]) == z["train_idx"].tolist() and list(P["y"]) == z["y"].tolist(), \
        f"{args.vectors} and {args.prompts} are not one split"
    assert not set(P["eval_idx"]) & set(z["train_idx"].tolist()), "eval prompts overlap the fit half"
    y = np.array(P["y"]); ev = np.array(P["eval_idx"])
    harmless = [i for i in ev if y[i] == 0][:args.n_each]
    harmful = [i for i in ev if y[i] == 1][:args.n_each]

    tok = AutoTokenizer.from_pretrained(refusal.MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(refusal.MODEL, torch_dtype=torch.float16).to(refusal.DEV).eval()
    rids = set(refusal.refusal_ids(tok))

    out = {"model": refusal.MODEL, "layer": L, "seed": args.seed, "alpha_unit": c, "alphas": alphas, "arms": {}}
    for name, w in (("r", r_hat), ("g", g_hat)):
        out["arms"][name] = {}
        with QwenSteerer(model, L) as st:
            for a in alphas:
                rows = []
                for idx, sign in ((harmless, +1), (harmful, -1)):
                    st.set(w, sign * a * c, refusal.DEV, torch.float16)
                    texts = [refusal.chat_wrap(tok, P["prompts"][i]) for i in idx]
                    gens = generate(model, tok, texts, args.max_new)
                    for i, g in zip(idx, gens):
                        first = tok(g, add_special_tokens=False)["input_ids"][:1]
                        rows.append({"prompt": P["prompts"][i], "y": int(y[i]), "sign": sign, "text": g,
                                     "first_token_refusal": bool(first and first[0] in rids)})
                out["arms"][name][str(a)] = rows
                fr = np.mean([r["first_token_refusal"] for r in rows if r["y"] == 0])
                print(f"L{L} {name} alpha={a:g}: harmless first-token refusal {fr:.2f}; e.g. {rows[0]['text'][:80]!r}", flush=True)
            st.set(None, 0, refusal.DEV, torch.float16)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    print("wrote", out_path)


# =============================================================================
# fd-check
# =============================================================================
def run_fd_check(argv=None):
    """Finite-difference check of `refusal_corner.py gradient`'s gradient convention: for a
    few prompts and directions w, compare (ell(+h w) - ell(-h w)) / 2h, with h w
    added to the residual at every position of block L, against G_i . w from the
    per-prompt gradient. Agreement to a few percent at small h validates the hook
    (sum over positions <-> add at all positions) and the fp16 loss scaling. The push
    is `QwenSteerer`'s (`steering.Steerer`), at the input of block L, where `batch_forward` takes the
    gradient (docs/model_conventions.md section 2).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--h", type=float, default=0.05)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(refusal.MODEL); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(refusal.MODEL, torch_dtype=torch.float16).to(refusal.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rids = refusal.refusal_ids(tok)
    harmful, harmless = refusal.load_prompts(2 * args.n, 1)
    texts = [refusal.chat_wrap(tok, p) for p in harmful[: args.n // 2] + harmless[: args.n // 2]]
    block = model.model.layers[args.layer]
    ell0, _, G = refusal.batch_forward(model, tok, block, texts, rids, True)
    d = G.shape[1]
    rng = np.random.default_rng(0)
    dirs = {"g_hat": G.mean(0) / np.linalg.norm(G.mean(0)), "rand": rng.standard_normal(d) / np.sqrt(d)}
    print(f"layer {args.layer}, h = {args.h}, {len(texts)} prompts")
    with QwenSteerer(model, args.layer) as ad:
        for name, w in dirs.items():
            pred = G @ w
            vec = torch.tensor(args.h * w, dtype=torch.float32, device=refusal.DEV)
            ad.vec = vec; ep, _, _ = refusal.batch_forward(model, tok, block, texts, rids, False)
            ad.vec = -vec; em, _, _ = refusal.batch_forward(model, tok, block, texts, rids, False)
            ad.vec = None
            fd = (ep - em) / (2 * args.h)
            rel = np.abs(fd - pred) / (np.abs(pred) + 1e-6)
            print(f"  {name:6s} grad.w  {np.array2string(pred, precision=3)}")
            print(f"         fd      {np.array2string(fd, precision=3)}   median rel err {np.median(rel):.3f}")


COMMANDS = {
    "gradient": run_gradient,
    "steer": run_steer,
    "generations": run_generations,
    "fd-check": run_fd_check,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
