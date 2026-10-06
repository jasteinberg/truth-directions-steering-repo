"""
Steering harness (torch): the residual-stream hook, the behavioral score, and the
bookkeeping shared by the steering sweep, the extended null and the gradient run.

Also: per-pair score gradients (grad_for_texts, pair_gradients), the steering
checkpoints (load_seed_cells, load_null_cell), the 400-pair pool and its arms
(pool_gradients, complement_arms), and readers for a steering run's response
(response_slopes, small_alpha_slope, noise_floor_ratio).

Drafted with the assistance of Claude (Anthropic).
"""
import gc
import glob
import json
import os

import numpy as np
import torch

from . import acts, data
from .estimators import auroc, chi_origin, class_gap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "artifacts")
CKPT = os.path.join(ART, "steer_ckpt")  # class-gap alpha units
ACTS = os.path.join(ART, "act_cache")
DEV = "mps"
LOSS_SCALE = 1024.0          # fp16 backward underflows without this

SEED_PLAN = {0.25: 10, 0.375: 10, 0.5: 10, 0.625: 5, 0.75: 5, 0.875: 10}


def tag(m): return m.split("/")[-1]


def cell_path(model, ds, layer, seed):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__s{seed}.json")


def null_path(model, ds, layer):
    return os.path.join(CKPT, f"{tag(model)}__{ds}__L{layer}__NULL.json")


def act_path(model, ds):
    return os.path.join(ACTS, f"{tag(model)}__{ds}.npz")


def planned_layers(n_blocks: int) -> list[int]:
    """The steering sweep's layers for a model of n_blocks: one per SEED_PLAN depth fraction
    (pythia-2.8b, 32 blocks: 8..28; pythia-1.4b, 24 blocks: 6..21)."""
    return sorted({max(1, int(round(f * n_blocks))) for f in SEED_PLAN})


def load_cached_acts(model: str, ds: str, n_blocks: int) -> tuple[dict, np.ndarray]:
    """The act cache for (model, dataset), refused unless it exists and holds exactly the
    planned layers: a missing or truncated cache would silently change every table read
    from it."""
    p = act_path(model, ds)
    if not os.path.exists(p):
        raise SystemExit(f"missing activation cache {os.path.relpath(p, os.path.dirname(ART))}")
    z = np.load(p)
    layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
    if layers != planned_layers(n_blocks):
        raise SystemExit(f"{os.path.basename(p)} holds layers {layers}, expected {planned_layers(n_blocks)}")
    return {L: z[f"L{L}"] for L in layers}, z["y"]


def read_ckpt() -> str:
    """Directory the analyses read cells from: $STEER_CKPT if set, else CKPT.
    Writers always use CKPT."""
    return str(os.environ.get("STEER_CKPT", CKPT))


def load_seed_cells(model: str, ds: str, layer: int, ckpt: str | None = None) -> list[dict]:
    """Every seed's cell at one (model, dataset, layer), in sorted filename order
    (s0, s1, s10, s2, ...), the order the figures and tables were built with."""
    pat = os.path.join(ckpt or read_ckpt(), f"{tag(model)}__{ds}__L{layer}__s*.json")
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]


def load_null_cell(model: str, ds: str, layer: int, ckpt: str | None = None) -> dict | None:
    """The extended-null cell at one (model, dataset, layer), or None if not run."""
    p = os.path.join(ckpt or read_ckpt(), f"{tag(model)}__{ds}__L{layer}__NULL.json")
    return json.load(open(p)) if os.path.exists(p) else None


def get_acts(model, tok, mname, ds, layers, cap, seed=0):
    """Cache last-token activations for the probed layers only.

    Full A is (n_layers+1, N, d) and is large; we keep just the probed layers,
    so extending the seed list later costs no forward passes at all.
    """
    p = act_path(mname, ds)
    if os.path.exists(p):
        z = np.load(p)
        if sorted(int(k[1:]) for k in z.files if k.startswith("L")) == sorted(layers):
            print(f"    (cached activations {os.path.basename(p)})", flush=True)
            return {int(k[1:]): z[k] for k in z.files if k.startswith("L")}, z["y"]
    stmts, y = data.load_dataset(ds, cap=cap, seed=seed)
    print(f"    extracting {len(stmts)} activations...", flush=True)
    A = acts.extract_all_layers(stmts, tok, model, DEV, 16)
    out = {L: A[L].astype(np.float32) for L in layers}
    os.makedirs(ACTS, exist_ok=True)
    np.savez_compressed(p, y=y, **{f"L{L}": v for L, v in out.items()})
    del A; gc.collect()
    return out, y


def load_cells(art=ART):
    """Notebook entry point: every cell on disk as a tidy list of dicts.

    Robust to however many seeds happen to exist -- add more by re-running with
    --seeds and calling this again.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(art, "steer_ckpt", "*__s*.json"))):
        r = json.load(open(f))
        base = os.path.basename(f)[:-5].split("__")
        r["model"], r["dataset"] = base[0], base[1]
        rows.append(r)
    nulls = {}
    for f in sorted(glob.glob(os.path.join(art, "steer_ckpt", "*__NULL.json"))):
        base = os.path.basename(f)[:-5].split("__")
        nulls[(base[0], base[1], int(base[2][1:]))] = json.load(open(f))
    return rows, nulls


def pairs_for_seed(pool, seed, n):
    """Stable given the seed: seed k always selects the same evaluation subset."""
    rng = np.random.default_rng(777 + seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


# ---- steering + scoring --------------------------------------------------
class Steerer:
    """Adds alpha * theta to the residual stream at layer l, all positions.

    Layer l is the residual stream entering block l (hidden_states[l]), the site the
    activations are extracted at (acts.extract_all_layers) and the score gradient is
    taken at (grad_for_texts): a forward pre-hook on block l adds the push to the
    block's input, so every later layer sees it. l = n_blocks (after the last block)
    has no block to hook and is refused.
    """
    def __init__(self, model, layer):
        bl = acts.blocks(model)
        if not 0 <= layer < len(bl):
            raise ValueError(f"layer {layer}: steering sites are the inputs of blocks 0..{len(bl) - 1}")
        self.block = bl[layer]
        self.vec = None
        self.h = None

    def __enter__(self):
        def hook(mod, args, kwargs):
            if self.vec is None:
                return None
            if args:
                return (args[0] + self.vec.to(args[0].dtype),) + tuple(args[1:]), kwargs
            h = kwargs["hidden_states"]
            return args, {**kwargs, "hidden_states": h + self.vec.to(h.dtype)}
        self.h = self.block.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *a):
        if self.h: self.h.remove()

    def set(self, theta, alpha, device, dtype):
        if theta is None or alpha == 0:
            self.vec = None
        else:
            t = torch.tensor(theta, device=device, dtype=dtype)
            self.vec = alpha * t


@torch.no_grad()
def score_pairs(model, tok, pairs, bs=8):
    """log P(true completion) - log P(false completion), summed over its tokens.

    Both completions are scored against the SAME prompt, so prompt length and
    any generic token bias cancel in the difference. Multi-token targets are
    summed (not length-normalised): the pair is scored on total log-prob, and
    length differences between the two targets are a property of the pair, not
    of the steering, so they cancel when we look at the *shift* under steering.
    """
    scores = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        vals = []
        for which in ("true", "false"):
            texts = [p["prompt"] + p[which] for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            logits = model(**enc).logits.float().log_softmax(-1)
            batch_vals = []
            for j, p in enumerate(chunk):
                n_prompt = len(tok(p["prompt"]).input_ids)
                n_full = len(tok(p["prompt"] + p[which]).input_ids)
                lp = 0.0
                for t in range(n_prompt, n_full):
                    tid = enc["input_ids"][j, t]
                    lp += logits[j, t - 1, tid].item()
                batch_vals.append(lp)
            vals.append(np.array(batch_vals))
        scores.append(vals[0] - vals[1])
    return np.concatenate(scores)


def antisym(model, tok, st, vec, alpha, scale, pairs, base, bs, dtype):
    st.set(vec, +alpha * scale, DEV, dtype)
    sp = float(np.mean(score_pairs(model, tok, pairs, bs) - base))
    st.set(vec, -alpha * scale, DEV, dtype)
    sm = float(np.mean(score_pairs(model, tok, pairs, bs) - base))
    return 0.5 * (sp - sm), 0.5 * (sp + sm)


def fit_dirs(X, y, seed):
    """Plain and whitened directions on the seed's training half, each oriented so
    its training AUROC is at least 1/2."""
    from . import estimators as est
    tr, te = est.split_indices(y, seed=seed)
    th = est.mass_mean(X[tr], y[tr])
    if est.auroc(X[tr] @ th, y[tr]) < 0.5: th = -th
    thw = est.fisher(X[tr], y[tr])     # a failed Fisher fit raises; never substitute the plain direction
    if est.auroc(X[tr] @ thw, y[tr]) < 0.5: thw = -thw
    return th, thw, tr, te


# ---- per-pair score gradient -------------------------------------------------------------
def grad_for_texts(model, tok, block, texts, n_prompts, n_fulls):
    """sum_t d(sum log P(completion tokens)) / d(residual entering `block` at t) -> (B, d).

    For block = blocks(model)[l] this is the gradient at layer l, hidden_states[l]:
    the site Steerer(l) pushes at and extract_all_layers reads.
    """
    store = {}

    def hook(mod, args, kwargs):
        # Replace the block's input with a leaf that requires grad: this block and
        # every later one depend on it, .grad lands on exactly the tensor the
        # Steerer adds to, and backprop stops here rather than continuing to
        # the embeddings.
        h = args[0] if args else kwargs["hidden_states"]
        h2 = h.detach().requires_grad_(True)
        store["h"] = h2
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": h2}

    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.enable_grad():
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            logits = model(**enc).logits.float().log_softmax(-1)
            loss = 0.0
            for j in range(len(texts)):
                for t in range(n_prompts[j], n_fulls[j]):
                    loss = loss + logits[j, t - 1, enc["input_ids"][j, t]]
            (LOSS_SCALE * loss).backward()
            g = store["h"].grad.detach().float().sum(1) / LOSS_SCALE   # (B, d)
            return g.cpu().numpy().astype(np.float64)
    finally:
        handle.remove()
        store.clear()


# ---- steering cells and their summary ----------------------------------------------------
def run_cell(model, tok, Xl, y, pool, layer, alphas, seed, bs, dtype, n_pairs):
    """One steering cell: fit theta and theta_F on the seed's train half, then the
    odd and even response of the behavioral score at each alpha, in class-gap units."""
    X = Xl.astype(np.float64)
    th, thw, tr, te = fit_dirs(X, y, seed)
    pairs = pairs_for_seed(pool, seed, n_pairs)
    scale = class_gap(X[tr], y[tr])          # fit the unit on train, like theta
    rec = {"layer": layer, "seed": seed, "alpha_unit": scale,
           "alpha_unit_kind": "class_gap_norm",
           "sigma_along_theta": float(np.std(X @ th)),
           "d_prime_theta": scale / float(np.std(X @ th)),
           "n_pairs": len(pairs),
           "probe_auroc": auroc(X[te] @ th, y[te]),
           "probe_auroc_whitened": auroc(X[te] @ thw, y[te]), "alphas": {}}
    with Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = score_pairs(model, tok, pairs, bs)
        for a in alphas:
            ap, sp_ = antisym(model, tok, st, th, a, scale, pairs, base, bs, dtype)
            aw, sw_ = antisym(model, tok, st, thw, a, scale, pairs, base, bs, dtype)
            rec["alphas"][str(a)] = {"plain": {"antisym": ap, "sym": sp_},
                                     "whitened": {"antisym": aw, "sym": sw_}}
        st.set(None, 0, DEV, dtype)
    rec["baseline"] = float(base.mean())
    return rec, scale


def run_null(model, tok, Xl, y, pool, layer, alphas, bs, dtype, n_pairs, n_rand):
    """Steering null: the odd response along n_rand random unit directions, same unit."""
    X = Xl.astype(np.float64)
    scale = class_gap(X, y)                  # same unit as the theta arms
    pairs = pairs_for_seed(pool, 0, n_pairs)
    rng = np.random.default_rng(9000 + layer * 17)
    out = {"layer": layer, "n_rand": n_rand, "alphas": {}}
    with Steerer(model, layer) as st:
        st.set(None, 0, DEV, dtype)
        base = score_pairs(model, tok, pairs, bs)
        for a in alphas:
            vals = []
            for _ in range(n_rand):
                r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)
                av, _ = antisym(model, tok, st, r, a, scale, pairs, base, bs, dtype)
                vals.append(av)
            v = np.array(vals)
            out["alphas"][str(a)] = {"antisym_mean": float(v.mean()),
                                     "antisym_std": float(v.std(ddof=1)),
                                     "antisym_p95": float(np.percentile(np.abs(v), 95)),
                                     "n_draws": len(v),
                                     "draws": [float(x) for x in v]}
        st.set(None, 0, DEV, dtype)
    return out


def chi_summary(model, dataset, layer, alphas):
    """chi (median, IQR) for both arms over seeds, and A(1) against the null:
    z-score and signed rank p."""
    seeds = load_seed_cells(model, dataset, layer)
    null = load_null_cell(model, dataset, layer)
    if not seeds or null is None:
        return None
    plain_chi, whit_chi, plain_a1, whit_a1 = [], [], [], []
    for s in seeds:
        al = s["alphas"]
        pv = [al[str(a)]["plain"]["antisym"] for a in alphas]
        wv = [al[str(a)]["whitened"]["antisym"] for a in alphas]
        plain_chi.append(chi_origin(alphas, pv))
        whit_chi.append(chi_origin(alphas, wv))
        plain_a1.append(al["1.0"]["plain"]["antisym"])
        whit_a1.append(al["1.0"]["whitened"]["antisym"])
    null1 = null["alphas"]["1.0"]
    null_draws = np.array(null1["draws"])

    def rank_p(obs):
        if obs >= 0:
            return float(np.mean(null_draws >= obs))
        return float(np.mean(null_draws <= obs))

    plain_mean, whit_mean = np.mean(plain_a1), np.mean(whit_a1)
    return dict(
        layer=layer, n=len(seeds),
        chi_plain=np.median(plain_chi), chi_plain_iqr=np.subtract(*np.percentile(plain_chi, [75, 25])),
        chi_whit=np.median(whit_chi), chi_whit_iqr=np.subtract(*np.percentile(whit_chi, [75, 25])),
        plain_pos_frac=float(np.mean(np.array(plain_a1) > 0)),
        whit_pos_frac=float(np.mean(np.array(whit_a1) > 0)),
        plain_z=(plain_mean - null1["antisym_mean"]) / null1["antisym_std"],
        whit_z=(whit_mean - null1["antisym_mean"]) / null1["antisym_std"],
        plain_p=rank_p(plain_mean), whit_p=rank_p(whit_mean),
        null_p95=null1["antisym_p95"], null_std=null1["antisym_std"],
    )


# ---- the 400-pair pool and its arms ------------------------------------------------------
GRAD_SHORT = {"counterfact": "cf", "cities": "cities"}


def grad_npz(ds: str, s: int) -> str:
    """Per-pair gradient file for draw s (rogue_dimension.py gradient): seed 0 is
    the pooled run's file, the others carry the short dataset name."""
    name = f"score_gradient_{ds}_vectors.npz" if s == 0 else f"score_gradient_{GRAD_SHORT[ds]}_seed{s}_vectors.npz"
    return os.path.join(ART, name)


def pool_gradients(ds: str, seeds, layer: int = 28, pool: int = 400, n: int = 250, d: int = 2560):
    """Per-pair gradients for every pair of the fixed pool, assembled from the draws.

    Draw s fits on the n pairs rng(777 + s) chooses; a pair drawn by several seeds
    keeps the first seed's row (they agree to fp16 jitter). Returns (Gp, chosen,
    coverage): Gp is (pool, d) with NaN rows for pairs no seed drew, chosen maps
    seed -> its set of pool indices.
    """
    Gp = [None] * pool
    chosen = {}
    for s in seeds:
        p = grad_npz(ds, s)
        if not os.path.exists(p):
            continue
        idx = np.random.default_rng(777 + s).choice(pool, size=n, replace=False)
        chosen[s] = set(idx.tolist())
        G = np.load(p)[f"L{layer}_g_per_pair"].astype(np.float64)
        for j, i in enumerate(idx):
            if Gp[i] is None:
                Gp[i] = G[j]
    cov = sum(g is not None for g in Gp)
    Gp = np.array([g if g is not None else np.full(d, np.nan) for g in Gp])
    return Gp, chosen, cov


def complement_arms(ds: str, s: int, d_json: dict, layer: int = 28, d: int = 2560) -> dict:
    """Rebuild the unit direction of every arm in a steer_g_arm run (seed s), keyed
    as in its JSON, from the same seeded draws the run used."""
    z = np.load(grad_npz(ds, s))
    G = z[f"L{layer}_g_per_pair"].astype(np.float64)
    g = G.mean(0); g_hat = g / np.linalg.norm(g)
    out = {}
    arms = d_json["arms"]
    if "g" in arms: out["g"] = g_hat
    if "plain" in arms:
        t = z[f"L{layer}_theta"].astype(np.float64); out["plain"] = t / np.linalg.norm(t)
    rng = np.random.default_rng(s)
    for i in range(sum(k.startswith("rand") for k in arms)):
        r = rng.standard_normal(d); out[f"rand{i}"] = r / np.linalg.norm(r)
    targets = {"theta": z[f"L{layer}_theta"], "theta_F": z[f"L{layer}_theta_whitened"], "e2": z[f"L{layer}_e2"]}
    crng = np.random.default_rng(1000 + s)
    # the run iterates --controls in the order given; recover it from key order in the JSON
    ctrl_order = [k for k in arms if k in targets]
    for t in ctrl_order:
        v = np.asarray(targets[t], np.float64); v /= np.linalg.norm(v); rho = float(v @ g_hat)
        out[t] = v; vp = v - rho * g_hat; out[f"{t}_perp"] = vp / np.linalg.norm(vp)
        for i in range(sum(k.startswith(f"match_{t}_") for k in arms)):
            u = crng.standard_normal(d); u -= (u @ g_hat) * g_hat; u /= np.linalg.norm(u)
            out[f"match_{t}_{i}"] = rho * g_hat + np.sqrt(1 - rho ** 2) * u
    srng = np.random.default_rng(2000 + s)
    for i in range(sum(k.startswith("gshuf") for k in arms)):
        sgn = srng.choice([-1.0, 1.0], size=G.shape[0]); gs = (sgn[:, None] * G).mean(0); out[f"gshuf{i}"] = gs / np.linalg.norm(gs)
    return out


def check_gradient_file(vec_path: str, dataset: str, seed: int, n_pairs: int) -> None:
    """Refuse a per-pair gradient file made for a different draw.

    The npz carries no provenance; its companion JSON (same stem, written by
    `rogue_dimension.py gradient`) records dataset, seed and pairs per seed. Files
    from before the seed field existed are seed 0's. Guards against the 11 Sept
    contamination (g fit on one draw, scored on pairs overlapping it).
    """
    meta_path = str(vec_path)[: -len("_vectors.npz")] + ".json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"no provenance file {meta_path} for {vec_path}")
    meta = json.load(open(meta_path))
    short = lambda name: name.replace("_true_false", "")
    problems = []
    if short(meta["dataset"]) != short(dataset):
        problems.append(f"dataset {meta['dataset']} != {dataset}")
    if meta.get("seed", 0) != seed:
        problems.append(f"seed {meta.get('seed', 0)} != {seed}")
    if meta["n_pairs"] != n_pairs:
        problems.append(f"pairs per seed {meta['n_pairs']} != {n_pairs}")
    if problems:
        raise SystemExit(f"{vec_path} was made for a different draw: " + "; ".join(problems))


# ---- reading a steering run --------------------------------------------------------
def response_slopes(arm: dict) -> dict[float, float]:
    """A(alpha)/alpha for every alpha the arm was steered at, ascending in alpha."""
    a = arm["antisym"]
    return {float(k): a[k] / float(k) for k in sorted(a, key=float)}


def small_alpha_slope(arm: dict) -> float:
    """A(alpha)/alpha at the smallest alpha of the grid (chi read inside the window)."""
    return next(iter(response_slopes(arm).values()))


def noise_floor_ratio(run: dict, n: int | None = None) -> float:
    """Predicted chi(g_hat) / (c ||g_hat||) = ||g||^2 / (||g||^2 + tr(Sigma_G)/n).

    ||g||^2 is estimated from the run's fit half as ||g_hat||^2 - tr(Sigma_G)/n_fit.
    n=None: at the run's own n_fit, where the ratio is (||g_hat||^2 - tr/n_fit)/||g_hat||^2.
    """
    g2, trn = run["g_fit_norm"] ** 2, run["tr_sigma_g_over_n"]
    if n is None:
        return (g2 - trn) / g2
    gt2, trS = g2 - trn, trn * run["n_fit_pairs"]
    return gt2 / (gt2 + trS / n)


def pair_gradients(model, tok, block, pairs: list, bs: int, log: str | None = None) -> np.ndarray:
    """Per-pair gradient of the truth score, (n_pairs, d): d(ell_true - ell_false)/d(residual entering
    block), summed over positions (grad_for_texts on each completion, difference).
    log: a prefix for a progress line every 10 batches."""
    gs = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        per_side = {}
        for which in ("true", "false"):
            texts = [p["prompt"] + p[which] for p in chunk]
            n_prompts = [len(tok(p["prompt"]).input_ids) for p in chunk]
            n_fulls = [len(tok(p["prompt"] + p[which]).input_ids) for p in chunk]
            per_side[which] = grad_for_texts(model, tok, block, texts, n_prompts, n_fulls)
        gs.append(per_side["true"] - per_side["false"])
        if log is not None and (i // bs) % 10 == 0:
            print(f"  {log}  {i + len(chunk)}/{len(pairs)}", flush=True)
    return np.vstack(gs)
