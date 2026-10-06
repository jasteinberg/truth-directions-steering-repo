"""
The per-layer protocol of the SNR sweep: fit a direction on the train half,
orient it on train, score it on the held-out half against a random-direction
null on the same points; pick the layer that most clears its null; probe
whether the signal survives projecting out the top principal components.

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np

from .estimators import auroc, evaluate_direction, fisher, mass_mean, split_indices
from .nulls import random_direction_null

PCA_KS = [0, 1, 2, 4, 8, 16, 32, 64]   # superposition probe


def fit_eval_split(X, y, kind="plain", seed=0):
    """Fit the direction on train, evaluate d'/AUROC on held-out test.

    The mass-mean estimator has little capacity, but with d comparable to N the
    estimated mean difference absorbs O(sqrt(d/N)) of noise; scoring in-sample
    then rewards exactly the fluctuations that defined the direction. A random
    direction gets no such bonus, so an in-sample d' would overstate how far the
    truth direction clears its null -- the very comparison this study rests on.

    The direction's sign is fixed on TRAIN (so that train AUROC >= 0.5) and then
    held fixed on test. d' is sign-blind, but AUROC is not: without this the
    held-out AUROC can land below 0.5 for a direction that genuinely separates,
    and any d'-based layer selection would be blind to the flip.
    """
    tr, te = split_indices(y, seed=seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    if kind == "plain":
        theta = mass_mean(Xtr, ytr)
    else:
        theta = fisher(Xtr, ytr)
    if auroc(Xtr @ theta, ytr) < 0.5:      # orient on train only
        theta = -theta
    return evaluate_direction(theta, Xte, yte), (te, theta)


def analyze_dataset(A, y, layer_stride=1, seed=0):
    """A: (L, N, d) activations. Per-layer plain + whitened + null."""
    L = A.shape[0]
    layers = list(range(0, L, layer_stride))
    if layers[-1] != L - 1:
        layers.append(L - 1)
    rows = []
    for li in layers:
        X = A[li].astype(np.float64)
        plain, (te, _) = fit_eval_split(X, y, "plain", seed=seed)
        # a failed whitened fit raises: an {"error": ...} row would let a figure or table
        # silently drop the layer (no published sweep ever recorded one)
        whit, _ = fit_eval_split(X, y, "whitened", seed=seed)
        # null on the SAME held-out points, so the comparison is apples-to-apples
        null = random_direction_null(X[te], y[te], seed=seed + li)
        rows.append({"layer": int(li), "plain": plain,
                     "whitened": whit, "null": null})
        print(f"      L{li:>2}  AUROC={plain['auroc']:.3f}  d'={plain['d_prime']:.2f}"
              f"  (null p95={null['auroc_p95']:.3f})", flush=True)
    return rows


def best_layer(rows):
    """Layer whose held-out AUROC most exceeds its own random-direction null.

    AUROC is sign-aware (d' is not), so a layer where the fitted direction is
    anti-predictive on held-out data cannot win. Layer 0 is the embedding, not
    a transformer block: on templated statements the last-token embedding is
    near-constant, giving a degenerate d'; it is excluded from selection.
    """
    cand = [r for r in rows if r["layer"] > 0] or rows
    scored = [(r["plain"]["auroc"] - r["null"]["auroc_p95"], r["layer"])
              for r in cand]
    return max(scored)[1]


def superposition_probe(X, y, ks=PCA_KS, seed=0):
    """d' of the mass-mean direction after projecting out the top-k PCs.

    Tests whether the truth signal merely rides the highest-variance
    (most salient) subspace, or occupies its own low-variance directions.
    The PCA basis and the direction are both fit on train, applied to test.
    """
    tr, te = split_indices(y, seed=seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    out = []
    for k in ks:
        if k == 0:
            Atr, Ate = Xtr, Xte
        else:
            V = Vt[:k]                             # (k, d), fit on train only
            Atr = Xtr - (Xtr @ V.T) @ V
            Ate = Xte - (Xte @ V.T) @ V
        theta = mass_mean(Atr, ytr)
        r = evaluate_direction(theta, Ate, yte)
        out.append({"k": int(k), **r})
    return out


# ---- transfer ----------------------------------------------------------------------------
def fit_oriented(X, y, seed=0):
    """Mass-mean and Fisher directions fit on the train half, each oriented on train.
    A failed Fisher fit raises. Returns (theta, theta_F, test indices)."""
    tr, te = split_indices(y, seed=seed)
    th = mass_mean(X[tr], y[tr])
    if auroc(X[tr] @ th, y[tr]) < 0.5:
        th = -th
    thw = fisher(X[tr], y[tr])
    if auroc(X[tr] @ thw, y[tr]) < 0.5:
        thw = -thw
    return th, thw, te
