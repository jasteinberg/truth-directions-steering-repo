"""
Within-class geometry at one layer, and the rogue dimension.

    gram_observables   PR, lambda ratios, cos(theta, v1), d' via the N x N Gram matrix
    massive_row        massive coordinates and droppers at one layer
    rogue_spectrum     spectrum and v1 oriented with theta; rogue_arms builds the
                       theta / v1 / theta_perp / random steering arms
    v1_outlier_mask, outlier_geometry, held_out_cleaning
                       the outlier appendix: the circular v1 selection, in-sample
                       geometry, and held-out d' under three cleaning regimes

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
from sklearn.covariance import LedoitWolf

from .estimators import (
    auroc,
    d_prime,
    eigh_desc,
    fisher,
    mass_mean,
    massive_mask,
    split_indices,
    within_class_center,
    within_class_cov,
)


def gram_observables(X, y):
    """PR, lambda ratios, cos(theta, v1), d_mm on the full sample."""
    X = X.astype(np.float64)
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)

    Xc = within_class_center(X, y)
    n = len(y)
    G = (Xc @ Xc.T) / (n - 1)              # N x N, same nonzero spectrum as C
    w, U = np.linalg.eigh(G)
    w, U = w[::-1], U[:, ::-1]
    w = np.clip(w, 0.0, None)

    v1 = Xc.T @ U[:, 0]
    v1 /= np.linalg.norm(v1)

    p = X @ th
    d_mm = d_prime(p, y)

    tr, sq = float(w.sum()), float((w ** 2).sum())
    if sq <= 0 or tr <= 0 or w[1] <= 0:
        # layer 0 on templated sets: the last-token embedding is near-constant
        # within a dataset, so the within-class covariance vanishes. The post
        # excludes this layer from selection for the same reason.
        return {"degenerate": True, "delta_norm": float(np.linalg.norm(delta))}
    return {
        "PR": tr ** 2 / sq,
        "lambda1_over_trace": float(w[0]) / tr,
        "lambda1_over_lambda2": float(w[0] / w[1]),
        "cos_theta_v1": float(abs(th @ v1)),
        "d_mass_mean": d_mm,
        "delta_norm": float(np.linalg.norm(delta)),
    }


def massive_row(X):
    """Massive coordinates and droppers at one layer, plus the coordinate ratio."""
    X = X.astype(np.float64)
    dropped, med_all, massive = massive_mask(X)
    n_massive = int(massive.sum())
    row = {"n_massive_coords": n_massive,
           "n_droppers": int(dropped.sum()),
           "median_abs_activation": med_all}
    if n_massive:
        med_j = np.median(X, axis=0)
        big = np.abs(med_j[massive])
        row["max_coord_ratio"] = float(big.max() / med_all)
        row["massive_coords"] = [int(c) for c in np.flatnonzero(massive)]
    else:
        # how close does the largest coordinate get to qualifying?
        med_j = np.median(X, axis=0)
        row["max_coord_ratio"] = float(np.abs(med_j).max() / med_all)
        row["massive_coords"] = []
    return row


def rogue_spectrum(X, y):
    """Within-class covariance rogue_spectrum + how the mass-mean direction sits in it."""
    C = within_class_cov(X, y, ddof=1)
    w, V = eigh_desc(C)
    v1 = V[:, 0]
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)
    if th @ v1 < 0:           # v1's sign is arbitrary; orient it with theta
        v1 = -v1
    return {
        "lambda1_over_trace": float(w[0] / w.sum()),
        "lambda1": float(w[0]), "lambda2": float(w[1]),
        "cos_theta_v1": float(abs(th @ v1)),
        "var_along_theta_frac": float((th @ C @ th) / w.sum()),
        "delta_over_sqrt_trace": float(np.linalg.norm(delta) / np.sqrt(w.sum())),
    }, v1, C


def rogue_arms(X, y, seed=0):
    """theta, v1 (label-free), theta_perp (after removing v1), random."""
    tr, te = split_indices(y, seed=seed)
    _, v1, _ = rogue_spectrum(X[tr], y[tr])

    th = mass_mean(X[tr], y[tr])
    if auroc(X[tr] @ th, y[tr]) < 0.5: th = -th

    Xp = X - np.outer(X @ v1, v1)          # project out the rogue dimension
    thp = mass_mean(Xp[tr], y[tr])
    if auroc(Xp[tr] @ thp, y[tr]) < 0.5: thp = -thp

    rng = np.random.default_rng(4242 + seed)
    r = rng.standard_normal(X.shape[1]); r /= np.linalg.norm(r)

    # orient v1 for steering the same way theta is oriented
    if auroc(X[tr] @ v1, y[tr]) < 0.5: v1 = -v1

    arms = {"theta": (th, X), "v1": (v1, X), "theta_perp": (thp, Xp), "random": (r, X)}
    aur = {k: auroc(Xa[te] @ u, y[te]) for k, (u, Xa) in arms.items()}
    return arms, aur, te


def v1_outlier_mask(X, y, zmax=4.0):
    """The circular selection, kept only so the two sets can be compared."""
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    v1 = np.linalg.eigh(C)[1][:, -1]
    p = Xc @ v1
    return np.abs(p) / p.std() > zmax


def outlier_geometry(X, y):
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    w, V = np.linalg.eigh(C)
    lam = np.clip(w[::-1], 0.0, None)
    th = mass_mean(X, y)
    P = LedoitWolf(assume_centered=True).fit(Xc).precision_
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    return {
        "lambda1_over_lambda2": float(lam[0] / lam[1]),
        "lambda1_over_trace": float(lam[0] / lam.sum()),
        "PR": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "cos_theta_v1": float(abs(th @ V[:, -1])),
        "d_mahalanobis_INSAMPLE": float(np.sqrt(dlt @ P @ dlt)),
    }


def held_out_cleaning(X, y, drop, seed=0, n_null=200):
    """Fit on train, score on test. Three cleaning regimes."""
    tr, te = split_indices(y, seed=seed)
    keep = ~drop
    out = {}
    regimes = {
        "all":          (tr, te),
        "clean_both":   (tr[keep[tr]], te[keep[te]]),
        "clean_train":  (tr[keep[tr]], te),
    }
    for tag, (a, b) in regimes.items():
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        row = {"n_train": int(len(a)), "n_test": int(len(b))}
        for kind, fn in (("mm", mass_mean), ("F", fisher)):
            t = fn(X[a], y[a])
            if d_prime(X[a] @ t, y[a]) and (X[a] @ t)[y[a] == 1].mean() < \
               (X[a] @ t)[y[a] == 0].mean():
                t = -t                       # orient on train only
            row[f"d_{kind}"] = d_prime(X[b] @ t, y[b])
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            u = rng.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
            nulls.append(d_prime(X[b] @ u, y[b]))
        row["null_p95"] = float(np.percentile(nulls, 95))
        out[tag] = row
    return out
