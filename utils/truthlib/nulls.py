"""
Null distributions for a fitted direction: how much separation an arbitrary
direction achieves (random_direction_null), and how much a mass-mean fit to
shuffled labels achieves as N crosses Cover's capacity 2d (cover_n_sweep, and
per dataset excess_curve, its power-law fit and the PR collapse constant).

Also the in-sample attenuation control on a planted direction (planted_attenuation).

Drafted with the assistance of Claude (Anthropic).
"""
import math

import numpy as np
from scipy.stats import kurtosis

from .estimators import (
    auroc,
    d_prime,
    eigvals_desc,
    evaluate_direction,
    mass_mean,
    participation_ratio,
    within_class_cov,
)

N_NULL = 200        # random directions for the null


def random_direction_null(X, y, n_draws=N_NULL, seed=0):
    """Distribution of AUROC / d' over random unit directions.

    AUROC orientation is arbitrary for a random vector, so fold to >= 0.5:
    the null we care about is 'how much separation does an arbitrary direction
    achieve', not its sign.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    aus, dps = [], []
    for _ in range(n_draws):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        z = X @ u
        a = auroc(z, y)
        aus.append(max(a, 1 - a))
        dps.append(d_prime(z, y))
    aus, dps = np.array(aus), np.array(dps)
    return {"auroc_p50": float(np.percentile(aus, 50)),
            "auroc_p95": float(np.percentile(aus, 95)),
            "d_prime_p50": float(np.percentile(dps, 50)),
            "d_prime_p95": float(np.percentile(dps, 95))}


def cover_n_sweep(X, y, ns, d_model, seed=0):
    """True vs shuffled labels as N crosses the separating capacity 2d.

    Cover: for N points in general position in R^d, nearly every dichotomy is
    linearly separable while N << 2d. A shuffled labeling therefore separates
    about as well as the true one at small N/d, and collapses toward chance as
    N grows past capacity -- while a real truth direction persists.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for N in ns:
        if N > len(y):
            continue
        idx = rng.choice(len(y), size=N, replace=False)
        Xs, ys = X[idx], y[idx]
        if len(np.unique(ys)) < 2:
            continue
        th_true = mass_mean(Xs, ys)
        r_true = evaluate_direction(th_true, Xs, ys)

        y_shuf = rng.permutation(ys)
        th_shuf = mass_mean(Xs, y_shuf)
        r_shuf = evaluate_direction(th_shuf, Xs, y_shuf)

        rows.append({"N": int(N), "d_model": int(d_model),
                     "N_over_2d": float(N / (2 * d_model)),
                     "true": r_true, "shuffled": r_shuf})
    return rows


# ---- per-dataset shuffled-label law ------------------------------------------------------
def cover_spectrum(X, y):
    """Within-class spectrum diagnostics (np.cov, class-sorted rows: the dimensional-slack convention)."""
    lam = eigvals_desc(within_class_cov(X, y, ddof=1, class_sorted=True))
    tr = lam.sum()
    return {
        "participation_ratio": participation_ratio(lam),
        "lambda1_over_trace": float(lam[0] / tr),
        "kappa_12": float(lam[0] / max(lam[1], 1e-12)),
        "d_model": int(X.shape[1]),
    }


def geometric_grid(n_total, n_min, n_pts):
    """Geometric N grid from n_min up to and including the full set."""
    if n_total <= n_min:
        return [int(n_total)]
    g = np.geomspace(n_min, n_total, n_pts)
    return sorted({int(round(v)) for v in g} | {int(n_total)})


def excess_curve(X, y, d_model, ns, n_rep, seed=0):
    """Mean in-sample shuffled-label excess AUROC at each N, over n_rep draws.

    The cover_spectrum is recomputed on the first subsample at each N so that PR can
    be checked for drift with sample size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for N in ns:
        if N > len(y):
            continue
        exc, tru = [], []
        spec_at_N, first = None, True
        for _ in range(n_rep):
            idx = rng.choice(len(y), size=N, replace=False)
            Xs, ys = X[idx], y[idx]
            if len(np.unique(ys)) < 2:
                continue
            if first:
                spec_at_N = cover_spectrum(Xs, ys)
                first = False
            ysh = rng.permutation(ys)
            th = mass_mean(Xs, ysh)
            exc.append(evaluate_direction(th, Xs, ysh)["auroc"] - 0.5)
            tht = mass_mean(Xs, ys)
            tru.append(evaluate_direction(tht, Xs, ys)["auroc"])
        rows.append({"N": int(N), "N_over_2d": float(N / (2 * d_model)),
                     "excess_mean": float(np.mean(exc)),
                     "excess_sd": float(np.std(exc, ddof=1)),
                     "true_auroc_mean": float(np.mean(tru)),
                     "n_rep": len(exc),
                     "exhaustive": bool(N == len(y)),
                     "pr_at_N": float(spec_at_N["participation_ratio"]),
                     "lambda1_over_trace_at_N": float(spec_at_N["lambda1_over_trace"])})
    return rows


def powerlaw_fit(rows):
    """OLS and inverse-variance-weighted LS of log(excess) on log(N/2d).

    Also refits over truncated N windows. The fitted exponent depends on how
    far up in N the grid reaches, so comparing datasets that stop at different
    N compares windows, not datasets.
    """
    def ols(rr):
        x = np.log(np.array([r["N_over_2d"] for r in rr]))
        yv = np.log(np.array([max(r["excess_mean"], 1e-6) for r in rr]))
        n = len(x)
        b, loga = np.polyfit(x, yv, 1)
        resid = yv - (loga + b * x)
        sxx = ((x - x.mean()) ** 2).sum()
        se = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1) / sxx)) if sxx > 0 else float("nan")
        ss = 1.0 - resid.var() / yv.var() if yv.var() > 0 else float("nan")
        return float(np.exp(loga)), float(b), se, float(ss), n

    a, b, se, ss, n = ols(rows)
    out = {"amplitude": a, "exponent": b, "exponent_se": se,
           "r2": ss, "n_points": n}
    # log-space sd is sd/mean to first order; weight by its inverse square.
    x = np.log(np.array([r["N_over_2d"] for r in rows]))
    yv = np.log(np.array([max(r["excess_mean"], 1e-6) for r in rows]))
    rel = np.array([max(r["excess_sd"], 1e-9) / max(r["excess_mean"], 1e-9)
                    / max(np.sqrt(r["n_rep"]), 1.0) for r in rows])
    w = 1.0 / rel ** 2
    if len(rows) > 2 and np.all(np.isfinite(w)):
        bw, logaw = np.polyfit(x, yv, 1, w=np.sqrt(w))
        out["amplitude_wls"] = float(np.exp(logaw))
        out["exponent_wls"] = float(bw)
    out["windows"] = {}
    for cap in (400, 1500):
        sub = [r for r in rows if r["N"] <= cap]
        if len(sub) >= 3:
            aa, bb, ss_e, rr2, nn = ols(sub)
            out["windows"][f"N<={cap}"] = {"amplitude": aa, "exponent": bb,
                                           "exponent_se": ss_e, "n_points": nn}
    return out


def collapse(rows, pr):
    """C = excess * sqrt(N / PR): the effective-dimension collapse constant."""
    return [{"N": r["N"], "C": float(r["excess_mean"] * math.sqrt(r["N"] / pr)),
             "C_at_N": float(r["excess_mean"] * math.sqrt(r["N"] / r["pr_at_N"]))}
            for r in rows]


def shuffled_draw(X, y, rng, N):
    """One shuffled-label fit: return measured AUROC, d', and projection shape."""
    idx = rng.choice(len(y), size=N, replace=False)
    Xs = X[idx]
    ysh = rng.permutation(y[idx])
    if len(np.unique(ysh)) < 2:
        return None
    th = mass_mean(Xs, ysh)
    z = Xs @ th
    z1, z0 = z[ysh == 1], z[ysh == 0]
    # pooled within-class sd of the projection
    sp = math.sqrt(0.5 * (z1.var(ddof=1) + z0.var(ddof=1)))
    dprime = float((z1.mean() - z0.mean()) / sp) if sp > 0 else float("nan")
    auroc = float(evaluate_direction(th, Xs, ysh)["auroc"])
    zc = np.concatenate([z1 - z1.mean(), z0 - z0.mean()])
    return dprime, auroc, float(kurtosis(zc, fisher=True))


def pool_stats(X, y):
    Sig = within_class_cov(X, y, ddof=1, class_sorted=True)
    lam = eigvals_desc(Sig)
    tr, tr2 = lam.sum(), (lam ** 2).sum()
    return Sig, float(tr), float(tr2), float(tr ** 2 / tr2)


def planted_attenuation(d, N, rng):
    """Planted d' = 1 along a random unit u: (in-sample d', held-out d', |cos(theta-hat, u)|)."""
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    y = np.r_[np.zeros(N // 2), np.ones(N // 2)].astype(int)
    X = rng.standard_normal((N, d)) + np.outer(y - 0.5, u)
    perm = rng.permutation(N)
    tr, te = perm[: N // 2], perm[N // 2:]
    th = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    th /= np.linalg.norm(th)
    return d_prime(X[tr] @ th, y[tr]), d_prime(X[te] @ th, y[te]), float(abs(th @ u))
