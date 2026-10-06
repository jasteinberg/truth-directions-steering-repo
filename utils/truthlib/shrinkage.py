"""
Ledoit-Wolf shrinkage of the within-class covariance: its intensity and why it is
large (beta^2 against the Gaussian reference), the shrinkage path from one
eigendecomposition, and cross-validated rho.

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
from sklearn.covariance import LedoitWolf

from .estimators import (
    d_prime,
    eig_ratio,
    eigvals_desc,
    participation_ratio,
    shrunk_spectrum,
    split_indices,
    within_class_center,
)


def ledoit_wolf_parts(Xc: np.ndarray) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """sklearn's Ledoit-Wolf pieces written out, so beta^2 and delta^2 are inspectable.

    Xc is centred data (N, d). Returns (C, mu, delta2, beta2_obs, beta2_gauss,
    rho_obs, rho_gauss): C divides by N; rho_obs is the observed LW intensity,
    rho_gauss the intensity a Gaussian with the same C would get, from
    E sum ||x||^4 = N[(tr C)^2 + 2 tr C^2].
    """
    N, d = Xc.shape
    C = (Xc.T @ Xc) / N
    mu = float(np.trace(C) / d)
    normF2 = float((C ** 2).sum())
    delta2 = (normF2 - d * mu ** 2) / d          # ||C - mu I||_F^2 / d
    sq = (Xc ** 2).sum(1)                        # ||x_i||^2
    beta2_obs = float((sq ** 2).sum() - N * normF2) / (d * N ** 2)
    rho_obs = min(beta2_obs, delta2) / delta2

    # Gaussian reference: E sum ||x||^4 = N[(tr C)^2 + 2 tr C^2]
    trC = float(np.trace(C))
    trC2 = normF2
    beta2_gauss = (trC ** 2 + 2 * trC2 - trC2) / (d * N)
    rho_gauss = min(beta2_gauss, delta2) / delta2
    return C, mu, delta2, beta2_obs, beta2_gauss, rho_obs, rho_gauss


def shrinkage_eig(X, y):
    """Eigendecompose the within-class covariance once; return pieces for any rho."""
    Xc = within_class_center(X, y)
    C = (Xc.T @ Xc) / len(y)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)
    mu = float(lam.mean())
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    return lam, V, mu, dlt


def shrunk_fisher(lam, V, mu, dlt, rho):
    """Fisher direction at shrinkage rho from shrinkage_eig's pieces, unit norm:
    V diag(1 / ((1 - rho) lam + rho mu)) V^T delta, with no new eigendecomposition."""
    w = V.T @ dlt
    t = V @ (w / ((1.0 - rho) * lam + rho * mu))
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def stratified_folds(y, K, rng):
    """Class-stratified K folds."""
    out = [[] for _ in range(K)]
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        for f, chunk in enumerate(np.array_split(idx, K)):
            out[f].append(chunk)
    return [np.concatenate(c) for c in out]


def shrinkage_row(X, y, seed=0):
    """LW intensity on the train half and the raw / shrunk spectrum it implies."""
    tr, _ = split_indices(y, seed=seed)
    Xtr, ytr = X[tr], y[tr]
    Xc = within_class_center(Xtr, ytr)

    lw = LedoitWolf(assume_centered=True).fit(Xc)
    rho = float(lw.shrinkage_)

    # Raw within-class covariance on the same rows (1/N convention, as LW uses).
    C = (Xc.T @ Xc) / Xc.shape[0]
    lam = eigvals_desc(C)
    mu = float(np.trace(C) / C.shape[0])
    lam_shr = shrunk_spectrum(lam, rho, mu)

    return {
        "n_train": int(Xtr.shape[0]),
        "d": int(X.shape[1]),
        "rho": rho,
        "lambda1_over_lambda2_raw": eig_ratio(lam),
        "lambda1_over_lambda2_shrunk": eig_ratio(lam_shr),
        "lambda1_over_trace_raw": float(lam[0] / lam.sum()),
        "PR_raw": participation_ratio(lam),
    }


def shrunk_ratios(C, rho, mu):
    """(spectrum, lambda1/lambda2 raw, lambda1/lambda2 after shrinkage rho, PR)."""
    lam = eigvals_desc(C)
    return lam, eig_ratio(lam), eig_ratio(shrunk_spectrum(lam, rho, mu)), participation_ratio(lam)


def lw_decomposition(X, y, seed=0):
    """Ledoit-Wolf intensity against its Gaussian reference, on the train half and the
    full set, with v1 and bulk kurtosis; and the full spectrum shrunk by train rho."""
    tr, _ = split_indices(y, seed=seed)
    out = {}
    for tag, rows in (("train", tr), ("full", np.arange(len(y)))):
        Xc = within_class_center(X[rows], y[rows])
        C, mu, d2, b2o, b2g, rho_o, rho_g = ledoit_wolf_parts(Xc)
        chk = float(LedoitWolf(assume_centered=True).fit(Xc).shrinkage_)
        lam, r_raw, r_shr, PR = shrunk_ratios(C, rho_o, mu)

        # excess kurtosis of the within-class-centered projection onto v1
        v1 = np.linalg.eigh(C)[1][:, -1]
        p = Xc @ v1
        s = p.std()
        kurt_v1 = float(((p / s) ** 4).mean() - 3.0) if s > 0 else float("nan")
        # and of a typical bulk direction, for contrast
        vb = np.linalg.eigh(C)[1][:, -50]
        pb = Xc @ vb
        sb = pb.std()
        kurt_bulk = float(((pb / sb) ** 4).mean() - 3.0) if sb > 0 else float("nan")

        out[tag] = {
            "N": int(len(rows)), "rho": rho_o, "rho_sklearn_check": chk,
            "rho_gauss": rho_g, "rho_over_gauss": rho_o / rho_g if rho_g > 0 else None,
            "beta2_obs": b2o, "beta2_gauss": b2g, "delta2": d2,
            "beta2_obs_over_gauss": b2o / b2g if b2g > 0 else None,
            "lambda1_over_lambda2_raw": r_raw, "lambda1_over_lambda2_shrunk": r_shr,
            "PR_raw": PR, "excess_kurtosis_v1": kurt_v1,
            "excess_kurtosis_bulk50": kurt_bulk,
        }

    # full-sample shrunk_ratios shrunk by the TRAIN-selected rho (Julia's proposal)
    Xc_f = within_class_center(X, y)
    C_f = (Xc_f.T @ Xc_f) / len(y)
    mu_f = float(np.trace(C_f) / C_f.shape[0])
    _, _, r_cross, _ = shrunk_ratios(C_f, out["train"]["rho"], mu_f)
    out["full_shrunk_by_train_rho"] = {"lambda1_over_lambda2": r_cross}
    return out


def cv_layer(X, y, tr, te, K, seed, grid, n_null=200):
    """K-fold CV of rho over `grid` on the train half, then one held-out scoring pass
    at rho* and at rho_LW, against a random-direction null."""
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    rng = np.random.default_rng(seed + 1)

    # ---- inner CV, train half only -------------------------------------
    fs = stratified_folds(ytr, K, rng)
    scores = {r: [] for r in grid}
    for f in range(K):
        va = fs[f]
        fit = np.concatenate([fs[g] for g in range(K) if g != f])
        if len(np.unique(ytr[fit])) < 2 or len(np.unique(ytr[va])) < 2:
            continue
        lam, V, mu, dlt = shrinkage_eig(Xtr[fit], ytr[fit])
        for r in grid:
            t = shrunk_fisher(lam, V, mu, dlt, r)
            scores[r].append(d_prime(Xtr[va] @ t, ytr[va]))
    cv = {r: float(np.mean(v)) for r, v in scores.items() if v}
    rho_star = max(cv, key=cv.get)

    # ---- one scoring pass on the test half -----------------------------
    lam, V, mu, dlt = shrinkage_eig(Xtr, ytr)
    rho_lw = float(LedoitWolf(assume_centered=True)
                   .fit(within_class_center(Xtr, ytr)).shrinkage_)

    def scored(r):
        t = shrunk_fisher(lam, V, mu, dlt, r)
        if (Xtr @ t)[ytr == 1].mean() < (Xtr @ t)[ytr == 0].mean():
            t = -t                                   # orient on train only
        return d_prime(Xte @ t, yte)

    nulls = []
    rngn = np.random.default_rng(seed)
    for _ in range(n_null):
        u = rngn.standard_normal(X.shape[1]); u /= np.linalg.norm(u)
        nulls.append(d_prime(Xte @ u, yte))

    return {"rho_star": rho_star, "rho_LW": rho_lw,
            "d_F_cv": scored(rho_star), "d_F_LW": scored(rho_lw),
            "cv_curve": {str(k): v for k, v in cv.items()},
            "null_p95": float(np.percentile(nulls, 95))}
