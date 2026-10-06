"""
Estimators for the truth-directions analysis.

NumPy and scikit-learn only, no torch, so every figure and table can be rebuilt
from the committed JSON on any machine. Where analyses use different conventions
(ddof, row order of the centred data), the choice is an explicit argument, so each
committed artifact reproduces bit for bit.

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


# ---- splitting and centring ---------------------------------------------
def split_indices(y, frac=0.5, seed=0):
    """Class-stratified train/test split (seed 0 is the post's split)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        tr.append(idx[:k]); te.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)


def within_class_center(X, y):
    """Centre each class on its own mean (returns data, not a covariance)."""
    Xc = X.copy()
    for lab in (0, 1):
        Xc[y == lab] = X[y == lab] - X[y == lab].mean(0)
    return Xc


def within_class_cov(X, y, ddof=0, class_sorted: bool = False):
    """Within-class covariance C-hat.

    ddof=0 divides by N, the post's definition and the Ledoit-Wolf convention.
    ddof=1 is np.cov, the convention of the geometry observables, the rogue-dimension
    spectrum and the dimensional-slack analyses; kept so those artifacts reproduce
    exactly. Ratios
    (lambda1/tr, lambda1/lambda2, PR) and eigenvectors are identical under both.

    class_sorted=True stacks the centred class-0 rows above the class-1 rows
    before summing, as the dimensional-slack analyses do. Same matrix, different summation
    order, so it differs in the last bits; kept for exact reproduction.
    """
    if class_sorted:
        Xc = np.vstack([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    else:
        Xc = within_class_center(X, y)
    if ddof == 1:
        return np.cov(Xc, rowvar=False)
    return (Xc.T @ Xc) / len(y)


# ---- separation -----------------------------------------------------------
def d_prime(z, y):
    """Held-out d' of a projection z: |m1 - m0| / sqrt(pooled variance).

    Returns 0 when the pooled variance vanishes (the layer-0 embedding on
    templated sets), which is what makes those rows read AUROC 0.500.
    """
    a, b = z[y == 1], z[y == 0]
    den = 0.5 * (a.var(ddof=1) + b.var(ddof=1))
    return float(abs(a.mean() - b.mean()) / np.sqrt(den)) if den > 0 else 0.0


def auroc(z, y):
    return float(roc_auc_score(y, z))


# ---- directions -----------------------------------------------------------
def class_gap(X, y):
    """||mu1 - mu0||, the steering unit c: a property of dataset and layer,
    never of the direction (std(X @ theta) was the extend_null scale bug)."""
    return float(np.linalg.norm(X[y == 1].mean(0) - X[y == 0].mean(0)))


def mass_mean(X, y):
    """theta-hat = delta-hat / ||delta-hat||."""
    t = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def fisher(X, y, rho=None):
    """Whitened direction Sigma-hat^{-1} delta-hat, unit norm.

    rho=None: sklearn Ledoit-Wolf precision, the post's theta-hat_F.
    rho=float: explicit shrinkage (1-rho) C + rho (tr C / d) I, C divided by N,
    solved rather than inverted. The two paths agree to floating point, not
    bitwise, at rho = rho_LW.
    """
    Xc = within_class_center(X, y)
    dlt = X[y == 1].mean(0) - X[y == 0].mean(0)
    if rho is None:
        t = LedoitWolf(assume_centered=True).fit(Xc).precision_ @ dlt
    else:
        C = (Xc.T @ Xc) / len(y)
        mu = np.trace(C) / C.shape[0]
        S = (1.0 - rho) * C + rho * mu * np.eye(C.shape[0])
        t = np.linalg.solve(S, dlt)
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


# ---- massive coordinates ------------------------------------------------------
def massive_mask(X, mag=100.0, rel=100.0, frac=0.5):
    """Flag statements that LACK the near-universal massive activation.

    Coordinate j is massive when |med_i x_ij| exceeds `mag` and `rel` times the
    median absolute activation (Sun et al. use 1000x; relaxed to 100x because the
    dominant coordinate runs only 564-587x the median at pythia-2.8b L24-L28,
    and every value from 100x to 560x flags the same eleven). Statement i is a
    dropper when |x_ij - med_j| > frac |med_j| for some massive j. Never touches
    v1, the class means, or the labels.

    Returns (dropped, med_all, massive).
    """
    A = np.abs(X)
    med_all = float(np.median(A))
    med_j = np.median(X, axis=0)
    massive = (np.abs(med_j) > mag) & (np.abs(med_j) > rel * med_all)
    if not massive.any():
        return np.zeros(len(X), bool), med_all, massive
    dev = np.abs(X[:, massive] - med_j[massive]) > frac * np.abs(med_j[massive])
    return dev.any(1), med_all, massive


# ---- spectra and steering summaries ---------------------------------------------
def participation_ratio(lam):
    """(sum lam)^2 / sum lam^2: the number of directions the spectrum occupies."""
    tr = lam.sum()
    return float(tr ** 2 / (lam ** 2).sum())


def eigh_desc(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of a symmetric matrix, largest first, unclipped."""
    w, V = np.linalg.eigh(C)
    return w[::-1], V[:, ::-1]


def eigvals_desc(C: np.ndarray) -> np.ndarray:
    """Eigenvalues only, largest first, clipped at 0 (round-off leaves the null
    space slightly negative when N < d)."""
    return np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)


def eig_ratio(lam: np.ndarray) -> float:
    """lambda1 / lambda2 of a descending spectrum; inf when lambda2 vanishes."""
    return float(lam[0] / lam[1]) if lam[1] > 0 else float("inf")


def shrunk_spectrum(lam: np.ndarray, rho: float, mu: float) -> np.ndarray:
    """Spectrum of (1 - rho) C + rho mu I: same eigenvectors, each eigenvalue
    pulled toward mu = tr C / d."""
    return (1.0 - rho) * lam + rho * mu


def chi_origin(alphas, values):
    """Steering susceptibility: least-squares slope of A against h through 0."""
    a = np.array(alphas, dtype=float)
    v = np.array(values, dtype=float)
    return float((a * v).sum() / (a * a).sum())


# ---- evaluation and geometry -------------------------------------------------------------
def accuracy_midpoint(z, y):
    """Threshold at the midpoint of the class means (the mass-mean rule)."""
    thr = 0.5 * (z[y == 1].mean() + z[y == 0].mean())
    pred = (z > thr).astype(int)
    if pred.mean() and (pred == y).mean() < 0.5:
        pred = 1 - pred          # orientation is arbitrary; take the better sign
    return float((pred == y).mean())


def evaluate_direction(theta, X, y):
    z = X @ theta
    return {"auroc": auroc(z, y), "d_prime": d_prime(z, y),
            "acc": accuracy_midpoint(z, y)}


def observables(X, y, shrink=True):
    """Full-set within-class geometry at one layer (geometry_observables.json).

    C-hat here is np.cov (divides by N-1); every ratio reported is invariant.
    """
    d = X.shape[1]
    delta = X[y == 1].mean(0) - X[y == 0].mean(0)
    th = delta / np.linalg.norm(delta)

    Xc = within_class_center(X, y)
    C = np.cov(Xc, rowvar=False)
    w, V = eigh_desc(C)
    v1 = V[:, 0]

    PR = participation_ratio(w)

    p = X @ th
    d_mm = d_prime(p, y)

    # optimal linear separation, shrunk inverse (raw C is singular when N < d)
    if shrink:
        P = LedoitWolf(assume_centered=True).fit(Xc).precision_
        d_maha = float(np.sqrt(max(delta @ P @ delta, 0.0)))
    else:
        d_maha = float("nan")

    return {
        "d_model": int(d), "n": int(len(y)),
        "PR": PR, "PR_over_d": PR / d,
        "lambda1_over_trace": float(w[0] / w.sum()),
        "lambda1_over_lambda2": float(w[0] / w[1]),
        "cos_theta_v1": float(abs(th @ v1)),
        "d_mass_mean": d_mm,
        "d_mahalanobis": d_maha,
        "hidden_signal_ratio": d_maha / d_mm if d_mm > 0 else float("nan"),
        "var_along_theta": float(p.var(ddof=1)),
        "delta_norm": float(np.linalg.norm(delta)),
        "sqrt_trace_Sigma": float(np.sqrt(w.sum())),
        "theta": th.astype(np.float32).tolist(),
    }


# ---- gradient geometry -------------------------------------------------------------------
def rogue_and_gap(X, y, tr):
    """v1 (leading eigenvector of the within-class covariance) and e2."""
    Xtr = X[tr]; ytr = y[tr]
    Xc = np.vstack([Xtr[ytr == k] - Xtr[ytr == k].mean(0) for k in (0, 1)])
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C)
    v1 = V[:, -1]
    lam = w[::-1]
    delta = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    d_perp = delta - (v1 @ delta) * v1
    e2 = d_perp / np.linalg.norm(d_perp)
    return v1, e2, lam


def cosine(a, b):
    """cos(a, b) as a float."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cos_bootstrap_interval(G, w, rng, B=10000):
    """[point, lo, hi] for cosine(mean_i g_i, w): B bootstrap resamples over pairs, 95% percentile interval."""
    n = len(G)
    point = cosine(G.mean(0), w)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.array([cosine(G[i].mean(0), w) for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [point, float(lo), float(hi)]
