"""
Response against fluctuation: where the steering response lies in the activation spectrum.

The section "Response against fluctuation" of part 2, "Steering Vectors and the Limits of Linear Response".
Projects the per-pair score gradient onto the within-class eigendirections and
compares the response spectrum with the fluctuation spectrum, layer by layer.

Subcommands:

    spectrum               Response against fluctuation

Run from the repo root:  python scripts/response_fluctuation.py <subcommand> [-h]

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
from utils.truthlib import estimators as est

# =============================================================================
# spectrum
# =============================================================================
ART = REPO / "artifacts"


MODEL_TAG = "pythia-2.8b"


DATASETS = [("counterfact_true_false", "counterfact"), ("cities", "cities")]


LAYERS = [8, 12, 16, 20, 24, 28]


TOPK = [1, 2, 10, 100]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_spectrum(argv=None):
    """Response against fluctuation. Project the score gradient g onto the eigenbasis
    of the within-class covariance C-hat and compare the response spectrum
    r_i = |g . v_i|^2 / ||g||^2 with the fluctuation spectrum lambda_i.

    A random direction has E[r_i] = 1/d for every i, flat in lambda. A
    difference-in-means estimator is pulled toward large lambda_i; the question
    is whether g is. C-hat is taken on the full sample (N = 1198 < d = 2560), so
    about d - N + 2 eigenvalues are exactly zero: the response in that null
    space is response along directions the sample cannot assign a variance to,
    and is reported separately rather than plotted.

    Reads: artifacts/act_cache/{model}__{dataset}.npz (activations, local only)
           artifacts/score_gradient_{short}_vectors.npz (g, v1 per layer; written by
           `rogue_dimension.py gradient`)
    Writes: artifacts/response_spectrum.json (summary numbers)
            artifacts/response_spectrum_vectors.npz (lambda and r per layer)
    """
    argparse.ArgumentParser(description=run_spectrum.__doc__).parse_args(argv)
    out = {"model": MODEL_TAG, "note": "C-hat on the full sample; eigenvalues below "
           "1e-8 * lambda_1 counted as the null space", "datasets": {}}
    vec = {}
    for ds, short in DATASETS:
        z = np.load(ART / "act_cache" / f"{MODEL_TAG}__{ds}.npz")
        y = z["y"]
        gz = np.load(ART / f"score_gradient_{short}_vectors.npz")
        out["datasets"][short] = {}
        for L in LAYERS:
            X = z[f"L{L}"].astype(np.float64)
            C = est.within_class_cov(X, y)
            w, V = np.linalg.eigh(C)
            w, V = w[::-1], V[:, ::-1]
            g = gz[f"L{L}_g"].astype(np.float64)
            g_hat = g / np.linalg.norm(g)
            r = (V.T @ g_hat) ** 2
            # noise-corrected spectrum: g_hat is a 250-pair mean, so each
            # (g.v_i)^2 carries a floor v_i^T Sigma_G v_i / n. Subtract it.
            G = gz[f"L{L}_g_per_pair"].astype(np.float64)
            n = G.shape[0]
            Gc = G - G.mean(0)
            noise_i = ((Gc @ V) ** 2).sum(0) / (n - 1) / n       # v_i^T Sigma_G v_i / n
            g2_true = float(g @ g - noise_i.sum())
            r_true = ((V.T @ g) ** 2 - noise_i) / g2_true
            noise_share = float(noise_i.sum() / (g @ g))
            keep = w > 1e-8 * w[0]
            d, n_keep = len(w), int(keep.sum())
            v1_check = abs(float(V[:, 0] @ gz[f"L{L}_v1"]))
            cum_r = np.cumsum(r[keep]); cum_lam = np.cumsum(w[keep]) / w[keep].sum()
            rec = {
                "d": d, "n_nonzero": n_keep, "N": int(len(y)),
                "v1_matches_npz": v1_check,
                "null_space_fraction": float(r[~keep].sum()),
                "random_per_axis": 1.0 / d,
                "top_k_fraction": {str(k): float(r[:k].sum()) for k in TOPK},
                "top_k_random": {str(k): k / d for k in TOPK},
                "lam_weighted_fraction": float((r[keep] * w[keep]).sum() / w[keep].sum()),
                "spearman_r_lambda": spearman(r[keep], w[keep]),
                # rank k at which the response reaches half its nonzero mass, vs variance
                "rank_half_response": int(np.searchsorted(cum_r, 0.5 * cum_r[-1]) + 1),
                "rank_half_variance": int(np.searchsorted(cum_lam, 0.5) + 1),
                "lam1_over_tr": float(w[0] / w.sum()),
                "noise_share_of_g2": noise_share,
                "true": {
                    "null_space_fraction": float(r_true[~keep].sum()),
                    "top_k_fraction": {str(k): float(r_true[:k].sum()) for k in TOPK},
                    "spearman_r_lambda": spearman(r_true[keep], w[keep]),
                    "pr_response": float(1.0 / (r_true ** 2).sum()),
                    "pr_fluctuation": float(est.participation_ratio(w)),
                    "noise_top100_fraction": float(noise_i[:100].sum() / noise_i.sum()),
                    "noise_null_fraction": float(noise_i[~keep].sum() / noise_i.sum()),
                },
            }
            out["datasets"][short][str(L)] = rec
            vec[f"{short}_L{L}_lam"] = w
            vec[f"{short}_L{L}_r"] = r
            vec[f"{short}_L{L}_r_true"] = r_true
            vec[f"{short}_L{L}_noise"] = noise_i
            t = rec["true"]
            print(f"{short:11s} L{L:<3d} noise {noise_share:.2f} | raw: null {rec['null_space_fraction']:.3f} "
                  f"top100 {rec['top_k_fraction']['100']:.3f} | true: null {t['null_space_fraction']:.3f} "
                  f"top1 {t['top_k_fraction']['1']:+.4f} top10 {t['top_k_fraction']['10']:.3f} "
                  f"top100 {t['top_k_fraction']['100']:.3f} (rand {100/d:.3f}) sp {t['spearman_r_lambda']:+.3f} "
                  f"PR_resp {t['pr_response']:.0f} PR_fluc {t['pr_fluctuation']:.1f} | noise top100 "
                  f"{t['noise_top100_fraction']:.3f} null {t['noise_null_fraction']:.3f}", flush=True)
    with open(ART / "response_spectrum.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(ART / "response_spectrum_vectors.npz", **vec)
    print("wrote", ART / "response_spectrum.json")


COMMANDS = {
    "spectrum": run_spectrum,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
