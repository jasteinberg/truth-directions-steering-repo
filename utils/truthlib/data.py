"""
Datasets: the Marks & Tegmark true/false sets and the contrastive completion pairs
built from them for the behavioral score. pandas and NumPy only, no torch.

Drafted with the assistance of Claude (Anthropic).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..env import ENV

REPO = Path(__file__).resolve().parents[2]

DATA = Path(ENV.GOT)

# Main tier capped at its smallest member (companies_true_false, N=1199) so
# that d', transfer, and N/d are directly comparable across datasets.
MAIN_TIER = [
    "cities", "neg_cities", "larger_than", "smaller_than",
    "cities_cities_conj", "cities_cities_disj",
    "common_claim_true_false", "companies_true_false", "counterfact_true_false",
]
SMALL_TIER = ["sp_en_trans", "neg_sp_en_trans"]   # N=354; higher null, noisier d'
DISTRACTOR = ["likely"]                            # plausibility decorrelated from truth
MAIN_CAP, SMALL_CAP = 1199, 354


def cap_for(name):
    if name in SMALL_TIER:
        return SMALL_CAP
    return MAIN_CAP


def load_dataset(name, cap=None, seed=0):
    """Return (statements, labels) balanced-subsampled to `cap`."""
    df = pd.read_csv(DATA / f"{name}.csv")
    df = df[df["label"].isin([0, 1])] if df["label"].dtype != object else df
    df = df.dropna(subset=["statement", "label"])
    df["label"] = df["label"].astype(int)
    if cap is not None and len(df) > cap:
        rng = np.random.default_rng(seed)
        per = cap // 2
        idx = []
        for lab in (0, 1):
            pool = df.index[df["label"] == lab].to_numpy()
            idx.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
        df = df.loc[np.concatenate(idx)].sample(frac=1.0, random_state=seed)
    return df["statement"].tolist(), df["label"].to_numpy().astype(int)


# ---- contrastive completion pairs for the behavioral score --------------------
def pairs_cities(df, negated=False):
    """'The city of Krasnodar is in' -> (' Russia', ' South Africa').

    Each city appears with a true and a false country. We pull the true row to
    get correct_country and a false row to get a distractor.

    Under negation the stem becomes '... is not in', which INVERTS which
    completion makes a true sentence: 'Krasnodar is not in South Africa' is
    true, 'Krasnodar is not in Russia' is false. So the targets swap. (Without
    this the negated readout is sign-flipped and every steering result on
    neg_cities comes out backwards.)
    """
    out = []
    for city, g in df.groupby("city"):
        t = g[g["correct_country"] == g["country"]]
        f = g[g["correct_country"] != g["country"]]
        if len(t) == 0 or len(f) == 0:
            continue
        cc = t.iloc[0]["correct_country"]
        fc = f.iloc[0]["country"]
        if negated:
            stem = f"The city of {city} is not in"
            true_tgt, false_tgt = f" {fc}", f" {cc}"
        else:
            stem = f"The city of {city} is in"
            true_tgt, false_tgt = f" {cc}", f" {fc}"
        out.append({"prompt": stem, "true": true_tgt, "false": false_tgt})
    return out


def pairs_counterfact(df):
    """Uses the shipped relation template, subject, target, true_target."""
    out = []
    for subj, g in df.groupby("subject"):
        r = g.iloc[0]["relation"]
        tt = g.iloc[0]["true_target"]
        wrong = g[g["target"] != g["true_target"]]
        if len(wrong) == 0 or not isinstance(r, str) or "{}" not in r:
            continue
        ft = wrong.iloc[0]["target"]
        out.append({"prompt": r.format(subj).rstrip(),
                    "true": f" {tt}", "false": f" {ft}"})
    return out


BUILDERS = {
    "cities": lambda d: pairs_cities(d),
    "neg_cities": lambda d: pairs_cities(d, negated=True),
    "counterfact_true_false": pairs_counterfact,
}


def load_pairs(name, cap, seed=0):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    pr = BUILDERS[name](df)
    rng = np.random.default_rng(seed)
    if len(pr) > cap:
        idx = rng.choice(len(pr), size=cap, replace=False)
        pr = [pr[i] for i in idx]
    return pr
