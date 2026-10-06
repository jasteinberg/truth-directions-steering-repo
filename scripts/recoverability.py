"""
Regime of recoverability: decoding across the Pythia ladder.

The post's section "Regime of recoverability" (https://jasteinberg.github.io/blog/2026/truth-directions-snr/).
Per (model, dataset, layer): held-out d' and AUROC of the mass-mean and Fisher
directions against a random-direction null; the best layer by null margin;
cross-dataset transfer, the superposition probe and the Cover N-sweep at that
layer (snr_sweep.json). Also the distractor transfer to `likely` and the
unsteered verdict readout.

Subcommands:

    sweep                  Truth-direction SNR sweep across the Pythia ladder
    transfer-likely        Closes the distractor test the post currently flags as not run
    verdicts               Can pythia-2.8b judge these statements true or false on its own, ...

Run from the repo root:  python scripts/recoverability.py <subcommand> [-h]

Drafted with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import provenance
from utils.env import ENV
from utils.truthlib import acts, data
from utils.truthlib import estimators as est
from utils.truthlib.acts import extract_all_layers, get_model, judgment_scores, verdict_logits
from utils.truthlib.data import DISTRACTOR, MAIN_CAP, MAIN_TIER, SMALL_CAP, SMALL_TIER, cap_for, load_dataset
from utils.truthlib.estimators import auroc, evaluate_direction, mass_mean, split_indices
from utils.truthlib.nulls import N_NULL, cover_n_sweep
from utils.truthlib.sweep import PCA_KS, analyze_dataset, best_layer, fit_oriented, superposition_probe

# =============================================================================
# sweep
# =============================================================================
DATA = Path(ENV.GOT)


SWEEP_ART = REPO / "artifacts"


CKPT = SWEEP_ART / "ckpt"


COVER_DATASET = "counterfact_true_false"           # full size N=31964


COVER_NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 31964]


MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]


def ckpt_path(model, dset):
    tag = model.split("/")[-1]
    return CKPT / f"{tag}__{dset}.json"


def run_model(mname, datasets, device, dtype, args):
    tag = mname.split("/")[-1]
    print(f"\n=== {mname} ===", flush=True)
    tok, model = get_model(mname, device, dtype)
    d_model = model.config.hidden_size

    per_dataset, acts_at_best = {}, {}
    for dset in datasets:
        cp = Path(args.ckpt_dir) / ckpt_path(mname, dset).name
        cap = args.cap_override or cap_for(dset)
        stmts, y = load_dataset(dset, cap=cap, seed=args.seed)
        if args.smoke:
            stmts, y = stmts[:args.smoke], y[:args.smoke]

        print(f"  [{dset}] N={len(stmts)}  d={d_model}  N/2d={len(stmts)/(2*d_model):.2f}",
              flush=True)
        A = extract_all_layers(stmts, tok, model, device, args.batch_size)

        if cp.exists() and not args.force:
            print(f"    (resume) {cp.name}", flush=True)
            per_dataset[dset] = json.loads(cp.read_text())
        else:
            rows = analyze_dataset(A, y, layer_stride=args.layer_stride, seed=args.seed)
            bl = best_layer(rows)
            entry = {"n": len(y), "d_model": int(d_model), "layers": rows,
                     "best_layer": bl,
                     "superposition": superposition_probe(A[bl].astype(np.float64), y, seed=args.seed)}
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(entry, indent=2))
            per_dataset[dset] = entry
            print(f"    best layer {bl}; checkpointed", flush=True)

        bl = per_dataset[dset]["best_layer"]
        acts_at_best[dset] = (A[bl].astype(np.float64), y)
        del A; gc.collect()

    # transfer across the run_sweep tier. The direction is fit on each source's TRAIN
    # half and scored on every target's TEST half, so the diagonal is held-out
    # too and is directly comparable to the off-diagonal.
    tier = [d for d in datasets if d in MAIN_TIER]
    transfer = {}
    for src in tier:
        Xs, ys = acts_at_best[src]
        tr_s, _ = split_indices(ys, seed=args.seed)
        th = mass_mean(Xs[tr_s], ys[tr_s])
        if auroc(Xs[tr_s] @ th, ys[tr_s]) < 0.5:   # orient on source train
            th = -th
        row = {}
        for tgt in tier:
            Xt, yt = acts_at_best[tgt]
            _, te_t = split_indices(yt, seed=args.seed)
            row[tgt] = evaluate_direction(th, Xt[te_t], yt[te_t])["auroc"]
        transfer[src] = row

    # Cover control at full size on the designated dataset
    cover = None
    if args.cover and COVER_DATASET in datasets:
        print(f"  [cover] {COVER_DATASET} full size", flush=True)
        stmts, y = load_dataset(COVER_DATASET, cap=None, seed=args.seed)
        if args.smoke:
            stmts, y = stmts[:args.smoke], y[:args.smoke]
        bl = per_dataset[COVER_DATASET]["best_layer"]
        A = extract_all_layers(stmts, tok, model, device, args.batch_size)
        ns = [n for n in COVER_NS if n <= len(y)] or [len(y)]
        cover = cover_n_sweep(A[bl].astype(np.float64), y, ns, d_model, seed=args.seed)
        for r in cover:
            print(f"      N={r['N']:>6} N/2d={r['N_over_2d']:.2f}  "
                  f"true AUROC={r['true']['auroc']:.3f}  "
                  f"shuf AUROC={r['shuffled']['auroc']:.3f}", flush=True)
        del A; gc.collect()

    del model, tok; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {"d_model": int(d_model), "datasets": per_dataset,
            "transfer": transfer, "cover": cover}


def run_sweep(argv=None):
    """Truth-direction SNR sweep across the Pythia ladder.

    For each (model, dataset): one forward pass, last-token residual activations
    cached for all layers; per layer, plain and whitened (Ledoit-Wolf) mass-mean
    d', AUROC, accuracy, and a random-direction null (p50/p95).

    At each model's best layer: cross-dataset transfer matrix, and a superposition
    probe (d' vs number of top-k PCs removed).

    Cover / N-sweep control on counterfact_true_false at full size: true vs
    shuffled labels as N crosses the separating capacity 2d.

    Writes artifacts/snr_sweep.json; checkpoints per (model, dataset) under
    artifacts/ckpt/ so an interrupted run resumes.
    """
    ap = argparse.ArgumentParser(description=run_sweep.__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(MAIN_TIER + SMALL_TIER + DISTRACTOR))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--ckpt_dir", default=str(CKPT), help="per-(model, dataset) checkpoints (default: the published artifacts/ckpt/)")
    ap.add_argument("--layer_stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap_override", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0,
                    help="truncate every dataset to this many rows")
    ap.add_argument("--no-cover", dest="cover", action="store_false")
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    ap.add_argument("--out", default=str(SWEEP_ART / "snr_sweep.json"))
    args = ap.parse_args(argv)

    SWEEP_ART.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.device == "mps" else torch.float32
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    results = {}
    for mname in [m.strip() for m in args.models.split(",") if m.strip()]:
        results[mname] = run_model(mname, datasets, args.device, dtype, args)
        # write after every model so a later crash cannot lose earlier rungs
        Path(args.out).write_text(json.dumps(
            {"config": {"main_tier": MAIN_TIER, "small_tier": SMALL_TIER,
                        "distractor": DISTRACTOR, "main_cap": MAIN_CAP,
                        "small_cap": SMALL_CAP, "n_null": N_NULL,
                        "pca_ks": PCA_KS, "cover_dataset": COVER_DATASET,
                        "seed": args.seed},
             "models": results}, indent=2))
        print(f"  -> wrote {args.out}", flush=True)

    print("\ndone.")


# =============================================================================
# transfer-likely
# =============================================================================
LIKELY_ART = Path(__file__).resolve().parent.parent / "artifacts"


DEV = "mps"


def run_transfer_likely(argv=None):
    """Closes the distractor test the post currently flags as not run.

    Fitting on `likely` and scoring on `likely` says only that a probability axis
    exists. The test Marks & Tegmark designed the set for is CROSS-dataset: take a
    direction fitted on truth statements and evaluate it on `likely`. If a truth
    direction is really reading textual plausibility, it should separate `likely`;
    if it is reading truth, it should sit at chance there.

    Runs both directions of the comparison (truth-fitted -> likely, and
    likely-fitted -> each truth set), for the plain and whitened arms, at each
    dataset's own selected layer from snr_sweep.json.
    """
    ap = argparse.ArgumentParser(description=run_transfer_likely.__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--cap", type=int, default=1199)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", default=str(LIKELY_ART / "snr_sweep.json"))
    ap.add_argument("--out", default=str(LIKELY_ART / "transfer_likely.json"))
    args = ap.parse_args(argv)

    sweep = json.load(open(args.sweep))
    m = sweep["models"][args.model]
    best = {ds: v["best_layer"] for ds, v in m["datasets"].items()}
    truth_sets = [d for d in m["transfer"].keys()]      # the 9 run_transfer_likely-tier sets

    tok, model = acts.get_model(args.model, DEV, torch.float16)

    # activations at every layer, per dataset, one pass each
    A, Y = {}, {}
    for ds in truth_sets + ["likely"]:
        stmts, y = data.load_dataset(ds, cap=args.cap, seed=args.seed)
        A[ds] = acts.extract_all_layers(stmts, tok, model, DEV, 16)
        Y[ds] = y
        print(f"  extracted {ds} ({len(y)})", flush=True)

    out = {"model": args.model, "note": "AUROC of a direction fitted on `train_on` "
           "and evaluated on `eval_on`, at the layer selected for `train_on`",
           "rows": []}

    # truth-fitted -> likely, and the reverse, both arms
    for a, b in [(t, "likely") for t in truth_sets] + \
                [("likely", t) for t in truth_sets]:
        L = best[a]
        if L >= len(A[b]):
            continue
        Xa, Xb = A[a][L].astype(np.float64), A[b][L].astype(np.float64)
        th, thw, _ = fit_oriented(Xa, Y[a], args.seed)
        _, teb = est.split_indices(Y[b], seed=args.seed)
        row = {"train_on": a, "eval_on": b, "layer": int(L),
               "auroc_plain": float(est.auroc(Xb[teb] @ th, Y[b][teb])),
               "auroc_whitened": float(est.auroc(Xb[teb] @ thw, Y[b][teb]))}
        out["rows"].append(row)
        print(f"  {a:26s} -> {b:26s} L{L:2d} "
              f"plain={row['auroc_plain']:.3f} whit={row['auroc_whitened']:.3f}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        gc.collect()

    print(f"\nwrote {args.out}\ndone.")


# =============================================================================
# verdicts
# =============================================================================
D = ENV.GOT


dev = "mps"


OUT = os.path.join(REPO, "artifacts", "check_model_verdicts.json")


TEMPLATE = '{stmt} This statement is:'


def judgment(out_path):
    mname = "EleutherAI/pythia-2.8b"
    tok, model = acts.get_model(mname, dev, torch.float16)
    res = {}
    for ds in ["cities", "common_claim_true_false", "counterfact_true_false"]:
        st, y = data.load_dataset(ds, cap=1199, seed=0)
        sc = judgment_scores(model, tok, st, TEMPLATE, dev)
        res[ds] = {"n": int(len(y)), "auroc": float(roc_auc_score(y, sc)),
                   "mean_score_true": float(sc[y == 1].mean()),
                   "mean_score_false": float(sc[y == 0].mean())}
        print(f"  {ds:<26} n={len(y)}  judgment AUROC={res[ds]['auroc']:.3f}", flush=True)
    with open(out_path, "w") as f:
        json.dump({"config": {"model": mname, "template": TEMPLATE, "targets": [" true", " false"],
                              "cap": 1199, "seed": 0, "dtype": "float16", "device": dev},
                   "results": res}, f, indent=1)
    print(f"wrote {out_path}")


def precheck():
    for mname in ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]:
        tok, model = acts.get_model(mname, dev, torch.float16)
        tt = tok(" TRUE").input_ids[0]; tf = tok(" FALSE").input_ids[0]
        print(f"\n=== {mname}  (' TRUE'={tt}, ' FALSE'={tf}) ===")
        for ds in ["cities", "companies_true_false", "counterfact_true_false"]:
            df = pd.read_csv(f"{D}/{ds}.csv").dropna(subset=["statement","label"])
            df["label"] = df["label"].astype(int)
            df = df.sample(n=min(200, len(df)), random_state=0)
            z = verdict_logits(model, tok, df["statement"].tolist(), tt, tf, TEMPLATE, dev)
            y = df["label"].to_numpy()
            a = roc_auc_score(y, z)
            acc = ((z > np.median(z)).astype(int) == y).mean()
            print(f"  {ds:<26} behavioural AUROC={a:.3f}  (acc@median={acc:.3f})  "
                  f"mean(logit diff)={z.mean():+.2f}")
        del model, tok
        gc.collect(); torch.mps.empty_cache()


def run_verdicts(argv=None):
    """Can pythia-2.8b judge these statements true or false on its own, before any steering?

    This is why the steering harness scores factual-preference completions rather than
    a judgment: the judgment readout is a behavior a base model barely has, so steering
    it would measure nothing.

      --mode judgment (default)  the number the post quotes: "{stmt} This statement is:"
          scored as log P(" true") - log P(" false") at the final token, AUROC against the
          labels, pythia-2.8b, class-balanced cap 1199 at seed 0. Writes
          artifacts/check_model_verdicts.json.
      --mode precheck            the original July precheck: logit(" TRUE") - logit(" FALSE")
          on 200 sampled statements, three Pythia sizes, printed only.
    """
    ap = argparse.ArgumentParser(description=run_verdicts.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["judgment", "precheck"], default="judgment")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    judgment(args.out) if args.mode == "judgment" else precheck()


COMMANDS = {
    "sweep": run_sweep,
    "transfer-likely": run_transfer_likely,
    "verdicts": run_verdicts,
}


def main() -> None:
    provenance.main(COMMANDS, __doc__)


if __name__ == "__main__":
    main()
