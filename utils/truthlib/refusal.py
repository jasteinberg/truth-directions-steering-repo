"""
Refusal-direction machinery (torch): the chat-formatted AdvBench/Alpaca prompts,
the refusal log-odds score, the per-prompt gradient pass, and the residual-stream
hook for Qwen (used by scripts/refusal_corner.py).

Drafted with the assistance of Claude (Anthropic).
"""
import gc
from pathlib import Path

import numpy as np
import torch

from ..env import ENV
from .steering import Steerer

DEV = "mps"
MODEL = "Qwen/Qwen1.5-1.8B-Chat"
# English refusal openers only. Steered along g_hat at alpha = 2 the model refuses
# coherently in Chinese (refusal_generations, draft OPEN g), which this set scores as
# compliance, so P(refusal) under-reads that regime. Do not change the set without
# Julia: every refusal artifact was scored with it (tests/test_steering.py pins it).
REFUSAL_TOKS = ["I", "I'm", "As", "Sorry", "I cannot", "I can", "Unfortunately"]
LOSS_SCALE = 1024.0


def load_prompts(n, seed):
    """harmful = AdvBench harmful_behaviors.csv (the llm-attacks release, kept at
    $EXTERNAL_ROOT/advbench/); harmless = Alpaca instructions from the Hub."""
    import pandas as pd
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    adv = pd.read_csv(Path(ENV.EXTERNAL) / "advbench" / "harmful_behaviors.csv")["goal"].tolist()
    alp = [x for x in load_dataset("tatsu-lab/alpaca", split="train")["instruction"] if x.strip()]
    h = list(rng.choice(adv, size=min(n // 2, len(adv)), replace=False))
    g = list(rng.choice(alp, size=n // 2, replace=False))
    return h, g


def chat_wrap(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)


def refusal_ids(tok):
    ids = set()
    for t in REFUSAL_TOKS:
        for v in (t, " " + t):
            e = tok(v, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                ids.add(e[0])
    return sorted(ids)


def logodds_from_logits(logits_last, rids):
    lp = logits_last.float().log_softmax(-1)
    lr = torch.logsumexp(lp[:, rids], dim=-1)               # log P(refusal)
    l1 = torch.log1p(-lr.exp().clamp(max=1 - 1e-6))         # log (1 - P)
    return lr - l1


def batch_forward(model, tok, block, texts, rids, want_grad):
    """Returns (ell, last-token acts entering `block`, per-prompt grads or None).

    For block = model.model.layers[l] this reads and differentiates at layer l, the
    residual stream entering block l (hidden_states[l]): the site QwenSteerer(l) pushes at.
    """
    store = {}

    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h2 = h.detach().requires_grad_(True) if want_grad else h
        store["h"] = h2
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": h2}

    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            b = torch.arange(len(texts), device=DEV)
            ell = logodds_from_logits(logits[b, last], rids)
            acts = store["h"][b, last].detach().float().cpu().numpy()
            grads = None
            if want_grad:
                (LOSS_SCALE * ell.sum()).backward()
                grads = (store["h"].grad.detach().float().sum(1) / LOSS_SCALE).cpu().numpy().astype(np.float64)
            return ell.detach().float().cpu().numpy(), acts.astype(np.float64), grads
    finally:
        handle.remove(); store.clear()


def boot_cos(G, u, B, seed):
    rng = np.random.default_rng(seed); n = len(G)
    vals = np.empty(B)
    for b in range(B):
        g = G[rng.integers(0, n, n)].mean(0)
        vals[b] = g @ u / np.linalg.norm(g)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---- steering ----------------------------------------------------------------------------
# Qwen uses the same steering hook as Pythia: a forward pre-hook adding the push to the
# residual stream entering block l (steering.Steerer finds Qwen's blocks via acts.blocks).
QwenSteerer = Steerer


@torch.no_grad()
def score(model, tok, texts, rids, bs):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        b = torch.arange(len(texts[i:i + bs]), device=DEV)
        out.append(logodds_from_logits(logits[b, last], rids).float().cpu().numpy())
        gc.collect(); torch.mps.empty_cache()
    return np.concatenate(out).astype(np.float64)
