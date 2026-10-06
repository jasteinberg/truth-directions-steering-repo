"""
Model loading and residual-stream activation extraction (torch).

Drafted with the assistance of Claude (Anthropic).
"""
import numpy as np
import torch


def get_model(name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, output_hidden_states=True).to(device).eval()
    return tok, model


def blocks(model):
    """The transformer blocks in order: GPT-NeoX / Pythia, GPT-2, and Llama-style (OLMo, Qwen).
    See docs/model_conventions.md."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return model.model.layers


@torch.no_grad()
def extract_all_layers(statements, tok, model, device, batch_size=16):
    """Last-token residual activations for every layer in one pass per batch.

    Returns float32 array of shape (n_blocks+1, N, d). Convention: layer l is the
    residual stream entering block l (hidden_states[l]; l = 0 is the embedding), the
    site Steerer(l) pushes at and the score gradient is taken at. Index n_blocks is
    the output of the final block BEFORE the final LayerNorm: hidden_states[-1] is
    normalised and is not a residual read. Padding is right-side, so the final real
    token is found from the attention mask.
    """
    final = {}
    hook = blocks(model)[-1].register_forward_hook(
        lambda m, i, o: final.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        chunks = []
        for i in range(0, len(statements), batch_size):
            enc = tok(statements[i:i + batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=128).to(device)
            hs = model(**enc).hidden_states           # (n_blocks+1) tensors (B,T,d)
            last = enc["attention_mask"].sum(1) - 1   # index of final real token
            b = torch.arange(last.shape[0], device=device)
            layers = list(hs[:-1]) + [final.pop("h")]  # pre-LN output of the last block
            batch = torch.stack([h[b, last] for h in layers])    # (n_blocks+1, B, d)
            chunks.append(batch.float().cpu().numpy())
            del hs, enc, batch, layers
    finally:
        hook.remove()
    return np.concatenate(chunks, axis=1)


# ---- unsteered verdict readout -----------------------------------------------------------
def verdict_logits(model, tok, stmts, tid_true, tid_false, template, dev, bs=8):
    """Logit difference between two verdict tokens at the final real token of template(stmt)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(stmts), bs):
            prompts = [template.format(stmt=s) for s in stmts[i:i+bs]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            b = torch.arange(last.shape[0], device=dev)
            fin = logits[b, last].float()
            out.append((fin[:, tid_true] - fin[:, tid_false]).cpu().numpy())
    return np.concatenate(out)


def judgment_scores(model, tok, stmts, template, dev, bs=16):
    """log P(" true") - log P(" false") after the judgment prompt, final real token."""
    ids_t = tok(" true", add_special_tokens=False).input_ids
    ids_f = tok(" false", add_special_tokens=False).input_ids
    assert len(ids_t) == 1 and len(ids_f) == 1, (ids_t, ids_f)
    out = []
    for i in range(0, len(stmts), bs):
        b = [template.format(stmt=s) for s in stmts[i:i + bs]]
        enc = tok(b, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            lg = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        lp = lg[torch.arange(len(b)), last].float().log_softmax(-1)
        out += (lp[:, ids_t[0]] - lp[:, ids_f[0]]).cpu().tolist()
    return np.array(out)
