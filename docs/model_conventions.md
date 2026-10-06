# Conventions for reading, steering and differentiating pretrained models

*Drafted with the assistance of Claude (Anthropic).*

Every number that reads a model's internals depends on choices that two pieces of code can
make differently without either one failing: which tensor "layer $$l$$" names, whether a
push lands where the activation was measured, which token is "the last". This note fixes
those choices for every pretrained transformer we hook (Pythia / GPT-NeoX, Qwen, OLMo,
Llama-style, GPT-2), states each one as an equation, and names the test that enforces it in
this repo (`tests/test_model_io.py`, `tests/test_steering.py`). Code written for a new model
or a new repo should adopt the same definitions and copy the tests.

## 1. The residual stream and its layer index

A decoder of $$n$$ blocks $$B_0, \dots, B_{n-1}$$ maps the embedding through

$$
h^{(0)} = \mathrm{embed}(x), \qquad h^{(l+1)} = B_l\big(h^{(l)}\big), \qquad
\mathrm{logits} = W_U\, \mathrm{LN}_f\big(h^{(n)}\big).
$$

**Layer $$l$$ means $$h^{(l)}$$, the residual stream entering block $$l$$**, for
$$l = 0, \dots, n$$; $$l = 0$$ is the embedding and $$l = n$$ is the output of the last block
*before* the final LayerNorm $$\mathrm{LN}_f$$.

Hugging Face's `output_hidden_states` returns $$(h^{(0)}, \dots, h^{(n-1)}, \mathrm{LN}_f(h^{(n)}))$$:
the first $$n$$ entries are $$h^{(l)}$$, the last is **normalised and is not a residual read**.
LayerNorm removes the mean and divides by the norm, so it erases exactly the quantities a
residual-stream analysis measures (massive coordinates, the norm, the class gap's scale).
Never read `hidden_states[-1]` as layer $$n$$; read $$h^{(n)}$$ with a forward hook on
$$B_{n-1}$$ (`truthlib.acts.extract_all_layers` does this).

Block lists by architecture (`truthlib.acts.blocks`):

| family | blocks | final norm |
|---|---|---|
| GPT-NeoX / Pythia | `model.gpt_neox.layers` | `model.gpt_neox.final_layer_norm` |
| Llama-style: Qwen, OLMo-2, Llama | `model.model.layers` | `model.model.norm` |
| GPT-2 | `model.transformer.h` | `model.transformer.ln_f` |

Tests: `test_extraction_reads_block_inputs_and_pre_ln_top`,
`test_probing_refuses_the_post_ln_entry_and_agrees_below_it`, and `tests/test_conventions.py`
(no other tracked code indexes the `hidden_states` tuple).

## 2. One site for reading, pushing and differentiating

Steering at layer $$l$$ with direction $$w$$ and coefficient $$\alpha$$ in class-gap units $$c$$
replaces $$h^{(l)} \to h^{(l)} + \alpha c\, w$$ at every position, before $$B_l$$ sees it: a
**forward pre-hook on $$B_l$$** (`truthlib.steering.Steerer`). The behavioural gradient is
taken at the same tensor,

$$
g = \Big\langle \sum_t \nabla_{h^{(l)}_t}\, \ell \Big\rangle,
$$

by replacing $$B_l$$'s input with a leaf that requires grad (a pre-hook again;
`steering.grad_for_texts`, `refusal.batch_forward`). Directions fit on extracted activations
at layer $$l$$ are therefore applied, and differentiated, exactly where they were measured,
and the linear-response identity $$\chi(w) = c\,(w \cdot g)$$ compares like with like.

A forward hook on $$B_l$$'s *output* acts at $$h^{(l+1)}$$, one layer later. Until 1 Oct the
Pythia code read at $$h^{(l)}$$ and pushed at $$h^{(l+1)}$$ (`notes/test-audit-2026-10.md`).
Steering after the last block ($$l = n$$) has no block to hook and is refused.

Tests: `test_push_lands_at_the_measurement_site_and_nowhere_earlier`,
`test_score_gradient_is_taken_at_the_push_site` (the finite difference of the score under the
push equals $$G \cdot w$$ at $$l$$, and is ~50% off at $$l+1$$),
`test_refusal_reads_pushes_and_differentiates_at_block_input`.

## 3. Hook hygiene

- $$\alpha = 0$$ (or $$w = 0$$) is the identity bit for bit in fp32: no dtype cast, no copy
  on the unsteered path.
- One hook per steering context, firing once per forward pass, removed on exit **and after an
  exception** (context managers, `try/finally`).
- Hooks never modify a tensor in place: the recorded $$h^{(l)}$$ of a steered run equals the
  clean one.
- Models are in eval mode (`model.training` is False) for every read.

Tests: `test_alpha_zero_and_zero_vector_are_the_identity`,
`test_one_hook_fires_once_per_forward_and_is_removed`,
`test_hooks_are_removed_after_an_exception_in_the_steered_forward`.

## 4. Positions and padding

Batches are **right-padded**; the last real token of sequence $$b$$ is
$$t_b = \sum_t m_{bt} - 1$$ from the attention mask $$m$$, never `[:, -1]`. Under right padding
a causal model's real tokens never attend to pads, so pushes that also land on pad positions
are harmless. Batched and per-example reads agree to fp32 kernel reordering (observed
$$\lesssim 2\times 10^{-4}$$ of the vector norm on pythia-70m with SDPA). Nothing is truncated:
every statement fits the extraction's 128-token limit.

Tests: `test_extraction_leaves_model_in_eval_and_batched_equals_per_example`,
`test_steering_at_every_position_including_padding`, `test_no_statement_is_truncated_at_extraction`.

## 5. Tokenization

- One tokenizer call convention for fitting, steering and differentiating: the same string
  gives the same `input_ids` on every path. Pythia adds no BOS.
- A score over completion tokens $$\ell = \sum_{t \in \text{completion}} \log p(x_t \mid x_{<t})$$
  locates the completion as positions $$|\mathrm{tok}(\text{prompt})|, \dots, |\mathrm{tok}(\text{prompt} + \text{completion})| - 1$$,
  which is correct only if $$\mathrm{tok}(\text{prompt})$$ is a prefix of
  $$\mathrm{tok}(\text{prompt}+\text{completion})$$: completions start with a space and prompts do not end
  with one.
- Verdict or label tokens (" true", " false") are single tokens that decode back to the
  intended string; never take `input_ids[0]` of a multi-token string.
- Chat models: the chat template is applied exactly once, by one function
  (`refusal.chat_wrap`), on every path; generation prompts end with the assistant header.

Tests: `test_scoring_paths_build_the_same_input_ids`, `test_completion_tokens_are_a_suffix_of_the_prompt`,
`test_verdict_tokens_are_single_tokens_that_decode_back`, `test_refusal_token_ids_decode_to_the_refusal_set`.

## 6. Scores, units and signs

- Truth score: $$\ell = \log p(\text{true completion}) - \log p(\text{false completion})$$, summed
  over completion tokens. Refusal score: the log-odds $$\log P_R - \log(1 - P_R)$$ that the first
  generated token is in the (English) refusal set.
- Steering unit: $$c = \lVert \mu_1 - \mu_0 \rVert$$ on the fit half at layer $$l$$, so
  $$\alpha = 1$$ moves an activation by the class gap. The response is antisymmetrised,
  $$A(\alpha) = \tfrac12[\bar\ell(+\alpha) - \bar\ell(-\alpha)]$$, on held-out items, and
  $$\chi$$ is the through-origin slope of $$A$$ on $$\alpha$$.
- Directions are oriented on the fit half (training AUROC $$\ge \tfrac12$$) and the sign is held
  on evaluation. A failed fit raises; nothing is silently replaced by another direction.

## 7. Numerics

Statistics on activations (means, covariances, spectra, Ledoit-Wolf) are computed in fp64
from fp16/fp32 reads. The model runs in fp16 on MPS for production and in fp32 or fp64 on CPU
for tests; finite differences need fp64 (in fp32 the score's rounding dominates below
$$h \sim 10^{-2}$$).
