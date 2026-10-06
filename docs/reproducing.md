# Reproducing the results

How the notebooks run from a fresh clone.

## Most notebooks are self-contained

`scalable_oversight` and `honesty_empirical` train small models on **synthetic
data generated in-notebook** — no external artifacts or downloads. "Run All"
reproduces them end-to-end in the conda env `interp` (laptop-cheap; no GPU
required).

## The one external dependency: `truth_directions`

`truth_directions/` is the exception. It needs:

1. **The Geometry-of-Truth datasets** at `data/geometry_of_truth/{cities,
   larger_than, sp_en_trans}.csv` (Marks & Tegmark 2023; see
   `docs/external_sources.md` for the source and license). These are **not
   vendored** — fetch them and drop them in `data/geometry_of_truth/` before
   running. The notebook resolves the path through the repo root (`REPO/data/
   geometry_of_truth`), so no path edits are needed once the files are present.
2. **Model downloads.** It loads `gpt2` and `EleutherAI/pythia-410m` from
   HuggingFace on first run. The optional §6 scale sweep (`RUN_SCALE = True`)
   additionally pulls `pythia-70m … pythia-1.4b` — several GB — and is off by
   default.

## Why activations come from raw `transformers`

Activations here are read from `output_hidden_states` on a plain HuggingFace
forward pass rather than through TransformerLens, which the interpretability
repo uses. Two reasons. TransformerLens supports no OLMo checkpoint, and the
OLMo-2-1B arm is what carries the cross-family transfer result, so one
extraction path covering every model was preferable to two. And
`HookedTransformer.from_pretrained` folds LayerNorm and centers the writing and
unembedding weights by default, which leaves an equivalent model with different
parameters — not what you want underneath claims about residual-stream geometry
and activation norms across depth.

## Paths

Notebooks locate the repo root by walking up to the directory containing
`utils/`, so they run from any checkout location without editing paths.
