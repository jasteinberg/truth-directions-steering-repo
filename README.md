# truth-directions-steering-repo

Code, artifacts and figures for a post on linear truth directions in language models:

- **[Truth Directions: Signal-to-Noise and Geometry of Recoverability](https://jasteinberg.github.io/blog/2026/truth-directions-snr/)**
  (published). When a mass-mean truth direction can be recovered from activations, the
  rogue dimension that defeats it, and the corrected (whitened) estimator.

Author: Julia Steinberg

*Drafted with the assistance of Claude (Anthropic).*

The tabular-control, policy-gradient / PPO and reward-modelling material that used to
live here moved to [rl-foundations-repo](https://github.com/jasteinberg/rl-foundations-repo).

## Layout

```
truth-directions-steering-repo/
├── utils/
│   ├── truthlib/     # the shared library (estimators, nulls, sweep, geometry,
│   │                 #   shrinkage, data, acts, steering)
│   └── env.py        # paths and device (ENV)
├── scripts/          # one file per post section, one subcommand per analysis
│   └── figures/      # the post's figures, grouped by section
├── artifacts/        # the JSON every number is read from
├── tests/            # identities and byte-for-byte reproduction
├── notebooks/        # exploratory notebooks
└── docs/             # data sources, reproducing, model conventions
```

Every number in the post traces to a subcommand and an artifact in `artifacts/`.

| file | section of the post |
|---|---|
| `scripts/recoverability.py` | Regime of recoverability |
| `scripts/dimensional_slack.py` | Dimensional slack and sample size |
| `scripts/causal_steering.py` | A causal test of the mass-mean direction; Steering without a rogue dimension |
| `scripts/rogue_dimension.py` | The rogue dimension (origin, score gradient), and the outlier / OLMo appendices |
| `scripts/corrected_estimator.py` | A corrected linear estimator |

`scripts/figures/<section>.py` draws that section's figures (`all` draws every one).

## Running

From the repo root:

```
python scripts/fetch_geometry_of_truth.py          # the Marks & Tegmark datasets
python scripts/<section>.py                         # list the subcommands
python scripts/<section>.py <subcommand> [-h]
python scripts/figures/<section>.py all             # figures, to $FIGURE_DIR (default figures/)
```

`make test` runs the identity tests (seconds), `make reproduce` reruns every
artifact-only analysis and requires its committed JSON to come back byte for byte
(a few minutes), and `make figures` redraws the figures. Commands that load a model
expect Apple-silicon MPS (Pythia, OLMo-2-1B via `transformers`). Analyses
that read `artifacts/act_cache/` or the per-pair gradient files (`*_vectors.npz`) need
those local files, which are not committed. See `docs/reproducing.md` and
`docs/external_sources.md`. The layer, hook, padding and tokenization conventions (and the
tests that enforce them) are in `docs/model_conventions.md`.
