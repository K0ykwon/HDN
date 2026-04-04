# Current Context

## Repository State

- Repo: `TWR-LM`
- Main goal: build a competitive `tokenless latent backbone` for sequence modeling.
- Mainline model files:
  - `src/twr/models/twr_lm.py`
  - `src/twr/modules/latent_encoder.py`
  - `src/twr/modules/latent_backbone.py`
- Key baselines:
  - `Transformer`
  - `Mamba`
- Key entrypoints:
  - `scripts/train.py`
  - `scripts/run_benchmark_full_suite.py`
  - `scripts/analyze_results.py`

## Current Architecture

The active implementation is no longer the original `write -> memory bank -> adaptive think` stack.

The current model is:

```text
tokens
  -> overlapping latent encoder
  -> latent sequence
  -> shared refinement + selective pairwise merge
  -> multiscale query readout
  -> logits
```

Important implications:

- token-wise hidden states are not kept as persistent backbone state after compression
- the persistent state is a latent sequence / latent pyramid
- adaptive depth and slot gating are not part of the current mainline model
- the project is currently optimizing for `parameter-efficient Transformer replacement`

## Important Code Changes Already Made

- Removed the original write/think modules:
  - `src/twr/modules/event_encoder.py`
  - `src/twr/modules/latent_memory.py`
  - `src/twr/modules/readout.py`
  - `src/twr/modules/refine_block.py`
  - `src/twr/modules/sequential_write.py`
  - `src/twr/modules/think_loop.py`
- Added latent-backbone replacements:
  - `src/twr/modules/latent_encoder.py`
  - `src/twr/modules/latent_backbone.py`
- Added lean and benchmark configs for TWR / Transformer / Mamba comparison
- Added local dataset snapshot caching in `src/twr/data/datasets.py`
- Updated trainer checkpoint/summary handling to track best validation accuracy

## Current Benchmark Snapshot

### ListOps

- `TWR lean`: `0.2285`, `82,587 params`
- `Transformer`: `0.2383`, `86,026 params`
- `Mamba`: `0.1289`, `95,530 params`

### RULER needle

- `TWR lean`: `0.0840`, `100,065 params`
- `Transformer`: `0.0801`, `117,136 params`
- `Mamba`: `0.0781`, `252,048 params`

### LongBench TREC

- `TWR lean`: `0.1800`, `494,083 params`
- `Transformer`: `0.1800`, `627,250 params`
- `Mamba`: `0.1800`, `1,272,242 params`

### Hyperpartisan

- `TWR lean`: `0.5400`, `475,283 params`
- `Transformer`: `0.5450`, `624,130 params`
- `Mamba`: `0.5150`, `1,266,050 params`

## Current Interpretation

- Lean TWR is now competitive on `ListOps` and `Hyperpartisan`.
- Lean TWR still loses some of the larger model's advantage on `RULER` and `LongBench`.
- Transformer remains the strongest overall baseline.
- Mamba is weaker than both in the current setup.

## Current Priority

Improve the latent backbone further without reintroducing token-persistent state or reviving the old write/think architecture.
