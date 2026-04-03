# Experiment Plan

## Main questions
1. Can a tokenless persistent-memory backbone work at all?
2. Does adaptive compute actually change with input difficulty?
3. Is there a meaningful performance/efficiency trade-off versus baselines?

## Phase order
### Phase 1 — Sanity check
- Small classification or synthetic task
- Verify end-to-end training
- Verify write-think-read path

### Phase 2 — Core TWR validation
- no_think
- fixed_think
- adaptive_depth
- adaptive_depth + slot_gate

### Phase 3 — Baselines
- Transformer encoder
- Perceiver-style latent baseline
- optional Mamba/SSM

### Phase 4 — Controlled long-sequence benchmarks
- LRA: ListOps
- LRA: Text
- LRA: Retrieval

### Phase 5 — Realistic long-context axis
- RULER or LongBench subset

### Phase 6 — Optional appendix
- text8 / enwik8

## Recommended initial claim scope
Position the model as a new backbone with interpretable adaptive compute, not as a universal replacement for Transformers.
