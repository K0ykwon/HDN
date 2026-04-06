# Current Roadmap

## Highest-Priority Bottlenecks

1. Compression loss at the window encoder
2. Overly uniform hierarchical merge
3. Throughput gap versus Transformer baselines
4. Readout specialization toward classification
5. Limited benchmark scope beyond the current four-task lean suite

## Immediate Experiment Plan

- Run stride/overlap compression ablations on `ListOps` and `RULER needle`
- Compare `stride=2` baseline against `stride=1` and larger-window variants
- Keep the backbone token-free and attention-free throughout
- Prefer ablations that preserve or reduce parameter count when possible

## Current Findings

- `ListOps` lean baseline currently reaches `0.2344` with `69,200` parameters.
- Simple compression ablations were not wins so far:
  - `stride=1` on `ListOps`: `0.1816`
  - `window_size=6` on `ListOps`: `0.1777`
  - `window_size=3, stride=1` on `ListOps`: `0.1484`
  - `stride=1` on `RULER`: below baseline
  - `window_size=10` and `window_size=6` on `RULER`: below baseline
- Separate pretraining was tested but is not part of the intended mainline path, so it is not being carried forward.

## Ready-to-Run Commands

```bash
python3 scripts/run_compression_ablation_suite.py
python3 scripts/run_benchmark_lean_twr.py
```
