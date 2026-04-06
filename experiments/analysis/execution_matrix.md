# Execution Matrix

This file turns the benchmark plan into an execution-oriented matrix.

## Batch A: Already Executed

| ID | Category | Run / Suite | Status |
| --- | --- | --- | --- |
| A01 | core | `run_benchmark_suite.py` | completed |
| A02 | compression | `run_compression_ablation_suite.py` subset | completed |
| A03 | scaling | `RULER 256 / 512 / 1024` for TWR / Transformer / Mamba | completed |
| A04 | hierarchy | `run_hierarchy_ablation_suite.py` | completed |

## Batch B: Repo-Ready Next

| ID | Category | What to add | Purpose |
| --- | --- | --- | --- |
| B01 | benchmark expansion | more `LongBench` subsets | realistic task diversity |
| B02 | benchmark expansion | `BABILong` or `InfiniteBench` | stronger long-context reasoning |
| B03 | retrieval | `RepoQA` or `BRIGHT` | precision preservation under compression |
| B04 | seeds | 3-seed reruns for core TWR / Transformer / Mamba | paper-ready statistics |

## Batch C: Requires New Model Implementations

| ID | Category | Missing family |
| --- | --- | --- |
| C01 | external baseline | Hyena |
| C02 | external baseline | RWKV |
| C03 | external baseline | RetNet |
| C04 | external baseline | H3 / H3++ |
| C05 | stronger dense baseline | Transformer-Strong |

## Batch D: Requires Training Infra Expansion

| ID | Requirement |
| --- | --- |
| D01 | matched 30M / 90M / 220M / 500M family scaling |
| D02 | fixed-token-budget large runs |
| D03 | bf16 + activation checkpointing standardized recipe |
| D04 | long-length efficiency study to 32k / 128k |

## Recommended Order

1. Finish Batch B before Batch C.
2. Finish Batch C before Batch D.
3. Do not claim the 12-model paper benchmark until B + C + D are all in place.
