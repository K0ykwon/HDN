# Plan

## TWR-LM Benchmark Experiment Plan

This file is the canonical paper-oriented benchmark plan for the repository.

It integrates:

- the large benchmark plan provided for the paper
- the repo-grounded execution constraints captured in [master_experiment_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/master_experiment_plan.md)
- the benchmark selection logic in [paper_benchmark_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/paper_benchmark_plan.md)
- the Transformer/Mamba comparison framing in [mamba_comparison_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/mamba_comparison_plan.md)

## 1. Objective

The benchmark suite exists to position TWR-LM as a serious alternative sequence modeling backbone along three axes:

1. Quality
2. Efficiency
3. Length generalization

This is a benchmarking plan, not a model-improvement plan.

## 2. Model Set

### External families in the paper-scale ambition

1. Transformer-Std
2. Transformer-Strong
3. Hyena
4. RWKV
5. RetNet
6. Mamba
7. H3 / H3++
8. TWR-LM

### Internal TWR ablations

9. TWR-LM without overlap
10. TWR-LM without hierarchy
11. TWR-LM without multiscale readout
12. TWR-LM shallow

### Current repo reality

Implemented and runnable now:

- TWR-LM
- Transformer baseline
- Mamba placeholder baseline

Configured TWR studies now:

- compression ablations
- hierarchy ablations
- initial length scaling ablations

Not yet implemented in the current repo:

- Hyena
- RWKV
- RetNet
- H3 / H3++
- Transformer-Strong as a distinct reference recipe
- explicit no-overlap / no-hierarchy / no-multiscale TWR variants as named model families

## 3. Scaling Axes

### Full paper ambition

Model sizes:

- 30M
- 90M
- 220M
- 500M

Context lengths:

- 512
- 2,048
- 8,192
- 32,768
- 131,072

### Current repo status

What is actually runnable today:

- small research-scale models only
- current benchmark lengths centered around 256 / 512 / 1024 for the scaling study already added
- no training pipeline yet for 30M / 90M / 220M / 500M matched families

Therefore the near-term execution plan is:

1. finish small-scale architecture and scaling evidence in the existing codebase
2. only then expand to matched-size family sweeps

## 4. Training Setup

### Full paper target recipe

- AdamW
- betas `(0.9, 0.95)`
- weight decay `0.1`
- grad clip `1.0`
- cosine decay
- warmup `2,000`
- bf16
- activation checkpointing
- shared tokenizer / shared splits

### Current repo status

The repository currently uses a lighter benchmarking recipe from config files and does not yet expose:

- bf16-specific training control
- activation checkpointing
- fixed-token-budget large-scale training
- standardized large-model sweeps

So the paper plan is divided into:

- `repo-ready now`
- `requires infra expansion`

## 5. Benchmark Groups

### Group A: LRA

Paper ambition:

- ListOps
- Text
- Retrieval
- Pathfinder
- Image
- Path-X optional

Repo-ready now:

- ListOps only

### Group B: Long-document classification

Paper ambition:

- Hyperpartisan
- IMDB long-document setup
- ArXiv / PubMed long-document classification
- GovReport proxy if defined

Repo-ready now:

- Hyperpartisan
- debug long-text hooks exist, but not yet a complete classification suite

### Group C: RULER

Paper ambition:

- 8 subtasks
- 6 lengths from 4k to 128k

Repo-ready now:

- simplified RULER-style benchmark already implemented
- initial scaling sweep completed at 256 / 512 / 1024 for TWR / Transformer / Mamba

### Group D: LongBench

Paper ambition:

- 12 core tasks
- bucketed by short / medium / long

Repo-ready now:

- LongBench `trec` only

## 6. Efficiency Evaluation

### Full paper ambition

- training throughput at 2k / 8k / 32k
- inference latency at batch 1 / 8 / 32
- peak memory
- OOM behavior

### Current repo status

Already logged:

- throughput
- peak GPU memory
- approximate FLOPs

Not yet implemented:

- decode latency
- prefill/decode split
- equal-throughput comparisons
- high-length latency study out to 128k

## 7. Seeds and Repetitions

### Full paper ambition

- 3 seeds for main results
- 3 evaluation repeats for expensive benchmarks

### Current repo status

- the current benchmark snapshots are mostly single-seed
- next infra expansion should standardize seeds `13`, `37`, `101`

## 8. Execution Stages

### Stage 1: Sanity screening

Full ambition:

- all baselines
- 30M
- 2k
- 5B tokens

Repo-ready substitute:

- existing small-model benchmark suites
- eliminate unstable configs and dead-end ablations

### Stage 2: Main benchmark

Full ambition:

- all baselines at 90M / 220M / 500M

Repo-ready substitute:

- TWR / Transformer / Mamba benchmark suite
- compression and hierarchy ablation suites

### Stage 3: Extreme length benchmark

Full ambition:

- strongest baselines only
- 8k / 32k / 128k

Repo-ready substitute:

- RULER scaling suite at 256 / 512 / 1024
- extend to larger lengths once loaders and hardware path are expanded

### Stage 4: Internal TWR ablations

Repo-ready now:

- compression suite
- hierarchy suite

Still missing:

- named no-overlap / no-hierarchy / no-multiscale variants

## 9. Current Evidence

### Core lean TWR results

| benchmark | score | params |
| --- | ---: | ---: |
| ListOps | `0.2344` | `69,200` |
| RULER needle | `0.0977` | `88,470` |
| LongBench TREC | `0.2000` | `482,488` |
| Hyperpartisan | `0.5275` | `462,920` |

### Compression study conclusion

Naive stride/window changes were not wins in the current repo.

### RULER scaling snapshot

| model | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: |
| TWR lean | `0.0859` | `0.0996` | `0.0840` |
| Transformer | `0.0879` | `0.0801` | `0.1133` |
| Mamba | `0.0898` | `0.0781` | `0.0723` |

### Hierarchy study snapshot

| run | score | params | interpretation |
| --- | ---: | ---: | --- |
| `twr_backbone_lra_listops_lean` | `0.2344` | `69,200` | current best text-lean baseline |
| `twr_backbone_lra_listops_lean_think2` | `0.2109` | `69,072` | too shallow |
| `twr_backbone_lra_listops_lean_think6` | `0.2227` | `69,328` | deeper is not better |
| `twr_backbone_lra_listops_lean_queries2` | `0.2305` | `67,536` | useful small-model efficiency variant |
| `twr_backbone_ruler_needle_lean` | `0.0977` | `88,470` | current long-lean baseline |
| `twr_backbone_ruler_needle_lean_think4` | `0.0918` | `88,342` | worse |
| `twr_backbone_ruler_needle_lean_think8` | `0.0938` | `88,598` | worse |
| `twr_backbone_ruler_needle_lean_queries4` | `0.0762` | `86,038` | much worse |

## 10. Success Criteria

### Repo-realistic near-term criteria

1. TWR remains competitive with Transformer on the four-task core suite.
2. TWR wins at least one long-context benchmark at equal or smaller parameter count.
3. TWR shows at least one useful efficiency knob that lowers params with minor quality loss.
4. Scaling and ablation studies explain where TWR wins and where it fails.

### Full paper criteria

Use the larger benchmark ambition only after the missing model families and loaders are implemented.

## 11. Next Work

Priority order:

1. add broader LongBench coverage
2. add one stronger long-context reasoning benchmark
3. add one retrieval benchmark
4. expand to repeated-seed execution
5. only then move toward the large 30M / 90M / 220M / 500M grid

## 12. Execution References

- [master_experiment_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/master_experiment_plan.md)
- [paper_benchmark_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/paper_benchmark_plan.md)
- [mamba_comparison_plan.md](/home/dause/Desktop/TWR-LM/experiments/analysis/mamba_comparison_plan.md)
- [experiment_status.json](/home/dause/Desktop/TWR-LM/experiments/analysis/experiment_status.json)
