# Master Experiment Plan

## Objective

Establish a paper-ready experimental program for TWR-LM that tests the practical value of a `token-free`, `attention-free` latent backbone against Transformer and Mamba baselines.

The plan is structured so that each phase answers one concrete paper question:

1. Is TWR competitive on the current benchmark core?
2. Where does compression fail?
3. Does the latent hierarchy help on long contexts?
4. Does TWR scale better in length or memory than token-persistent baselines?
5. What claims are justified for the first paper version?

## Non-Negotiable Evaluation Rules

- No separate pretraining as the main solution path.
- Keep all mainline comparisons end-to-end.
- Preserve `token-free after compression` and `attention-free backbone/readout`.
- Always compare against both `Transformer` and `Mamba` where feasible.
- Report accuracy together with parameter count and throughput.

## Phase Structure

### Phase 0: Current Baseline Snapshot

Goal:
Lock the current reproducible baseline before broader expansion.

Required runs:

- `twr_backbone_lra_listops_lean`
- `twr_backbone_ruler_needle_lean`
- `twr_backbone_longbench_trec_lean`
- `twr_backbone_hyperpartisan_lean`
- matching Transformer baselines
- matching Mamba baselines

Status:

- Clean rerun completed for TWR, Transformer, and Mamba on the four-task core suite.

Current lean TWR snapshot:

| benchmark | run | score | params |
| --- | --- | ---: | ---: |
| ListOps | `twr_backbone_lra_listops_lean` | `0.2344` | `69,200` |
| RULER needle | `twr_backbone_ruler_needle_lean` | `0.0977` | `88,470` |
| LongBench TREC | `twr_backbone_longbench_trec_lean` | `0.2000` | `482,488` |
| Hyperpartisan | `twr_backbone_hyperpartisan_lean` | `0.5275` | `462,920` |

### Phase 1: Compression Bottleneck

Goal:
Determine whether the main limitation comes from early token-to-latent compression.

Primary variables:

- `window_size`
- `stride`
- latent count implied by compression

Required runs:

- `twr_backbone_lra_listops_lean_stride1`
- `twr_backbone_lra_listops_lean_window6`
- `twr_backbone_lra_listops_lean_window3_stride1`
- `twr_backbone_ruler_needle_lean_window6`
- `twr_backbone_ruler_needle_lean_window10`

Status:

- Completed.

Findings so far:

| benchmark | run | score | verdict |
| --- | --- | ---: | --- |
| ListOps | `twr_backbone_lra_listops_lean_stride1` | `0.1816` | worse |
| ListOps | `twr_backbone_lra_listops_lean_window6` | `0.1777` | worse |
| ListOps | `twr_backbone_lra_listops_lean_window3_stride1` | `0.1484` | worse |
| RULER needle | `twr_backbone_ruler_needle_lean_window6` | `0.0762` | worse |
| RULER needle | `twr_backbone_ruler_needle_lean_window10` | `0.0820` | worse |
| LongBench TREC | `twr_backbone_longbench_trec_lean_window6` | `0.1800` | worse |
| Hyperpartisan | `twr_backbone_hyperpartisan_lean_window6` | `0.5200` | worse |

Interpretation:

- Naive compression changes do not improve the current mainline.
- The bottleneck is real, but simple overlap/window sweeps are not a sufficient fix.

### Phase 2: Latent Hierarchy Study

Goal:
Measure whether the hierarchical latent backbone itself is doing useful work beyond the encoder.

Primary variables:

- `think_steps`
- `num_readout_queries`
- multiscale readout vs reduced readout

Current status:

- Completed via clean rerun of the configured hierarchy sweep.
- The current lean default remains the most stable `ListOps` setting.
- `queries=2` is the best smaller `ListOps` variant, but the same direction does not carry over to `RULER`.

Next runs:

- `think_steps` sweep for text lean
- `think_steps` sweep for long lean
- reduced-query readout ablation on long tasks

Configured runs:

- `twr_backbone_lra_listops_lean_think2`
- `twr_backbone_lra_listops_lean_think6`
- `twr_backbone_lra_listops_lean_queries2`
- `twr_backbone_ruler_needle_lean_think4`
- `twr_backbone_ruler_needle_lean_think8`
- `twr_backbone_ruler_needle_lean_queries4`

Current rerun findings:

| benchmark | run | score | verdict |
| --- | --- | ---: | --- |
| ListOps | `twr_backbone_lra_listops_lean_think2` | `0.2109` | worse |
| ListOps | `twr_backbone_lra_listops_lean_think6` | `0.2109` | worse |
| ListOps | `twr_backbone_lra_listops_lean_queries2` | `0.2305` | near-parity, smaller |
| RULER needle | `twr_backbone_ruler_needle_lean_think4` | `0.0918` | worse |
| RULER needle | `twr_backbone_ruler_needle_lean_think8` | `0.0938` | worse |
| RULER needle | `twr_backbone_ruler_needle_lean_queries4` | `0.0762` | worse |

### Phase 3: Length Extrapolation

Goal:
Test whether TWR degrades more gracefully as context length increases.

Required benchmarks:

- `RULER`
- one stronger long-context reasoning benchmark such as `BABILong` or `InfiniteBench`

Protocol:

- train at `256`
- test at `512`, `1k`, `2k`

Status:

- Initial `RULER` scaling sweep is completed for `256 / 512 / 1024`.

Current `RULER` scaling snapshot:

| model | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: |
| TWR lean | `0.0859` | `0.0996` | `0.0840` |
| Transformer | `0.0879` | `0.0801` | `0.1133` |
| Mamba | `0.0898` | `0.0781` | `0.0723` |

### Phase 4: Broader Benchmark Coverage

Goal:
Move from a four-benchmark snapshot to a paper-grade suite.

Add:

- more than one `LongBench` subset
- one stronger long-range reasoning benchmark: `BABILong` or `InfiniteBench`
- one retrieval benchmark: `RepoQA` or `BRIGHT`

Status:

- Initial benchmark expansion rerun completed.

Current added benchmark:

| benchmark | TWR lean | Transformer | Mamba |
| --- | ---: | ---: | ---: |
| LongBench LSHT | `0.1000` | `0.0000` | `0.1000` |

### Phase 5: Efficiency And Practicality

Goal:
Measure whether parameter efficiency translates into practical usability.

Required reporting:

- throughput
- peak memory
- approximate FLOPs
- score under equal-parameter budgets
- score under equal-throughput budgets if feasible

Status:

- Partial.
- Throughput and memory are already logged in current runs.
- Equal-throughput study is not yet implemented.

## Execution Order

The current recommended order is:

1. lock the best current lean baseline
2. finish hierarchy ablations
3. expand benchmark coverage
4. run length extrapolation
5. produce paper tables and plots

## Paper Readiness Criteria

The first paper draft is ready when all of the following are true:

- the four-task core benchmark suite is stable and documented
- at least one stronger long-context benchmark beyond the current four is added
- compression and hierarchy ablations are complete
- efficiency reporting is included
- the paper claim is limited to what the current experiments truly support

## Commands

Core lean suite:

```bash
python3 scripts/run_benchmark_lean_twr.py
```

Compression ablation suite:

```bash
python3 scripts/run_compression_ablation_suite.py
```

Analysis refresh:

```bash
python3 scripts/analyze_results.py
```
