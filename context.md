# Current Context

## Repository State

- Repo: `TWR-LM`
- Goal: experiment-first PyTorch scaffold for a tokenless write-think-read latent model.
- Core model path:
  - `src/twr/models/twr_lm.py`
  - `src/twr/modules/event_encoder.py`
  - `src/twr/modules/sequential_write.py`
  - `src/twr/modules/think_loop.py`
  - `src/twr/modules/refine_block.py`
  - `src/twr/modules/readout.py`
- Trainer path:
  - `scripts/train.py`
  - `src/twr/training/trainer.py`
- Config stack:
  - `configs/experiment/*.yaml`
  - `configs/model/*.yaml`
  - `configs/data/*.yaml`
  - `configs/train/*.yaml`

## What Was Read

- The codebase was inspected and summarized.
- The experiment-design document used was:
  - `TWR-LM_전체정리_및_실험설계_대화반영_개정본.txt`
- Important constraint found:
  - The document asks for LRA, RULER or LongBench, and Hyperpartisan-style evaluation.
  - The current repo only implements:
    - synthetic parity
    - synthetic count-compare
    - long synthetic count-compare
    - hashed-token Hugging Face text classification debug path for IMDB

## Environment

- GPU visible and used:
  - `NVIDIA GeForce RTX 5060 Ti`
- Additional GPU present:
  - `NVIDIA GeForce RTX 3060`
- Python dependencies were missing initially.
- Installed with user-site pip:
  - `torch`
  - `torchvision`
  - `datasets`
  - `pytest`
  - `matplotlib`
- Verified:
  - `torch 2.11.0+cu130`
  - CUDA available

## File Changes Made

### 1. Fixed Hugging Face cache path portability

Updated:
- `src/twr/data/datasets.py`

Change:
- Removed hardcoded cache assumptions tied to `/workspaces/HDN/...`.
- Added repo-local default cache resolution via:
  - `REPO_ROOT / experiments / cache / huggingface`
- Added environment override support:
  - `TWR_HF_CACHE_DIR`
  - `TWR_IMDB_ARROW_CACHE_DIR`

Reason:
- IMDB debug runs were failing with a permission error because the old path pointed to an unavailable external workspace path.

### 2. Added aggregate analysis script

Added:
- `scripts/analyze_results.py`

What it does:
- scans `experiments/runs/*`
- writes:
  - `experiments/analysis/run_summary.csv`
  - `experiments/analysis/run_summary.json`
  - `experiments/analysis/summary_report.md`
- generates plots:
  - `experiments/analysis/parity_baselines.png`
  - `experiments/analysis/parity_ablations.png`
  - `experiments/analysis/count_compare.png`
  - `experiments/analysis/count_compare_long.png`
  - `experiments/analysis/imdb_debug.png`
  - `experiments/analysis/accuracy_vs_throughput.png`

## Experiment Cleanup Performed

- Deleted prior experiment outputs before rerunning:
  - `experiments/runs/*`
  - `experiments/validation`

## Experiments Actually Run

All supported configs in the current repo were run on GPU:

### Parity debug / baseline

- `configs/experiment/twr_debug.yaml`
- `configs/experiment/transformer_debug.yaml`
- `configs/experiment/perceiver_debug.yaml`
- `configs/experiment/mamba_placeholder_debug.yaml`

### Parity ablations

- `configs/experiment/ablation_adaptive_depth.yaml`
- `configs/experiment/ablation_fixed_think.yaml`
- `configs/experiment/ablation_no_slot_gate.yaml`
- `configs/experiment/ablation_no_think.yaml`
- `configs/experiment/ablation_token_residual.yaml`
- `configs/experiment/ablation_write_one_shot.yaml`
- `configs/experiment/ablation_write_residual.yaml`

### Count compare

- `configs/experiment/twr_count_compare_adaptive.yaml`
- `configs/experiment/twr_count_compare_fixed.yaml`
- `configs/experiment/twr_count_compare_no_slot_gate.yaml`
- `configs/experiment/twr_count_compare_deeper.yaml`
- `configs/experiment/transformer_count_compare.yaml`
- `configs/experiment/mamba_placeholder_count_compare.yaml`

### Long count compare

- `configs/experiment/twr_long_adaptive.yaml`
- `configs/experiment/twr_long_fixed.yaml`
- `configs/experiment/twr_long_no_think.yaml`

### IMDB debug

- `configs/experiment/twr_imdb_long_debug.yaml`
- `configs/experiment/transformer_imdb_long_debug.yaml`

## High-Level Results

Canonical aggregate report:
- `experiments/analysis/summary_report.md`

Important results:

### Count compare

- `transformer_count_compare`
  - val acc: `0.9980`
  - loss: `0.0049`
  - depth: `1.000`
- `twr_count_compare_adaptive`
  - val acc: `0.9980`
  - loss: `0.0071`
  - depth: `2.549`
- `twr_count_compare_fixed`
  - val acc: `0.9941`
  - loss: `0.0206`
  - depth: `4.000`
- `twr_count_compare_no_slot_gate`
  - val acc: `0.9961`
  - loss: `0.0170`
  - depth: `2.562`
- `twr_count_compare_deeper`
  - val acc: `0.9961`
  - loss: `0.0072`
  - depth: `4.089`
- `mamba_placeholder_count_compare`
  - val acc: `0.8965`
  - loss: `0.2014`

Interpretation:
- On the implemented count-compare benchmark, adaptive TWR matched the Transformer’s best validation accuracy.
- Adaptive TWR used lower effective depth than fixed/deeper TWR variants.
- The current Mamba placeholder is clearly weaker.

### Long count compare

- `twr_long_no_think`
  - val acc: `0.9697`
  - loss: `0.0806`
  - depth: `1.000`
- `twr_long_fixed`
  - val acc: `0.9688`
  - loss: `0.0760`
  - depth: `6.000`
- `twr_long_adaptive`
  - val acc: `0.9688`
  - loss: `0.0775`
  - depth: `5.197`

Interpretation:
- On the current long synthetic task, think depth did not create a clear accuracy advantage.
- The no-think variant remained competitive.

### Parity baseline / ablation takeaways

- `perceiver_debug` was strongest on the tiny parity debug baseline at `0.9062`.
- Default `twr_debug` reached `0.8203`.
- Among parity ablations, `twr_no_slot_gate` performed best at `0.8711`.
- `twr_write_one_shot` and `twr_write_residual` were weak.

Interpretation:
- On the tiny parity setup, the current TWR design is not yet clearly helped by slot gating or extra think depth.
- The write phase design looks more important than the current parity ablation defaults.

### IMDB debug

- `twr_imdb_long_debug`
  - val acc: `1.0000`
  - loss: `0.0003`
  - depth: `1.933`
- `transformer_imdb_long_debug`
  - val acc: `1.0000`
  - loss: `0.0034`
  - depth: `1.000`

Interpretation:
- This is only a tiny 1-epoch debug setting, not a publication-grade text result.
- It shows the text path now runs successfully after the cache path fix.

## Important Limitations

- The full benchmark plan in the TXT document was not executed because the repo does not yet implement:
  - LRA
  - RULER
  - LongBench
  - Hyperpartisan
- So the work completed was:
  - full rerun of the currently supported experiment matrix
  - aggregate analysis
  - visualization

## Current Analysis Artifacts

Directory:
- `experiments/analysis`

Files:
- `run_summary.csv`
- `run_summary.json`
- `summary_report.md`
- `parity_baselines.png`
- `parity_ablations.png`
- `count_compare.png`
- `count_compare_long.png`
- `imdb_debug.png`
- `accuracy_vs_throughput.png`

## Useful Next Steps

1. Implement the missing benchmark adapters and configs for:
   - LRA
   - RULER or LongBench
   - Hyperpartisan
2. Replace the current Mamba placeholder with a stronger real SSM baseline.
3. Revisit the TWR think/slot-gate mechanism because current synthetic results do not show a robust advantage from deeper or gated thinking.
4. Add stronger behavioral tests around:
   - adaptive depth behavior
   - slot usage behavior
   - cache-path portability
5. If desired, promote `scripts/analyze_results.py` into a standard post-run pipeline.

## Update: Benchmark Implementation In Progress

This section records the later work started after the earlier summary above.

### New benchmark paths added

The following benchmark-oriented files were added:

- `configs/data/lra_listops.yaml`
- `configs/data/ruler_needle.yaml`
- `configs/data/longbench_trec.yaml`
- `configs/data/hyperpartisan.yaml`
- `configs/train/benchmark.yaml`
- `configs/experiment/twr_lra_listops.yaml`
- `configs/experiment/transformer_lra_listops.yaml`
- `configs/experiment/twr_ruler_needle.yaml`
- `configs/experiment/transformer_ruler_needle.yaml`
- `configs/experiment/twr_longbench_trec.yaml`
- `configs/experiment/transformer_longbench_trec.yaml`
- `configs/experiment/twr_hyperpartisan.yaml`
- `configs/experiment/transformer_hyperpartisan.yaml`
- `scripts/run_benchmark_suite.py`

### Code changes made for benchmark support

Updated:
- `src/twr/data/datasets.py`
- `src/twr/training/trainer.py`
- `scripts/analyze_results.py`

What was implemented:

1. `LRA`:
   - Added a synthetic `lra_listops` dataset path.
   - This is a ListOps-style long-range classification generator compatible with the current classification-only trainer.

2. `RULER`:
   - Added a synthetic `ruler_needle` dataset path.
   - This is a RULER-style long-context key-value retrieval classification task.

3. `LongBench`:
   - Added direct zip-backed loading for `THUDM/LongBench` via `huggingface_hub`.
   - Used `trec` from LongBench as the first supported task because it is a fixed-class classification task and fits the current classifier head.
   - The loader avoids the current `datasets` package issue where `LongBench.py` script-based loading fails.

4. `Hyperpartisan`:
   - Added a streaming Hugging Face dataset path for `pietrolesci/hyperpartisan_news_detection`.
   - Validation split was confirmed to be `validation`, not `test`.

5. Analysis outputs:
   - Extended `scripts/analyze_results.py` to recognize:
     - `lra`
     - `ruler`
     - `longbench`
     - `hyperpartisan`
     - `smoke`
   - Added per-run training curve image generation and benchmark summary/table image generation.

### Parameter-count constraint check

The intended benchmark comparison uses:
- `configs/model/twr_text_small.yaml`
- `configs/model/transformer_text_small.yaml`

A direct parameter-count probe for a 512-length text setup showed:
- `twr`: `675,204`
- `transformer`: `624,130`

So the baseline Transformer is smaller than TWR, satisfying the stated comparison constraint.

### Dataset inspection results

Confirmed during implementation:

- LongBench repo contains:
  - `data.zip`
  - `data/trec.jsonl`
  - `data/passage_count.jsonl`
- `passage_count` answers are variable integers and do not expose a fixed class list, so it was not used as the first classification integration target.
- `trec` exposes fixed `all_classes`, so it was chosen for the current LongBench integration.
- Hyperpartisan streaming load works with:
  - dataset: `pietrolesci/hyperpartisan_news_detection`
  - train split: `train`
  - validation split: `validation`

### Smoke checks completed

Targeted dataloader smoke checks succeeded for:
- `lra_listops`
- `ruler_needle`
- `longbench/trec`
- `hyperpartisan`

Small end-to-end smoke training runs also succeeded:

- `smoke_twr_lra_listops`
  - epoch 1 val acc: `0.09375`
- `smoke_transformer_longbench_trec`
  - epoch 1 val acc: `0.125`

Note:
- `pytest -q` did not run cleanly because the environment did not expose `src` on `PYTHONPATH` during test collection.
- The failure was:
  - `ModuleNotFoundError: No module named 'src'`
- This appears to be an environment/test invocation issue, not a benchmark-code import issue, because direct `PYTHONPATH=.` smoke checks succeeded.

### Full benchmark suite run attempt

The new suite script was launched:
- `PYTHONPATH=. python3 scripts/run_benchmark_suite.py`

Completed runs before interruption/failure:

1. `twr_lra_listops`
   - val acc: `0.1035`
   - val loss: `2.3368`
   - params: `137100`

2. `transformer_lra_listops`
   - val acc: `0.0820`
   - val loss: `2.3024`
   - params: `86026`

3. `twr_ruler_needle`
   - val acc: `0.0801`
   - val loss: `2.9105`
   - params: `168210`

4. `transformer_ruler_needle`
   - val acc: `0.0645`
   - val loss: `2.7753`
   - params: `117136`

5. `twr_longbench_trec`
   - val acc: `0.1800`
   - val loss: `4.4057`
   - params: `678324`

### Failure encountered during full suite

The suite failed on:
- `configs/experiment/transformer_longbench_trec.yaml`

Observed runtime error:

```text
RuntimeError: stack expects each tensor to be equal size, but got [8] at entry 0 and [6] at entry 18
```

Root cause:
- `TransformerEncoderBaseline.forward()` currently returns:
  - `slot_histogram: ones.squeeze(-1)`
  - `think_slot_histogram: ones.squeeze(-1)`
- That makes the histogram shape depend on the batch size.
- `trainer.run_epoch()` expects a fixed-width histogram across batches and stacks them with:
  - `torch.stack(slot_histograms)`
- The last short batch on LongBench caused the stack failure.

### Important state at interruption

- The attempted Transformer histogram-shape bug fix was started but **was not applied** because the turn was interrupted.
- Current transformer file still has the old batch-size-dependent histogram outputs:
  - `src/twr/baselines/transformer_encoder.py`

### Immediate next action

Resume from here:

1. Fix `src/twr/baselines/transformer_encoder.py` so `slot_histogram` and `think_slot_histogram` are fixed-size tensors independent of batch size.
2. Re-run:
   - `PYTHONPATH=. python3 scripts/run_benchmark_suite.py`
3. Confirm remaining runs finish:
   - `transformer_longbench_trec`
   - `twr_hyperpartisan`
   - `transformer_hyperpartisan`
4. Re-run or finish `scripts/analyze_results.py` so all requested image outputs are regenerated from the completed suite.
