# AGENTS.md

## Project
TWR-LM (Tokenless Write-Think-Read Latent Machine)

## Goal
Implement a research codebase for a tokenless sequence model that:
1. Encodes input tokens as transient events.
2. Writes events into a small latent memory bank.
3. Discards token representations after write.
4. Runs iterative computation only on latent memory.
5. Uses adaptive think depth and slot-wise gating.
6. Supports controlled experiments, baselines, ablations, and reproducible logging.

## Non-negotiable architecture constraints
- Do **not** keep token-wise hidden states as the persistent model state after the write phase.
- The persistent state after write must be only the latent memory tensor.
- The v1 model should prioritize simplicity and debuggability over novelty.
- The codebase must be structured for experiments first, not demos first.
- All new code should be typed where practical and written in clear PyTorch.
- Prefer small, composable modules over large monolithic files.

## v1 model specification
- Embedding dim: 128
- Event encoder: 1-layer MLP or linear + nonlinearity
- Memory slots: 16
- Slot dim: 128
- Write: sequential soft write
- Think steps: max 4
- Refine block: LayerNorm -> low-rank slot mixer -> GLU MLP -> residual
- Step gate: scalar sigmoid per step
- Slot gate: slot-wise sigmoid
- Readout: mean pool over slots + linear classifier

## Experimental priorities
Priority order:
1. Make the model train end-to-end on a small synthetic/classification task.
2. Add fixed-think variant.
3. Add adaptive step gate.
4. Add adaptive slot gate.
5. Add logging for effective depth and slot usage.
6. Add baselines under a unified training interface.
7. Add ablation configs and reproducible scripts.

## Required repository qualities
- Clear `src/`, `configs/`, `scripts/`, `experiments/`, `tests/` layout.
- YAML configs for model/data/train/experiment presets.
- One training entrypoint that works across TWR and baselines.
- Deterministic seed control where possible.
- Metric logging to stdout and structured files.
- Save checkpoints and a compact experiment summary JSON.

## Logging requirements
Every training/eval run should log at minimum:
- loss
- accuracy/F1 when applicable
- parameter count
- estimated FLOPs if feasible, otherwise a clearly labeled approximation hook
- throughput
- peak GPU memory if CUDA is available
- average effective depth
- per-step gate statistics
- average active slots
- slot usage histogram or summary

## Baseline requirements
Implement these with the same training interface:
- Transformer encoder baseline
- Perceiver-style latent baseline
- Optional Mamba/SSM placeholder interface if full implementation is deferred

## Ablations to support
- no_think
- fixed_think
- adaptive_depth
- no_slot_gate
- token_residual_access_variant
- slots: 8 / 16 / 32
- slot_dim: 64 / 128 / 192
- think_steps: 2 / 4 / 6
- write variants: soft_write / stronger_residual_write / pooled_one_shot_write

## Code style
- Keep functions short.
- Prefer descriptive names.
- Add docstrings to public modules/classes.
- Avoid hidden magic behavior.
- Fail loudly on config mismatches.
- Add TODO comments only when tied to a concrete next step.

## Working style
When implementing:
- First create skeletons and interfaces.
- Then make the smallest trainable path work.
- Then add instrumentation.
- Then add baselines and ablations.
- Then clean up and add tests.

## What to avoid
- Do not overengineer the first version.
- Do not add fancy routing libraries before the basic refine path works.
- Do not introduce token residual access in the core TWR path except as an ablation.
- Do not bury the write-think-read separation under overly generic abstractions.
