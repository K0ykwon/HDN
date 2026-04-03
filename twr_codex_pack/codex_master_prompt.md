# Codex Master Prompt

You are implementing a research codebase for **TWR-LM (Tokenless Write-Think-Read Latent Machine)** in PyTorch.

Your job is to build the repo incrementally, keeping the project scientifically clean and easy to run. Follow `AGENTS.md` strictly.

## High-level concept
TWR-LM is a sequence model where input tokens are converted into transient events, sequentially written into a small latent memory bank, then discarded. All later computation happens only inside the latent memory. Final predictions are read from the refined memory.

## Core identity
- Token representations are **not** persistent state after the write phase.
- Persistent state is **only** the latent memory bank.
- Computation is split into **Write -> Think -> Read**.
- Think-phase compute should support **adaptive depth** and **slot-wise gating**.

## Immediate implementation target
Build a clean v1 that can train on a classification task with:
- embedding dim 128
- 16 memory slots
- slot dim 128
- sequential soft write
- max think steps 4
- refine block = LayerNorm -> low-rank slot mixing -> GLU MLP -> residual
- scalar step gate
- slot-wise sigmoid gate
- mean-pool readout + linear head

## Required repo shape
Create or maintain this structure:

```text
repo/
  AGENTS.md
  README.md
  pyproject.toml
  requirements.txt
  src/
    twr/
      models/
      modules/
      baselines/
      data/
      training/
      eval/
      utils/
  configs/
    model/
    data/
    train/
    experiment/
  scripts/
  experiments/
  tests/
```

## Development order
1. Create repository skeleton and minimal config system.
2. Implement core TWR modules.
3. Implement a single training loop and evaluator.
4. Make a small end-to-end run succeed.
5. Add step-gate and slot-gate logging.
6. Add Transformer and Perceiver-style baselines.
7. Add ablation configs and scripts.
8. Add tests and cleanup.

## Core modules to implement
- Token embedding / event encoder
- Latent memory initialization
- Sequential soft write module
- Refine block
- Think loop with fixed and adaptive variants
- Readout head
- Training runner
- Metrics/logger utilities

## Experimental constraints
- Support at least: no_think, fixed_think, adaptive_depth, no_slot_gate
- Make it easy to swap slot count, slot dim, and think steps from config
- Keep baselines under the same trainer and dataset pipeline

## Output expectations
When you work, prefer to:
- create/edit files directly
- keep configs explicit
- add concise comments
- leave a short progress note in README or experiment docs when meaningful

## Quality bar
The code does not need to be production-grade, but it must be:
- runnable
- readable
- modular
- reproducible enough for research iteration

## Start now
Begin by scaffolding the repository and implementing the smallest trainable TWR path. Then add config files and a single example experiment command.
