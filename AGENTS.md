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
- Do not keep token-wise hidden states as the persistent model state after the write phase.
- The persistent state after write must be only the latent memory tensor.
- The v1 model should prioritize simplicity and debuggability over novelty.
- The codebase must be structured for experiments first, not demos first.
- All new code should be typed where practical and written in clear PyTorch.
- Prefer small, composable modules over large monolithic files.
