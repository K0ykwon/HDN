# Prompt 03 — Add Training and Logging

Implement the training and evaluation pipeline for TWR-LM.

## Requirements
- One trainer that works from configs.
- Train/eval loops with checkpointing.
- Structured logging to JSON or CSV plus readable console logs.
- Seed control.
- GPU detection and memory logging if available.
- Throughput logging.

## Metrics to log
- loss
- accuracy and/or F1
- parameter count
- throughput
- peak GPU memory
- average effective depth
- per-step gate stats
- average active slots

## Deliverable
A working training script plus at least one example config and one example command in README.
