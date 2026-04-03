# Prompt 05 — Add Ablations

Add ablation variants and configs for TWR-LM.

## Ablations
- no_think
- fixed_think
- adaptive_depth
- no_slot_gate
- token_residual_access_variant
- slots: 8 / 16 / 32
- slot_dim: 64 / 128 / 192
- think_steps: 2 / 4 / 6
- write variants: soft_write / stronger_residual_write / pooled_one_shot_write

## Requirements
- Minimal code duplication
- Config-driven toggles
- Experiment naming that is easy to parse later

## Deliverable
Ablation-ready configs and scripts that can launch sweeps or manual runs.
