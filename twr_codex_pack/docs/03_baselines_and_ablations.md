# Baselines and Ablations

## Baselines
### Transformer encoder
Purpose:
- canonical token-persistent baseline

### Perceiver-style latent baseline
Purpose:
- compare against latent bottleneck models without the same tokenless write-think-read identity

### Mamba / SSM
Purpose:
- compare against modern efficient sequence backbones

## Core ablations
- `no_think`
- `fixed_think`
- `adaptive_depth`
- `no_slot_gate`
- `token_residual_access`

## Capacity ablations
- `slots = 8, 16, 32`
- `slot_dim = 64, 128, 192`
- `think_steps = 2, 4, 6`

## Write ablations
- `soft_write`
- `stronger_residual_write`
- `pooled_one_shot_write`

## What to measure for ablations
- task metrics
- parameter count
- throughput
- memory
- average effective depth
- active slot statistics
