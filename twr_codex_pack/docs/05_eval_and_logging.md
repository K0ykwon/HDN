# Evaluation and Logging Spec

## Task metrics
- loss
- accuracy
- F1 where relevant

## Efficiency metrics
- parameter count
- estimated FLOPs or a clearly labeled approximation
- throughput
- peak GPU memory

## Structural metrics
- average effective depth
- step gate values per layer/step
- average active slots
- slot activation histogram or usage summary
- difficulty vs depth correlation when labels or proxy buckets exist

## Run artifacts
Each run should save:
- config snapshot
- metrics JSON/CSV
- checkpoint(s)
- summary JSON
- optional plots for depth and slot activity

## Minimum summary JSON fields
- run_name
- model_name
- dataset_name
- seed
- final_train_loss
- final_val_loss
- final_val_accuracy
- final_val_f1
- parameter_count
- avg_effective_depth
- avg_active_slots
- throughput
- peak_gpu_memory_mb
